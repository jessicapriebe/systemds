/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.instructions.ooc;

import java.util.HashMap;
import java.util.concurrent.ExecutorService;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class BinaryMatrixMatrixOOCInstruction extends ComputationOOCInstruction {


	protected BinaryMatrixMatrixOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(type, op, in1, in2, out, opcode, istr);
	}

	public static BinaryMatrixMatrixOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 4);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]); // the larget matrix (streamed)
		CPOperand in2 = new CPOperand(parts[2]); // the small matrix (in-memory)
		CPOperand out = new CPOperand(parts[3]);

		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator ba = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);

		return new BinaryMatrixMatrixOOCInstruction(OOCType.MAPMM, ba, in1, in2, out, opcode, str);
	}

	@Override
	public void processInstruction( ExecutionContext ec ) {
		// 1. Identify the inputs
		MatrixObject in1 = ec.getMatrixObject(input1); // big matrix
		MatrixBlock in2 = ec.getMatrixObject(input2)
			.acquireReadAndRelease(); // in-memory matrix

		// 2. Pre-partition the in-memory matrix into a hashmap
		HashMap<Long, MatrixBlock> partitionedMatrix = new HashMap<>();
		int blksize = in1.getDataCharacteristics().getBlocksize();
		if (blksize < 0)
			blksize = ConfigurationManager.getBlocksize();
		long cols2 = (long) Math.ceil((double) in2.getNumColumns() / blksize);
		for (int i=0; i < in2.getNumRows(); i+=blksize) {
			for (int j=0; j < in2.getNumColumns(); j+=blksize) {
				long key = (long) (i/blksize) * cols2 + (long) (j/blksize);
				MatrixBlock slice = in2.slice(i, Math.min(i + blksize, in2.getNumRows())-1, j, Math.min(j + blksize, in2.getNumColumns())-1);
				partitionedMatrix.put(key, slice);
			}
		}

		// number of colBlocks for early block output
		long emissionThreshold = in1.getDataCharacteristics().getNumColBlocks();
		OOCMatrixBlockTracker aggTracker = new OOCMatrixBlockTracker(emissionThreshold);

		LocalTaskQueue<IndexedMatrixValue> qIn = in1.getStreamHandle();
		LocalTaskQueue<IndexedMatrixValue> qOut = new LocalTaskQueue<>();
		BinaryOperator plus = InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString());
		ec.getMatrixObject(output).setStreamHandle(qOut);

		ExecutorService pool = CommonThreadPool.get();
		try {
			// Core logic: background thread
			pool.submit(() -> {
				IndexedMatrixValue tmp = null;
				try {
					while((tmp = qIn.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
						MatrixBlock block1 = (MatrixBlock) tmp.getValue();
						long r1 = tmp.getIndexes().getRowIndex();
						long c1 = tmp.getIndexes().getColumnIndex();

						// aggregation with all corresponding blocks
						for (int j=0; j < cols2; j++) {
							// r2 == c1
							long key2 = (c1-1) * cols2 + j;
							MatrixBlock block2 = partitionedMatrix.get(key2);

							// Now, call the operation with the correct, specific operator.
							MatrixBlock partialResult = block1.aggregateBinaryOperations(block1, block2,
								new MatrixBlock(), (AggregateBinaryOperator) _optr);

							// for single column block, no aggregation neeeded
							if(emissionThreshold == 1) {
								qOut.enqueueTask(new IndexedMatrixValue(tmp.getIndexes(), partialResult));
							}
							else {
								// index in the aggregated block: (r1, j+1)
								long aggKey = (r1-1) * cols2 + j;
								// aggregation
								MatrixBlock currAgg = aggTracker.get(aggKey);
								if(currAgg == null) {
									aggTracker.putAndIncrementCount(aggKey, partialResult);
								}
								else {
									currAgg = currAgg.binaryOperations(plus, partialResult);
									if(aggTracker.putAndIncrementCount(aggKey, currAgg)) {
										// early block output: emit aggregated block
										MatrixIndexes idx = new MatrixIndexes(r1, j+1);
										qOut.enqueueTask(new IndexedMatrixValue(idx, currAgg));
										aggTracker.remove(aggKey);
									}
								}
							}
						}
					}
				}
				catch(Exception ex) {
					throw new DMLRuntimeException(ex);
				}
				finally {
					qOut.closeInput();
				}
			});
		} catch (Exception e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			pool.shutdown();
		}
	}
}
