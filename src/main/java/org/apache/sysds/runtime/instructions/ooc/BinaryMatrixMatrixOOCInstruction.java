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

import java.util.concurrent.ExecutorService;

import org.apache.sysds.common.Opcodes;
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
		CPOperand in2 = new CPOperand(parts[2]); // the smaller matrix (streamed)
		CPOperand out = new CPOperand(parts[3]);

		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator ba = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);

		return new BinaryMatrixMatrixOOCInstruction(OOCType.MAPMM, ba, in1, in2, out, opcode, str);
	}

	@Override
	public void processInstruction( ExecutionContext ec ) {
		// 1. Identify the inputs
		MatrixObject in1 = ec.getMatrixObject(input1); // big matrix
		MatrixObject in2 = ec.getMatrixObject(input2); // smaller matrix

		long cols1 = in1.getDataCharacteristics().getNumColBlocks();
		long cols2 = in2.getDataCharacteristics().getNumColBlocks();

		// caches each block of the 2nd input only until that block is no longer needed
		OOCMatrixBlockTracker in2Tracker = new OOCMatrixBlockTracker(in1.getDataCharacteristics().getNumRowBlocks());
		// collects partial aggregation blocks until the block is ready to be emitted
		OOCMatrixBlockTracker aggTracker = new OOCMatrixBlockTracker(cols1);

		LocalTaskQueue<IndexedMatrixValue> qIn1 = in1.getStreamHandle();
		LocalTaskQueue<IndexedMatrixValue> qIn2 = in2.getStreamHandle();
		LocalTaskQueue<IndexedMatrixValue> qOut = new LocalTaskQueue<>();
		BinaryOperator plus = InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString());
		ec.getMatrixObject(output).setStreamHandle(qOut);

		ExecutorService pool = CommonThreadPool.get();
		try {
			// Core logic: background thread
			pool.submit(() -> {
				IndexedMatrixValue tmp1 = null;
				IndexedMatrixValue tmp2 = null;
				try {
					while((tmp1 = qIn1.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
						MatrixBlock block1 = (MatrixBlock) tmp1.getValue();
						long r1 = tmp1.getIndexes().getRowIndex();
						long c1 = tmp1.getIndexes().getColumnIndex();

						// aggregation with all corresponding blocks
						for (int j=0; j < cols2; j++) {
							// r2 == c1
							long key2 = (c1-1) * cols2 + j;
							MatrixBlock block2 = in2Tracker.get(key2);
							if(block2 == null) {
								// corresponding block still in 2nd queue, dequeue until found
								while((tmp2 = qIn2.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
									block2 = (MatrixBlock) tmp2.getValue();
									long r2 = tmp2.getIndexes().getRowIndex();
									long c2 = tmp2.getIndexes().getColumnIndex();
									long key = (r2-1) * cols2 + (c2-1);
									// store all dequeued blocks in cache
									in2Tracker.putAndInitCount(key, block2);
									// found corresponding block
									if (key == key2) break;
								}
							}

							// Now, call the operation with the correct, specific operator.
							MatrixBlock partialResult = block1.aggregateBinaryOperations(block1, block2,
								new MatrixBlock(), (AggregateBinaryOperator) _optr);

							// early discard: remove block once it has been used for all of its corresponding aggregations
							if(in2Tracker.incrementCount(key2)) in2Tracker.remove(key2);

							// for single column block, no aggregation neeeded
							if(cols1 == 1) {
								qOut.enqueueTask(new IndexedMatrixValue(tmp1.getIndexes(), partialResult));
							}
							else {
								// index in the aggregated block: (r1, j+1)
								long aggKey = (r1-1) * cols2 + j;
								// accumulate partial aggregations
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
