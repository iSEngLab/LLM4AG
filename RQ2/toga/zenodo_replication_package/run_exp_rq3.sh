#!/bin/bash
set -u -e -x


DATA_DIR=$1
REG_DIR=$2
N_JOBS=$3
GENERATOR=$4

if [ ! -d ${DATA_DIR} ]
then
    mkdir ${DATA_DIR}
fi

MODEL_NAME=${GENERATOR}
OUT_DIR=${DATA_DIR}/${MODEL_NAME}_generated
D4J_PATH=/data/swf/defects4j/framework/projects


python eval/gen_tests_from_metadata.py ${DATA_DIR}/${MODEL_NAME}_oracle_preds.csv ${REG_DIR} ${OUT_DIR} --test_harness data/test_harness --d4j_path ${D4J_PATH} --generator ${GENERATOR}
python eval/aggregate_test_cases.py ${OUT_DIR} --generator ${GENERATOR}
python eval/eval_tests.py ${OUT_DIR}/aggregated_d4j_tests/ -o ${OUT_DIR}/results/ -n ${N_JOBS}
python eval/fix_failed_tests.py ${OUT_DIR}/aggregated_d4j_tests ${OUT_DIR}/results ${OUT_DIR}/fixed_d4j_tests -m ${GENERATOR}
python eval/eval_tests.py ${OUT_DIR}/fixed_d4j_tests/ -o ${OUT_DIR}/fixed_results/ -n ${N_JOBS}
python eval/merge_results.py ${OUT_DIR}/results ${OUT_DIR}/fixed_results ${OUT_DIR}/merged_results
 # collect both old results and merged results
python eval/collect_test_results.py ${OUT_DIR} results
python eval/collect_test_results.py ${OUT_DIR} merged_results
python -m eval.postprocess_test_results ${OUT_DIR} results ${MODEL_NAME}
python -m eval.postprocess_test_results ${OUT_DIR} merged_results ${MODEL_NAME}

# remove all temp directories
rm -rf /tmp/run_bug_detection.pl_*
