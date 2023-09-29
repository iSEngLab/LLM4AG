NJOBS=${1:-10}
TOTAL=10
BUGGY_GEN_DIR="data/evosuite_buggy_regression_all"
BUGGY_TEST_DIR="data/evosuite_buggy_tests"
for i in `seq 1 ${TOTAL}`;do
    python toga.py ${BUGGY_TEST_DIR}/${i}/inputs.csv ${BUGGY_TEST_DIR}/${i}/meta.csv GraphCodeBERT
done