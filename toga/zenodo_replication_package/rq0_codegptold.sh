NJOBS=${1:-10}
TOTAL=10
BUGGY_GEN_DIR="data/evosuite_buggy_regression_all"
BUGGY_TEST_DIR="data/evosuite_buggy_tests"

date

 generate regression tests


for i in `seq 2 4`;do
    python toga.py ${BUGGY_TEST_DIR}/${i}/inputs.csv ${BUGGY_TEST_DIR}/${i}/meta.csv CodeGPTold
done

date