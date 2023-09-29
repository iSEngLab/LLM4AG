#!/bin/bash

N_JOBS=${1:-10}
START=${2:-1}
END=10
t=buggy
g=naive

for i in `seq ${START} ${END}`;do
    echo ${i}
    echo ${t}
    echo ${g}
    bash run_exp_rq3.sh data/evosuite_${t}_tests/${i} data/evosuite_${t}_regression_all/${i} ${N_JOBS} ${g}
    rm -rf /tmp/run_bug_detection.pl_*
done
