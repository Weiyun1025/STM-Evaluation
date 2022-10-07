#!/usr/bin/env bash

set -x
mkdir logs
mkdir logs/profile_single

PARTITION=VC
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

export CUDA_LAUNCH_BLOCKING=1

function run() {
    name=$1

    srun -p ${PARTITION} \
        --gres=gpu:${GPUS_PER_NODE} \
        --quotatype=spot \
        kernprof -l speed_eval.py --model_type ${name}
    
    srun -p ${PARTITION} \
        --gres=gpu:${GPUS_PER_NODE} \
        --quotatype=spot \
        python -m line_profiler speed_eval.py.lprof 1>"logs/profile_single/${name}.out"
}

run $1
