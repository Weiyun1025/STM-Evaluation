#!/usr/bin/env bash

set -x
mkdir logs
mkdir logs/profile

PARTITION=VC
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MODEL_TYPE=$1

export CUDA_LAUNCH_BLOCKING=1

kernprof -l speed_eval.py --model_type ${MODEL_TYPE}

srun -p ${PARTITION} \
    --gres=gpu:${GPUS_PER_NODE} \
    --quotatype=spot \
    python -m line_profiler speed_eval.py.lprof 1>"logs/profile/${MODEL_TYPE}.out"
    