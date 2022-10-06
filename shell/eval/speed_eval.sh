#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

export CUDA_LAUNCH_BLOCKING=1

kernprof -l speed_eval.py --model_type $1

srun -p ${PARTITION} \
    --gres=gpu:${GPUS_PER_NODE} \
    --quotatype=spot \
    python -m line_profiler speed_eval.py.lprof
    