#!/usr/bin/env bash

set -x
mkdir logs
mkdir logs/speed_eval

PARTITION=VC
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MODEL=$1

srun -p ${PARTITION} \
    --gres=gpu:${GPUS_PER_NODE} \
    --quotatype=spot \
    python speed_eval.py --model_type ${MODEL} 1>"logs/speed_eval/${MODEL}.out"
    