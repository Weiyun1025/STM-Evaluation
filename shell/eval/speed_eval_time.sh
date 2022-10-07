#!/usr/bin/env bash

set -x
mkdir logs
mkdir logs/speed_eval

PARTITION=VC
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
EPOCHS=$1

model_list=('conv_swin_tiny' 'conv_halo_v2_timm_tiny' 'conv_halo_v2_mask_tiny' 'conv_halo_v2_mask_out_tiny' 'conv_halo_v2_github_tiny')
for model in ${model_list[@]}
do
    srun -p ${PARTITION} \
        --gres=gpu:${GPUS_PER_NODE} \
        --quotatype=spot \
        kernprof -l speed_eval_time.py --model_type ${model} --epochs ${EPOCHS} \
        1>"logs/speed_eval/${model}.out"
done
