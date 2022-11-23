#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

model_list=('halo_v2' 'pvt' 'swin' 'convnext_v3' 'dcn_v3')
scale_list=('micro' 'tiny' 'small' 'base')

for model_name in "${model_list[@]}"
do
    for scale in "${scale_list[@]}"
    do
        srun -p ${PARTITION} \
            --gres=gpu:${GPUS_PER_NODE} \
            --quotatype=spot \
            python calculate_mac_param.py --model_name "unified_${model_name}_${scale}"
    done
done
