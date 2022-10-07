#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

model_list=('conv_swin' 'conv_halo_v2_timm' 'conv_pvt' 'conv_pvt_v2' 'conv_convnext_v2')
scale_list=('micro' 'tiny' 'small' 'base')

for model_name in ${model_list[@]}
do
    for scale in ${scale_list[@]}
    do
        srun -p ${PARTITION} \
            --gres=gpu:${GPUS_PER_NODE} \
            --quotatype=spot \
            python calculate_mac_param.py --model_name "${model_name}_${scale}"
    done
done
