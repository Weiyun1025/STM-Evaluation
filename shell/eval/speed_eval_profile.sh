#!/usr/bin/env bash

set -x
mkdir logs
mkdir logs/profile

PARTITION=VC
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

export CUDA_LAUNCH_BLOCKING=1

function run() {
    name=$1
    size=$2

    srun -p ${PARTITION} \
        --gres=gpu:${GPUS_PER_NODE} \
        --quotatype=spot \
        kernprof -l speed_eval_profile.py --model_type ${name} --halo_size ${size}
    
    srun -p ${PARTITION} \
        --gres=gpu:${GPUS_PER_NODE} \
        --quotatype=spot \
        python -m line_profiler speed_eval.py.lprof 1>"logs/profile/${name}_${size}.out"
}

model_list=('conv_swin_tiny' 'conv_halo_v2_timm_tiny' 'conv_halo_v2_mask_tiny' 'conv_halo_v2_mask_out_tiny' 'conv_halo_v2_github_tiny')
for model_name in ${model_list[@]}
do
    run ${model_name} 3
    run ${model_name} 0
done
