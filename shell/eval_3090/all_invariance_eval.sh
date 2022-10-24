#!/usr/bin/env bash

set -x
mkdir logs
CKPT_DIR="./backbone_outputdir"

# key hyperparameters
TOTAL_BATCH_SIZE="64"

GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

MODEL=('conv_convnext_v3' 'conv_swin' 'conv_halo_v2' 'dcn_v3' 'conv_pvt')
CKPT_NAME=('49' '99' '149' '199' '249' '299')
VARIANCE_TYPE=('translation' 'pre_rotation' 'post_rotation' 'scale')
for model_type in "${MODEL[@]}"
do
    model_type="${model_type}_tiny"
    for ckpt in "${CKPT_NAME[@]}"
    do
        for variance in "${VARIANCE_TYPE[@]}"
        do
            torchrun \
                --nnodes=1 \
                --nproc_per_node="${GPUS_PER_NODE}" \
                invariance_eval_all.py \
                --model "${model_type}" \
                --variance_type "${variance}" \
                --resume "${CKPT_DIR}/${model_type}/checkpoint-${ckpt}.pth" \
                --batch_size $((TOTAL_BATCH_SIZE/GPUS)) \
                --data_path /root/ImageNet \
                --data_on_memory false \
                --use_amp false \
                --output_dir "backbone_outputdir/invariance_all/${model_type}_${ckpt}"
        done
    done
done
