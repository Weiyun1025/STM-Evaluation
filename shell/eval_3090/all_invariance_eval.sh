#!/usr/bin/env bash

set -x
mkdir logs

MODEL="conv_swin_tiny" 
CKPT_DIR="./backbone_outputdir/conv_swin_tiny_1k_unified_config_3090"

DESC="eval_invariance" 

# key hyperparameters
TOTAL_BATCH_SIZE="64"

PROJECT_NAME="${MODEL}_1k_${DESC}"

GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

CPUS_PER_TASK=${CPUS_PER_TASK:-12}

VARIANCE_TYPE=('translation' 'pre_rotation' 'post_rotation' 'scale')
CKPT_NAME=('49' '99' '149' '199' '249' '299' 'best')

for variance in "${VARIANCE_TYPE[@]}"
do
    for ckpt in "${CKPT_NAME[@]}"
    do
        torchrun \
            --nnodes=1 \
            --nproc_per_node=${GPUS_PER_NODE} \
            invariance_eval_all.py \
            --model ${MODEL} \
            --variance_type ${variance} \
            --resume "${CKPT_DIR}/checkpoint-${ckpt}.pth" \
            --batch_size $((TOTAL_BATCH_SIZE/GPUS_PER_NODE)) \
            --data_path /root/ImageNet \
            --data_on_memory false \
            --use_amp true \
            --output_dir "backbone_outputdir/${PROJECT_NAME}_${ckpt}"
    done
done
