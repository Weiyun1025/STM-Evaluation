#!/usr/bin/env bash

set -x
mkdir logs

MODEL="conv_swin_tiny" 
CKPT_DIR="./backbone_outputdir/conv_swin_tiny_1k_unified_config_3090"

DESC="eval_invariance" 

# key hyperparameters
TOTAL_BATCH_SIZE="1024"

PROJECT_NAME="${MODEL}_1k_${DESC}"

GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

CPUS_PER_TASK=${CPUS_PER_TASK:-12}

CKPT_NAME=('checkpoint-49.pth' 'checkpoint-99.pth' 'checkpoint-149.pth' 'checkpoint-199.pth' 'checkpoint-249.pth' 'checkpoint-299.pth' 'checkpoint-best.pth')

for ckpt in "${CKPT_NAME[@]}"
do
    torchrun \
        --nnodes=1 \
        --nproc_per_node=${GPUS_PER_NODE} \
        invariance_eval_all.py \
        --model ${MODEL} \
        --resume "${CKPT_DIR}/${ckpt}" \
        --batch_size $((TOTAL_BATCH_SIZE/GPUS_PER_NODE)) \
        --data_path /root/ImageNet \
        --output_dir "backbone_outputdir/${PROJECT_NAME}"
done
