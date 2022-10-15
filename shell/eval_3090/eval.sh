#!/usr/bin/env bash

set -x
mkdir logs

CKPT_DIR=$1
MODEL=$2

# key hyperparameters
TOTAL_BATCH_SIZE="1024"


GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

torchrun \
    --nnodes=1 \
    --nproc_per_node="${GPUS_PER_NODE}" \
    main.py \
    --model "${MODEL}" \
    --eval true \
    --resume "${CKPT_DIR}/${CKPT}.pth" \
    --batch_size $((TOTAL_BATCH_SIZE/GPUS_PER_NODE)) \
    --input_size 224 \
    --data_set IMNET1k \
    --data_path /root/ImageNet \
    --data_on_memory false \
    --nb_classes 1000 \
    --use_amp false \
