#!/usr/bin/env bash

set -x
mkdir logs

MODEL="pvt_tiny"
DESC="unified_config_3090" 

# key hyperparameters
TOTAL_BATCH_SIZE="1024"
LR="1e-3"
INIT_LR="1e-6"
END_LR="1e-5"
DROP_PATH="0.1"

PROJECT_NAME="${MODEL}_1k_${DESC}"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}

SRUN_ARGS=${SRUN_ARGS:-""}

torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPUS_PER_NODE} \
    main.py \
    --model ${MODEL} \
    --epochs 300 \
    --batch_size $((TOTAL_BATCH_SIZE/GPUS_PER_NODE)) \
    --warmup_epochs 20 \
    --lr ${LR} \
    --warmup_init_lr ${INIT_LR} \
    --min_lr ${END_LR} \
    --opt adamw \
    --clip_grad 5.0 \
    --drop_path ${DROP_PATH} \
    --weight_decay 0.05 \
    --layer_scale_init_value 1e-6 \
    --smoothing 0.1 \
    --model_ema true \
    --model_ema_decay 0.9999 \
    --model_ema_eval true \
    --input_size 224 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --mixup_prob 1.0 \
    --mixup_switch_prob 0.5 \
    --aa rand-m9-mstd0.5-inc1 \
    --repeated_aug false \
    --reprob 0.25 \
    --color_jitter 0.4 \
    --crop_pct 0.875 \
    --data_set IMNET1k \
    --data_path /root/ImageNet \
    --data_on_memory false \
    --nb_classes 1000 \
    --use_amp true \
    --save_ckpt true \
    --enable_wandb true \
    --project 'model evaluation' \
    --name ${PROJECT_NAME} \
    --output_dir "backbone_outputdir/${PROJECT_NAME}" \
    1>"logs/${PROJECT_NAME}.out" 2>"logs/${PROJECT_NAME}.err"
