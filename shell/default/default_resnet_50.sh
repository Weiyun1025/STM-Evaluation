#!/usr/bin/env bash
# resnet-50 using RSB-ResNet config
# NOTE:
# for RSB-ResNet: bs=2048, lr=5e-3, weight_decay=0.02, opt=LAMB, droppath=0.05, test_crop_ratio=0.95, warm_up_epoch=5
# however, we retain the data augmentation to be the same as swin and convnext
# rand aug: modified to m7 std0.5, mixup alpha: 0.1 -> 0.8

set -x
mkdir logs

PARTITION=VC
MODEL="resnet50"
DESC="pt_224_bs2048_rsb_training_recipe_conv_aug" 

JOB_NAME=${MODEL}
PROJECT_NAME="${MODEL}_1k_${DESC}"

GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
QUOTA_TYPE="auto"

CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}

# todo: dropout
srun -p ${PARTITION} \
    -x SH-IDC1-10-140-24-110 \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA_TYPE} \
    --async \
    --output="logs/${PROJECT_NAME}.out" \
    --error="logs/${PROJECT_NAME}.err" \
    ${SRUN_ARGS} \
    python -u main.py \
    --model ${MODEL} \
    --epochs 300 \
    --batch_size 512 \
    --warmup_epochs 5 \
    --lr 5e-3\
    --warmup_init_lr 0 \
    --min_lr 1e-6 \
    --opt lamb \
    --drop_path 0.05 \
    --weight_decay 0.02 \
    --layerscale_opt false \
    --smoothing 0.1 \
    --model_ema true \
    --model_ema_decay 0.9999 \
    --model_ema_eval true \
    --input_size 224 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --mixup_prob 1.0 \
    --mixup_switch_prob 0.5 \
    --aa rand-m7-mstd0.5-inc1 \
    --repeated_aug false \
    --reprob 0.25 \
    --color_jitter 0.4 \
    --crop_pct 0.95 \
    --data_set IMNET1k \
    --data_path /mnt/cache/share/images/ \
    --nb_classes 1000 \
    --use_amp false \
    --save_ckpt true \
    --output_dir "backbone_outputdir/${PROJECT_NAME}"

# sh train.sh
