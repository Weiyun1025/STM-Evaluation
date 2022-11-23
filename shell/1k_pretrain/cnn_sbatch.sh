#!/bin/bash 
#SBATCH -J stm_evaluation
#SBATCH --quotatype=auto
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH -t 7-12:00:00
#SBATCH -p VC
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:16 -N 2

source ~/.bashrc
conda activate torch1.12

python test.py

set -x

MODEL=$1
DESC="unified_config" 
PROJECT_NAME="${MODEL}_1k_${DESC}"

# key hyperparameters
IMAGENET_PATH="/mnt/cache/share/images/"
TOTAL_BATCH_SIZE="4096"
LR="4e-3"
INIT_LR="0"
END_LR="1e-6"
UPDATE_FREQ=1

torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    main.py \
    --model ${MODEL} \
    --epochs 300 \
    --update_freq ${UPDATE_FREQ} \
    --batch_size $((TOTAL_BATCH_SIZE/GPUS/UPDATE_FREQ)) \
    --lr ${LR} \
    --warmup_init_lr ${INIT_LR} \
    --min_lr ${END_LR} \
    --data_set IMNET1k \
    --data_path ${IMAGENET_PATH} \
    --nb_classes 1000 \
    --output_dir "/mnt/petrelfs/wangweiyun/model_evaluation/${PROJECT_NAME}"
    