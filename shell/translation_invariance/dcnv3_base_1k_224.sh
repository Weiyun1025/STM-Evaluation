#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
MODEL="dcn_v3_base"
DESC="unified_config" 

# key hyperparameters
TOTAL_BATCH_SIZE="256"
JOB_NAME=${MODEL}
PROJECT_NAME="${MODEL}_1k_${DESC}"

GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
QUOTA_TYPE="spot"

CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
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
    python -u invariance_test.py \
    --model ${MODEL} \
    --epochs 300 \
    --batch_size $((TOTAL_BATCH_SIZE/GPUS)) \
    --input_size 224 \
    --crop_pct 0.875 \
    --invariance_type translation \
    --translation_strength \
    --rotation_strength \
    --scaling_strength \
    --data_set IMNET1k \
    --data_path /mnt/cache/share/images/ \
    --data_on_memory false \
    --nb_classes 1000 \
    --use_amp true \
    --output_dir backbone_outputdir/${PROJECT_NAME} \
    --resume "/mnt/petrelfs/share_data/shimin/share_checkpoint/dcnv3/dcnv3_base/checkpoint-best.pth"
    #--output_dir "/mnt/petrelfs/${USER}/model_evaluation/${PROJECT_NAME}"
    
# sh ./shell/translation_invariance/dcnv3_base_1k_224.sh