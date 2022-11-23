# Requirements

The code requires to install PyTorch 1.12, torch vision, timm and MultiScaleDeformableAttention.

To install the MultiScaleDeformableAttention

```shell
cd ./ops
sh make.sh
```


# Usage

## 1K Pretraining

- To train a unified model on ImageNet-1K with slurm

```shell
# training setting for unified_halonet, unified_swin and unified_pvt
sh shells/1k_pretrain/transformer.sh [model_name]

# training setting for unified_convnext_v3 and unified_dcn_v3
sh shells/1k_pretrain/cnn.sh [model_name]
```

otherwise

```shell
torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
		main.py \
    --model ${MODEL} \
    --epochs 300 \
    --update_freq ${UPDATE_FREQ} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --warmup_init_lr ${INIT_LR} \
    --min_lr ${END_LR} \
    --data_set IMNET1k \
    --data_path /mnt/cache/share/images/ \
    --nb_classes 1000 \
    --output_dir "output_dir"
```

## Evaluation

- To evaluate a unified model on ImageNet-1K with slurm
  - remember to modify the path to the pertained ckpt in the shell script

```shell
sh shells/eval/eval.sh [model_name] [ckpt_path]
```

otherwise

```shell
torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    main.py \
    --model ${MODEL} \
    --eval true \
    --resume ${CKPT} \
    --batch_size ${BATCH_SIZE} \
    --input_size 224 \
    --data_set IMNET1k \
    --data_path /mnt/cache/share/images/ \
    --data_on_memory false \
    --nb_classes 1000 \
    --use_amp false \
```



