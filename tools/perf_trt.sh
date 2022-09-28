#! /usr/bin/bash

models=("resnet50" "convnext_base" "swin-base" "pvt-v2-b2" "pvt-v2-b4")
for name in "${models[@]}"; do
    echo "========================== Model: ${name} Bz: 1 ======================="
    PYTHONPATH=. python tools/perf_onnxtrt.py \
        -config configs/${name}.yaml
    echo "========================== Model: ${name} Bz: 32 ======================"
    PYTHONPATH=. python tools/perf_onnxtrt.py \
        -config configs/${name}.yaml \
        -input_shape 32 3 224 224
done