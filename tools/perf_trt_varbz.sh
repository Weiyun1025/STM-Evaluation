#! /usr/bin/bash

models=("resnet50" "convnext_base" "swin-base" "pvt-v2-b2" "pvt-v2-b4")
for name in "${models[@]}"; do
    echo "========================== Model: ${name} ========================="
    for bz in {0..256..16}; do
        bz=$(( bz > 0 ? bz : 1 ))
        PYTHONPATH=. python tools/perf_onnxtr_tf32_fp16.py \
            -config configs/${name}.yaml \
            -input_shape ${bz} 3 224 224
        if [[ $? != 0 ]]; then
            echo "Error occurs"
            break
        fi
    done
done
