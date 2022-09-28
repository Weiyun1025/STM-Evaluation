#! /usr/bin/bash

models=("conv_convnext_tiny" "conv_halo_v2_timm_tiny" "conv_pvt_v2_tiny" "conv_swin_micro" )
for name in "${models[@]}"; do
    echo "========================== Model: ${name} ========================="
    for bz in {0..256..16}; do
        bz=$(( bz > 0 ? bz : 1 ))
        PYTHONPATH=. python tools/perf_onnxtrt_tf32_fp16.py \
            -model_name ${name} \
            -input_shape ${bz} 3 224 224
        if [[ $? != 0 ]]; then
            echo "Error occurs"
            break
        fi
        break
    done
done
