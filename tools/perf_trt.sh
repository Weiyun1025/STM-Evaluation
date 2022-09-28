#! /usr/bin/bash

models=("conv_convnext_tiny" "conv_halo_v2_timm_tiny" "conv_pvt_v2_tiny" "conv_swin_micro")
for name in "${models[@]}"; do
    echo "========================== Model: ${name} Bz: 1 ======================="
    PYTHONPATH=. python tools/perf_onnxtrt.py \
        -model_name ${name}
    echo "========================== Model: ${name} Bz: 32 ======================"
    PYTHONPATH=. python tools/perf_onnxtrt.py \
        -model_name ${name} \
        -input_shape 32 3 224 224
done