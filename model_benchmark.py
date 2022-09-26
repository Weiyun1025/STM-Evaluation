# Benchmarks used to test the inference speed, number of parameters and the FLOPs

import torch
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string
import time
from timm.models import create_model
import models

def speed_test(model, img_size=224):
    input = torch.rand(16, 3, img_size, img_size).cuda()
    print(f"Input Feature Map: {input.shape}")
    for i in range(10):  # warmup
        output = model(input)

    torch.cuda.synchronize()
    start = time.time()
    for i in range(30):
        output = model(input)
    torch.cuda.synchronize()
    end = time.time()
    print(f"FPS: {30 / (end - start)}")
    print("--------------------------")


if __name__ == '__main__':
    model_name_list = ["conv_halo_v3_timm_tiny", "conv_halo_v2_timm_tiny"]
    
    for model_name in model_name_list:
        print("********************************************************")
        model = create_model(model_name,
                            pretrained=False,
                            num_classes=1000).cuda()
    
        flops, params = get_model_complexity_info(model, (3, 224, 224),
                                                as_strings=False,
                                                print_per_layer_stat=False)

        flops, params = flops_to_string(flops), params_to_string(params)
        print(model_name, ": ", "FLOPs: ", flops, "#Params: ", params)
        speed_test(model)
        print("********************************************************\n")

# srun -p VC -N 1 --gres=gpu:1 --ntasks=1 --cpus-per-task=10 --quotatype=spot python model_benchmark.py