# Benchmarks used to test the inference speed, number of parameters and the FLOPs

import os
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


#@profile
def inference_and_backward(model, criterion, inputs, targets):
    for i in range(100):
        outputs = model(inputs)
        loss = criterion(outputs, targets.long())
        loss.backward()


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #model_name = "conv_halo_v2_timm_tiny"
    #model_name = "optim_halo_v2_timm_tiny"
    #model_name = "conv_swin_tiny"
    #model_name = "swin_tiny_patch4_window7_224"
    #model_name = "conv_halo_v3_timm_tiny"
    #model_name = "unified_halo_tiny"
    #model_name = "optim_halo_v2_fixed_posembed_tiny"
    model_name = "optim_halo_v2_with_mask_tiny"

    model = create_model(model_name,
                        pretrained=False,
                        num_classes=1000).cuda()

    inputs = torch.rand(16, 3, 224, 224).cuda()
    
    criterion = torch.nn.CrossEntropyLoss()
    targets = torch.ones(16,).cuda()

    inference_and_backward(model, criterion, inputs, targets)


# kernprof -l model_profile.py
# python -m line_profiler model_profile.py.lprof