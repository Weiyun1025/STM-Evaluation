'''
ERF tool functions
mostly copied from RepLKNet 
https://github.com/DingXiaoH/RepLKNet-pytorch/blob/b76808ac2c763eaed6af9d286c6163c9345b4d5f/erf/visualize_erf.py
'''

import torch
import numpy as np

def get_input_grad(model, samples, layer=-1):
    logits, outputs = model(samples)
    outputs = outputs[layer]
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map

def get_rectangle(data, thresh):
    h, w = data.shape
    all_sum = np.sum(data)
    for i in range(1, h // 2):
        selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
        area_sum = np.sum(selected_area)
        if area_sum / all_sum > thresh:
            return (i * 2 + 1) / h, (i * 2 + 1) / h * (i * 2 + 1) / w
    return None