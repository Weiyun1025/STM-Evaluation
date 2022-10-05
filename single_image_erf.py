# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import argparse
import numpy as np
import torch
from timm.utils import AverageMeter
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from torch import optim as optim
from matplotlib import pyplot as plt

# timm register while not used
import models
from timm.models import create_model
from erf_tools import get_input_grad, get_rectangle

plt.switch_backend('agg')


def parse_args():
    parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)
    parser.add_argument('--model', default='conv_swin_tiny', type=str, help='model name')
    parser.add_argument('--weights', default="/data1/shimin/model_parameters/backbone/swin_tiny/checkpoint-best.pth", type=str, help='path to weights file. For resnet101/152, ignore this arg to download from torchvision')
    #parser.add_argument('--model', default='conv_halo_v2_timm_tiny', type=str, help='model name')
    #parser.add_argument('--weights', default="/data1/shimin/model_parameters/backbone/halonet_v2_tiny/checkpoint-best.pth", type=str, help='path to weights file. For resnet101/152, ignore this arg to download from torchvision')
    parser.add_argument('--data_path', default='minidata/val/ILSVRC2012_val_00000019.JPEG', type=str, help='dataset path')
    args = parser.parse_args()
    return args


def denormalize(image):
    '''
    image: (h,w,3), torch tensor
    '''
    std = torch.tensor(IMAGENET_DEFAULT_STD).cuda()
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).cuda()
    image = (image * std[:, None, None]) + mean[:, None, None]
    return image.permute(1, 2, 0).detach().cpu().numpy() * 255.0


def main(args):
    #   ================================= transform: standard test transform
    # NOTE: this is different from RepLKNet ERF code, where they resize the input images to 1000x1000
    t = [
        transforms.Resize((448, 448), interpolation=Image.BICUBIC),
        #transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    transform = transforms.Compose(t)

    print("reading from datapath", args.data_path)
    image = Image.open(args.data_path)

    # TODO: rm this part
    if 'halo' in args.model:
        info = args.model.split('_')
        args.model = f'conv_halo_v2_{info[-1]}'

        if 'github' in info:
            args.halo_type = 'github'
        elif 'mask' in info and 'rpe' in info:
            args.halo_type = 'with_mask_with_rpe'
        elif 'mask' in info:
            args.halo_type = 'with_mask'
        elif 'switch' in info:
            args.halo_type = 'switch'
        else:
            args.halo_type = 'timm'

        import sys
        print(f'model_type: {args.model}', file=sys.stderr)
        print(f'halo_type: {args.halo_type}', file=sys.stderr)
    else:
        args.halo_type = None

    # TODO: rm halo_type option
    model = create_model(
        args.model,
        halo_type=args.halo_type,
        pretrained=False,
        num_classes=1000,
    )

    
    if args.weights is not None:
        print('load weights')
        weights = torch.load(args.weights, map_location='cpu')
        if 'model' in weights:
            weights = weights['model']
        if 'state_dict' in weights:
            weights = weights['state_dict']
        model.load_state_dict(weights)
        print('loaded')
    

    model.cuda()
    model.eval()    #   fix BN and droppath

    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

    optimizer.zero_grad()

    samples = transform(image).unsqueeze(dim=0)
    samples = samples.cuda(non_blocking=True)
    samples.requires_grad = True

    erf_map = []
    for layer in range(4):
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, samples, layer=layer) # size of (input_size, input_size)
        erf_map.append(contribution_scores)

    optimizer.zero_grad()
    contribution_scores = get_input_grad(model, samples, layer=0) # size of (input_size, input_size)

    """
    heatmap visualization
    """
    cmap = plt.cm.get_cmap('jet')
    original_image = denormalize(samples[0])
    figure = plt.figure(dpi=800)
    img_ax = figure.add_subplot(1, 5, 1)
    img_ax.get_xaxis().set_visible(False)
    img_ax.get_yaxis().set_visible(False)
    img_ax.imshow(original_image.astype(np.uint8))

    for lidx, erf_map in enumerate(erf_map):
        contribution_scores = np.log10(erf_map + 1)

        # compute erf ratio for each stage
        print("====================Stage %d======================"%(lidx+1))
        for thresh in [0.2, 0.3, 0.5, 0.99]:
            side_ratio, area_ratio = get_rectangle(contribution_scores, thresh)
            print('thresh: %.2f, side_ratio: %.5f, area ratio: %.5f '%(thresh, side_ratio, area_ratio))

        contribution_scores = contribution_scores / (contribution_scores.max() + 1e-10)
        erf_map = cmap(contribution_scores)[:,:,0:3] * 255.0
        erf_map = erf_map * 0.8 + original_image * 0.2
            
        ax = figure.add_subplot(1,5,lidx+2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(erf_map.astype(np.uint8))

    plt.savefig("erf_map_"+os.path.basename(args.data_path))
    plt.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)