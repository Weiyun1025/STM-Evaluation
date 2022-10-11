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
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from torch import optim as optim
from matplotlib import pyplot as plt

# timm register while not used
import models
from timm.models import create_model
from erf_tools import get_input_grad, get_rectangle

from datasets import ImageCephDataset

plt.switch_backend('agg')


def parse_args():
    parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)
    parser.add_argument('--model', default='conv_swin_tiny', type=str, help='model name')
    parser.add_argument('--weights', default="/data1/shimin/model_parameters/backbone/swin_tiny/checkpoint-best.pth", type=str)
    parser.add_argument('--data_path', default='minidata/', type=str, help='dataset path')
    parser.add_argument('--num_data', default=50, type=int, help='dataset path')
    parser.add_argument('--input_size', default=224, type=int, help='dataset path')
    args = parser.parse_args()
    return args


def denormalize(image):
    '''
    image: (h,w,3), torch tensor
    '''
    image = (image / IMAGENET_DEFAULT_STD[None, None, :]) + IMAGENET_DEFAULT_MEAN[None, None, :]
    return image.permute(2, 0, 1).cpu().numpy()


def main(args):
    #   ================================= transform: standard test transform
    # NOTE: this is different from RepLKNet ERF code, where they resize the input images to 1000x1000
    t = [
        transforms.Resize((args.input_size, args.input_size), interpolation=Image.BICUBIC),
        #transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    transform = transforms.Compose(t)

    print("reading from datapath", args.data_path)

    dataset = ImageCephDataset(args.data_path, "val", transform=transform, on_memory=False)
    sampler_val = torch.utils.data.SequentialSampler(dataset)
    data_loader_val = torch.utils.data.DataLoader(dataset, 
                                                  sampler=sampler_val,
                                                  batch_size=1,
                                                  num_workers=1,
                                                  pin_memory=True,
                                                  drop_last=False)

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
        if 'ema' in args.weights:
            weights = weights['model_ema']
        else:
            weights = weights['model']
        model.load_state_dict(weights)
        print('loaded')

    model.cuda()
    model.eval()    #   fix BN and droppath

    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

    meter = AverageMeter()
    optimizer.zero_grad()

    mean_erf = [np.zeros((args.input_size, args.input_size)) for i in range(4)]

    counter = 0
    for idx, (samples, label) in enumerate(data_loader_val):
        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True

        for layer in range(4):
            optimizer.zero_grad()
            contribution_scores = get_input_grad(model, samples, layer=layer) # size of (input_size, input_size)

            if np.isnan(np.sum(contribution_scores)):
                print('got NAN, next image')
                continue
            else:
                mean_erf[layer] += contribution_scores
            counter += 1

        if counter == args.num_data:
            break

    """
    heatmap visualization
    """
    cmap = plt.cm.get_cmap('viridis')
    #original_image = np.from_array(image)
    figure = plt.figure(dpi=800)

    for lidx, erf_map in enumerate(mean_erf):
        contribution_scores = erf_map / (counter)
        contribution_scores = np.log10(contribution_scores + 1)
        contribution_scores = contribution_scores / (contribution_scores.max() + 1e-10)

        # compute erf ratio for each stage
        print("====================Stage %d======================"%(lidx+1))
        for thresh in [0.2, 0.3, 0.5, 0.99]:
            side_ratio, area_ratio = get_rectangle(contribution_scores, thresh)
            print('thresh: %.2f, side_ratio: %.5f, area ratio: %.5f '%(thresh, side_ratio, area_ratio))

        erf_map = cmap(contribution_scores)[:,:,0:3] * 255.0
        ax = figure.add_subplot(1,4,lidx+1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(erf_map.astype(np.uint8))

    save_name = "visualizations/" + args.model + "_erf_map.jpg" 
    plt.savefig(save_name)
    plt.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)