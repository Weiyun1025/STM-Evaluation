import argparse

import torch
from ptflops import get_model_complexity_info
from timm.models import create_model
import models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    return parser.parse_args()


def func(module, input, output):
    module.__flops__ += 0


custom_modules_hooks = {
    models.blocks.halonet_timm.HaloAttention: func,
}


def main(args):
    model = create_model(args.model_name, pretrained=False,
                         num_classes=1000).cuda()

    flops, params = get_model_complexity_info(
        model, (3, 224, 224),
        as_strings=False,
        print_per_layer_stat=False,
        custom_modules_hooks=custom_modules_hooks)

    flops = str(round(flops / 10.**9, 2))
    params = str(round(params / 10**6, 2))
    print("model_name: MACs {}G, Params {}M".format(flops, params))


if __name__ == '__main__':
    main(parse_args())

# srun -p VC -N 1 --gres=gpu:1 --ntasks=1 \
#      --cpus-per-task=10 --quotatype=spot \
#      python tools/calulate_mac_param.py
