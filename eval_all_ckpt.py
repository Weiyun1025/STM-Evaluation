import os
import argparse
from pathlib import Path

import utils
from main import get_args_parser, main as eval_main

BASE_DIR = '/mnt/petrelfs/share_data/shimin/share_checkpoint'
MODEL_TYPE_DICT = {
    'convnext': 'conv_convnext_v2',
    'dcnv3': 'dcn_v3',
    'halonet': 'conv_halo_v2_timm_tiny',
    'pvt': 'conv_pvt',
    'swin': 'conv_swin',
}
SCALE = ('micro', 'tiny', 'small', 'base')


def get_scale(name):
    for s in SCALE:
        if s in name:
            return s
    return None


def main():
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    utils.init_distributed_mode(args)

    res = {}
    for model_type in os.listdir(BASE_DIR):
        if model_type not in MODEL_TYPE_DICT:
            continue

        model_dir = os.path.join(BASE_DIR, model_type)
        model_type = MODEL_TYPE_DICT[model_type]
        for model_scale in os.listdir(model_dir):
            ckpt_path = os.path.join(model_dir, model_scale, 'checkpoint-best.pth')
            ckpt_ema_path = os.path.join(model_dir, model_scale, 'checkpoint-best-ema.pth')

            model_scale = get_scale(model_scale)
            if model_scale is None:
                continue

            args.model = f'{model_type}_{model_scale}'
            args.resume = ckpt_path
            res[args.model] = eval_main(args)

            args.resume = ckpt_ema_path
            res[f'{args.model}_ema'] = eval_main(args)

    for key, value in res.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
