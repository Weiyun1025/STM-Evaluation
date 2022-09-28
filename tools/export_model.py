import os
import argparse

import torch
torch.nn.Module.apply = lambda self, f: None
from timm.models import create_model

import models
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str)
    parser.add_argument('-root', type=str, default='logs/inference_benchmark')
    parser.add_argument('-input_shape',
                        nargs='+',
                        type=int,
                        default=(1, 3, 224, 224))
    parser.add_argument('-seed', type=int, default=1001)
    return parser.parse_args()


def export_to_onnx(save_path, model, data):
    torch.onnx.export(
        model,
        data,
        save_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {
                0: 'batch_size'
            },
            'output': {
                0: 'batch_size'
            }
        },
        opset_version=12,
    )


@torch.no_grad()
def main(args):
    if args.seed:
        set_seed(args.seed)

    data = torch.randn(*args.input_shape,
                       dtype=torch.float32,
                       device=torch.device('cuda'))
    model = model = create_model(args.model_name,
                                 pretrained=False,
                                 num_classes=1000)

    model = model.eval().cuda()
    with torch.jit.optimized_execution(True):
        ts_model = torch.jit.trace(model, data)

    root = os.path.join(args.root, args.model_name)
    os.makedirs(root, exist_ok=True)
    torch.jit.save(ts_model, '{}/{}.ts'.format(root, args.model_name))
    export_to_onnx('{}/{}.onnx'.format(root, args.model_name), ts_model, data)


if __name__ == '__main__':
    main(parse_args())