import time
import argparse
import contextlib
import torch
from timm.models import create_model
import models

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def benchmark_qps(model, data, backward=False, num_warm=10, num_iter=100):
    if backward:
        x = torch.ones_like(model(data))
        ctx = contextlib.nullcontext()
    else:
        model = model.eval()
        ctx = torch.no_grad()
    with ctx:
        for _ in range(num_warm):
            out = model(data)
            if backward:
                out.backward(x)
            torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(num_iter):
            out = model(data)
            if backward:
                out.backward(x)
            torch.cuda.synchronize()
        t2 = time.time()
    tt = (t2 - t1) / num_iter / data.shape[0]
    return 1.0 / tt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='conv_halo_opt_v1_tiny')
    parser.add_argument('--backward', action='store_true', default=False)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(args.model_type).to(device)
    data = torch.randn(32, 3, 224, 224).to(device)

    print(benchmark_qps(model, data, backward=args.backward))


if __name__ == '__main__':
    main()
