import time
import datetime
import argparse
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from timm.models import create_model
import models


def test(model, criterion, x_input, y_gold, epochs):
    epoch_time_list = []
    for _ in tqdm(range(epochs)):
        epoch_start_time = time.time()

        y_pred = model(x_input)
        loss = criterion(y_pred, y_gold)
        loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_time = time.time() - epoch_start_time
        epoch_time_list.append(int(epoch_time))

    print(f'\tEpoch time mean: {datetime.timedelta(seconds=np.mean(epoch_time_list))}')


def main():
    args = _get_args()
    model = create_model(args.model_type).to(args.device)
    criterion = nn.CrossEntropyLoss()
    print(f'{args.model_type}:')

    x = torch.randn(args.bsz, 3, 224, 224).to(args.device)
    y = torch.nn.functional.softmax(torch.randn(args.bsz, 1000).to(args.device), dim=-1)

    start_time = time.time()
    test(model, criterion, x, y, args.epochs)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print(f'\tTotal time {total_time_str}')


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--bsz', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
