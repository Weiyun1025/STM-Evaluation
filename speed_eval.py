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

    print(f'Epoch time mean: {datetime.timedelta(seconds=np.mean(epoch_time_list))}')


def main():
    args = _get_args()
    model = create_model(args.model_type)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(args.bsz, 3, 224, 224)
    y = torch.nn.functional.softmax(torch.randn(args.bsz, 1000), dim=-1)

    start_time = time.time()
    test(model, criterion, x, y, args.epochs)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print(f'Total time {total_time_str}')


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--bsz', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
