import argparse
import torch
from torch import nn
from timm.models import create_model
import models


def test(model_type, x_input, y_gold, epochs):
    model = create_model(model_type)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        y_pred = model(x_input)
        loss = criterion(y_pred, y_gold)
        loss.backward()


def main():
    args = _get_args()

    x = torch.randn(args.bsz, 3, 224, 224)
    y = torch.nn.functional.softmax(torch.randn(args.bsz, 1000), dim=-1)
    test(args.model_type, x, y, args.epochs)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--bsz', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
