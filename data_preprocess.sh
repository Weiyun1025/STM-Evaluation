#!/usr/bin/env bash

set -x
mkdir /root/ImageNet/meta
cp ./minidata/train.txt /root/ImageNet/meta
cp ./minidata/val.txt /root/ImageNet/meta
