#!/usr/bin/env bash

set -x
mkdir /root/ImageNet/meta
cp ./minidata/train.txt /root/ImageNet/meta/train.txt
cp ./minidata/modified_val.txt /root/ImageNet/meta/val.txt
