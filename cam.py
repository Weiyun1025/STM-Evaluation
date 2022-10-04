import os
from PIL import Image

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from timm.models import create_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import models


def build_transform(input_size, crop_pct, norm):
    t = []

    # warping (no cropping) when evaluated at 384 or larger
    if input_size >= 384:
        t.append(transforms.Resize((input_size, input_size),
                                   interpolation=transforms.InterpolationMode.BICUBIC),)
    else:
        size = int(input_size / crop_pct)
        t.append(transforms.Resize(size,
                                   interpolation=transforms.InterpolationMode.BICUBIC),)
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    if norm:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    else:
        return transforms.Compose(t), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    return transforms.Compose(t)


def cam_img(model, target_layers, img_path, output_dir,
            desc='', img_label=None, save=True, save_raw=False):
    transform, norm = build_transform(input_size=224, crop_pct=0.875, norm=False)

    img = Image.open(img_path).convert('RGB')
    img_rgb = transform(img)
    input_tensor = norm(img_rgb)
    input_tensor = torch.stack([input_tensor], dim=0)
    targets = [ClassifierOutputTarget(img_label)] if img_label is not None else None

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    img_rgb = img_rgb.permute(1, 2, 0).numpy()
    visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

    if not save:
        Image.fromarray(visualization).show()
    else:
        save_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, desc, save_name)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(visualization).save(save_path)

    if save_raw:
        save_path = os.path.join(output_dir, 'raw', save_name)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        img_rgb = img_rgb / np.max(img_rgb)
        img_rgb = np.uint8(255 * img_rgb)
        Image.fromarray(img_rgb).save(save_path)


def _read_all_images(image_dir):
    images = []
    images_name = [x for x in os.listdir(image_dir) if not x.startswith('.')]

    for image_name in images_name:
        images.append(Image.open(os.path.join(image_dir, image_name)))

    return images


def concat_images(cam_dir):
    height, width = 224, 224
    cam_dir_name = [x for x in os.listdir(cam_dir) if '.' not in x]

    images = []
    for dir_name in cam_dir_name:
        dir_path = os.path.join(cam_dir, dir_name)
        images.append(_read_all_images(dir_path))

    row = len(images)
    col = len(images[0])

    # 创建成品图的画布, delta为图像之间的间隙
    delta = 10
    target = Image.new('RGB', ((height + delta) * col, (width + delta) * row))
    for i in range(row):
        for j in range(col):
            # paste方法第一个参数指定需要拼接的图片，第二个参数为二元元组（指定复制位置的左上角坐标）
            target.paste(images[i][j],
                         (width * j + delta * (j-1), height * i + delta*(i-1)))

    target.save(os.path.join(cam_dir, 'all.pdf'), 'PDF')


def main(model_type):
    if model_type == 'halo':
        ckpt = torch.load('/Users/weiyun/Downloads/模型算子评测日志备份/conv_halo_v2_timm_tiny_1k_unified_config/checkpoint-best.pth',
                          map_location='cpu')['model']
        model = create_model('conv_halo_v2_tiny')
    else:
        ckpt = torch.load('/Users/weiyun/Downloads/模型算子评测日志备份/conv_swin_tiny_1k_unified_config/checkpoint-best.pth',
                          map_location='cpu')['model']
        model = create_model('conv_swin_tiny')

    model.load_state_dict(ckpt)
    target_layers = model.target_layers()

    base_dir = './minidata/val'
    with open('./minidata/meta/val.txt', 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            img_path, img_label = line.strip().split()
            cam_img(model, target_layers,
                    img_path=os.path.join(base_dir, img_path),
                    img_label=int(img_label),
                    output_dir='./outputs/cam',
                    desc=model_type,
                    save=True, save_raw=True)


if __name__ == '__main__':
    main('halo')
    main('swin')
    concat_images(cam_dir='./outputs/cam')
