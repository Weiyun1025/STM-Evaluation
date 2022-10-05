"""
data transform modules for invariance analysis
"""

from torchvision import transforms
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def standard_transform(img_size=224, crop_ratio=0.875):
    size = int(img_size / crop_ratio)
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def position_jitter_transform(img_size=224, crop_ratio=0.875, jitter_strength=0):
    return PositionJitterTransform(input_size=img_size,
                                   resizing_size=int(img_size / crop_ratio),
                                   jitter_strength=jitter_strength)


class PositionJitterTransform:
    def __init__(self, jitter_strength=0, resizing_size=256, input_size=224):

        self.jitter_strength = jitter_strength
        self.resizing_size = resizing_size
        self.input_size = input_size
        self.jitter_strength = max(self.jitter_strength - self.input_size, jitter_strength)
        self.resize = transforms.Resize(self.resizing_size,
                                        interpolation=transforms.InterpolationMode.BICUBIC)
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def __call__(self, image):
        resized_image = self.resize(image)
        resized_image = self.to_tensor(resized_image)
        resized_image = self.positon_jitter_crop(resized_image)
        resized_image = self.norm(resized_image)

        return resized_image

    def positon_jitter_crop(self, image):
        mode = np.random.randint(7)
        center = [image.shape[-2] // 2, image.shape[-1] // 2]
        h, w = (self.input_size // 2, self.input_size // 2)

        # randomly choose one direction from 8 directions
        if mode == 0:
            center[0] = center[0] + self.jitter_strength
        elif mode == 1:
            center[1] = center[1] + self.jitter_strength
        elif mode == 2:
            center[0] = center[0] - self.jitter_strength
        elif mode == 3:
            center[1] = center[1] - self.jitter_strength
        elif mode == 4:
            center[0] = center[0] + self.jitter_strength
            center[1] = center[1] + self.jitter_strength
        elif mode == 5:
            center[0] = center[0] - self.jitter_strength
            center[1] = center[1] + self.jitter_strength
        elif mode == 6:
            center[0] = center[0] - self.jitter_strength
            center[1] = center[1] - self.jitter_strength
        elif mode == 7:
            center[0] = center[0] + self.jitter_strength
            center[1] = center[1] - self.jitter_strength

        return image[:, center[0]-h:center[0]+h, center[1]-w:center[1]+w]
