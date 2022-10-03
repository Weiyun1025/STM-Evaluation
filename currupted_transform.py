"""
data transform modules for invariance analysis
"""

from tkinter import image_names
from cv2 import resize
import torch
from torchvision import datasets, transforms

class PositionJitterTransform(object):
    def __init__(self, jitter_strength=0, resizing_size=256, input_size=224):
        
        self.resizing_size = resizing_size
        self.input_size = input_size
        self.jitter_strength = max(self.jitter_strength - self.input_size, jitter_strength)
        self.resize = transforms.Resize(self.resizing_size, interpolation=transforms.InterpolationMode.BICUBIC)
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, image):
        resized_image = self.resize(image)
        resized_image = self.to_tensor(resized_image)
        return resized_image

class StandardTestTransform(object):
    def __init__(self, resizing_size=256, input_size=224):
        
        self.resizing_size = resizing_size
        self.input_size = input_size

        
        self.resize = transforms.Resize(self.resizing_size, interpolation=transforms.InterpolationMode.BICUBIC)
        self.to_tensor = transforms.ToTensor()
        self.crop = transforms.CenterCrop(self.input_size)
        
    
    def __call__(self, image):
        resized_image = self.resize(image)
        resized_image = self.to_tensor(resized_image)
        cropped_image = self.crop(resized_image)
        return cropped_image

