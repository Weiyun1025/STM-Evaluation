"""
data transform modules for invariance analysis
"""

from tkinter import image_names
from cv2 import resize
import torch
from torchvision import datasets, transforms
import numpy as np

class PositionJitterTransform(object):
    def __init__(self, jitter_strength=0, resizing_size=256, input_size=224):
        
        self.jitter_strength = jitter_strength
        self.resizing_size = resizing_size
        self.input_size = input_size
        self.jitter_strength = max(self.jitter_strength - self.input_size, jitter_strength)
        self.resize = transforms.Resize(self.resizing_size, interpolation=transforms.InterpolationMode.BICUBIC)
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, image):
        resized_image = self.resize(image)
        resized_image = self.to_tensor(resized_image)
        resized_image = self.positon_jitter_crop(resized_image)

        return resized_image
    
    def positon_jitter_crop(self, image):
        mode = np.random.randint(7)
        center = [image.shape[-2] // 2, image.shape[-1] // 2]
        h, w = (self.input_size // 2, self.input_size // 2)
        
        # randomly choose one direction from 8 directions      
        if mode == 0:
            center[0] += self.jitter_strength
        elif mode == 1:
            center[1] += self.jitter_strength
        elif mode == 2:
            center[0] -= self.jitter_strength
        elif mode == 3:
            center[1] -= self.jitter_strength
        elif mode == 4:
            center[0], center[1] = center[0] + self.jitter_strength, center[1] + self.jitter_strength
        elif mode == 5:
            center[0], center[1] = center[0] - self.jitter_strength, center[1] + self.jitter_strength
        elif mode == 6:
            center[0], center[1] = center[0] - self.jitter_strength, center[1] - self.jitter_strength
        elif mode == 7:
            center[0], center[1] = center[0] + self.jitter_strength, center[1] - self.jitter_strength
             
        return image[:, center[1]-h:center[1]+h, center[0]-w:center[0]+w]    

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

