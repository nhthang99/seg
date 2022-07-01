import numpy as np
import torch
import torchvision.transforms as transforms


class Normalization:
    def __call__(self, image: np.ndarray, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])(image)


class SelfNormalization:
    def __call__(self, image: np.ndarray):
        tensor = torch.from_numpy(image).to(torch.float32)
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-7)
        tensor = tensor.permute(2, 0, 1)
        return tensor
