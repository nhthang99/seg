import random
from pathlib import Path
from typing import Callable, Dict, Optional, List, Tuple

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from imgaug.augmenters import PadToSquare
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from .normalization import Normalization


class VOCDataset(Dataset):
    def __init__(self, dirname: str, mode: str, classes: Dict[str, Dict], image_size: Tuple[int, int],
                 pad_to_square: bool=False, transforms: Optional[List[Callable]]=None,
                 normalization: Optional[Callable] = None):
        super(VOCDataset, self).__init__()
        self.dirname = dirname
        self.mode = mode
        self.classes = classes
        self.image_size = image_size
        self.pad_to_square = pad_to_square
        self.normalization = normalization or Normalization()
        self.transforms = transforms or []
        self.augment_weights = []
        if len(self.transforms):
            # Uniform, Ex: - iaa.Add(value=(-50, 50), per_channel=True)
            if not isinstance(self.transforms[0], tuple):
                self.augment_weights = [1 / len(self.transforms)] * len(self.transforms)
            # Weights, Ex: - iaa.Add(value=(-50, 50), per_channel=True), 0.2 (with weight=0.2)
            else:
                self.augment_weights = [float(trans[1]) for trans in self.transforms]
            # Ensure sum to be 1
            self.augment_weights = (np.asarray(self.augment_weights) / sum(self.augment_weights)).tolist()

        assert mode in ["train", "val", "trainval"]
        with open(Path(dirname).joinpath("ImageSets", "Segmentation", f"{mode}.txt")) as f:
            file_stems = [x.strip() for x in f.readlines()]
        image_paths = [Path(dirname).joinpath("JPEGImages", f"{stem}.jpg") for stem in file_stems]
        label_paths = [Path(dirname).joinpath("SegmentationClass", f"{stem}.png") for stem in file_stems]
        self.data_pairs = list(zip(image_paths, label_paths))
        print(f"No.samples of {mode}:", len(self.data_pairs))
    
    def __len__(self) -> int:
        return len(self.data_pairs)

    def label_encoding(self, label_mask, multi_label=False):
        if multi_label:
            masks = []
            for idx, (label_name, color) in enumerate(self.classes.items(), start=1):
                mask = np.zeros(shape=(label_mask.shape[0], label_mask.shape[1]), dtype=np.uint8)
                color_arr = np.array(color)[None, None] # [1, 1, 3]
                mask[(label_mask == color_arr).all(axis=-1)] = 1
                masks.append(mask)
            return np.stack(masks, axis=0)
        else:
            mask = np.zeros(shape=(label_mask.shape[0], label_mask.shape[1]), dtype=np.uint8)
            for idx, (label_name, color) in enumerate(self.classes.items(), start=1):
                color_arr = np.array(color)[None, None] # [1, 1, 3]
                mask[(label_mask == color_arr).all(axis=-1)] = idx
            return mask[None] # [1, H, W]

    def __getitem__(self, idx: int):
        image_path, label_path = self.data_pairs[idx]
        image = cv2.imread(str(image_path))
        masks = self.label_encoding(cv2.imread(str(label_path)), multi_label=False)

        image_info = [str(image_path), image.shape[1::-1]] # (W, H)
        masks = [SegmentationMapsOnImage(mask, image.shape) for mask in masks]

        for transform in self.transforms and np.random.choice(self.transforms, \
                                                                size=random.randint(0, len(self.transforms)), \
                                                                p=self.augment_weights, replace=False):
            _transform = transform.to_deterministic()
            image = _transform(image=image)
            masks = [_transform(segmentation_maps=mask) for mask in masks]
        masks = [mask.get_arr() for mask in masks]

        if self.pad_to_square:
            # Pad to square to keep object"s ratio
            image = PadToSquare(position="right-bottom")(image=image)
            masks = [PadToSquare(position="right-bottom")(image=mask) for mask in masks]

        image = cv2.resize(image, dsize=self.image_size)
        masks = [cv2.resize(mask, dsize=self.image_size, interpolation=cv2.INTER_NEAREST) for mask in masks]
        masks = np.stack(masks, axis=-1) # H x W x C

        image = self.normalization(image)

        masks = torch.from_numpy(masks)
        masks = masks.permute(2, 0, 1).contiguous().to(torch.float)

        return image, masks, image_info
