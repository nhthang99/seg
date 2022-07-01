import json
from typing import Callable, Optional, Union, List, Tuple

import cv2
import torch
import random
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from imgaug.augmenters import PadToSquare
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from .normalization import Normalization


class LabelmeDataset(Dataset):
    def __init__(self, dirname: Union[List[str], str], classes: List[str], image_size: Tuple[int, int], \
                    image_patterns: Union[List[str], str], pad_to_square: bool=False, \
                    force_oriented_rectangle: bool=True, transforms: Optional[List[Callable]]=None, \
                    normalization: Optional[Callable] = None):
        super(LabelmeDataset, self).__init__()
        self.dirname = dirname
        self.classes = classes
        self.image_size = image_size
        self.image_patterns = image_patterns if isinstance(image_patterns, list) else [image_patterns]
        self.pad_to_square = pad_to_square
        self.force_oriented_rectangle = force_oriented_rectangle
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

        # Get all image and annotation file
        image_paths = []
        if isinstance(dirname, list):
            for dir_ in dirname:
                dir_ = Path(dir_)
                image_paths_ = []
                for image_pattern in self.image_patterns:
                    paths = [path for path in dir_.glob(image_pattern)]
                    image_paths_ += paths
                image_paths += image_paths_
                print(f"{str(dir_)}: {len(image_paths_)}")
        elif isinstance(dirname, str):
            dirname = Path(dirname)
            for image_pattern in self.image_patterns:
                image_paths += [path for path in dirname.glob(image_pattern)]
            print(f"{str(dirname)}: {len(image_paths)}")
        else:
            raise TypeError(f"{dirname} must be string or list of string.")

        self.data_pairs = [(path, path.with_suffix(".json")) for path in image_paths
                            if path.with_suffix(".json").exists()]
        print("Length:", len(self.data_pairs), end="\n")

    def __len__(self):
        return len(self.data_pairs)

    def _get_masks_from_json(self, json_path, multi_label=False):
        """Get mask from json

        Args:
            json_path (str): json path of annotation file
            mask_value (int, optional): mask value. Defaults to 1.

        Raises:
            ValueError: Type of label region must be rectangle or polygon
            ValueError: Shape should contain at least 3 points

        Returns:
            np.ndarray: Stacked mask array, Shape: C x H x W
        """
        with open(file=str(json_path), encoding="utf-8", mode="r") as fp:
            data = json.load(fp)

        height, width = data["imageHeight"], data["imageWidth"]

        if multi_label:
            masks = []
            for class_idx, label_name in enumerate(self.classes):
                mask = np.zeros(shape=(height, width), dtype=np.uint8)
                for region in data["shapes"]:
                    if (isinstance(self.classes, list) and region["label"] == label_name) \
                        or (isinstance(self.classes, dict) and region["label"] in self.classes[label_name]["intra_classes"]):
                        if region["shape_type"] == "rectangle":
                            xmin, ymin = np.array(region["points"]).min(axis=0)
                            xmax, ymax = np.array(region["points"]).max(axis=0)
                            points = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                        elif region["shape_type"] == "polygon":
                            points = region["points"]
                        else:
                            raise ValueError(f"Type of label region must be rectangle or polygon, \
                                                given \'{region['shape_type']}\'")

                        if self.force_oriented_rectangle:
                            points = cv2.boxPoints(cv2.minAreaRect(np.expand_dims(np.float32(points), axis=1)))
                        
                        if len(points) <= 2:
                            raise ValueError(f"Shape should contain at least 3 points, \
                                                given \'{len(points)}\' points")
                        cv2.fillPoly(img=mask, pts=[np.int32(points)], color=1)
                masks.append(mask)

            return np.stack(masks, axis=0) # C, H, W
        else:
            mask = np.zeros(shape=(height, width), dtype=np.uint8)
            for class_idx, label_name in enumerate(self.classes):
                for region in data["shapes"]:
                    if (isinstance(self.classes, list) and region["label"] == label_name) \
                        or (isinstance(self.classes, dict) and region["label"] in self.classes[label_name]["intra_classes"]):
                        if region["shape_type"] == "rectangle":
                            xmin, ymin = np.array(region["points"]).min(axis=0)
                            xmax, ymax = np.array(region["points"]).max(axis=0)
                            points = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                        elif region["shape_type"] == "polygon":
                            points = region["points"]
                        else:
                            raise ValueError(f"Type of label region must be rectangle or polygon, \
                                                given \'{region['shape_type']}\'")

                        if self.force_oriented_rectangle:
                            points = cv2.boxPoints(cv2.minAreaRect(np.expand_dims(np.float32(points), axis=1)))
                        
                        if len(points) <= 2:
                            raise ValueError(f"Shape should contain at least 3 points, \
                                                given \'{len(points)}\' points")
                        cv2.fillPoly(img=mask, pts=[np.int32(points)], color=class_idx)
            return mask[None] # 1, H, W

    def __getitem__(self, idx):
        image_path, label_path = self.data_pairs[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if label_path.suffix == ".json":
            masks = self._get_masks_from_json(label_path)

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
