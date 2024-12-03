"""
This module contains augmentations that try to mimic different ct-scan devices.
"""
import random
from typing import List, Tuple, Union, Literal

import nibabel as nib
import numpy as np
from monai.transforms import (
    CastToTyped,
    EnsureTyped, MapTransform, OneOf, RandAdjustContrastd, RandGaussianSmoothd, RandGaussianNoised
)
from monai.transforms import (
    EnsureChannelFirstd,
    Compose
)

from utils import torch_io_resize


class StepAndShootAug(MapTransform):
    def __init__(self, keys=("img",), cutoff_index: int = -1,
                 cut_off_pixel_value_weight: Union[Tuple[float, float], List[float]] = (0.2, 0.6),
                 normalize: bool = True, prob: float = 1):
        """

        :param keys:
        :param cutoff_index:
        :param cut_off_pixel_value_weight: The mean pixel value is calculated then a value between the
        given input values is randomly chosen and added to a part of the image.
        :param normalize: whether to normalize the output between min and max. Default is True,
        :param prob:
        """
        super().__init__(keys=keys)
        self.cutoff_index = cutoff_index
        assert 0 <= cut_off_pixel_value_weight[0]
        self.cut_off_pixel_value_weight = cut_off_pixel_value_weight
        self.normalize = normalize
        self.prob = prob

    def __call__(self, data, lazy: bool | None = None):
        key = self.keys[0]
        if random.random() < self.prob:
            image: np.ndarray = data[key].copy().astype(np.float32)
            if self.normalize:
                min_value = np.min(image)
                max_value = np.max(image)

            cut_off_position = random.randint(0, image.shape[self.cutoff_index] - 1)
            mean_pixel_value = np.mean(image)
            intensity_value = random.uniform(self.cut_off_pixel_value_weight[0] * mean_pixel_value,
                                             self.cut_off_pixel_value_weight[1] * mean_pixel_value)
            intensity_value = intensity_value if random.random() < 0.5 else -intensity_value
            # whether to lower the pixel values or increase it!

            image = step_and_shoot_core(img=image,
                                        cutoff_index=self.cutoff_index,
                                        cut_off_position=cut_off_position,
                                        intensity_value=intensity_value)
            if self.normalize:
                image = (image - np.min(image)) / (np.max(image) - np.min(image))  # normalize to 0-1
                image = image * (max_value - min_value) + min_value  # scale to the requested range

            data[key] = image
        return data


def step_and_shoot_core(img: np.ndarray, cutoff_index: int, cut_off_position: int, intensity_value: float):
    """
    Core function for doing step and shoot augmentation
    :param img:
    :param cutoff_index:
    :param cut_off_position:
    :param intensity_value:
    :return:
    """
    image = img.copy()
    if random.random() < 0.5:
        if cutoff_index == 0:
            image[cut_off_position:, ...] += intensity_value
        elif cutoff_index == 1:
            image[:, cut_off_position:, ...] += intensity_value
        elif cutoff_index == 2 or cutoff_index == -1:
            image[:, :, cut_off_position:] += intensity_value
        else:
            raise ValueError("Only 3D images are supported!")
    else:
        if cutoff_index == 0:
            image[:cut_off_position, ...] += intensity_value
        elif cutoff_index == 1:
            image[:, :cut_off_position, ...] += intensity_value
        elif cutoff_index == 2 or cutoff_index == -1:
            image[:, :, :cut_off_position] += intensity_value
        else:
            raise ValueError("Only 3D images are supported!")
    return image


class MotionAugmentation(MapTransform):
    def __init__(self, keys=("img",),
                 motion_move_position: Union[Tuple[float, float], List[float]] = (0.05, 0.2),
                 motion_move_range: Union[Tuple[float, float], List[float]] = (0.01, 0.05),
                 cutoff_index=2,
                 move_index=1,
                 normalize: bool = True,
                 prob: float = 1,
                 cropped_value: Literal["zero", "mean", "cut", "cut_resize"] = "cut_resize"):
        """

        :param keys:
        :param cutoff_index:
        :param move_index:
        :param motion_move_range: The mean pixel value is calculated then a value between the
        given input values is randomly chosen and added to a part of the image.
        :param normalize: whether to normalize the output between min and max. Default is True,
        :param prob:
        :param cropped_value: zero: replace with zero, mean: replace with image min, cut: cut the section, cut_resize:
        cut and resize the image to its original size, default is set to cut_resize!
        """
        super().__init__(keys=keys)
        # self.motion_index = motion_index
        assert 0 <= motion_move_range[0]
        assert 0 <= motion_move_position[0]
        self.motion_move_range = motion_move_range
        self.motion_move_position = motion_move_position
        self.normalize = normalize
        self.cutoff_index = cutoff_index
        self.move_index = move_index
        self.prob = prob
        self.cropped_value = cropped_value

    def __call__(self, data, lazy: bool | None = None):
        key = self.keys[0]
        if random.random() < self.prob:

            image: np.ndarray = data[key].copy().astype(np.float32)
            if self.normalize:
                min_value = np.min(image)
                max_value = np.max(image)

            mean_value = np.mean(image)
            img_size = image.shape

            move_value = random.randint(int(image.shape[self.move_index] * self.motion_move_range[0]),
                                        int(image.shape[self.move_index] * self.motion_move_range[1]) - 1)

            cut_off_position = random.randint(int(image.shape[self.cutoff_index] * self.motion_move_position[0]),
                                              int(image.shape[self.cutoff_index] * self.motion_move_position[1]) - 1)

            cut_off_position = cut_off_position if random.random() < 0.5 else image.shape[
                                                                                  self.cutoff_index] - cut_off_position
            image = motion_core(image, self.cutoff_index, self.move_index, self.cropped_value,
                                cut_off_position, img_size, mean_value, move_value)
            if self.normalize:
                image = (image - np.min(image)) / (np.max(image) - np.min(image))  # normalize to 0-1
                image = image * (max_value - min_value) + min_value  # scale to the requested range
            data[key] = image

        return data


def get_crop_value(cropped_value: str, mean_value: float):
    """
    Get the crop value
    :param cropped_value:
    :param mean_value:
    :return:
    """
    if cropped_value == "zero":
        return 0
    else:
        return mean_value


def motion_core(img: np.ndarray,
                cutoff_index: int,
                move_index: int,
                cropped_value: str,
                cut_off_position: int,
                img_size: tuple[int, ...],
                mean_value: float,
                move_value: int):
    """
    Core function for motion augmentation!
    :param img:
    :param cutoff_index:
    :param move_index:
    :param cropped_value:
    :param cut_off_position:
    :param img_size:
    :param mean_value:
    :param move_value:
    :return:
    """
    image = img.copy()
    if (cutoff_index == 2 or cutoff_index == -1) and move_index == 1:
        if random.random() < 0.5:
            # Move left
            image[:, :image.shape[move_index] - move_value, cut_off_position:] = image[:, move_value:,
                                                                                 cut_off_position:]
            if cropped_value in ["zero", "mean"]:
                image[:, image.shape[move_index] - move_value:,
                cut_off_position:] = get_crop_value(cropped_value, mean_value)
            elif cropped_value in ["cut", "cut_resize"]:
                image = image[:, :image.shape[move_index] - move_value, :]
                if cropped_value == "cut_resize":
                    image = torch_io_resize(image, img_size)
        else:
            # Move right
            image[:, move_value:, cut_off_position:] = image[:, :image.shape[move_index] - move_value,
                                                       cut_off_position:]
            if cropped_value in ["zero", "mean"]:
                image[:, :move_value, cut_off_position:] = get_crop_value(cropped_value, mean_value)
            elif cropped_value in ["cut", "cut_resize"]:
                image = image[:, move_value:, :]
                if cropped_value == "cut_resize":
                    image = torch_io_resize(image, img_size)

    else:
        raise ValueError("only cutoff_index=2 and move_index=1 is supported!")
    return image


class StepAndShootAugPlusMotionAugmentation(MapTransform):
    def __init__(self, keys=("img",),
                 cut_off_pixel_value_weight: tuple[float, float] | list[float] = (0.2, 0.6),
                 motion_move_position: Union[Tuple[float, float], List[float]] = (0.05, 0.2),
                 motion_move_range: Union[Tuple[float, float], List[float]] = (0.02, 0.1),
                 cutoff_index=2,
                 move_index=1,
                 cropped_value: Literal["zero", "mean", "cut", "cut_resize"] = "cut_resize",
                 normalize: bool = True,
                 prob: float = 1):
        """

        :param keys:
        :param cut_off_pixel_value_weight: The mean pixel value is calculated then a value between the
        given input values is randomly chosen and added to a part of the image.
        :param normalize: whether to normalize the output between min and max. Default is True,
        :param prob:
        """
        super().__init__(keys=keys)
        assert 0 <= cut_off_pixel_value_weight[0]
        assert 0 <= motion_move_range[0]
        assert 0 <= motion_move_position[0]
        self.cut_off_pixel_value_weight = cut_off_pixel_value_weight
        self.motion_move_range = motion_move_range
        self.motion_move_position = motion_move_position
        self.normalize = normalize
        self.cutoff_index = cutoff_index
        self.move_index = move_index
        self.prob = prob
        self.cropped_value = cropped_value

    def __call__(self, data, lazy: bool | None = None):
        key = self.keys[0]
        if random.random() < self.prob:
            image: np.ndarray = data[key].copy().astype(np.float32)
            if self.normalize:
                min_value = np.min(image)
                max_value = np.max(image)

            cut_off_position = random.randint(0, image.shape[self.cutoff_index] - 1)
            mean_value = np.mean(image)
            img_size = image.shape

            move_value = random.randint(int(image.shape[self.move_index] * self.motion_move_range[0]),
                                        int(image.shape[self.move_index] * self.motion_move_range[1]) - 1)
            intensity_value = random.uniform(self.cut_off_pixel_value_weight[0] * mean_value,
                                             self.cut_off_pixel_value_weight[1] * mean_value)
            intensity_value = intensity_value if random.random() < 0.5 else -intensity_value
            # whether to lower the pixel values or increase it!

            image = step_and_shoot_core(img=image,
                                        cutoff_index=self.cutoff_index,
                                        cut_off_position=cut_off_position,
                                        intensity_value=intensity_value)
            image = motion_core(image, self.cutoff_index, self.move_index, self.cropped_value,
                                cut_off_position, img_size, mean_value, move_value)
            if self.normalize:
                image = (image - np.min(image)) / (np.max(image) - np.min(image))  # normalize to 0-1
                image = image * (max_value - min_value) + min_value  # scale to the requested range

            data[key] = image
        return data


def get_transform(aug_prob: float):
    """
    Create the final monai augmentations
    :param aug_prob:
    :return:
    """
    keys = ["img"]
    train_transforms = Compose(
        [
            OneOf([
                RandGaussianNoised(keys=keys, std=0.01, prob=aug_prob),
                RandGaussianSmoothd(
                    keys=keys,
                    sigma_x=(0.5, 1.15),
                    sigma_y=(0.5, 1.15),
                    sigma_z=(0.5, 1.15),
                    prob=aug_prob,
                ),
                RandAdjustContrastd(keys=keys, prob=aug_prob),
                StepAndShootAug(keys=keys, prob=aug_prob),
                MotionAugmentation(keys=keys, prob=aug_prob),
                StepAndShootAugPlusMotionAugmentation(keys=keys, prob=aug_prob),
            ]),
            CastToTyped(keys=['img', ], dtype=(np.float32,)),
        ]
    )
    val_transforms = Compose(
        [
            EnsureChannelFirstd(keys=keys),
            CastToTyped(keys=['img', ], dtype=(np.float32,)),
            EnsureTyped(keys=['img', "label"]),
        ]
    )

    test_transform = Compose(
        [
            EnsureChannelFirstd(keys=['img']),
            CastToTyped(keys=['img', ], dtype=(np.float32,)),
            EnsureTyped(keys=['img']),
        ]
    )
    return train_transforms, val_transforms, test_transform