"""
燕进提供的原代码
"""

import math
import numbers
import random
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from PIL import Image, ImageOps
import numpy as np


class Resize:
    def __init__(self, size, interpolation="bicubic", max_size=None):
        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation

    def _compute_resized_output_size(
        self,
        image_size: Tuple[int, int],
        size: List[int],
        max_size: Optional[int] = None,
    ) -> List[int]:
        w, h = image_size
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = size if isinstance(size, int) else size[0]
        new_short, new_long = requested_new_short, int(
            requested_new_short * long / short
        )
        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        return [new_h, new_w]

    def __call__(self, img: Image.Image) -> Image.Image:
        image_height, image_width = img.size
        output_size = self._compute_resized_output_size(
            (image_height, image_width), [self.size], self.max_size
        )
        return img.resize(tuple(output_size[::-1]), Image.BICUBIC)


class CenterCrop:
    def __init__(self, size):
        self.size = int(size), int(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        image_height, image_width = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return img.crop(
            (crop_top, crop_left, crop_top + crop_height, crop_left + crop_width)
        )


class ToTensor:
    def __call__(self, pic: Union[Image.Image, np.ndarray]) -> Tensor:
        default_float_dtype = torch.get_default_dtype()
        img = torch.from_numpy(np.array(pic, np.uint8, copy=True))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        img = img.permute((2, 0, 1)).contiguous()
        return img.to(dtype=default_float_dtype).div(255)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: Tensor) -> Tensor:
        tensor = tensor.clone()
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        return tensor.sub_(mean).div_(std)
