import numpy as np
from PIL import Image
import random

import numbers
import collections

import torch
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Randomflip_Rotate(object):
    def __init__(self, p=0.5, degrees=0):
        self.p = p
        self.degrees = degrees

    def __call__(self, img, lab):
        if random.random() < self.p:
            img = F.hflip(img)
            lab = F.hflip(lab)
        if random.random() < self.p:
            img = F.vflip(img)
            lab = F.vflip(lab)

        if isinstance(self.degrees, numbers.Number):
            if self.degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-self.degrees, self.degrees)
        else:
            if len(self.degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = self.degrees
        angle = random.uniform(degrees[0], degrees[1])
        img = F.rotate(img, angle)
        lab = F.rotate(lab, angle)

        return img, lab


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class randomcrop(object):
    """Crop the given PIL Image and mask at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lab):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lab (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image and mask.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lab = F.pad(lab, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            lab = F.pad(lab, (int((1 + self.size[1] - lab.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))
            lab = F.pad(lab, (0, int((1 + self.size[0] - lab.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lab, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class resize(object):
    """Resize the input PIL Image and mask to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lab):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lab (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image and mask.
        """
        # return TF.resize(img, self.size, self.interpolation), TF.resize(lab, self.size, self.interpolation)
        return F.resize(img, self.size, self.interpolation), F.resize(lab, self.size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

