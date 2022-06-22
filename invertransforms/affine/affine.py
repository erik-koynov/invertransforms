"""
This module contains transform classes to apply affine transformations to images.
The transformation can be random or fixed.
Including specific transformations for rotations.

"""
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.transforms.functional_tensor as F_t
from .utils import _invert_affine_matrix
from ..lib import Invertible


class Affine(Invertible):
    """
    Apply affine transformation on the image.

    Args:
        parameters to be used (without randomness)
    """

    def __init__(self, *params, fill=None):
        self.params = params
        self._matrix = F._get_inverse_affine_matrix([0., 0.], *self.params)
        self.fill = fill

    def __call__(self, img: torch.Tensor):
        """
        Args
            img torch tensor: Image to be transformed.

        Returns:
            torch tensor: Affine transformed image.
        """
        return F_t.affine(img, matrix= self._matrix, fill=self.fill)

    def inverse_transform(self, img: torch.Tensor):

        matrix_inv = _invert_affine_matrix(self._matrix)

        return F_t.affine(img, matrix=matrix_inv, fill=self.fill)

    def __repr__(self):
        return f'{self.__class__.__name__}(matrix={self._matrix})'


class RandomAffine(transforms.RandomAffine, Invertible):
    # __init__ inherited from transforms.RandomAffine
    def __call__(self, img, reuse_params = False):
        """
        Args
            img torch.tensor: Image to be transformed.

        Returns:
            torch tensor: Affine transformed image.
        """
        if (not getattr(self, "params", False)) or (not reuse_params):
            self.params = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.shape)

        center = [0., 0.]
        self._matrix = F._get_inverse_affine_matrix(center, *self.params)

        return F.affine(img, *self.params, interpolation=self.interpolation, fill=self.fill)

    def _can_invert(self):
        getattr(self, "_matrix", False)


    def inverse_transform(self, img: torch.Tensor):

        matrix_inv = _invert_affine_matrix(self._matrix)

        return F_t.affine(img, matrix=matrix_inv, fill=self.fill)





