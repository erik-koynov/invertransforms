import torch
from torchvision.transforms import functional_tensor as F_t
from copy import deepcopy
from typing import List

def _affine_with_matrix(img: torch.Tensor,
                        inverse_affine_attrix_matrix: List[float],
                        fill=None)->torch.Tensor:
    """
    By convention PIL and torchvision compute the transforms of the pixel coordinates via the inverse
    affine matrix, computed in F._get_inverse_affine_matrix (see F.affine())
    :param img: B x C x H x W
    :param inverse_affine_attrix_matrix: list of floats : the flattened fist 2 rows of the transform matrix()
                                (in homogeneous coordinates)
    :param fill:
    :return: B x X H x W
    """
    return F_t.affine(img, matrix=inverse_affine_attrix_matrix, fill=fill)


def _invert_affine_matrix(matrix: list):
    matrix = deepcopy(matrix)
    if len(matrix) == 6:
        matrix += [0., 0., 1.]
    matrix = torch.tensor(matrix).reshape(3, 3)
    return matrix.inverse()[:-1, :].reshape(-1).tolist()