"""
This module contains functions for tensors

Notes:

* The functions are for tensors defined as numpy.array objects.

"""

import numpy as np


def delta(shape):
    """
    Return delta tensor of given shape, i.e. element values of 1 when all non-dummy indices are equal, 0 otherwise.

    :param shape: Shape of tensor.
    :type shape: tuple of int
    :return: Delta tensor
    :rtype: numpy.array
    """
    # start with ones, in case all dummy indices
    n = np.ones(shape, dtype=int)
    # non-dummy index mask
    ndi_mask = np.array(n.shape) != 1
    # fill values
    if any(ndi_mask):
        for i in np.ndindex(n.shape):
            ndi = np.array(i)[ndi_mask]  # non-dummy indices
            n[i] = 0 if np.any(ndi != ndi[0]) else 1  # 0 if any non-dummy indices differ
    return n


def as_scalar(tsr):
    """
    Return tensor as scalar.

    :param tsr: Tensor
    :type tsr: numpy.array (4d)
    :return: Scalar
    :rtype: numpy.number
    :raises ValueError: if tensor is not a scalar.
    """
    # check we got a scalar
    if tsr.size != 1:
        raise ValueError('Tensor is not a scalar')
    return tsr.flatten()[0]
