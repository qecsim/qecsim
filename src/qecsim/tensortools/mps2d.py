"""
This module contains functions for two-dimensional lattice tensor networks

Notes:

* The functions are for 2-D tensor networks where each row or column is an MPS/MPO.

"""

import numpy as np
from mpmath import mp

from qecsim.tensortools import tsr as tt_tsr, mps as tt_mps


def transpose(tn):
    """
    Return transposed network of tensors.

    Notes:

    * Both the network and each tensor is transposed, so that contracting the transposed network column-by-column is
      equivalent to contracting the original network row-by-row.
    * Sparse tensor networks with None objects are supported; None objects are unchanged.

    :param tn: Tensor network whose columns are MPS/MPO.
    :type tn: numpy.array (2d) of numpy.array (4d)
    :return: Transposed tensor network.
    :rtype: numpy.array (2d) of numpy.array (4d)
    """
    return np.vectorize(lambda t: None if t is None else np.transpose(t), otypes=[object])(tn.transpose())


def contract(tn, chi=None, tol=None, start=None, stop=None, step=None, mask=None):
    """
    Return column-by-column contraction of tensor network where each column as an MPS/MPO.

    Notes:

    * Sparse MPS/MPO with None objects are supported provided that they contain a single contiguous list of tensors;
      None objects at the start and/or end are ignored.
    * The contraction is performed as a series of :func:`qecsim.tensortools.mps.contract_pairwise` /
      :func:`qecsim.tensortools.mps.truncate` operations followed by a :func:`qecsim.tensortools.mps.inner_product`
      operation on the final pair of columns without truncation.
    * In case of a full contraction, the network is assumed to contract to a scalar.
    * In case of a partial contraction, the final column is truncated but not contracted to a scalar.
    * In case of a partial contraction, to avoid underflow, a multiplier is returned that is only the cumulative norm
      from the truncation operations. Therefore, it should not be assumed that the returned value is fully normalised;
      rather the true value is given by the product of the returned value and multiplier.
    * In case of a partial contraction where the range of columns is empty, None is returned with multiplier 1.

    :param tn: Tensor network whose columns are MPS/MPO.
    :type tn: numpy.array (2d) of numpy.array (4d)
    :param chi: Truncated bond dimension. (default=None, unrestricted=None)
    :type chi: int or None
    :param tol: Tolerance for treating normalised singular values as zero. (default=None, unrestricted=None)
    :type tol: float or None
    :param start: Start column index (default=None resolves to 0)
    :type start: int or None
    :param stop: Stop column index (exclusive) (default=None resolves to number of columns)
    :type stop: int or None
    :param step: Step (default=None resolves to 1)
    :type step: int or None
    :param mask: Mask of same shape as tn. True indicates tensor may be truncated. (default=None resolves to all True)
    :type mask: numpy.array (2d) of bool or None
    :return: Contraction value
        (if partial contraction: Partially contracted tensor network as MPS/MPO, Multiplier)
    :rtype: mpmath.mpf
        (if partial contraction: list of numpy.array (4d), mpmath.mpf)
    :raises ValueError: if pairwise-contracted MPS/MPOs do not contain a single contiguous list of tensors.
    :raises ValueError: if full contraction and final contraction value is not a scalar.
    """
    assert mask is None or mask.shape == tn.shape, 'mask (if specified) must have same shape as tn'
    # prepare result. store mult in mpmath.mpf to avoid float underflow
    result, mult = None, mp.mpf(1.0)
    # number of columns in tn
    n_cols = tn.shape[1]
    # resolve start, stop, step to column range
    col_range = range(*slice(start, stop, step).indices(n_cols))
    # iterate columns
    for col in col_range:
        # reference mps from tn
        mps = tn[:, col]
        # extract truncate mask for mps
        mps_mask = None if mask is None else mask[:, col]
        if col == col_range[0]:  # is_first_column
            # assign (cast to list so final result is list as per documentation).
            result = list(mps)
        else:  # subsequent columns
            # pair correctly for slice direction
            left, right = (result, mps) if (step is None or step > 0) else (mps, result)
            # contract pairwise
            result = tt_mps.contract_pairwise(left, right)
            # truncate unless (is_last_column and full_contraction)
            if not ((col == col_range[-1]) and (n_cols == len(col_range))):
                result, norm = tt_mps.truncate(result, chi=chi, tol=tol, mask=mps_mask)
                mult *= norm
    # contract to scalar if full_contraction
    if n_cols == len(col_range):
        result = tt_mps.contract_ladder(result)
        result = tt_tsr.as_scalar(result)
        return mult * result
    # otherwise return partial_contraction with multiplier
    else:
        return result, mult
