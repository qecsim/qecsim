"""
This module contains functions for Matrix Product States (MPS) and Matrix Product Operators (MPO)

Notes:

* An MPS/MPO is represented by a lists of tensors represented by numpy arrays.
* Most functions support sparse MPS/MPO with None objects in place of tensors but some require a single contiguous list
  of tensors with None only supported at the start and/or end of the MPS/MPO, see function documentation for details.
* All tensors must have 4 indices in the following order (N, E, S, W) where N and S are North and South link indices and
  E and W are East and West physical indices. Irrelevant indices must have dimension 1 to be treated as dummy indices.
* Equivalently, tensors can be thought of as having 4 indices in the following order (L, U, R, D) where L and R are Left
  and Right link indices and U and D are Up and Down physical indices.
* For example, if the tensor network has physical indices of dimension p and link indices of dimension l:

    * A bra MPS is a list of numpy.array with shapes:
      (1, p, l, 1), (l, p, l, 1), ..., (l, p, l, 1), (l, p, 1, 1)
    * An MPO is a list of numpy.array with shapes:
      (1, p, l, p), (l, p, l, p), ..., (l, p, l, p), (l, p, 1, p)
    * A ket MPS is a list of numpy.array with shapes:
      (1, 1, l, p), (l, 1, l, p), ..., (l, 1, l, p), (l, 1, 1, p)

For more details on matrix product states and the functions in this module, see Schollwoeck, U. (2011) "The
density-matrix renormalization group in the age of matrix product states" https://arxiv.org/abs/1008.3477
"""

import functools
import logging
import sys

import numpy as np
from mpmath import mp
from qecsim.tensortools import tsr as tt_tsr
from scipy import linalg as sp_linalg

logger = logging.getLogger(__name__)


def zeros_like(mps):
    """
    Given an MPS/MPO as a lists of tensors with indices (N, E, S, W), return an MPS/MPO consisting of zero tensors with
    virtual indices (N, S) of dimension 1 and physical indices (E, W) of the same dimensions as those of the
    corresponding tensor in the given MPS/MPO.

    (Equivalently, given an MPS/MPO as a lists of tensors with indices (L, U, R, D), return an MPS/MPO consisting of
    zero tensors with virtual indices (L, R) of dimension 1 and physical indices (U, D) of the same dimensions as those
    of the corresponding tensor in the given MPS/MPO.)

    Notes:

    * Sparse MPS/MPO with None objects are supported by simply inserting None objects in the zeros MPS/MPO in the
      corresponding position in the list.

    :param mps: MPS/MPO.
    :type mps: list of numpy.array (4d)
    :return: Zeros MPS/MPO.
    :rtype: list of numpy.array (4d)
    """
    zeros_mps = []
    for tsr in mps:
        if tsr is None:
            zeros_mps.append(None)
        else:
            zeros_mps.append(np.zeros((1, tsr.shape[1], 1, tsr.shape[3]), dtype=tsr.dtype))
    return zeros_mps


def contract_pairwise(left_mps, right_mps):
    """
    Given two MPS/MPO as equal length lists of tensors with indices (N, E, S, W), return a single MPS/MPO evaluated by
    contracting the tensors of the left MPS/MPO and right MPS/MPO pairwise summing over the E index of each left tensor
    and the W index of the corresponding right tensor.

    (Equivalently, given two MPS/MPO as equal length lists of tensors with indices (L, U, R, D), return a single MPS/MPO
    evaluated by contracting the tensors of the left MPS/MPO and right MPS/MPO pairwise summing over the U index of each
    left tensor and the D index of the corresponding right tensor.)

    Notes:

    * Sparse MPS/MPO with None objects are supported by simply copying the corresponding object (tensor or None) from
      the other MPS/MPO to the resultant contracted MPS/MPO.

    :param left_mps: Left MPS/MPO.
    :type left_mps: list of numpy.array (4d)
    :param right_mps: Right MPS/MPO.
    :type right_mps: list of numpy.array (4d)
    :return: Pairwise contracted MPS/MPO.
    :rtype: list of numpy.array (4d)
    """
    assert len(left_mps) == len(right_mps), 'MPS/MPO are different lengths so pairwise contraction cannot be evaluated.'

    def _contract(l, r):
        if l is None:
            return r
        elif r is None:
            return l
        else:
            return np.einsum('nesw,NESe->nNEsSw', l, r).reshape(
                (l.shape[0] * r.shape[0], r.shape[1], l.shape[2] * r.shape[2], l.shape[3])  # (nN)E(sS)w
            )

    # contract left.east with right.west, merge indices and return new mps
    return [_contract(l, r) for l, r in zip(left_mps, right_mps)]


def contract_ladder(mps):
    """
    Given an MPS/MPO as a lists of tensors with indices (N, E, S, W), return the tensor evaluated by contracting the
    MPS/MPO by summing over the S index of the first tensor and the N index of each subsequent tensor.

    (Equivalently, given an MPS/MPO as a lists of tensors with indices (L, U, R, D), return the tensor evaluated by
    contracting the MPS/MPO by summing over the R index of the first tensor and the L index of each subsequent tensor.)

    Notes:

    * Sparse MPS/MPO with None objects are supported provided that they contain a single contiguous list of tensors;
      None objects at the start and/or end are ignored.

    :param mps: MPS/MPO.
    :type mps: list of numpy.array (4d)
    :return: Contracted tensor.
    :rtype: numpy.array (4d)
    """

    # NOTE: Even for large networks, e.g. PlanarCode(29, 29), this contract_ladder gives reasonable results, e.g. 1e-16,
    # so there is no need to store a cumulative norm in an mpmath.mpf variable.

    # iterate tensors in mps (skip None at start or end of mps)
    start, stop = _mps_start_stop_indices(mps)
    # contract mps along links and return value
    return functools.reduce(
        lambda v, t: np.einsum('nesw,sESW->neESwW', v, t).reshape(
            (v.shape[0], v.shape[1] * t.shape[1], t.shape[2], v.shape[3] * t.shape[3])  # n(eE)S(wW)
        ), mps[start:stop])


def inner_product(bra_mps, ket_mps):
    """
    Given bra and ket MPS as equal length lists of tensors with indices (N, E, S, W), return the inner product evaluated
    by pairwise contracting (see :func:`contract_pairwise`) the bra and ket MPS, ladder contracting the resultant MPS
    (see :func:`contract_ladder`), and extracting final contracted tensor as a scalar.

    (Equivalently, given bra and ket MPS as equal length lists of tensors with indices (L, U, R, D), return the inner
    product evaluated by pairwise contracting (see :func:`contract_pairwise`) the bra and ket MPS and ladder contracting
    the resultant MPS (see :func:`contract_ladder`), and extracting final contracted tensor as a scalar).

    Notes:

    * Sparse MPS with None objects are supported, as described in :func:`contract_pairwise`, but the pairwise-contracted
      MPS must contain a single contiguous list of tensors, as described in :func:`contract_ladder`.

    * The bra MPS must have dummy (dimension 1) indices in the N and W positions of the first tensor, W position of
      subsequent tensors, and S and W positions of the last tensor.

      (Equivalently, the bra MPS must have dummy (dimension 1) indices in the L and D positions of the first tensor, D
      position of subsequent tensors, and R and D positions of the last tensor.)

    * The ket MPS must have dummy (dimension 1) indices in the N and E positions of the first tensor, E position of
      subsequent tensors, and S and E positions of the last tensor.

      (Equivalently, the ket MPS must have dummy (dimension 1) indices in the L and U positions of the first tensor, U
      position of subsequent tensors, and R and U positions of the last tensor.)

    :param bra_mps: Bra MPS.
    :type bra_mps: list of numpy.array (4d)
    :param ket_mps: Ket MPS.
    :type ket_mps: list of numpy.array (4d)
    :return: Inner product.
    :rtype: numpy.number
    :raises ValueError: if pairwise-contracted MPS does not contain a single contiguous list of tensors.
    :raises ValueError: if final contraction is not a scalar.
    """

    # NOTE: Even for large networks, e.g. PlanarCode(29, 29), this inner_product gives reasonable results, e.g. 1e-16,
    # so there is no need to store a cumulative norm in an mpmath.mpf variable.

    assert len(bra_mps) == len(ket_mps), 'Bra and ket MPS are different lengths so inner product cannot be evaluated.'

    # contract pairwise
    mps = contract_pairwise(bra_mps, ket_mps)
    # contract mps along links and return value
    tensor = contract_ladder(mps)
    # extract scalar
    return tt_tsr.as_scalar(tensor)


def left_canonical_form(mps, chi=None, tol=None, qr=False, normalise=False, mask=None):
    """
    Given an MPS/MPO as a list of tensors, return the MPS/MPO in left canonical form evaluated by QR decomposition or
    SVD of each tensor in turn, retaining the unitary matrix and contracting the other matrices of the decomposition
    into the next tensor.

    Notes:

    * Sparse MPS/MPO with None objects are supported provided that they contain a single contiguous list of tensors;
      None objects at the start and/or end are ignored.
    * The QR algorithm is faster than the default SVD algorithm but is incompatible with defined chi and tol parameters.
    * If chi is defined, it specifies the number of singular values retained in the SVD of each tensor. This effectively
      truncates the bond dimension of the MPS/MPO to chi.
    * If tol is defined, only the singular values in the SVD of each tensor that, when normalised, are strictly larger
      than tol will be retained. This effectively defines a tolerance below which normalised singular values are
      considered to be zero.
    * If normalise is False, the returned MPS/MPO will not be normalised and the only return value.
    * If normalise is True, the returned MPS/MPO will be normalised and the returned normalisation factor set
      accordingly. However, if the norm is zero then the returned MPS/MPO will consist of zero tensors like those in the
      given MPS/MPO but with bond dimension 1, and the normalisation factor will be 0.0.
    * If mask is specified and the element corresponding to a given tensor is falsy, the tensor will be decomposed using
      the QR algorithm and truncation based on chi or tol will be skipped.

    :param mps: MPS/MPO
    :type mps: list of numpy.array (4d)
    :param chi: Truncated bond dimension. (default=None, unrestricted=None)
    :type chi: int or None
    :param tol: Tolerance for treating normalised singular values as zero. (default=None, unrestricted=None)
    :type tol: float or None
    :param qr: Use QR decomposition instead of SVD. (Incompatible with chi and tol parameters) (default=False)
    :type qr: bool
    :param normalise: Normalise resultant MPS/MPO. (default=False)
    :type normalise: bool
    :param mask: Mask of same length as mps. True indicates tensor may be truncated. (default=None resolves to all True)
    :type mask: list of bool or None
    :return: MPS/MPO in left canonical form
        (if normalise=True: MPS/MPO in left canonical form, Norm of MPS/MPO)
    :rtype: list of numpy.array (4d)
        (if normalise=True: 2-tuple of list of numpy.array (4d), mpmath.mpf)
    :raises ValueError: if MPS/MPO does not contain a single contiguous list of tensors.
    """
    # validate parameters
    assert not (chi and qr), 'Truncated bond dimension chi={} is incompatible with QR'.format(chi)
    assert not (tol and qr), 'Normalised singular value zero tolerance tol={} is incompatible with QR'.format(tol)
    assert mask is None or len(mask) == len(mps), 'Truncate mask (if specified) must have same length as mps'
    # prepare return values
    lcf_mps, norm = list(mps), mp.mpf(1.0)
    # iterate tensors in mps (skip None at start or end of mps)
    start, stop = _mps_start_stop_indices(lcf_mps)
    for row in range(start, stop):
        if not row == stop - 1:  # all but last row
            # reference tensor from mps
            tsr = lcf_mps[row]  # nesw
            # extract truncate mask for tensor
            tsr_mask = True if mask is None else mask[row]
            # reshape tensor=nesw as matrix=(new)s
            matrix = np.einsum('nesw->news', tsr).reshape((tsr.shape[0] * tsr.shape[1] * tsr.shape[3], tsr.shape[2]))
            # decomposition
            if qr or not tsr_mask:  # use QR decomposition
                # decompose matrix into Q, R
                # NOTE: np.linalg.qr is faster but sp_linalg.qr is used for consistency with sp_linalg.svd
                # q, r = np.linalg.qr(matrix, mode='reduced')  # (new)k, ks
                q, r = sp_linalg.qr(matrix, mode='economic')  # (new)k, ks

                # find norm of r
                r_norm = sp_linalg.norm(r)
                if not r_norm:
                    # if r_norm is zero then zero return values and break
                    lcf_mps, norm = zeros_like(mps), mp.mpf(0.0)
                    break
                # normalise r
                r = r / r_norm
                # update overall norm
                norm *= r_norm

                # update matrix
                matrix = q  # (new)s
                # contract R=ns into next_tensor=nesw
                lcf_mps[row + 1] = np.einsum('ns,sESW->nESW', r, lcf_mps[row + 1])  # nesw
            else:  # use SVD decomposition
                # decompose matrix into U, S, V
                # NOTE: np.linalg.svd is faster but sp_linalg.svd allows lapack_driver=gesvd which is more stable
                # u, s, v = np.linalg.svd(matrix, full_matrices=False)  # (new)k, kk, ks
                try:
                    u, s, v = sp_linalg.svd(matrix, full_matrices=False, lapack_driver='gesvd')  # (new)k, kk, ks
                except np.linalg.LinAlgError as lae:
                    logger.warning('SVD by gesvd failed: {!r}. Trying gesdd.'.format(lae))
                    u, s, v = sp_linalg.svd(matrix, full_matrices=False, lapack_driver='gesdd')  # (new)k, kk, ks

                # # Equivalent code using mpmath, set mp.dps = 18 for higher precision
                # from mpmath import mp
                # mpmatrix = mp.matrix(np.vectorize(repr, otypes=[str])(matrix))
                # u, s, v = mp.svd(mpmatrix)
                # u = np.array(u.tolist(), dtype=matrix.dtype)
                # s = np.array(s.tolist(), dtype=matrix.dtype).flatten()
                # v = np.array(v.tolist(), dtype=matrix.dtype)

                # find largest singular value (s is singular values in descending order)
                max_s = s[0]
                if not max_s:
                    # if max_s is zero then zero return values and break
                    lcf_mps, norm = zeros_like(mps), mp.mpf(0.0)
                    break
                # normalise singular values
                s = s / max_s
                # update overall norm
                norm *= max_s

                # if tol defined, discard (normalised) singular values in S which are not greater than tolerance.
                if tol:
                    s = s[s > tol]
                # if chi defined, retain chi largest singular values in S.
                if chi:
                    s = s[:chi]

                # truncate U, V in accordance with S
                u = u[:, :len(s)]
                v = v[:len(s), :]

                # update matrix
                matrix = u  # (new)s
                # contract S=ns and V=ns into next_tensor=nesw
                lcf_mps[row + 1] = np.einsum('ns,sS,SE...->nE...', np.diag(s), v, lcf_mps[row + 1])  # nesw
            # reshape matrix=(new)s as tensor=nesw and update mps
            lcf_mps[row] = np.einsum('news->nesw', matrix.reshape(
                (tsr.shape[0], tsr.shape[1], tsr.shape[3], matrix.shape[1])))
        else:  # last row
            if normalise:
                # NOTE: np.linalg.norm is faster but sp_linalg.norm is used for consistency with sp_linalg.svd
                last_row_norm = sp_linalg.norm(lcf_mps[row])
                if not last_row_norm:
                    # if last_row_norm is zero then zero return values and break
                    lcf_mps, norm = zeros_like(mps), mp.mpf(0.0)
                    break
                # normalise last row
                lcf_mps[row] = lcf_mps[row] / last_row_norm
                # update overall norm
                norm *= last_row_norm
            else:
                # absorb overall norm into last row
                # NOTE: np.array has (at best) range of np.float64 (same as python float) so cast norm to float.
                if norm > sys.float_info.max or norm < sys.float_info.min:
                    logger.warning('Casting out-of-range norm={} to float={}'.format(norm, float(norm)))
                lcf_mps[row] = lcf_mps[row] * float(norm)

    if normalise:
        return lcf_mps, norm
    else:
        return lcf_mps


def reverse(mps):
    """
    Given an MPS/MPO as a list of tensors with indices (N, E, S, W), return the reversed list with N and S indices
    swapped for each tensor, i.e. (N, E, S, W) -> (S, E, N, W).

    (Equivalently, given an MPS/MPO as a list of tensors with indices (L, U, R, D), return the reversed list with L and
    R indices swapped for each tensor, i.e. (L, U, R, D) -> (R, U, L, D).)

    Notes:

    * None objects in the MPS/MPO are copied to the reversed list unchanged.

    :param mps: MPS/MPO
    :type mps: list of numpy.array (4d)
    :return: Reversed MPS/MPO
    :rtype: list of numpy.array (4d)
    """
    # reverse list of tensors and swap north and south indices
    return [(np.einsum('nesw->senw', t) if t is not None else None) for t in reversed(mps)]


def right_canonical_form(mps, chi=None, tol=None, qr=False, normalise=False, mask=None):
    """
    Given an MPS/MPO as a list of tensors, return the MPS/MPO in right canonical form evaluated by reversing the MPS/MPO
    (see :func:`reverse`), converting the reversed MPS to left canonical form (see :func:`left_canonical_form`), and
    reversing the resultant MPS.

    Notes: see :func:`left_canonical_form`

    :param mps: MPS/MPO
    :type mps: list of numpy.array (4d)
    :param chi: Truncated bond dimension. (default=None, unrestricted=None)
    :type chi: int or None
    :param tol: Tolerance for treating normalised singular values as zero. (default=None, unrestricted=None)
    :type tol: float or None
    :param qr: Use QR decomposition instead of SVD. (Incompatible with chi and tol parameters) (default=False)
    :type qr: bool
    :param normalise: Normalise resultant MPS/MPO. (default=False)
    :type normalise: bool
    :param mask: Mask of same length as mps. True indicates tensor may be truncated. (default=None resolves to all True)
    :type mask: list of bool or None
    :return: MPS/MPO in right canonical form
        (if normalise=True: MPS/MPO in right canonical form, Norm of MPS/MPO)
    :rtype: list of numpy.array (4d)
        (if normalise=True: 2-tuple of list of numpy.array (4d), float)
    :raises ValueError: if MPS/MPO does not contain a single contiguous list of tensors.
    """
    # reverse, apply lcf, reverse and return
    reversed_mps = reverse(mps)
    reversed_mask = None if mask is None else list(reversed(mask))
    result = left_canonical_form(reversed_mps, chi=chi, tol=tol, qr=qr, normalise=normalise, mask=reversed_mask)
    if normalise:
        return reverse(result[0]), result[1]
    else:
        return reverse(result)


def bond_dimension(mps):
    """
    Given an MPS/MPO as a list of tensors with indices (N, E, S, W), return the maximum dimension of the N indices.

    (Equivalently, given an MPS/MPO as a list of tensors with indices (L, U, R, D), return the maximum dimension of the
    L indices.)

    Notes:

    * None objects in the MPS/MPO are considered to have dimension 0.

    :param mps: MPS/MPO
    :type mps: list of numpy.array (4d)
    :return: Bond dimension of MPS/MPO.
    :rtype: int
    """
    # return max dimension of north indices, or zero if empty mps
    return max((t.shape[0] if t is not None else 0) for t in mps) if len(mps) else 0


def truncate(mps, chi=None, tol=None, mask=None):
    """
    Given an MPS/MPO as a list of tensors, return the MPS/MPO with truncated bond dimension.

    The evaluation is as follows:

    1. Put the MPS/MPO into left canonical form (see :func:`left_canonical_form`) without truncation
    2. Normalise the final tensor of the MPS/MPO and set the norm.
    3. Put the resultant MPS/MPO into right canonical form (see :func:`right_canonical_form`) truncating to chi singular
       values for each tensor.
    4. Return the resultant MPS/MPO in right canonical form with the norm from step 2.

    Notes:

    * Sparse MPS/MPO with None objects are supported provided that they contain a single contiguous list of tensors;
      None objects at the start and/or end are ignored.
    * If tol is unspecified and the current bond dimension (see :func:`bond_dimension`) is less than or equal to chi or
      the mask is all falsy, the MPS/MPO is returned unmodified and the norm is returned as 1.0.
    * If the final tensor of the MPS/MPO cannot be normalised, the norm is returned as 1.0.

    :param mps: MPS/MPO
    :type mps: list of numpy.array (4d)
    :param chi: Truncated bond dimension. (default=None, unrestricted=None)
    :type chi: int or None
    :param tol: Tolerance for treating normalised singular values as zero. (default=None, unrestricted=None)
    :type tol: float or None
    :param mask: Mask of same length as mps. True indicates tensor may be truncated. (default=None resolves to all True)
    :type mask: list of bool or None
    :return: MPS/MPO with truncated bond dimension, norm from putting into left canonical form.
    :rtype: list of numpy.array (4d), float
    :raises ValueError: if MPS/MPO does not contain a single contiguous list of tensors.
    """
    trunc_mps, norm = mps, 1.0
    if len(trunc_mps) and (tol or (chi and chi < bond_dimension(mps))) and (mask is None or any(mask)):
        lcf_mps, norm = left_canonical_form(mps, qr=True, normalise=True)
        # N.B.: No need to handle zeros lcf_mps specially; in that case, right_canonical_form returns a zeros_mps fast.
        trunc_mps = right_canonical_form(lcf_mps, chi=chi, tol=tol, mask=mask)
    return trunc_mps, norm


# < Internal functions >

def _mps_start_stop_indices(mps):
    """
    Return the start and stop (exclusive) indices of the single contiguous list of tensors in the MPS/MPO.

    :param mps: MPS/MPO
    :type mps: list of numpy.array (4d)
    :return: start index, stop index (exclusive)
    :rtype: 2-tuple of int
    :raises ValueError: if MPS/MPO does not contain a single contiguous list of tensors.
    """
    start, stop = None, None
    for i in range(len(mps)):
        tensor = mps[i]
        if start is None:  # found nothing, looking for first tensor: not None
            if tensor is not None:  # found it
                start = i
        elif stop is None:  # found start, looking for next non-tensor: None
            if tensor is None:  # found it
                stop = i
        elif tensor is not None:  # found start and stop, checking all subsequent are non-tensor: None
            raise ValueError('MPS/MPO does not contain a single contiguous list of tensors')
    if start is None:  # no tensors in MPS
        start, stop = 0, 0
    if stop is None:  # no non-tensors at end of MPS
        stop = len(mps)
    return start, stop

# </ Internal functions >
