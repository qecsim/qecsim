import functools
import itertools
import json
import logging
import operator

import numpy as np
from mpmath import mp

from qecsim import paulitools as pt, tensortools as tt
from qecsim.model import Decoder, cli_description
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode

logger = logging.getLogger(__name__)


@cli_description('MPS ([chi] INT >=0, [mode] CHAR, ...)')
class RotatedPlanarMPSDecoder(Decoder):
    r"""
    Implements a rotated planar Matrix Product State (MPS) decoder.

    Decoding algorithm:

    * A sample recovery operation :math:`f` is found by applying a path of X(Z) operators between each Z(X)-plaquette,
      identified by the syndrome, and an X(Z)-boundary.
    * The probability of the left coset :math:`fG` of the stabilizer group :math:`G` of the planar code with respect
      to :math:`f` is found by contracting an appropriately defined MPS-based tensor network (see
      https://arxiv.org/abs/1405.4883).
    * The complexity of the algorithm can managed by defining a bond dimension :math:`\chi` to which the MPS bond
      dimension is truncated after each row/column of the tensor network is contracted into the MPS.
    * The probability of cosets :math:`f\bar{X}G`, :math:`f\bar{Y}G` and :math:`f\bar{Z}G` are calculated similarly.
    * The default contraction is column-by-column but can be set using the mode parameter to row-by-row or the average
      of both contractions.
    * A sample recovery operation from the most probable coset is returned.

    Notes:

    * Specifying chi=None gives an exact contract (up to rounding errors) but is exponentially slow in the size of
      the lattice.
    * Modes:

        * mode='c': contract by columns
        * mode='r': contract by rows
        * mode='a': contract by columns and by rows and, for each coset, take the average of the probabilities.

    Tensor network example:

    3x3 rotated planar code with H or V indicating qubits and hashed/blank plaquettes indicating X/Z stabilizers:
    ::

           /---\
           |   |
           H---V---H--\
           |###|   |##|
           |###|   |##|
           |###|   |##|
        /--V---H---V--/
        |##|   |###|
        |##|   |###|
        |##|   |###|
        \--H---V---H
               |   |
               \---/


    MPS tensor network as per https://arxiv.org/abs/1405.4883 (s=stabilizer):
    ::

             s
            / \
           H   V   H
            \ / \ / \
             s   s   s
            / \ / \ /
           V   H   V
          / \ / \ /
         s   s   s
          \ / \ / \
           H   V   H
                \ /
                 s

    MPS tensor network is contracted diagonally in SE(SW) direction in mode c(r). Equivalently the tensor network is
    rotated 45 degrees anticlockwise for contraction by column(row):
    ::

          0 1 2 3 4

        0     H-s
              | |
        1 s-V-s-V
          | | | |
        2 H-s-H-s-H
            | | | |
        3   V-s-V-s
            | |
        4   s-H
    """

    def __init__(self, chi=None, mode='c', tol=None):
        """
        Initialise new rotated planar MPS decoder.

        :param chi: Truncated bond dimension. (default=None, unrestricted=falsy)
        :type chi: int or None
        :param mode: Contraction mode. (default='c', 'c'=columns, 'r'=rows, 'a'=average)
        :type mode: str
        :param tol: Tolerance for treating normalised singular values as zero. (default=None, unrestricted=falsy)
        :type tol: float or None
        :raises ValueError: if chi is not falsy or > 0.
        :raises ValueError: if mode not in ('c', 'r', 'a').
        :raises ValueError: if tol is not falsy or > 0.0.
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI. (operator.index ensures the parameter can be treated as an int)
            if not (not chi or operator.index(chi) > 0):
                raise ValueError('RotatedPlanarMPSDecoder valid chi values are falsy or integer > 0')
            if mode not in ('c', 'r', 'a'):
                raise ValueError("RotatedPlanarMPSDecoder valid mode values are ('c', 'r', 'a')")
            if not (not tol or tol > 0.0):
                raise ValueError('RotatedPlanarMPSDecoder valid tol values are falsy or number > 0.0')
        except TypeError as ex:
            raise TypeError('RotatedPlanarMPSDecoder invalid parameter type') from ex
        self._chi = chi
        self._mode = mode
        self._tol = tol

    @staticmethod
    def _sample_recovery(code, syndrome):
        """
        Return a sample Pauli consistent with the syndrome, created by applying a path of X(Z) operators between each
        Z(X)-plaquette, identified by the syndrome, and an X(Z)-boundary.

        :param code: Rotated planar code.
        :type code: RotatedPlanarCode
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :return: Sample recovery operation as rotated planar pauli.
        :rtype: RotatedPlanarPauli
        """
        # prepare sample
        sample_recovery = code.new_pauli()
        # ask code for syndrome plaquette_indices
        plaquette_indices = code.syndrome_to_plaquette_indices(syndrome)
        # for each plaquette
        max_site_x, max_site_y = code.site_bounds
        for plaq_index in plaquette_indices:
            # NOTE: plaquette index coincides with site on lower left corner
            plaq_x, plaq_y = plaq_index
            if code.is_z_plaquette(plaq_index):
                # add X path to X-boundary (left or right)
                site_y = max(0, plaq_y)  # ensure sites are within lattice bounds
                site_x_range = range(0, plaq_x + 1)  # to left boundary
                # site_x_range = range(plaq_x + 1, max_site_x + 1)  # to right boundary
                sample_recovery.site('X', *((site_x, site_y) for site_x in site_x_range))
            else:
                # add Z path to Z-boundary (bottom or top)
                site_x = max(0, plaq_x)  # ensure sites are within lattice bounds
                site_y_range = range(0, plaq_y + 1)  # to bottom boundary
                # site_y_range = range(plaq_y + 1, max_site_y + 1)  # to top boundary
                sample_recovery.site('Z', *((site_x, site_y) for site_y in site_y_range))
        # return sample
        return sample_recovery

    def _coset_probabilities(self, prob_dist, sample_pauli):
        r"""
        Return the (approximate) probability and sample Pauli for the left coset :math:`fG` of the stabilizer group
        :math:`G` of the planar code with respect to the given sample Pauli :math:`f`, as well as for the cosets
        :math:`f\bar{X}G`, :math:`f\bar{Y}G` and :math:`f\bar{Z}G`.

        :param prob_dist: Tuple of probability distribution in the format (P(I), P(X), P(Y), P(Z)).
        :type prob_dist: 4-tuple of float
        :param sample_pauli: Sample planar Pauli.
        :type sample_pauli: PlanarPauli
        :return: Coset probabilities, Sample Paulis (both in order I, X, Y, Z)
            E.g. (0.20, 0.10, 0.05, 0.10), (PlanarPauli(...), PlanarPauli(...), PlanarPauli(...), PlanarPauli(...))
        :rtype: 4-tuple of mp.mpf, 4-tuple of PlanarPauli
        """
        # NOTE: all list/tuples in this method are ordered (i, x, y, z)
        # empty log warnings
        log_warnings = []
        # sample paulis
        sample_paulis = (
            sample_pauli,
            sample_pauli.copy().logical_x(),
            sample_pauli.copy().logical_x().logical_z(),
            sample_pauli.copy().logical_z()
        )
        # tensor networks: tns are common to both contraction by column and by row (after transposition)
        tns = [_create_tn(prob_dist, sp) for sp in sample_paulis]
        # probabilities
        coset_ps = (0.0, 0.0, 0.0, 0.0)  # default coset probabilities
        coset_ps_col = coset_ps_row = None  # undefined coset probabilities by column and row
        # N.B. After multiplication by mult, coset_ps will be of type mp.mpf so don't process with numpy!
        if self._mode in ('c', 'a'):
            # evaluate coset probabilities by column
            coset_ps_col = [0.0, 0.0, 0.0, 0.0]  # default coset probabilities
            for i in range(len(tns)):
                try:
                    coset_ps_col[i] = tt.mps2d.contract(tns[i], chi=self._chi, tol=self._tol)
                except (ValueError, np.linalg.LinAlgError) as ex:
                    log_warnings.append('CONTRACTION BY COL FOR {} COSET FAILED: {!r}'.format('IXYZ'[i], ex))
            # treat nan as inf so it doesn't get lost
            coset_ps_col = [mp.inf if mp.isnan(coset_p) else coset_p for coset_p in coset_ps_col]
        if self._mode in ('r', 'a'):
            # evaluate coset probabilities by row
            coset_ps_row = [0.0, 0.0, 0.0, 0.0]  # default coset probabilities
            # transpose tensor networks
            tns = [tt.mps2d.transpose(tn) for tn in tns]
            for i in range(len(tns)):
                try:
                    coset_ps_row[i] = tt.mps2d.contract(tns[i], chi=self._chi, tol=self._tol)
                except (ValueError, np.linalg.LinAlgError) as ex:
                    log_warnings.append('CONTRACTION BY ROW FOR {} COSET FAILED: {!r}'.format('IXYZ'[i], ex))
            # treat nan as inf so it doesn't get lost
            coset_ps_row = [mp.inf if mp.isnan(coset_p) else coset_p for coset_p in coset_ps_row]
        if self._mode == 'c':
            coset_ps = coset_ps_col
        elif self._mode == 'r':
            coset_ps = coset_ps_row
        elif self._mode == 'a':
            # average coset probabilities
            coset_ps = [sum(coset_p) / len(coset_p) for coset_p in zip(coset_ps_col, coset_ps_row)]
        # logging
        if log_warnings:
            log_data = {
                # instance
                'decoder': repr(self),
                # method parameters
                'prob_dist': prob_dist,
                'sample_pauli': pt.pack(sample_pauli.to_bsf()),
                # variables (convert to string because mp.mpf)
                'coset_ps_col': [repr(p) for p in coset_ps_col] if coset_ps_col else None,
                'coset_ps_row': [repr(p) for p in coset_ps_row] if coset_ps_row else None,
                'coset_ps': [repr(p) for p in coset_ps],
            }
            logger.warning('{}: {}'.format(' | '.join(log_warnings), json.dumps(log_data, sort_keys=True)))
        # results
        return tuple(coset_ps), sample_paulis

    def decode(self, code, syndrome, error_model=DepolarizingErrorModel(), error_probability=0.1, **kwargs):
        """
        See :meth:`qecsim.model.Decoder.decode`

        :param code: Rotated planar code.
        :type code: RotatedPlanarCode
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :param error_model: Error model. (default=DepolarizingErrorModel())
        :type error_model: ErrorModel
        :param error_probability: Overall probability of an error on a single qubit. (default=0.1)
        :type error_probability: float
        :return: Recovery operation as binary symplectic vector.
        :rtype: numpy.array (1d)
        """
        # any recovery
        any_recovery = self._sample_recovery(code, syndrome)
        # probability distribution
        prob_dist = error_model.probability_distribution(error_probability)
        # coset probabilities, recovery operations
        coset_ps, recoveries = self._coset_probabilities(prob_dist, any_recovery)
        # most likely recovery operation
        max_coset_p, max_recovery = max(zip(coset_ps, recoveries), key=lambda coset_p_recovery: coset_p_recovery[0])
        # logging
        if not (mp.isfinite(max_coset_p) and max_coset_p > 0):
            log_data = {
                # instance
                'decoder': repr(self),
                # method parameters
                'code': repr(code),
                'syndrome': pt.pack(syndrome),
                'error_model': repr(error_model),
                'error_probability': error_probability,
                # variables
                'prob_dist': prob_dist,
                'coset_ps': [repr(p) for p in coset_ps],  # convert to string because mp.mpf
                # context
                'error': pt.pack(kwargs['error']) if 'error' in kwargs else None,
            }
            logger.warning('NON-POSITIVE-FINITE MAX COSET PROBABILITY: {}'.format(json.dumps(log_data, sort_keys=True)))
        # return most likely recovery operation as bsf
        return max_recovery.to_bsf()

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        params = [('chi', self._chi), ('mode', self._mode), ('tol', self._tol), ]
        return 'Rotated planar MPS ({})'.format(', '.join('{}={}'.format(k, v) for k, v in params if v))

    def __repr__(self):
        return '{}({!r}, {!r}, {!r})'.format(type(self).__name__, self._chi, self._mode, self._tol)


# < Tensor network creation functions >

@functools.lru_cache()
def _h_node_value(prob_dist, f, n, e, s, w):
    """Return horizontal edge tensor element value."""
    paulis = ('I', 'X', 'Y', 'Z')
    op_to_pr = dict(zip(paulis, prob_dist))
    f = pt.pauli_to_bsf(f)
    I, X, Y, Z = pt.pauli_to_bsf(paulis)
    # n, e, s, w are in {0, 1} so multiply op to turn on or off
    op = (f + (n * Z) + (e * X) + (s * Z) + (w * X)) % 2
    return op_to_pr[pt.bsf_to_pauli(op)]


@functools.lru_cache()
def _create_h_node(prob_dist, f, compass_direction=None):
    """Return horizontal qubit tensor, i.e. has X plaquettes to left and right and Z plaquettes above and below."""

    def _shape(compass_direction=None):
        """Return shape of tensor including dummy indices."""
        return {  # (ne, se, sw, nw)
            'n': (2, 2, 2, 1),
            'ne': (1, 2, 2, 1),
            'e': (1, 2, 2, 2),
            'se': (1, 1, 2, 2),
            's': (2, 1, 2, 2),
            'sw': (2, 1, 1, 2),
            'w': (2, 2, 1, 2),
            'nw': (2, 2, 1, 1),
        }.get(compass_direction, (2, 2, 2, 2))

    # create bare h_node
    node = np.empty(_shape(compass_direction), dtype=np.float64)
    # fill values
    for n, e, s, w in np.ndindex(node.shape):
        node[(n, e, s, w)] = _h_node_value(prob_dist, f, n, e, s, w)
    return node


@functools.lru_cache()
def _create_v_node(prob_dist, f, compass_direction=None):
    """Return vertical qubit tensor, i.e. has Z plaquettes to left and right and X plaquettes above and below."""

    def _shape(compass_direction=None):
        """Return shape of tensor including dummy indices."""
        return {  # (ne, se, sw, nw)
            'n': (1, 2, 2, 2),
            'ne': (1, 1, 2, 2),
            'e': (2, 1, 2, 2),
            'se': (2, 1, 1, 2),
            's': (2, 2, 1, 2),
            # 'sw': (2, 2, 1, 1),  # cannot happen
            'w': (2, 2, 2, 1),
            'nw': (1, 2, 2, 1),
        }.get(compass_direction, (2, 2, 2, 2))

    # create bare v_node
    node = np.empty(_shape(compass_direction), dtype=np.float64)
    # fill values
    for n, e, s, w in np.ndindex(node.shape):
        # N.B. order of nesw is rotated relative to h_node
        node[(n, e, s, w)] = _h_node_value(prob_dist, f, e, s, w, n)
    return node


@functools.lru_cache()
def _create_s_node(compass_direction=None):
    """Return stabilizer tensor."""

    def _shape(compass_direction=None):
        """Return shape of tensor including dummy indices."""
        return {  # (ne, se, sw, nw)
            'n': (1, 2, 2, 1),
            'e': (1, 1, 2, 2),
            's': (2, 1, 1, 2),
            'w': (2, 2, 1, 1),
        }.get(compass_direction, (2, 2, 2, 2))

    node = tt.tsr.delta(_shape(compass_direction))
    return node


def _create_tn(prob_dist, sample_pauli):
    """Return a network (numpy.array 2d) of tensors (numpy.array 4d).
    Note: The network contracts to the coset probability of the given sample_pauli.
    """

    def _rotate_q_index(index, code):
        """Convert code site index in format (x, y) to tensor network q-node index in format (r, c)"""
        site_x, site_y = index  # qubit index in (x, y)
        site_r, site_c = code.site_bounds[1] - site_y, site_x  # qubit index in (r, c)
        return code.site_bounds[0] - site_c + site_r, site_r + site_c  # q-node index in (r, c)

    def _rotate_p_index(index, code):
        """Convert code plaquette index in format (x, y) to tensor network s-node index in format (r, c)"""
        q_node_r, q_node_c = _rotate_q_index(index, code)  # q-node index in (r, c)
        return q_node_r - 1, q_node_c  # s-node index in (r, c)

    def _compass_q_direction(index, code):
        """if the code site index lies on border of code lattice then give that direction, else empty string"""
        direction = {code.site_bounds[1]: 'n', 0: 's'}.get(index[1], '')
        direction += {0: 'w', code.site_bounds[0]: 'e'}.get(index[0], '')
        return direction

    def _compass_p_direction(index, code):
        """if the code plaquette index lies on border of code lattice then give that direction, else empty string"""
        direction = {code.site_bounds[1]: 'n', -1: 's'}.get(index[1], '')
        direction += {-1: 'w', code.site_bounds[0]: 'e'}.get(index[0], '')
        return direction

    # extract code
    code = sample_pauli.code
    # initialise empty tn
    tn_max_r, _ = _rotate_q_index((0, 0), code)
    _, tn_max_c = _rotate_q_index((code.site_bounds[0], 0), code)
    tn = np.empty((tn_max_r + 1, tn_max_c + 1), dtype=object)
    # iterate over
    max_site_x, max_site_y = code.site_bounds
    for code_index in itertools.product(range(-1, max_site_x + 1), range(-1, max_site_y + 1)):
        is_z_plaquette = code.is_z_plaquette(code_index)
        if code.is_in_site_bounds(code_index):
            q_node_index = _rotate_q_index(code_index, code)
            q_pauli = sample_pauli.operator(code_index)
            if is_z_plaquette:
                q_node = _create_h_node(prob_dist, q_pauli, _compass_q_direction(code_index, code))
            else:
                q_node = _create_v_node(prob_dist, q_pauli, _compass_q_direction(code_index, code))
            tn[q_node_index] = q_node
        if code.is_in_plaquette_bounds(code_index):
            s_node_index = _rotate_p_index(code_index, code)
            s_node = _create_s_node(_compass_p_direction(code_index, code))
            tn[s_node_index] = s_node
    return tn

# </ Tensor network creation functions >
