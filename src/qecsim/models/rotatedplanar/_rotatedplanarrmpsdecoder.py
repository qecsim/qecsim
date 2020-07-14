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

logger = logging.getLogger(__name__)


@cli_description('Rotated MPS ([chi] INT >=0, [mode] CHAR, ...)')
class RotatedPlanarRMPSDecoder(Decoder):
    r"""
    Implements a rotated planar Rotated Matrix Product State (RMPS) decoder.

    A version of this decoder yielded results reported in https://arxiv.org/abs/1812.08186.

    Decoding algorithm:

    * A sample recovery operation :math:`f` is found by applying a path of X(Z) operators between each Z(X)-plaquette,
      identified by the syndrome, and an X(Z)-boundary.
    * The probability of the left coset :math:`fG` of the stabilizer group :math:`G` of the planar code with respect
      to :math:`f` is found by contracting an appropriately defined MPS-based tensor network (see
      https://arxiv.org/abs/1405.4883).
    * Since this is a rotated MPS decoder, the links of the network are rotated 45 degrees by splitting each stabilizer
      node into 4 delta nodes that are absorbed into the neighbouring qubit nodes.
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

    * Contracting by columns (i.e. truncating vertical links) may give different coset probabilities to contracting by
      rows (i.e. truncating horizontal links). However, the effect is symmetric in that transposing the sample_pauli on
      the lattice and exchanging X and Z single Paulis reverses the difference between X and Z cosets probabilities.

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

    Links are rotated by splitting stabilizers and absorbing them into neighbouring qubits.
    For even columns of stabilizers (according to indexing defined in :class:`qecsim.models.planar.RotatedPlanarCode`),
    a 'lucky' horseshoe shape is used:
    ::

        H   V      H     V
         \ /        \   /       H V
          s    =>    s s    =>  | |
         / \         | |        V-H
        V   H        s-s
                    /   \
                   V     H

    For odd columns, an 'unlucky' horseshoe shape is used:
    ::

        H   V      H     V
         \ /        \   /       H-V
          s    =>    s-s    =>  | |
         / \         | |        V H
        V   H        s s
                    /   \
                   V     H

    Resultant MPS tensor network, where horizontal (vertical) bonds have dimension 2 (4) respectively.
    ::

          0 1 2
        0 H-V-H
          | | |
        1 V-H-V
          | | |
        2 H-V-H
    """

    def __init__(self, chi=None, mode='c', tol=None):
        """
        Initialise new rotated planar RMPS decoder.

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
                raise ValueError('RotatedPlanarRMPSDecoder valid chi values are falsy or integer > 0')
            if mode not in ('c', 'r', 'a'):
                raise ValueError("RotatedPlanarRMPSDecoder valid mode values are ('c', 'r', 'a')")
            if not (not tol or tol > 0.0):
                raise ValueError('RotatedPlanarRMPSDecoder valid tol values are falsy or number > 0.0')
        except TypeError as ex:
            raise TypeError('RotatedPlanarRMPSDecoder invalid parameter type') from ex
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
            # note: I,Z and X,Y cosets differ only in the last column (logical Z)
            try:
                bra_i, mult = tt.mps2d.contract(tns[0], chi=self._chi, tol=self._tol, stop=-1)  # tns.i
                coset_ps_col[0] = tt.mps.inner_product(bra_i, tns[0][:, -1]) * mult  # coset_ps_col.i
                coset_ps_col[3] = tt.mps.inner_product(bra_i, tns[3][:, -1]) * mult  # coset_ps_col.z
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY COL FOR I/Z COSET FAILED: {!r}'.format(ex))
            try:
                bra_z, mult = tt.mps2d.contract(tns[1], chi=self._chi, tol=self._tol, stop=-1)  # tns.x
                coset_ps_col[1] = tt.mps.inner_product(bra_z, tns[1][:, -1]) * mult  # coset_ps_col.x
                coset_ps_col[2] = tt.mps.inner_product(bra_z, tns[2][:, -1]) * mult  # coset_ps_col.y
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY COL FOR X/Y COSET FAILED: {!r}'.format(ex))
            # treat nan as inf so it doesn't get lost
            coset_ps_col = [mp.inf if mp.isnan(coset_p) else coset_p for coset_p in coset_ps_col]
        if self._mode in ('r', 'a'):
            # evaluate coset probabilities by row
            coset_ps_row = [0.0, 0.0, 0.0, 0.0]  # default coset probabilities
            # transpose tensor networks
            tns = [tt.mps2d.transpose(tn) for tn in tns]
            # note: I,X and Z,Y cosets differ only in the last row (logical X)
            try:
                bra_i, mult = tt.mps2d.contract(tns[0], chi=self._chi, tol=self._tol, stop=-1)  # tns.i
                coset_ps_row[0] = tt.mps.inner_product(bra_i, tns[0][:, -1]) * mult  # coset_ps_row.i
                coset_ps_row[1] = tt.mps.inner_product(bra_i, tns[1][:, -1]) * mult  # coset_ps_row.x
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY ROW FOR I/X COSET FAILED: {!r}'.format(ex))
            try:
                bra_x, mult = tt.mps2d.contract(tns[3], chi=self._chi, tol=self._tol, stop=-1)  # tns.x
                coset_ps_row[3] = tt.mps.inner_product(bra_x, tns[3][:, -1]) * mult  # coset_ps_row.z
                coset_ps_row[2] = tt.mps.inner_product(bra_x, tns[2][:, -1]) * mult  # coset_ps_row.y
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY ROW FOR Z/Y COSET FAILED: {!r}'.format(ex))
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

    def decode(self, code, syndrome,
               error_model=DepolarizingErrorModel(),  # noqa: B008
               error_probability=0.1, **kwargs):
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
        return 'Rotated planar RMPS ({})'.format(', '.join('{}={}'.format(k, v) for k, v in params if v))

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


@functools.lru_cache(maxsize=256)
def _create_q_node(prob_dist, f, h_node, even_column, compass_direction=None):
    """Create q-node for tensor network.

    Notes:

    * H-nodes have Z-plaquettes above and below (i.e. in NE and SW directions).
    * V-nodes have Z-plaquettes on either side (i.e. in NW and SE directions).
    * Columns are considered even/odd according to indexing defined in :class:`qecsim.models.planar.RotatedPlanarCode`.

    :param h_node: If h-node, else V-node.
    :type h_node: bool
    :param prob_dist: Probability distribution in the format (Pr(I), Pr(X), Pr(Y), Pr(Z)).
    :type prob_dist: (float, float, float, float)
    :param f: Pauli operator on qubit as 'I', 'X', 'Y', or 'Z'.
    :type f: str
    :param even_column: If even column, else odd column.
    :type even_column: bool
    :param compass_direction: Compass direction as 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw', or falsy for bulk.
    :type compass_direction: str
    :return: Q-node for tensor network.
    :rtype: numpy.array (4d)
    """

    # H indicates h-node with shape (n,e,s,w).
    # * indicates delta nodes with shapes (n,I,j), (e,J,k), (s,K,l), (w,L,i) for n-, e-, s-, and w-deltas respectively.
    # n,e,s,w,i,j,k,I,J,K are bond labels
    #
    #   i     I
    #   |     |
    # L-*     *-j
    #    \   /
    #    w\ /n
    #      H
    #    s/ \e
    #    /   \
    # l-*     *-J
    #   |     |
    #   K     k
    #
    # Deltas are absorbed into h-node over n,e,s,w legs and reshaped as follows:
    # nesw -> (iI)(jJ)(Kk)(Ll)

    # define shapes # q_node:(n, e, s, w); delta_nodes: n:(n,I,j), e:(e,J,k), s:(s,K,l), w:(w,L,i)
    if h_node:
        # bulk h-node
        q_shape = (2, 2, 2, 2)
        if even_column:
            n_shape, e_shape, s_shape, w_shape = (2, 2, 2), (2, 1, 2), (2, 2, 2), (2, 1, 2)
        else:
            n_shape, e_shape, s_shape, w_shape = (2, 2, 1), (2, 2, 2), (2, 2, 1), (2, 2, 2)
        # modifications for directions
        if compass_direction == 'n':
            q_shape = (2, 2, 2, 1)
            n_shape, w_shape = (2, 1, 2), (1, 1, 1)
        elif compass_direction == 'ne':
            q_shape = (1, 2, 2, 1)
            n_shape, e_shape, w_shape = (1, 1, 1), (2, 1, 2), (1, 1, 1)
        elif compass_direction == 'e':
            q_shape = (1, 2, 2, 2)
            n_shape, e_shape = (1, 1, 1), (2, 1, 2)
        elif compass_direction == 'se':  # always even
            q_shape = (1, 1, 2, 2)
            n_shape, e_shape, s_shape = (1, 1, 1), (1, 1, 1), (2, 1, 2)
        elif compass_direction == 's':  # always even
            q_shape = (2, 1, 2, 2)
            e_shape, s_shape = (1, 1, 1), (2, 1, 2)
        elif compass_direction == 'sw':  # always even
            q_shape = (2, 1, 1, 2)
            e_shape, s_shape, w_shape = (1, 1, 1), (1, 1, 1), (2, 1, 2)
        elif compass_direction == 'w':  # always even
            q_shape = (2, 2, 1, 2)
            s_shape, w_shape = (1, 1, 1), (2, 1, 2)
        elif compass_direction == 'nw':  # always even
            q_shape = (2, 2, 1, 1)
            n_shape, s_shape, w_shape = (2, 1, 2), (1, 1, 1), (1, 1, 1)
    else:
        # bulk v-node
        q_shape = (2, 2, 2, 2)
        if even_column:
            n_shape, e_shape, s_shape, w_shape = (2, 2, 2), (2, 1, 2), (2, 2, 2), (2, 1, 2)
        else:
            n_shape, e_shape, s_shape, w_shape = (2, 2, 1), (2, 2, 2), (2, 2, 1), (2, 2, 2)
        # modifications for directions
        if compass_direction == 'n':
            q_shape = (1, 2, 2, 2)
            n_shape, w_shape = (1, 1, 1), (2, 2, 1)
        elif compass_direction == 'ne':
            q_shape = (1, 1, 2, 2)
            n_shape, e_shape, w_shape = (1, 1, 1), (1, 1, 1), (2, 2, 1)
        elif compass_direction == 'e':
            q_shape = (2, 1, 2, 2)
            n_shape, e_shape = (2, 2, 1), (1, 1, 1)
        elif compass_direction == 'se':  # always odd
            q_shape = (2, 1, 1, 2)
            n_shape, e_shape, s_shape = (2, 2, 1), (1, 1, 1), (1, 1, 1)
        elif compass_direction == 's':  # always odd
            q_shape = (2, 2, 1, 2)
            e_shape, s_shape = (2, 2, 1), (1, 1, 1)
        elif compass_direction == 'sw':  # not possible
            raise ValueError('Cannot have v-node in SW corner of lattice.')
        elif compass_direction == 'w':  # always even
            q_shape = (2, 2, 2, 1)
            s_shape, w_shape = (2, 2, 1), (1, 1, 1)
        elif compass_direction == 'nw':  # always even
            q_shape = (1, 2, 2, 1)
            n_shape, s_shape, w_shape = (1, 1, 1), (2, 2, 1), (1, 1, 1)

    # create deltas
    n_delta = tt.tsr.delta(n_shape)
    e_delta = tt.tsr.delta(e_shape)
    s_delta = tt.tsr.delta(s_shape)
    w_delta = tt.tsr.delta(w_shape)
    # create q_node and fill values
    q_node = np.empty(q_shape, dtype=np.float64)
    for n, e, s, w in np.ndindex(q_node.shape):
        if h_node:
            # N.B. for h_node use standard order of nesw
            q_node[(n, e, s, w)] = _h_node_value(prob_dist, f, n, e, s, w)
        else:
            # N.B. for v_node order of nesw is rotated relative to h_node
            q_node[(n, e, s, w)] = _h_node_value(prob_dist, f, e, s, w, n)
    # derive combined node shape
    shape = (w_shape[2] * n_shape[1], n_shape[2] * e_shape[1], e_shape[2] * s_shape[1], s_shape[2] * w_shape[1])
    # create combined node by absorbing deltas into q_node: nesw -> (iI)(jJ)(Kk)(Ll)
    node = np.einsum('nesw,nIj,eJk,sKl,wLi->iIjJKkLl', q_node, n_delta, e_delta, s_delta, w_delta).reshape(shape)
    # return combined node
    return node


def _create_tn(prob_dist, sample_pauli):
    """Return a network (numpy.array 2d) of tensors (numpy.array 4d).
    Note: The network contracts to the coset probability of the given sample_pauli.
    """

    def _xy_to_rc_index(index, code):
        """Convert code site index in format (x, y) to tensor network q-node index in format (r, c)"""
        x, y = index
        return code.site_bounds[1] - y, x

    def _compass_direction(index, code):
        """if the code site index in format (x, y) lies on border then give that direction, else empty string"""
        direction = {code.site_bounds[1]: 'n', 0: 's'}.get(index[1], '')
        direction += {0: 'w', code.site_bounds[0]: 'e'}.get(index[0], '')
        return direction

    # extract code
    code = sample_pauli.code
    # initialise empty tn
    tn = np.empty(code.size, dtype=object)
    # iterate over site indices
    max_site_x, max_site_y = code.site_bounds
    for code_index in itertools.product(range(max_site_x + 1), range(max_site_y + 1)):
        # prepare parameters
        is_h_node = code.is_z_plaquette(code_index)
        q_node_index = _xy_to_rc_index(code_index, code)
        is_even_column = not (q_node_index[1] % 2)
        q_pauli = sample_pauli.operator(code_index)
        q_direction = _compass_direction(code_index, code)
        # create q-node
        q_node = _create_q_node(prob_dist, q_pauli, is_h_node, is_even_column, q_direction)
        # add q-node to tensor network
        tn[q_node_index] = q_node
    return tn

# </ Tensor network creation functions >
