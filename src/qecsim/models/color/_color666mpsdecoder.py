import functools
import json
import logging
import operator

import numpy as np
from mpmath import mp

from qecsim import paulitools as pt, tensortools as tt
from qecsim.model import Decoder, cli_description
from qecsim.models.generic import DepolarizingErrorModel

logger = logging.getLogger(__name__)


@cli_description('MPS ([chi] INT, ...)')
class Color666MPSDecoder(Decoder):
    r"""
    Implements a planar Matrix Product State (MPS) decoder.

    A version of this decoder yielded results reported in https://arxiv.org/abs/1812.08186.

    Decoding algorithm:

    * A sample recovery operation :math:`f` is found by resolving the syndrome to plaquettes
      (:meth:`qecsim.models.planar.PlanarCode.syndrome_to_plaquette_indices`), finding the boundary of the same color
      for each plaquette (:meth:`qecsim.models.planar.PlanarCode.virtual_plaquette_index`), constructing a recovery
      operation by applying the correcting operator along a path between each plaquette and its corresponding boundary.
    * The probability of the left coset :math:`fG` of the stabilizer group :math:`G` of the planar code with respect
      to :math:`f` is found by contracting an appropriately defined MPS-based tensor network.
    * The complexity of the algorithm can managed by defining a bond dimension :math:`\chi` to which the MPS bond
      dimension is truncated after each row/column of the tensor network is contracted into the MPS.
    * The probability of cosets :math:`f\bar{X}G`, :math:`f\bar{Y}G` and :math:`f\bar{Z}G` are calculated similarly.
    * Contraction is column-by-column starting from the rightmost column w.r.t. the tensor network described below.
    * A sample recovery operation from the most probable coset is returned.

    Notes:

    * Specifying chi=None gives an exact contract (up to rounding errors) but is exponentially slow in the size of
      the lattice.
    * The code is optimised to evaluate cosets that differ only in the first column together. It is important,
      therefore, that the logical operators of the Color666Pauli act on the first column.

    Lattice to tensor network mapping (by analogy to https://arxiv.org/abs/1405.4883):

    * The following figures show how a size 5 color 6.6.6 triangular lattice, as depicted in
      :class:`qecsim.models.color.Color666Code`, is mapped to a tensor network that contracts to a coset probability.
    * Stabilizers are denoted by ``@``, single qubits are denoted by ``o``, and paired qubits are denoted by ``8``.
    * Single links ``-`` have dimension 4, and double links ``=`` have dimension 16.
    * Stabilizer tensors are defined such that each element has value 1 if all indices are identical, and value 0
      otherwise.
    * Single qubit tensors are defined such that each element has the probability associated with the product of the
      Pauli of the sample recovery restricted to that qubit with the Paulis associated with link indices where 0 -> I,
      1 -> X, 2 -> Y, 3 -> Z.

    Figure 1, size 5 triangular lattice:
    ::

        o
        |
        o   @
          \
        @   o - o
            |     \
        o - o   @   o
        |     \     |
        o   @   o - o   @
          \     |     \
        @   o - o   @   o - o
            |     \     |     \
        o - o   @   o - o   @   o

    Figure 2, tensor network corresponding to figure 1:
    ::

        o
          \
        o - @
        |   | \
        @ - o   o
        | \   \ |
        o   o - @ - o
          \ |   | \   \
        o - @ - o   o - @
        |   | \   \ |   | \
        @ - o   o - @ - o   o
        | \   \ |   | \   \ |
        o   o - @ - o   o - @ - o

    Figure 3, square tensor network equivalent to figure 2:
    ::

        8 = @ - o
        |   |   |
        @ = 8 = @ = 8 = @ - o
        |   |   |   |   |   |
        8 = @ = 8 = @ = 8 = @ - o
        |   |   |   |
        @ = 8 = @ - o
        |
        o

    Figure 4, stabilizers can be split and summed into neighbouring qubits to reduce leg dimension from 16 to 4:
    ::

            8                    8                    8
            |                    |                    |
        8 = @ = 8   =>   8 = @ - @ - @ = 8   =>   8.- @ -.8
            |                    |                    |
            8                    8                    8

    Figure 5, tensor network equivalent to figure 3 (after splitting and summing stabilizers):
    ::

        8.- @ - o
        |   |   |
        @ -.8.- @ -.8.- @ - o
        |   |   |   |   |   |
        8.- @ -.8.- @ -.8.- @ - o
        |   |   |   |
        @ -.8.- @ - o
        |
        o
    """

    def __init__(self, chi=None, tol=None):
        """
        Initialise new planar MPS decoder.

        :param chi: Truncated bond dimension. (default=None, unrestricted=falsy)
        :type chi: int or None
        :param tol: Tolerance for treating normalised singular values as zero. (default=None, unrestricted=falsy)
        :type tol: float or None
        :raises ValueError: if chi is not falsy or > 0.
        :raises ValueError: if tol is not falsy or > 0.0.
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI. (operator.index ensures the parameter can be treated as an int)
            if not (not chi or operator.index(chi) > 0):
                raise ValueError('{} valid chi values are falsy or integer > 0'.format(type(self).__name__))
            if not (not tol or tol > 0.0):
                raise ValueError('{} valid tol values are falsy or number > 0.0'.format(type(self).__name__))
        except TypeError as ex:
            raise TypeError('{} invalid parameter type'.format(type(self).__name__)) from ex
        self._chi = chi
        self._tol = tol
        self._tnc = self.TNC()

    @classmethod
    def sample_recovery(cls, code, syndrome):
        """
        Return a sample Pauli consistent with the syndrome, created by applying a path between each plaquette identified
        by the syndrome and the nearest boundary of the same type as the plaquette.

        :param code: Color 666 code.
        :type code: Color666Code
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :return: Sample recovery operation as color 666 Pauli.
        :rtype: Color666Pauli
        """
        # prepare sample
        sample_recovery = code.new_pauli()
        # iterate syndrome_indices and corrective operator
        for syndrome_indices, op in zip(code.syndrome_to_plaquette_indices(syndrome), ('Z', 'X')):
            # add path of op from syndrome_index:(r1, c1) to virtual_index:(r2, c2) inclusive
            for r1, c1 in syndrome_indices:
                # find nearest off-boundary plaquette
                r2, c2 = code.virtual_plaquette_index((r1, c1))
                # path along horizontal
                step = np.sign(c2 - c1)
                if step:
                    for cc in range(c1, c2 + step, step):
                        if code.is_site((r1, cc)):
                            sample_recovery.site(op, (r1, cc))
                # path along vertical
                step = np.sign(r2 - r1)
                if step:
                    for rr in range(r1, r2 + step, step):
                        if code.is_site((rr, c1)):
                            sample_recovery.site(op, (rr, c1))
        # return sample
        return sample_recovery

    def _coset_probabilities(self, prob_dist, sample_pauli):
        r"""
        Return the (approximate) probability and sample Pauli for the left coset :math:`fG` of the stabilizer group
        :math:`G` of the planar code with respect to the given sample Pauli :math:`f`, as well as for the cosets
        :math:`f\bar{X}G`, :math:`f\bar{Y}G` and :math:`f\bar{Z}G`.

        :param prob_dist: Tuple of probability distribution in the format (P(I), P(X), P(Y), P(Z)).
        :type prob_dist: 4-tuple of float
        :param sample_pauli: Sample color 666 Pauli.
        :type sample_pauli: Color666Pauli
        :return: Coset probabilities, Sample Paulis (both in order I, X, Y, Z)
        E.g. (0.20, 0.10, 0.05, 0.10), (Color666Pauli(...), Color666Pauli(...), Color666Pauli(...), Color666Pauli(...))
        :rtype: 4-tuple of mp.mpf, 4-tuple of Color666Pauli
        """
        # NOTE: all list/tuples in this method are ordered (i, x, y, z)
        # empty log warnings
        log_warnings = []
        # sample_paulis
        sample_paulis = [
            sample_pauli,
            sample_pauli.copy().logical_x(),
            sample_pauli.copy().logical_x().logical_z(),
            sample_pauli.copy().logical_z()
        ]
        # tensor networks
        tns = [self._tnc.create_tn(prob_dist, pauli) for pauli in sample_paulis]
        # probabilities
        coset_ps = [0.0, 0.0, 0.0, 0.0]  # default coset probabilities
        # N.B. After multiplication by mult, coset_ps will be of type mp.mpf so don't process with numpy!
        try:
            # note: cosets differ only in the first column
            ket_i, mult = tt.mps2d.contract(tns[0], chi=self._chi, tol=self._tol, start=-1, stop=0, step=-1)  # tns.i
            coset_ps[0] = tt.mps.inner_product(tns[0][:, 0], ket_i) * mult  # coset_ps.i
            coset_ps[1] = tt.mps.inner_product(tns[1][:, 0], ket_i) * mult  # coset_ps.x
            coset_ps[2] = tt.mps.inner_product(tns[2][:, 0], ket_i) * mult  # coset_ps.y
            coset_ps[3] = tt.mps.inner_product(tns[3][:, 0], ket_i) * mult  # coset_ps.z
        except (ValueError, np.linalg.LinAlgError) as ex:
            log_warnings.append('CONTRACTION FOR I COSET FAILED: {!r}'.format(ex))

        # treat nan as inf so it doesn't get lost
        coset_ps = [mp.inf if mp.isnan(coset_p) else coset_p for coset_p in coset_ps]

        # logging
        if log_warnings:
            log_data = {
                # instance
                'decoder': repr(self),
                # method parameters
                'prob_dist': prob_dist,
                'sample_pauli': pt.pack(sample_pauli.to_bsf()),
                # variables (convert to string because mp.mpf)
                'coset_ps': [repr(p) for p in coset_ps],
            }
            logger.warning('{}: {}'.format(' | '.join(log_warnings), json.dumps(log_data, sort_keys=True)))
        # results
        return tuple(coset_ps), tuple(sample_paulis)

    def decode(self, code, syndrome,
               error_model=DepolarizingErrorModel(),  # noqa: B008
               error_probability=0.1, **kwargs):
        """
        See :meth:`qecsim.model.Decoder.decode`

        Note: The optional keyword parameters ``error_model`` and ``error_probability`` are used to determine the prior
        probability distribution for use in the decoding algorithm. Any provided error model must implement
        :meth:`~qecsim.model.ErrorModel.probability_distribution`.

        :param code: Color 666 code.
        :type code: Color666Code
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
        any_recovery = self.sample_recovery(code, syndrome)
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
        params = [('chi', self._chi), ('tol', self._tol), ]
        return 'Color 6.6.6 MPS ({})'.format(', '.join('{}={}'.format(k, v) for k, v in params if v))

    def __repr__(self):
        return '{}({!r}, {!r})'.format(
            type(self).__name__, self._chi, self._tol,
        )

    class TNC:
        """Tensor network creator"""

        @functools.lru_cache()
        def create_s_node(self, compass_direction=None):
            """
            Return stabilizer tensor.

            :param compass_direction: Location of tensor relative to squared lattice (figure 3), i.e. 'n', 's' or None.
            :type compass_direction: str
            :return: Stabilizer tensor
            :rtype: numpy.array (4d)
            """

            def _node_shape(direction=None):
                """Return shape of tensor including dummy indices."""
                return {  # direction order n,e,se,s,w,nw
                    'n': (1, 4, 4, 4),
                    's': (4, 4, 1, 4),
                    'w': (4, 4, 4, 1),
                }.get(direction, (4, 4, 4, 4))

            node = tt.tsr.delta(_node_shape(compass_direction))
            return node

        @functools.lru_cache()
        def q_node_value(self, prob_dist, f, n, e, s, w):
            """Return qubit tensor element value."""
            paulis = ('I', 'X', 'Y', 'Z')
            op_to_pr = dict(zip(paulis, prob_dist))
            f = pt.pauli_to_bsf(f)
            # n, e, s, w are in {0, 1, 2, 3} so create dict from index to op
            index_to_op = dict(zip((0, 1, 2, 3), pt.pauli_to_bsf(paulis)))
            # apply ops from indices to f
            op = (f + index_to_op[n] + index_to_op[e] + index_to_op[s] + index_to_op[w]) % 2
            # return probability of op
            return op_to_pr[pt.bsf_to_pauli(op)]

        @functools.lru_cache(256)  # 256=2^8 is enough for fixed prob_dist
        def create_q_node(self, prob_dist, fs, compass_direction=None):
            """
            Return qubit tensor.

            :param prob_dist: Tuple of probability distribution in the format (P(I), P(X), P(Y), P(Z)).
            :type prob_dist: 4-tuple of float
            :param fs: Two qubit Paulis, e.g. ('I', None) or ('Z', 'I').
            :type fs: 2-tuple of str
            :param compass_direction: Location of tensor relative to squared lattice (figure 3), e.g. 'n', 'se' or None.
            :type compass_direction: str
            :return: Qubit tensor
            :rtype: numpy.array (4d)
            """

            def _node_shapes(direction=None):
                """Return shape of upper and lower tensors including dummy indices."""
                return {  # direction order n,e,s,w
                    'n': ((1, 4, 1, 4), (1, 4, 4, 4)),
                    'ne': (None, (1, 1, 4, 4)),
                    'e': ((1, 1, 1, 4), None),
                    'se': ((4, 1, 1, 4), None),
                    's': ((4, 4, 1, 4), (1, 4, 1, 4)),
                    'sw': ((4, 1, 1, 1), None),
                    'w': ((4, 4, 1, 1), (1, 4, 4, 1)),
                    'nw': ((1, 4, 1, 1), (1, 4, 4, 1)),
                }.get(direction, ((4, 4, 1, 4), (1, 4, 4, 4)))

            # create tensors with values
            nodes = []
            for shape, f in zip(_node_shapes(compass_direction), fs):
                assert (shape is None) == (f is None), 'Restricted Paulis do not match shapes.'
                if shape is None:  # add dummy tensor
                    nodes.append(np.ones((1, 1, 1, 1), dtype=np.float64))
                else:  # add qubit tensor
                    node = np.empty(shape, dtype=np.float64)
                    for n, e, s, w in np.ndindex(node.shape):
                        node[(n, e, s, w)] = self.q_node_value(prob_dist, f, n, e, s, w)
                    nodes.append(node)
            # merge upper and lower tensors
            node = np.einsum('nesw,sESW->neESwW', nodes[0], nodes[1]).reshape(
                (
                    nodes[0].shape[0],  # n
                    nodes[0].shape[1] * nodes[1].shape[1],  # eE
                    nodes[1].shape[2],  # S
                    nodes[0].shape[3] * nodes[1].shape[3]  # wW
                )
            )
            # multiply dimension-16 legs with deltas to reduce dimension to 4
            if node.shape[1] == 16:
                node = np.einsum('nesw,Ee->nEsw', node, tt.tsr.delta((4, 4, 4)).reshape((4, 16)))
            if node.shape[3] == 16:
                node = np.einsum('nesw,Ww->nesW', node, tt.tsr.delta((4, 4, 4)).reshape((4, 16)))
            return node

        def create_tn(self, prob_dist, sample_pauli):
            """
            Return a tensor network that contracts to the coset probability corresponding to the given probability
            distribution and Pauli.

            See :class:`qecsim.models.color.Color666MPSDecoder` for more details.

            :param prob_dist: Tuple of probability distribution in the format (P(I), P(X), P(Y), P(Z)).
            :type prob_dist: 4-tuple of float
            :param sample_pauli: Sample recovery operation as color 666 Pauli.
            :type sample_pauli: Color666Pauli
            :return: Tensor network
            :rtype: numpy.array (2d) of numpy.array (4d)
            """
            code = sample_pauli.code
            # initialise empty tn
            tn = np.empty((code.size, code.bound + 1), dtype=object)
            for c in range(code.bound + 1):  # iterate pauli columns
                tn_r, tn_c = c // 3, c  # tn rows and columns (column has c // 3 blanks at top)
                r = c  # start at first in bounds row in pauli column
                while r <= code.bound:  # iterate pauli rows
                    if code.is_site((r, c)):  # create node for site(s)
                        # pauli restricted to site
                        f1 = sample_pauli.operator((r, c))
                        # pauli restricted to site below if there
                        if code.is_in_bounds((r + 1, c)) and code.is_site((r + 1, c)):
                            f2 = sample_pauli.operator((r + 1, c))
                        else:
                            f2 = None
                        # create node
                        if c == 0:  # left side
                            if r == 0:
                                node = self.create_q_node(prob_dist, (f1, f2), 'nw')
                            elif r == code.bound:
                                node = self.create_q_node(prob_dist, (f1, None), 'sw')
                            else:
                                node = self.create_q_node(prob_dist, (f1, f2), 'w')
                        elif r == c:  # diagonal side
                            if c == code.bound:
                                node = self.create_q_node(prob_dist, (f1, None), 'e')
                            elif c % 3 == 2:
                                node = self.create_q_node(prob_dist, (None, f1), 'ne')
                            else:
                                node = self.create_q_node(prob_dist, (f1, f2), 'n')
                        elif r == code.bound - 1:  # lower side - 1 because we merge in lower qubits
                            node = self.create_q_node(prob_dist, (f1, f2), 's')
                        elif r == code.bound:  # lower side
                            node = self.create_q_node(prob_dist, (f1, None), 'se')
                        else:  # bulk
                            node = self.create_q_node(prob_dist, (f1, f2))
                        if f2 is not None:
                            r += 1  # skip next merged qubit
                    else:  # create node for stabilizer
                        if c == 0:  # left side
                            node = self.create_s_node('w')
                        elif r == c:  # diagonal side
                            node = self.create_s_node('n')
                        elif r == code.bound:  # lower side
                            node = self.create_s_node('s')
                        else:  # bulk
                            node = self.create_s_node()
                    # add node to tn
                    tn[tn_r, tn_c] = node
                    r += 1  # increment pauli row
                    tn_r += 1  # increment tn row
            return tn
