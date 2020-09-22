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


@cli_description('MPS ([chi] INT >=0, [mode] CHAR, ...)')
class PlanarMPSDecoder(Decoder):
    r"""
    Implements a planar Matrix Product State (MPS) decoder.

    A version of this decoder yielded results reported in https://arxiv.org/abs/1708.08474
    and https://arxiv.org/abs/1812.08186.

    Decoding algorithm:

    * A sample recovery operation :math:`f` is found by resolving the syndrome to plaquettes
      (:meth:`qecsim.models.planar.PlanarCode.syndrome_to_plaquette_indices`), finding the nearest boundary of the same
      type for each plaquette (:meth:`qecsim.models.planar.PlanarCode.virtual_plaquette_index`), constructing a recovery
      operation by applying the path between each plaquette and its corresponding boundary
      (:meth:`qecsim.models.planar.PlanarPauli.path`).
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

    * Contracting by columns (i.e. truncating vertical links) may give different coset probabilities to contracting by
      rows (i.e. truncating horizontal links). However, the effect is symmetric in that transposing the sample_pauli on
      the lattice and exchanging X and Z single Paulis reverses the difference between X and Z cosets probabilities.
    * Specifying stp (skip truncate probability) gives the probability that a tensor is not truncated in the approximate
      contraction controlled by chi. This can be used to break the symmetry of the contraction approximation.
    * The code is optimised to evaluate cosets that differ only in the last column/row together. It is important,
      therefore, that the logical X/Z operators of the PlanarPauli act on the last column/row respectively.

    Tensor network example:

    3x4 planar code (H=qubit on horizontal edge, V=qubit on vertical edge):
    ::

        H---H---H---H
          |   |   |
          V   V   V
          |   |   |
        H---H---H---H
          |   |   |
          V   V   V
          |   |   |
        H---H---H---H

    MPS tensor network as per https://arxiv.org/abs/1405.4883 (s=stabilizer):
    ::

         0 1 2 3 4 5 6
        0H-s-H-s-H-s-H
         | | | | | | |
        1s-V-s-V-s-V-s
         | | | | | | |
        2H-s-H-s-H-s-H
         | | | | | | |
        3s-V-s-V-s-V-s
         | | | | | | |
        4H-s-H-s-H-s-H
    """

    def __init__(self, chi=None, mode='c', stp=None, tol=None):
        """
        Initialise new planar MPS decoder.

        :param chi: Truncated bond dimension. (default=None, unrestricted=falsy)
        :type chi: int or None
        :param mode: Contraction mode. (default='c', 'c'=columns, 'r'=rows, 'a'=average)
        :type mode: str
        :param stp: Skip truncate probability. (default=None, disabled=falsy)
        :type stp: float or None
        :param tol: Tolerance for treating normalised singular values as zero. (default=None, unrestricted=falsy)
        :type tol: float or None
        :raises ValueError: if chi is not falsy or > 0.
        :raises ValueError: if mode not in ('c', 'r', 'a').
        :raises ValueError: if stp is not falsy or 1.0 >= stp > 0.0.
        :raises ValueError: if tol is not falsy or > 0.0.
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI. (operator.index ensures the parameter can be treated as an int)
            if not (not chi or operator.index(chi) > 0):
                raise ValueError('{} valid chi values are falsy or integer > 0'.format(type(self).__name__))
            if mode not in ('c', 'r', 'a'):
                raise ValueError("{} valid mode values are ('c', 'r', 'a')".format(type(self).__name__))
            if not (not stp or 1.0 >= stp > 0.0):
                raise ValueError('{} valid stp values are falsy or 1.0 >= number > 0.0'.format(type(self).__name__))
            if not (not tol or tol > 0.0):
                raise ValueError('{} valid tol values are falsy or number > 0.0'.format(type(self).__name__))
        except TypeError as ex:
            raise TypeError('{} invalid parameter type'.format(type(self).__name__)) from ex
        self._chi = chi
        self._mode = mode
        self._stp = stp
        self._tol = tol
        self._tnc = self.TNC()

    @classmethod
    def sample_recovery(cls, code, syndrome):
        """
        Return a sample Pauli consistent with the syndrome, created by applying a path between each plaquette identified
        by the syndrome and the nearest boundary of the same type as the plaquette.

        :param code: Planar code.
        :type code: PlanarCode
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :return: Sample recovery operation as planar pauli.
        :rtype: PlanarPauli
        """
        # prepare sample
        sample_recovery = code.new_pauli()
        # ask code for syndrome plaquette_indices
        plaquette_indices = code.syndrome_to_plaquette_indices(syndrome)
        # for each plaquette
        for index in plaquette_indices:
            # find nearest off-boundary plaquette
            virtual_index = code.virtual_plaquette_index(index)
            # add path to boundary
            sample_recovery.path(index, virtual_index)
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
        # sample_paulis
        sample_paulis = [
            sample_pauli,
            sample_pauli.copy().logical_x(),
            sample_pauli.copy().logical_x().logical_z(),
            sample_pauli.copy().logical_z()
        ]
        # tensor networks
        tns = [self._tnc.create_tn(prob_dist, pauli) for pauli in sample_paulis]
        mask = self._tnc.create_mask(self._stp, tns[0].shape)  # same mask for all tns
        # probabilities
        coset_ps = (0.0, 0.0, 0.0, 0.0)  # default coset probabilities
        coset_ps_col = coset_ps_row = None  # undefined coset probabilities by column and row
        # N.B. After multiplication by mult, coset_ps will be of type mp.mpf so don't process with numpy!
        if self._mode in ('c', 'a'):
            # evaluate coset probabilities by column
            coset_ps_col = [0.0, 0.0, 0.0, 0.0]  # default coset probabilities
            # note: I,X and Z,Y cosets differ only in the last column (logical X)
            try:
                bra_i, mult = tt.mps2d.contract(tns[0], chi=self._chi, tol=self._tol, stop=-1, mask=mask)  # tns.i
                coset_ps_col[0] = tt.mps.inner_product(bra_i, tns[0][:, -1]) * mult  # coset_ps_col.i
                coset_ps_col[1] = tt.mps.inner_product(bra_i, tns[1][:, -1]) * mult  # coset_ps_col.x
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY COL FOR I COSET FAILED: {!r}'.format(ex))
            try:
                bra_z, mult = tt.mps2d.contract(tns[3], chi=self._chi, tol=self._tol, stop=-1, mask=mask)  # tns.z
                coset_ps_col[2] = tt.mps.inner_product(bra_z, tns[2][:, -1]) * mult  # coset_ps_col.y
                coset_ps_col[3] = tt.mps.inner_product(bra_z, tns[3][:, -1]) * mult  # coset_ps_col.z
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY COL FOR Z COSET FAILED: {!r}'.format(ex))
            # treat nan as inf so it doesn't get lost
            coset_ps_col = [mp.inf if mp.isnan(coset_p) else coset_p for coset_p in coset_ps_col]
        if self._mode in ('r', 'a'):
            # evaluate coset probabilities by row
            coset_ps_row = [0.0, 0.0, 0.0, 0.0]  # default coset probabilities
            # transpose tensor networks
            tns = [tt.mps2d.transpose(tn) for tn in tns]
            mask = None if mask is None else mask.transpose()
            # note: I,Z and X,Y cosets differ only in the last row (logical Z)
            try:
                bra_i, mult = tt.mps2d.contract(tns[0], chi=self._chi, tol=self._tol, stop=-1, mask=mask)  # tns.i
                coset_ps_row[0] = tt.mps.inner_product(bra_i, tns[0][:, -1]) * mult  # coset_ps_row.i
                coset_ps_row[3] = tt.mps.inner_product(bra_i, tns[3][:, -1]) * mult  # coset_ps_row.z
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY ROW FOR I COSET FAILED: {!r}'.format(ex))
            try:
                bra_x, mult = tt.mps2d.contract(tns[1], chi=self._chi, tol=self._tol, stop=-1, mask=mask)  # tns.x
                coset_ps_row[1] = tt.mps.inner_product(bra_x, tns[1][:, -1]) * mult  # coset_ps_row.x
                coset_ps_row[2] = tt.mps.inner_product(bra_x, tns[2][:, -1]) * mult  # coset_ps_row.y
            except (ValueError, np.linalg.LinAlgError) as ex:
                log_warnings.append('CONTRACTION BY ROW FOR X COSET FAILED: {!r}'.format(ex))
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
        return tuple(coset_ps), tuple(sample_paulis)

    def decode(self, code, syndrome,
               error_model=DepolarizingErrorModel(),  # noqa: B008
               error_probability=0.1, **kwargs):
        """
        See :meth:`qecsim.model.Decoder.decode`

        :param code: Planar code.
        :type code: PlanarCode
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
        params = [('chi', self._chi), ('mode', self._mode), ('stp', self._stp), ('tol', self._tol), ]
        return 'Planar MPS ({})'.format(', '.join('{}={}'.format(k, v) for k, v in params if v))

    def __repr__(self):
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            type(self).__name__, self._chi, self._mode, self._stp, self._tol,
        )

    class TNC:
        """Tensor network creator"""

        def node_shape(self, compass_direction=None):
            """Return shape of tensor including dummy indices."""
            return {
                'n': (1, 2, 2, 2),
                'ne': (1, 1, 2, 2),
                'e': (2, 1, 2, 2),
                'se': (2, 1, 1, 2),
                's': (2, 2, 1, 2),
                'sw': (2, 2, 1, 1),
                'w': (2, 2, 2, 1),
                'nw': (1, 2, 2, 1),
            }.get(compass_direction, (2, 2, 2, 2))

        @functools.lru_cache()
        def create_s_node(self, compass_direction=None):
            """Return stabilizer tensor."""
            return tt.tsr.delta(self.node_shape(compass_direction))

        @functools.lru_cache()
        def h_node_value(self, prob_dist, f, n, e, s, w):
            """Return horizontal edge tensor element value."""
            paulis = ('I', 'X', 'Y', 'Z')
            op_to_pr = dict(zip(paulis, prob_dist))
            f = pt.pauli_to_bsf(f)
            I, X, Y, Z = pt.pauli_to_bsf(paulis)
            # n, e, s, w are in {0, 1} so multiply op to turn on or off
            op = (f + (n * Z) + (e * X) + (s * Z) + (w * X)) % 2
            return op_to_pr[pt.bsf_to_pauli(op)]

        @functools.lru_cache()
        def create_h_node(self, prob_dist, f, compass_direction=None):
            """Return horizontal edge tensor."""
            node = np.empty(self.node_shape(compass_direction), dtype=np.float64)
            for n, e, s, w in np.ndindex(node.shape):
                node[(n, e, s, w)] = self.h_node_value(prob_dist, f, n, e, s, w)
            return node

        @functools.lru_cache()
        def create_v_node(self, prob_dist, f, compass_direction=None):
            """Return vertical edge tensor."""
            node = np.empty(self.node_shape(compass_direction), dtype=np.float64)
            for n, e, s, w in np.ndindex(node.shape):
                node[(n, e, s, w)] = self.h_node_value(prob_dist, f, e, s, w, n)
            return node

        def create_tn(self, prob_dist, sample_pauli):
            """Return a network (numpy.array 2d) of tensors (numpy.array 4d).
            Note: The network contracts to the coset probability of the given sample_pauli."""
            # initialise empty tn
            tn = np.empty((2 * sample_pauli.code.size[0] - 1, 2 * sample_pauli.code.size[1] - 1), dtype=object)
            # index to direction maps
            row_to_direction = {0: 'n', tn.shape[0] - 1: 's'}
            col_to_direction = {0: 'w', tn.shape[1] - 1: 'e'}
            # add nodes to tn
            for row, col in np.ndindex(tn.shape):
                # find direction
                direction = row_to_direction.get(row, '')
                direction += col_to_direction.get(col, '')
                # add node
                if 0 == row % 2 == col % 2:
                    tn[row, col] = self.create_h_node(prob_dist, sample_pauli.operator((row, col)), direction)
                elif 1 == row % 2 == col % 2:
                    tn[row, col] = self.create_v_node(prob_dist, sample_pauli.operator((row, col)), direction)
                else:
                    tn[row, col] = self.create_s_node(direction)
            return tn

        def create_mask(self, stp, shape):
            """Return truncate mask (numpy.array 2d) of elements True with probability 1-stp and False with probability
            stp. Note: None is returned if stp (skip truncate probability) is falsy."""
            rng = np.random.default_rng()
            return rng.choice((True, False), size=shape, p=(1 - stp, stp)) if stp else None
