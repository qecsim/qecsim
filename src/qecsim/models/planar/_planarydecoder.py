import functools
import itertools
import json
import logging
import math
import random

import numpy as np
from mpmath import mp

from qecsim import paulitools as pt
from qecsim import util
from qecsim.error import QecsimError
from qecsim.model import Decoder, cli_description
from qecsim.models.generic import BitPhaseFlipErrorModel

logger = logging.getLogger(__name__)


@cli_description('Y-noise')
class PlanarYDecoder(Decoder):
    """
    Implements a planar Y-noise decoder.

    A version of this decoder yielded results reported in https://arxiv.org/abs/1812.08186.

    NOTE: This decoder will not be optimal for noise models that are not pure Y-noise.

    Decoding algorithm given syndrome:

    * Find a sample all-Y recovery operator matching syndrome.
    * Construct an alternative all-Y recovery operator by adding an all-Y non-trivial logical operator.
    * Find group of all-Y stabilizer operators.
    * Calculate probability of coset of stabilizer group with each recovery operator.
    * Return a recovery operator from highest probability coset.

    Notes:

    * MP math is used to calculate coset probabilities with exact exponent and precision of 50 decimal places.

    Algorithm to find a sample all-Y recovery operator given syndrome (general case):

    * For each syndrome bit, construct a partial recovery operator that triggers the given syndrome bit and some bits on
      the right/lower boundary (whichever is shorter).
    * Combine partial recoveries to give combined partial recovery.
    * Subtract error syndrome from combined partial recovery syndrome to give residual syndrome on lower boundary.
    * Find residual recovery operator that triggers residual syndrome on right/lower boundary.
    * Return combined partial and residual recovery as all-Y recovery operator.

    Notes:

    * Constructing a partial recovery operator: An operator that triggers a given syndrome bit and some bits on the
      right/lower boundary can be constructed directly by by applying a zig-zag pattern of Y starting right/below the
      given syndrome bit. For a p x q code, there are n - 1 = pq + (p-1)(q-1) - 1 distinct syndrome bits.
    * For co-prime codes, it is always possible to trigger a single syndrome bit on a boundary by apply Y next to the
      syndrome bit and bouncing diagonally around the lattice until a corner is encountered. That is destabilizers exist
      for co-prime codes. This is implemented in this decoder.
    * For codes where one side is an integer multiple of the other, the residual syndrome will always be trivial because
      patterns of Y that trigger no syndrome bits on 3 boundaries also trigger no syndrome bits on the 4th boundary.
      This is implemented in this decoder.
    * For constant gcd > 1 codes where one side is not a multiple of the other, it is possible to combine the above
      approaches for rectangular regions of the lattice where one side is a multiple of the other and one co-prime
      region to find a sample recovery operator. This is not yet implemented in this decoder; instead a look-up table of
      residual recovery operators is constructed by combining operators consisting of applying a zig-zag pattern of Y
      starting at each of the edges on the left/upper boundary.

    Example::

        Error:     Syndrome:
        Y--------  ---------
          Y   |      |   |
        ----Y----  ------V--
          |   |      | P |
        ---------  ---------
          |   |      |   |
        ---------  ---------

        Partial    Partial       Partial    Partial       Combined   Combined
        recovery:  syndrome:     recovery:  syndrome:     partial    partial
                                                          recovery:  syndrome:
        ---------  ---------     ---------  ---------     ---------  ---------
          |   |      |   |         |   |      |   |         |   |      |   |
        ---------  ---------     ---------  ------V--     ---------  ------V--
          |   |      | P |    +    |   Y      |   |    =    |   Y      | P |
        ----Y----  ---------     ----Y---Y  ---------     --------Y  ---------
          Y   Y      |   |         Y   Y      |   |         |   |      |   |
        Y---Y---Y  --V---V--     Y---Y----  --V------     --------Y  ------V--

        Combined   Combined      Residual   Residual      Sample     Sample
        partial    partial       recovery:  syndrome:     recovery:  syndrome:
        recovery:  syndrome:
        ---------  ---------     Y--------  ---------     Y--------  ---------
          |   |      |   |         Y   |      |   |         Y   |      |   |
        ---------  ------V--     ----Y----  ---------     ----Y----  ------V--
          |   Y      | P |    +    |   Y      |   |    =    |   |      | P |
        --------Y  ---------     --------Y  ---------     ---------  ---------
          |   |      |   |         |   |      |   |         |   |      |   |
        --------Y  ------V--     --------Y  ------V--     ---------  ---------

    Algorithm to find group of all-Y stabilizers and logicals algorithm (general case):

    * Create generators: For row=0 and column=each of 0 to gcd(p,q) - 1, apply Y at row,column and in a SE direction,
      bouncing just beyond the boundaries until a loop is completed or a corner is encountered.
    * Create full group: Combine generators in a possible combinations without repetition.
    * Split into stabilizers/logicals: Check commutation relations with logical X and Z of code.

    Notes:

    * There are 2^gcd(p,q) all-Y (trivial and non-trivial) logical operator for a p x q code.
    * For coprime codes, the all-Y non-trivial logical operator consists of Y on all horizontal edges.
    * For square codes, an all-Y non-trivial logical operator consists of Y on the major diagonal edges.

    """

    CHUNK_LEN = 500  # default chunk length for processing numpy arrays on start-up. smaller = less memory but more time

    @classmethod
    @functools.lru_cache(maxsize=2 ** 23)  # big enough for 128 codes of size 100x100
    def _snake_fill(cls, code, start_index, down=True):
        """
        Return an operator with Y on the given start index and below such that the only syndrome bits are directly above
        the start index and on the lower boundary of the code.

        NOTE: If start_index is out of bounds, then identity is returned.

        For example::

            ---------
              |   |
            ---------
              Y   |
            Y-|-Y-|--
              Y   Y
            --|-Y-|-Y

        :param code: Planar code
        :type code: PlanarCode
        :param start_index: Start index in format (row, column)
        :type start_index: 2-tuple of int
        :param down: Should fill downwards (default=True, falsy=rightwards).
        :type down: boolean
        :return: Snake filled-down operator in binary symplectic form.
        :rtype: numpy.array (1d)
        """
        # start with identity
        pauli = code.new_pauli()
        # if start_index out of bounds, return identity
        if not code.is_in_bounds(start_index):
            return pauli.to_bsf()
        # expand indices
        start_r, start_c = start_index
        min_r, min_c = 0, 0
        max_r, max_c = code.bounds
        # seed edges in SE direction to any boundary
        for seed_r, seed_c in zip(range(start_r, max_r + 1), range(start_c, max_c + 1)):
            # index iterators
            # N.B. we bounce just beyond boundaries so exclusive limits are min-2 and max+2
            if down:  # snake from seed index in SW direction to lower boundary (bouncing as necessary)
                index_it = zip(
                    range(seed_r, max_r + 1),  # from seed_r to lower boundary
                    itertools.cycle(  # cycle column indices
                        itertools.chain(
                            range(seed_c, min_c - 2, -1),  # from seed_c to just beyond left boundary
                            range(min_c, max_c + 2),  # from left boundary to just beyond right boundary
                            range(max_c, seed_c, -1)  # from right boundary to just before seed_c
                        )
                    )
                )
            else:  # snake from seed index in NW direction to right boundary (bouncing as necessary)
                index_it = zip(
                    itertools.cycle(  # cycle row indices
                        itertools.chain(
                            range(seed_r, min_r - 2, -1),  # from seed_r to just beyond upper boundary
                            range(min_r, max_r + 2),  # from upper boundary to just beyond lower boundary
                            range(max_r, seed_r, -1)  # from lower boundary to just before seed_r
                        )
                    ),
                    range(seed_c, max_c + 1)  # from seed_c to right boundary
                )
            # snake-fill
            for snake_index in index_it:
                pauli.site('Y', snake_index)
        # return as bsf
        return pauli.to_bsf()

    @classmethod
    @functools.lru_cache()  # big enough for 128 codes
    def _residual_syndrome_to_recovery_map(cls, code):
        """
        Return map of residual syndromes to residual recovery operators.

        NOTE: In order to reduce the number of residual syndromes, they are placed on the smaller boundary. That is,
        for a p x q code, if p < q then syndrome bits are pushed to right boundary, otherwise to lower boundary.

        NOTE: For memory efficiency, syndromes and recoveries stored in the map in packed format, see
        :func:`qecsim.paulitools.pack`.

        :param code: Planar code
        :type code: PlanarCode
        :return: Residual syndrome to recovery operator map.
        :rtype: dict of packed bsf to packed bsf
        """
        # initialise return values
        residual_map = {}

        def _add(residual_recoveries, skip_trivial):
            """
            Add residual recoveries to residual_map keyed by syndrome as tuple.

            NOTE: This method processes recoveries in chunks so it is memory efficient to pass an iterator.

            :param residual_recoveries: Residual recoveries in bsf
            :type residual_recoveries: iterator of numpy.array (1d)
            :param skip_trivial: Do not add recoveries with trivial syndromes
            :type skip_trivial: bool
            """
            for chunk in util.chunker(residual_recoveries, PlanarYDecoder.CHUNK_LEN):
                residual_recoveries_chunk = np.array(tuple(chunk))
                # residual syndromes
                residual_syndromes_chunk = pt.bsp(residual_recoveries_chunk, code.stabilizers.T)
                # add to map
                for syndrome, recovery in zip(residual_syndromes_chunk, residual_recoveries_chunk):
                    if not (skip_trivial and not np.any(syndrome)):
                        residual_map.setdefault(pt.pack(syndrome), pt.pack(recovery))

        # N.B. we add identity at the end so it is not included in the all possible products part
        # snake fill to trigger syndrome bits on one boundary
        if code.size[0] < code.size[1]:  # push syndrome bits to right boundary
            # add snake-fill-right operators for each edge on left boundary
            _add((cls._snake_fill(code, (start_r, 0), down=False)
                  for start_r in range(0, code.bounds[0] + 1, 2)), skip_trivial=True)
        else:  # push syndrome bits to lower boundary
            # add snake-fill-down operators for each edge on upper boundary
            _add((cls._snake_fill(code, (0, start_c), down=True)
                  for start_c in range(0, code.bounds[1] + 1, 2)), skip_trivial=True)
        # add product of all combinations of operators (any length, without repetition)
        operators = list(pt.unpack(o) for o in residual_map.values())  # copy of operators in map so far
        _add(
            (np.sum(operator_set, axis=0) % 2
             for operator_set in itertools.chain.from_iterable(
                itertools.combinations(operators, n_operators)
                for n_operators in range(1, len(operators) + 1))),
            skip_trivial=True)
        # add identity
        _add([code.new_pauli().to_bsf()], skip_trivial=False)
        # return map
        return residual_map

    @classmethod
    def _residual_recovery(cls, code, syndrome):
        """
        Return residual recovery consistent with (lower boundary) syndrome (if possible).

        :param code: Planar code
        :type code: PlanarCode
        :param syndrome: Lower boundary syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :return: Residual recovery operation in binary symplectic form.
        :rtype: numpy.array (1d)
        """
        try:
            # get residual recovery from map
            return pt.unpack(cls._residual_syndrome_to_recovery_map(code)[pt.pack(syndrome)])
        except KeyError:
            # N.B. this should not happen if a pure Y-noise model is used
            log_data = {
                # parameters
                'code': repr(code),
                'syndrome': pt.pack(syndrome),
            }
            logger.warning('RESIDUAL RECOVERY NOT FOUND: {}'.format(json.dumps(log_data, sort_keys=True)))
            # return identity
            return code.new_pauli().to_bsf()

    @classmethod
    @functools.lru_cache(maxsize=2 ** 22)  # big enough for 128 codes of size 100x100
    def _partial_recovery(cls, code, syndrome_index):
        """
        Return partial recovery triggering given syndrome bit plus syndrome bits on one boundary.

        NOTE: In order to reduce the number of boundary syndrome bits, they are placed on the smaller boundary. That is,
        for a p x q code, if p < q then syndrome bits are pushed to right boundary, otherwise to lower boundary.

        :param code: Planar code
        :type code: PlanarCode
        :param syndrome_index: Index of syndrome bit in format (row, column)
        :type syndrome_index: 2-tuple of int
        :return: Partial recovery operation in binary symplectic form.
        :rtype: numpy.array (1d)
        """
        # if syndrome bit is out of bounds, return identity
        if not code.is_in_bounds(syndrome_index):
            return code.new_pauli().to_bsf()
        # snake fill to trigger target syndrome bit and syndrome bits on one boundary
        if code.size[0] < code.size[1]:  # push syndrome bits to right boundary
            # start on qubit to right of syndrome_index
            return cls._snake_fill(code, (syndrome_index[0], syndrome_index[1] + 1), down=False)
        else:  # push syndrome bits to lower boundary
            # start on qubit below syndrome_index
            return cls._snake_fill(code, (syndrome_index[0] + 1, syndrome_index[1]), down=True)

    @classmethod
    @functools.lru_cache(maxsize=2 ** 22)  # big enough for 128 codes of size 100x100
    def _destabilizer(cls, code, syndrome_index):
        """
        Return destabilizer for given co-prime code and syndrome index.

        :param code: Co-prime planar code
        :type code: PlanarCode
        :param syndrome_index: Index of syndrome bit in format (row, column)
        :type syndrome_index: 2-tuple of int
        :return: Destabilizer in binary symplectic form.
        :rtype: numpy.array (1d)
        :raises ValueError: if code size is not co-prime.
        """
        if math.gcd(*code.size) != 1:
            raise ValueError('Y destabilizers only exist in general for co-prime codes.')
        # if syndrome bit is out of bounds, return identity
        if not code.is_in_bounds(syndrome_index):
            return code.new_pauli().to_bsf()
        # destabilizer in bsf (new object to avoid modifying cached objects from private methods)
        destabilizer = code.new_pauli().to_bsf()
        # build destabilizer that triggers syndrome bit plus syndrome bits on lower boundary
        # start snake fill downwards, from qubit below, stopping at lower boundary
        start_index = syndrome_index[0] + 1, syndrome_index[1]
        destabilizer ^= cls._snake_fill(code, start_index, down=True)
        # resolve syndrome bits of destabilizer so far
        destabilizer_syndrome = pt.bsp(destabilizer, code.stabilizers.T)
        destabilizer_syndrome_indices = code.syndrome_to_plaquette_indices(destabilizer_syndrome)
        # correct any syndrome bits that remain (they should all be on lower boundary)
        for index in destabilizer_syndrome_indices ^ {syndrome_index}:  # symmetric_difference
            # start snaking in NW direction, from qubit to left, stopping in a corner
            start_index = index[0], index[1] - 1
            destabilizer ^= cls._snake(code, start_index, se=False, full=False)
        return destabilizer

    @classmethod
    def _sample_recovery(cls, code, syndrome):
        """
        Return an all-Y sample Pauli consistent with the syndrome (if possible).

        :param code: Planar code
        :type code: PlanarCode
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :return: Sample recovery operation in binary symplectic form.
        :rtype: numpy.array (1d)
        """
        # TODO: For constant gcd > 1, consider optimizing by using gaussian elimination to find residual recovery
        # TODO: For constant gcd > 1, consider optimizing by using pushing syndrome bits beyond boundary of sub-regions
        #       that have one side a multiple of the other until a co-prime sub-region then fix remaining syndrome bits
        #       beyond boundaries of all previous sub-regions.
        # coprime
        coprime = math.gcd(*code.size) == 1
        # recovery in bsf (new object to avoid modifying cached objects from private methods)
        recovery = code.new_pauli().to_bsf()
        # get syndrome indices
        syndrome_indices = code.syndrome_to_plaquette_indices(syndrome)
        # add destabilizers / partial recoveries
        for index in syndrome_indices:
            if coprime:  # destabilizers exist for co-prime codes
                recovery ^= cls._destabilizer(code, index)
            else:
                recovery ^= cls._partial_recovery(code, index)
        # find recovery_syndrome
        recovery_syndrome = pt.bsp(recovery, code.stabilizers.T)
        # find residual syndrome
        residual_syndrome = syndrome ^ recovery_syndrome
        # N.B. we should not get any residual syndrome for co-prime codes or codes with one side a multiple of the other
        if np.any(residual_syndrome):
            if coprime or max(code.size) % min(code.size) == 0:
                logger.warning('UNEXPECTED RESIDUAL SYNDROME FOR {}'.format(code))
            # find residual_recovery
            residual_recovery = cls._residual_recovery(code, residual_syndrome)
            # add residual_recovery
            recovery ^= residual_recovery
        return recovery

    @classmethod
    @functools.lru_cache(maxsize=2 ** 25)  # big enough for 128 codes of size 100x100
    def _snake(cls, code, start_index, se=True, full=True, skip_first=False):
        """
        Return operator after applying snake of Y from start index and bouncing in SE (NW) direction(s).

        NOTE: If start_index is out of bounds, then identity is returned.

        For example::

            ----Y--------
              |   |   |
            --------Y----
              |   |   |
            ------------Y

            Y---Y----
              |   Y
            Y-------Y
              Y   |
            ----Y---Y

        :param code: Planar code
        :type code: PlanarCode
        :param start_index: Start index in format (row, column).
        :type start_index: 2-tuple of int
        :param se: Should snake in SE direction initially (default=True, falsy=NW direction).
        :type se: bool
        :param full: Should snake in opposite direction if not looped in initial direction (default=True).
        :type full: bool
        :param skip_first: Should not apply Y to start_index (default=False).
        :type skip_first: bool
        :return: Operator in binary symplectic form.
        :rtype: numpy.array (1d)
        """
        # start with identity
        pauli = code.new_pauli()
        # if start_index out of bounds, return identity
        if not code.is_in_bounds(start_index):
            return pauli.to_bsf()
        # expand indices
        start_r, start_c = start_index
        min_r, min_c = 0, 0
        max_r, max_c = code.bounds
        # index iterators
        if se:  # snake in SE direction
            index_it = zip(
                itertools.cycle(  # cycle row indices
                    itertools.chain(
                        range(start_r, max_r + 2),  # from start_r to just beyond lower boundary
                        range(max_r, min_r - 2, -1),  # from lower boundary to just beyond upper boundary
                        range(min_r, start_r)  # from upper boundary to just before start_r
                    )
                ),
                itertools.cycle(  # cycle column indices
                    itertools.chain(
                        range(start_c, max_c + 2),  # from start_c to just beyond right boundary
                        range(max_c, min_c - 2, -1),  # from right boundary to just beyond left boundary
                        range(min_c, start_c)  # from left boundary to just before start_c
                    )
                )
            )
        else:  # snake in NW direction
            index_it = zip(
                itertools.cycle(  # cycle row indices
                    itertools.chain(
                        range(start_r, min_r - 2, -1),  # from start_r to just beyond upper boundary
                        range(min_r, max_r + 2),  # from upper boundary to just beyond lower boundary
                        range(max_r, start_r, -1)  # from lower boundary to just before start_r
                    )
                ),
                itertools.cycle(  # cycle column indices
                    itertools.chain(
                        range(start_c, min_c - 2, -1),  # from start_c to just beyond left boundary
                        range(min_c, max_c + 2),  # from left boundary to just beyond right boundary
                        range(max_c, start_c, -1)  # from right boundary to just before start_c
                    )
                )
            )
        # initialise return value
        looped = False
        # infinite loop protection
        max_count = code.n_k_d[0] * 100
        count = 0
        # initialise indices
        previous_index, current_index = None, None
        # snake
        for next_index in index_it:
            # stop if next_index is initial_index and in given direction from current_index (i.e. we've looped)
            if next_index == start_index and (
                    tuple(np.subtract(next_index, (1, 1) if se else (-1, -1))) == current_index):
                looped = True
                break
            # stop if next_index is previous_index (i.e. we've bounced off a corner)
            if next_index == previous_index:
                break
            # bump indices
            previous_index, current_index = current_index, next_index
            # apply Y (unless skip_first and null previous_index)
            if not (skip_first and previous_index is None):
                pauli.site('Y', current_index)
            # infinite loop protection
            count += 1
            if count > max_count:
                break
        # report infinite loop protection
        if count > max_count:
            raise QecsimError('Infinite loop applying Y to {} starting at {} in se={} direction.'.format(
                code, start_index, se))
        # convert to bsf
        operator = pauli.to_bsf()
        # if full requested and we have not looped, then apply snake in opposite direction
        if full and not looped:
            operator ^ cls._snake(code, start_index, full=False, se=not se, skip_first=True)
        # return as bsf
        return operator

    @classmethod
    @functools.lru_cache()  # big enough for 128 codes
    def _y_stabilizers(cls, code):
        """
        Return complete group of all-Y stabilizers.

        Example stabilizers::

            Y---Y----
              |   Y
            Y-------Y
              Y   |
            ----Y---Y

            ----Y---Y
              Y   |
            Y-------Y
              |   Y
            Y---Y----

            Y---Y---Y---Y
              |   |   |
            Y---Y---Y---Y

        :param code: Planar code
        :type code: PlanarCode
        :return: All all-Y stabilizers in binary symplectic form as rows in array.
        :rtype: numpy.array (2d)
        """
        # all-Y stabilizer generators: snake from (0, 2), (0, 4), ...
        y_stabilizer_generators = []
        for c in range(2, 2 * math.gcd(*code.size), 2):
            start_index = 0, c
            # add generator created by snaking
            y_stabilizer_generators.append(cls._snake(code, start_index))
        # all-Y stabilizers: add product of all combinations of generators (any length, without repetition)
        y_stabilizers = []
        for generator_set in itertools.chain.from_iterable(
                itertools.combinations(y_stabilizer_generators, n_generators)
                for n_generators in range(1, len(y_stabilizer_generators) + 1)):
            y_stabilizers.append(np.sum(generator_set, axis=0) % 2)
        # all-Y stabilizers: add identity
        y_stabilizers.append(code.new_pauli().to_bsf())
        # return as np.array
        return np.array(y_stabilizers)

    @classmethod
    @functools.lru_cache()  # big enough for 128 codes
    def _y_logical(cls, code):
        """
        Return a single all-Y logical (non-trivial) operator.

        Example logicals::

            Y---Y---Y---Y
              |   |   |
            Y---Y---Y---Y
              |   |   |
            Y---Y---Y---Y

            Y--------
              Y   |
            ----Y----
              |   Y
            --------Y

            Y-----------Y
              Y   |   Y
            ----Y---Y----

        :param code: Planar code
        :type code: PlanarCode
        :return: A single all-Y logical (non-trivial) operator in binary symplectic form.
        :rtype: numpy.array (1d)
        """
        # all-Y logical: snake from (0, 0)
        return cls._snake(code, (0, 0))

    @classmethod
    @functools.lru_cache(maxsize=2 ** 22)  # enough for 128 pairs of p_i, p_y with a code of size 100x100
    def _operator_probability(cls, p_i, n_i, p_y, n_y):
        """
        Return the probability of the given operator conditioned by the given probabilities.

        NOTE: Assumes operator consists only of I and Y.

        :param p_i: Probability of I
        :type p_i: float
        :param n_i: Number of I
        :type n_i: int
        :param p_y: Probability of Y
        :type p_y: float
        :param n_y: Number of Y
        :type n_y: int
        :return: All-Y operator probability
        :rtype: mp.mpf
        """
        return mp.mpf(p_y) ** n_y * mp.mpf(p_i) ** n_i

    @classmethod
    def _coset_probability(cls, prob_dist, coset):
        """
        Return the probability of the given coset conditioned by the given probability distribution.

        NOTE: Assumes coset elements consist only of I and Y.

        :param prob_dist: Tuple of probability distribution in the format (P(I), P(X), P(Y), P(Z)).
        :type prob_dist: 4-tuple of float
        :param coset: Coset elements in binary symplectic form as rows in arrays.
        :type coset: numpy.array (2d)
        :return: Coset probability
        :rtype: mp.mpf
        """
        # extract I and Y probabilities
        p_i, _, p_y, _ = prob_dist
        # number of qubits
        n = coset.shape[1] // 2
        # number of y errors per coset element (use numpy for integer arithmetic for speed)
        n_ys = np.sum(coset[:, :n], axis=1)
        # sum over coset element probabilities (use mpmath for float arithmetic for precision and to avoid underflow)
        return sum(cls._operator_probability(p_i, n - n_y, p_y, n_y) for n_y in n_ys)
        # ALTERNATIVE CODE BELOW: This would be much faster but can fail with underflow
        # return sum(p_i ** (n - n_ys) * p_y ** n_ys)

    @mp.workdps(50)
    def decode(self, code, syndrome,
               error_model=BitPhaseFlipErrorModel(),  # noqa: B008
               error_probability=0.1, **kwargs):
        """See :meth:`qecsim.model.Decoder.decode`"""
        # extract probability distribution
        prob_dist = error_model.probability_distribution(error_probability)
        # build sample recoveries
        sample_recovery_1 = self._sample_recovery(code, syndrome)
        sample_recovery_2 = sample_recovery_1 ^ self._y_logical(code)
        # calculate coset probabilities
        y_stabilizers = self._y_stabilizers(code)
        coset_1 = y_stabilizers ^ sample_recovery_1  # numpy broadcasting applies recovery to all stabilizers
        coset_probability_1 = self._coset_probability(prob_dist, coset_1)
        coset_2 = y_stabilizers ^ sample_recovery_2  # numpy broadcasting applies recovery to all stabilizers
        coset_probability_2 = self._coset_probability(prob_dist, coset_2)
        # return sample from highest probability coset
        if coset_probability_1 == coset_probability_2:
            # choose randomly if coset probabilities equal
            return random.choice([sample_recovery_1, sample_recovery_2])
        elif coset_probability_1 > coset_probability_2:
            return sample_recovery_1
        else:
            return sample_recovery_2

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'Planar Y'

    def __repr__(self):
        return '{}()'.format(type(self).__name__)
