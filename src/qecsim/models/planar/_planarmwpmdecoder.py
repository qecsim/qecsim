import functools
import itertools

from qecsim import graphtools as gt
from qecsim.model import Decoder, cli_description


@cli_description('MWPM')
class PlanarMWPMDecoder(Decoder):
    """
    Implements a planar Minimum Weight Perfect Matching (MWPM) decoder.

    Decoding algorithm:

    * The syndrome is resolved to plaquettes defects using:
      :meth:`qecsim.models.planar.PlanarCode.syndrome_to_plaquette_indices`.
    * For each defect the nearest off-boundary plaquette defect is added using:
      :meth:`qecsim.models.planar.PlanarCode.virtual_plaquette_index`.
    * If the total number of defects is odd an extra virtual off-boundary defect is added.
    * A graph between plaquettes is built with weights given by: :meth:`distance`.
    * A MWPM algorithm is used to match plaquettes into pairs.
    * A recovery operator is constructed by applying the shortest path between matching plaquette pairs using:
      :meth:`qecsim.models.planar.PlanarPauli.path` and returned.
    """

    @classmethod
    @functools.lru_cache(maxsize=2 ** 28)  # for MxN lattice, cache_size <~ 2(MN)(MN-1) so handle 100x100 codes.
    def distance(cls, code, a_index, b_index):
        """
        Distance between plaquettes in terms of plaquette steps.

        Note: This implementation returns the taxi-cab distance based on
        :meth:`qecsim.models.planar.PlanarCode.translation`.

        :param code: Planar code.
        :type code: PlanarCode
        :param a_index: Index identifying a plaquette in the format (row, column).
        :type a_index: 2-tuple of int
        :param b_index: Index identifying a plaquette in the format (row, column).
        :type b_index: 2-tuple of int
        :return: Distance between plaquettes.
        :rtype: int
        :raises IndexError: If indices are not plaquette indices on the same lattice.
        """
        row_steps, col_steps = code.translation(a_index, b_index)
        return abs(row_steps) + abs(col_steps)

    def decode(self, code, syndrome, **kwargs):
        """See :meth:`qecsim.model.Decoder.decode`"""
        # prepare recovery
        recovery_pauli = code.new_pauli()
        # get syndrome indices
        syndrome_indices = code.syndrome_to_plaquette_indices(syndrome)
        # split indices into primal and dual
        primal_indices = [i for i in syndrome_indices if code.is_primal(i)]
        dual_indices = [i for i in syndrome_indices if code.is_dual(i)]
        # extra virual indices are deliberately well off-boundary to be separate from nearest virtual indices
        primal_extra_vindex = (-9, -10)
        dual_extra_vindex = (-10, -9)
        # for each type of indices and extra virtual index
        for indices, extra_vindex in (primal_indices, primal_extra_vindex), (dual_indices, dual_extra_vindex):
            # prepare graph
            graph = gt.SimpleGraph()
            # prepare virtual nodes
            vindices = set()
            # add weighted edges between nodes and virtual nodes
            for index in indices:
                vindex = code.virtual_plaquette_index(index)
                vindices.add(vindex)
                distance = self.distance(code, index, vindex)
                graph.add_edge(index, vindex, distance)
            # add extra virtual node if odd number of total nodes
            if (len(indices) + len(vindices)) % 2:
                vindices.add(extra_vindex)
            # add weighted edges to graph between all (non-virtual) nodes
            for a_index, b_index in itertools.combinations(indices, 2):
                distance = self.distance(code, a_index, b_index)
                graph.add_edge(a_index, b_index, distance)
            # add zero weight edges between all virtual nodes
            for a_index, b_index in itertools.combinations(vindices, 2):
                graph.add_edge(a_index, b_index, 0)
            # find MWPM edges {(a, b), (c, d), ...}
            mates = gt.mwpm(graph)
            # iterate edges
            for a_index, b_index in mates:
                # add path to recover
                recovery_pauli.path(a_index, b_index)
        # return recover as bsf
        return recovery_pauli.to_bsf()

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'Planar MWPM'

    def __repr__(self):
        return '{}()'.format(type(self).__name__)
