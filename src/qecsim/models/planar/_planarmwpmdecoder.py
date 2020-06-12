import itertools

from qecsim import graphtools as gt
from qecsim.model import Decoder, cli_description


class _Node:
    # simple class to contain index and implement object reference equality for mwpm
    def __init__(self, index):
        self.index = index


@cli_description('MWPM')
class PlanarMWPMDecoder(Decoder):
    """
    Implements a planar Minimum Weight Perfect Matching (MWPM) decoder.

    Decoding algorithm:

    * The syndrome is resolved to plaquettes using:
      :meth:`qecsim.models.planar.PlanarCode.syndrome_to_plaquette_indices`.
    * For each plaquette the nearest off boundary plaquette is added using:
      :meth:`qecsim.models.planar.PlanarCode.virtual_plaquette_index`.
    * A graph between plaquettes is built with weights given by: :meth:`qecsim.models.planar.PlanarCode.distance`.
    * A MWPM algorithm is used to match plaquettes into pairs.
    * A recovery operator is constructed by applying the shortest path between matching plaquette pairs using:
      :meth:`qecsim.models.planar.PlanarPauli.path` and returned.
    """

    def decode(self, code, syndrome, **kwargs):
        """See :meth:`qecsim.model.Decoder.decode`"""
        # prepare recovery
        recovery_pauli = code.new_pauli()
        # get syndrome indices
        syndrome_indices = code.syndrome_to_plaquette_indices(syndrome)
        # split indices into primal and dual
        primal_indices = (i for i in syndrome_indices if code.is_primal(i))
        dual_indices = (i for i in syndrome_indices if code.is_dual(i))
        # for each type of indices
        for indices in primal_indices, dual_indices:
            # prepare graph
            graph = gt.SimpleGraph()
            # create lists of nodes and corresponding vnodes
            # NOTE: encapsulate indices in node objects that implement object reference equality since we may pass
            # multiple virtual plaquettes with the same index for matching.
            nodes, vnodes = [], []
            for index in indices:
                nodes.append(_Node(index))
                vnodes.append(_Node(code.virtual_plaquette_index(index)))
            # add weighted edges to graph
            for a_node, b_node in itertools.chain(
                    itertools.combinations(nodes, 2),  # all nodes to all nodes
                    itertools.combinations(vnodes, 2),  # all vnodes to all vnodes
                    zip(nodes, vnodes)):  # each node to corresponding vnode
                # find taxi-cab distance between a and b
                distance = code.distance(a_node.index, b_node.index)
                # add edge with weight=distance
                graph.add_edge(a_node, b_node, distance)
            # find MWPM edges {(a, b), (c, d), ...}
            mates = gt.mwpm(graph)
            # iterate edges
            for a_node, b_node in mates:
                # add path to recover
                recovery_pauli.path(a_node.index, b_node.index)
        # return recover as bsf
        return recovery_pauli.to_bsf()

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'Planar MWPM'

    def __repr__(self):
        return '{}()'.format(type(self).__name__)
