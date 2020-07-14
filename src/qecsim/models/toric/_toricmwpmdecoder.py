import itertools

from qecsim import graphtools as gt
from qecsim.model import Decoder, cli_description


@cli_description('MWPM')
class ToricMWPMDecoder(Decoder):
    """
    Implements a toric Minimum Weight Perfect Matching (MWPM) decoder.

    Decoding algorithm:

    * The syndrome is resolved to plaquettes using: :meth:`qecsim.models.toric.ToricCode.syndrome_to_plaquette_indices`.
    * A graph between plaquettes is built with weights given by: :meth:`qecsim.models.toric.ToricCode.distance`.
    * A MWPM algorithm is used to match plaquettes into pairs.
    * A recovery operator is constructed by applying the shortest path between matching plaquette pairs using:
      :meth:`qecsim.models.toric.ToricPauli.path`.
    """

    def decode(self, code, syndrome, **kwargs):
        """See :meth:`qecsim.model.Decoder.decode`"""
        # prepare recovery
        recovery_pauli = code.new_pauli()
        # ask code for plaquette_indices
        plaquette_indices = code.syndrome_to_plaquette_indices(syndrome)
        # for each lattice
        for lattice in (code.PRIMAL_INDEX, code.DUAL_INDEX):
            # prepare lattice graph
            l_graph = gt.SimpleGraph()
            # select lattice plaquettes
            l_plaquette_indices = [(la, r, c) for la, r, c in plaquette_indices if la == lattice]
            # add weighted edges to lattice graph
            for a_index, b_index in itertools.combinations(l_plaquette_indices, 2):
                # add edge with taxi-cab distance between a and b
                l_graph.add_edge(a_index, b_index, code.distance(a_index, b_index))
            # find MWPM edges {(a, b), (c, d), ...}
            l_mates = gt.mwpm(l_graph)
            # iterate edges
            for a_index, b_index in l_mates:
                # add path to recover
                recovery_pauli.path(a_index, b_index)
        # return recover as bsf
        return recovery_pauli.to_bsf()

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'Toric MWPM'

    def __repr__(self):
        return '{}()'.format(type(self).__name__)
