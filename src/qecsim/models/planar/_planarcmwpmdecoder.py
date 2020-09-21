import functools
import itertools
import logging
import operator

import numpy as np

from qecsim import graphtools as gt
from qecsim.model import Decoder, cli_description

logger = logging.getLogger(__name__)


class StepGrid:
    """
    Grid of providing a weighted background for MWPM.

    Methods:

    * Set background weights based on matched index pairs: :meth:`set_background`.
    * Resolve taxi-cab distance, weighted by the background, between a pair of indices: :meth:`distance`.
    * Minimum weight perfect matching in a graph of indices where sites are weighted based on distance through the
      background: :meth:`mwpm`.

    """

    def __init__(self, code):
        """
        Initialise new step grid.

        :param code: Planar code.
        :type code: PlanarCode
        """
        self._code = code
        # NOTE: size is bounds +3 (+1 because bounds is inclusive, +2 for border to include virtual indices)
        # NOTE: we use dtype=float because we get overflow to negative with int64 at 3^40.
        self._grid = np.zeros((code.bounds[0] + 3, code.bounds[1] + 3), dtype=float)
        self.set_background()

    def set_background(self, matched_indices=None, factor=3, initial=1, box_shape='t'):
        """
        Set grid background from matched syndrome indices.

        Note:

        * The grid is initialised with initial value at all sites and zero elsewhere.
        * For each matched pair of syndrome indices, all sites outside the box-shape, bounding the pair of indices,
          are multiplied by factor.

        :param matched_indices: Matched pairs of syndrome indices.
        :type matched_indices: set of 2-tuples of 2-tuple of int
        :param factor: Multiplication factor. (default=3)
        :type factor: int or float
        :param initial: Initial edge weight. (default=1)
        :type initial: int or float
        :param box_shape: Shape of background boxes. (default='t', 't'=tight, 'r'=rounded, 'f'=fitted, 'l'=loose)
        :type box_shape: str
        """
        assert box_shape in ('t', 'r', 'f', 'l'), 'StepGrid: Unsupported box shape'
        self._grid.fill(0)
        self._grid[::2, ::2] = initial
        self._grid[1::2, 1::2] = initial
        if matched_indices:
            for src_i, tgt_i in matched_indices:
                # if both indices virtual then skip
                if not (self._code.is_in_bounds(src_i) or self._code.is_in_bounds(tgt_i)):
                    continue
                # multiply all elements outside box defined by syndrome indices
                if box_shape == 'r':
                    self._box_rounded(src_i, tgt_i, factor)
                elif box_shape == 'f':
                    self._box_fitted(src_i, tgt_i, factor)
                elif box_shape == 'l':
                    self._box_loose(src_i, tgt_i, factor)
                else:
                    self._box_tight(src_i, tgt_i, factor)

    @classmethod
    def _syndrome_to_grid_index(cls, *syndrome_indices):
        """
        Convert given syndrome indices to grid indices allowing for border of virtual indices around grid.

        :param syndrome_indices: Any number of syndrome indices.
        :type syndrome_indices: 2-tuples of int
        :return: Grid indices
        :rtype: 2-tuples of int
        """
        return tuple((syndrome_index[0] + 1, syndrome_index[1] + 1) for syndrome_index in syndrome_indices)

    @classmethod
    def _box_corners(cls, *indices):
        """
        Top-left and bottom-right corners of box that bounds the given indices.

        :param indices: Any number of indices.
        :type indices: 2-tuples of int
        :return: Top-left and bottom-right indices.
        :rtype: 2-tuple of 2-tuple of int
        """
        min_r = min(indices, key=lambda i: i[0])[0]
        max_r = max(indices, key=lambda i: i[0])[0]
        min_c = min(indices, key=lambda i: i[1])[1]
        max_c = max(indices, key=lambda i: i[1])[1]
        return (min_r, min_c), (max_r, max_c)

    def _multiply_box(self, top_left_i, bottom_right_i, factor):
        """
        Multiply all sites inside box with given corner indices (boundary sites are multiplied).

        :param top_left_i: Top-left grid index.
        :type top_left_i: 2-tuple of int
        :param bottom_right_i: Bottom-right grid index.
        :type bottom_right_i: 2-tuple of int
        :param factor: Multiplication factor.
        :type factor: int or float
        """
        min_r = max(0, top_left_i[0])
        max_r = min(self._grid.shape[0] - 1, bottom_right_i[0])
        min_c = max(0, top_left_i[1])
        max_c = min(self._grid.shape[1] - 1, bottom_right_i[1])
        self._grid[min_r:max_r + 1, min_c:max_c + 1] *= factor

    def _multiply_box_complement(self, top_left_i, bottom_right_i, factor):
        """
        Multiply all sites outside box with given corner indices (boundary sites are not multiplied).

        :param top_left_i: Top-left grid index.
        :type top_left_i: 2-tuple of int
        :param bottom_right_i: Bottom-right grid index.
        :type bottom_right_i: 2-tuple of int
        :param factor: Multiplication factor.
        :type factor: int or float
        """
        min_r = max(0, top_left_i[0])
        max_r = min(self._grid.shape[0] - 1, bottom_right_i[0])
        min_c = max(0, top_left_i[1])
        max_c = min(self._grid.shape[1] - 1, bottom_right_i[1])
        # top rows
        self._grid[:min_r] *= factor
        # bottom rows
        self._grid[max_r + 1:] *= factor
        # left cols (between top and bottom rows)
        self._grid[min_r:max_r + 1, :min_c] *= factor
        # right cols (between top and bottom rows)
        self._grid[min_r:max_r + 1, max_c + 1:] *= factor

    def _box_tight(self, src_i, tgt_i, factor):
        """
        Multiply all sites outside tight-box.

        :param src_i: Source syndrome index.
        :type src_i: 2-tuple of int
        :param tgt_i: Target syndrome index.
        :type tgt_i: 2-tuple of int
        :param factor: Multiplication factor.
        :type factor: int or float
        """
        src_i, tgt_i = self._syndrome_to_grid_index(src_i, tgt_i)  # grid indices
        min_i, max_i = self._box_corners(src_i, tgt_i)  # box corners
        # tight box
        self._multiply_box_complement(min_i, max_i, factor)

    def _box_rounded(self, src_i, tgt_i, factor):
        """
        Multiply all sites outside loose-box with rounded corners.

        :param src_i: Source syndrome index.
        :type src_i: 2-tuple of int
        :param tgt_i: Target syndrome index.
        :type tgt_i: 2-tuple of int
        :param factor: Multiplication factor.
        :type factor: int or float
        """
        src_i, tgt_i = self._syndrome_to_grid_index(src_i, tgt_i)  # grid indices
        (min_r, min_c), (max_r, max_c) = self._box_corners(src_i, tgt_i)  # box corners
        # loose box
        self._multiply_box_complement((min_r - 1, min_c - 1), (max_r + 1, max_c + 1), factor)
        # rounded corners
        if min_r == max_r:  # syndromes on same row
            self._multiply_box((min_r - 1, min_c - 1), (max_r + 1, min_c), factor)  # left
            self._multiply_box((min_r - 1, max_c), (max_r + 1, max_c + 1), factor)  # right
        elif min_c == max_c:  # syndromes on same column
            self._multiply_box((min_r - 1, min_c - 1), (min_r, max_c + 1), factor)  # top
            self._multiply_box((max_r, min_c - 1), (max_r + 1, max_c + 1), factor)  # bottom
        else:  # syndromes in corners of box
            self._multiply_box((min_r - 1, min_c - 1), (min_r, min_c), factor)  # top-left
            self._multiply_box((max_r, max_c), (max_r + 1, max_c + 1), factor)  # bottom-right
            self._multiply_box((min_r - 1, max_c), (min_r, max_c + 1), factor)  # top-right
            self._multiply_box((max_r, min_c - 1), (max_r + 1, min_c), factor)  # bottom-left

    def _box_fitted(self, src_i, tgt_i, factor):
        """
        Multiply all sites outside loose-box with rounded corners adjacent to syndrome indices.

        :param src_i: Source syndrome index.
        :type src_i: 2-tuple of int
        :param tgt_i: Target syndrome index.
        :type tgt_i: 2-tuple of int
        :param factor: Multiplication factor.
        :type factor: int or float
        """
        src_i, tgt_i = self._syndrome_to_grid_index(src_i, tgt_i)  # grid indices
        (min_r, min_c), (max_r, max_c) = self._box_corners(src_i, tgt_i)  # box corners
        # loose box
        self._multiply_box_complement((min_r - 1, min_c - 1), (max_r + 1, max_c + 1), factor)
        # rounded corners (adjacent to syndrome indices only)
        if min_r == max_r:  # syndromes on same row
            self._multiply_box((min_r - 1, min_c - 1), (max_r + 1, min_c), factor)  # left
            self._multiply_box((min_r - 1, max_c), (max_r + 1, max_c + 1), factor)  # right
        elif min_c == max_c:  # syndromes on same column
            self._multiply_box((min_r - 1, min_c - 1), (min_r, max_c + 1), factor)  # top
            self._multiply_box((max_r, min_c - 1), (max_r + 1, max_c + 1), factor)  # bottom
        elif min(src_i, tgt_i) == (min_r, min_c):  # syndromes top-left and bottom-right
            self._multiply_box((min_r - 1, min_c - 1), (min_r, min_c), factor)  # top-left
            self._multiply_box((max_r, max_c), (max_r + 1, max_c + 1), factor)  # bottom-right
        else:  # syndromes top-right and bottom-left
            self._multiply_box((min_r - 1, max_c), (min_r, max_c + 1), factor)  # top-right
            self._multiply_box((max_r, min_c - 1), (max_r + 1, min_c), factor)  # bottom-left

    def _box_loose(self, src_i, tgt_i, factor):
        """
        Multiply all sites outside loose-box.

        :param src_i: Source syndrome index.
        :type src_i: 2-tuple of int
        :param tgt_i: Target syndrome index.
        :type tgt_i: 2-tuple of int
        :param factor: Multiplication factor.
        :type factor: int or float
        """
        src_i, tgt_i = self._syndrome_to_grid_index(src_i, tgt_i)  # grid indices
        (min_r, min_c), (max_r, max_c) = self._box_corners(src_i, tgt_i)  # box corners
        # loose box
        self._multiply_box_complement((min_r - 1, min_c - 1), (max_r + 1, max_c + 1), factor)

    def distance(self, src_i, tgt_i, algorithm=4):
        """
        Distance between syndrome indices weighted by the grid background.

        Note:

        * The distance algorithm defines the path(s) used to calculate distance between syndrome indices.

        :param src_i: Source syndrome index.
        :type src_i: 2-tuple of int
        :param tgt_i: Target syndrome index.
        :type tgt_i: 2-tuple of int
        :param algorithm: Distance algorithm. (default=4, 1=v+h, 2=min(v+h,h+v), 4=min(v+h,h+v,v+h+v,h+v+h)
        :type algorithm: int
        :return: Distance.
        :rtype: float
        """
        assert algorithm in (1, 2, 4), 'StepGrid: Unsupported distance algorithm'
        # if both indices virtual then zero weight
        if not (self._code.is_in_bounds(src_i) or self._code.is_in_bounds(tgt_i)):
            return 0
        # find sum of weighted steps over matrix elements along path
        if algorithm == 1:
            distance = self._distance_1(src_i, tgt_i)
        elif algorithm == 2:
            distance = self._distance_2(src_i, tgt_i)
        else:
            distance = self._distance_4(src_i, tgt_i)
        return distance

    def _distance_1(self, src_i, tgt_i):
        """
        Distance between syndrome indices as sum of site weights [down and across].

        :param src_i: Source syndrome index.
        :type src_i: 2-tuple of int
        :param tgt_i: Target syndrome index.
        :type tgt_i: 2-tuple of int
        :return: Distance
        :rtype: float
        """
        src_i, tgt_i = self._syndrome_to_grid_index(src_i, tgt_i)  # grid indices
        (min_r, min_c), (max_r, max_c) = self._box_corners(src_i, tgt_i)  # box corners
        # sum down src column + sum across tgt row
        return np.sum(self._grid[min_r:max_r, src_i[1]]) + np.sum(self._grid[tgt_i[0], min_c:max_c])

    def _distance_2(self, src_i, tgt_i):
        """
        Distance between syndrome indices taking the minimum of sums of site weights [down and across] and [across and
        down].

        :param src_i: Source syndrome index.
        :type src_i: 2-tuple of int
        :param tgt_i: Target syndrome index.
        :type tgt_i: 2-tuple of int
        :return: Distance
        :rtype: float
        """
        src_i, tgt_i = self._syndrome_to_grid_index(src_i, tgt_i)  # grid indices
        (min_r, min_c), (max_r, max_c) = self._box_corners(src_i, tgt_i)  # box corners
        # sum down src column + sum along tgt row
        distance1 = np.sum(self._grid[min_r:max_r, src_i[1]]) + np.sum(self._grid[tgt_i[0], min_c:max_c])
        # sum along src row + sum down tgt column
        distance2 = np.sum(self._grid[src_i[0], min_c:max_c]) + np.sum(self._grid[min_r:max_r, tgt_i[1]])
        return min(distance1, distance2)

    def _distance_4(self, src_i, tgt_i):
        """
        Distance between syndrome indices taking the minimum of sums of site weights [down and across], [across and
        down], [half-way down, across, half-way down] and [half-way across, down, half-way across].

        :param src_i: Source syndrome index.
        :type src_i: 2-tuple of int
        :param tgt_i: Target syndrome index.
        :type tgt_i: 2-tuple of int
        :return: Distance
        :rtype: float
        """
        src_i, tgt_i = self._syndrome_to_grid_index(src_i, tgt_i)  # grid indices
        (min_r, min_c), (max_r, max_c) = self._box_corners(src_i, tgt_i)  # box corners
        # sum down src column + sum along tgt row
        distance1 = np.sum(self._grid[min_r:max_r, src_i[1]]) + np.sum(self._grid[tgt_i[0], min_c:max_c])
        # sum along src row + sum down tgt column
        distance2 = np.sum(self._grid[src_i[0], min_c:max_c]) + np.sum(self._grid[min_r:max_r, tgt_i[1]])
        # sum half-way down src column + sum along mid-point row + sum half-way down tgt column
        mid_r = (min_r + max_r) // 2
        distance3 = (np.sum(self._grid[min_r:mid_r, src_i[1]])
                     + np.sum(self._grid[mid_r, min_c:max_c])
                     + np.sum(self._grid[mid_r:max_r, tgt_i[1]]))
        # sum half-way along src row + sum down mid-point column + sum half-way along tgt row
        mid_c = (min_c + max_c) // 2
        distance4 = (np.sum(self._grid[src_i[0], min_c:mid_c])
                     + np.sum(self._grid[min_r:max_r, mid_c])
                     + np.sum(self._grid[tgt_i[0], mid_c:max_c]))
        return min(distance1, distance2, distance3, distance4)

    @functools.lru_cache()
    def mwpm(self, matched_indices, syndrome_indices, factor=3, initial=1, box_shape='t', distance_algorithm=4):
        """
        Minimum-weight perfect matching of syndrome indices over a background of matched dual syndrome indices.

        Notes:

        * The background is set according to :meth:`set_background`.
        * A graph of the unmatched foreground indices is created, with appropriate virtual indices, and with edge
          weights given by :meth:`distance`.
        * A standard minimum-weight perfect matching is found in the graph.

        :param matched_indices: Matched pairs of background syndrome indices (dual to foreground).
        :type matched_indices: frozenset of 2-tuples of 2-tuple of int
        :param syndrome_indices: Unmatched foreground syndrome indices.
        :type syndrome_indices: frozenset of 2-tuple of int
        :param factor: Multiplication factor. (default=3)
        :type factor: int or float
        :param initial: Initial edge weight. (default=1)
        :type initial: int or float
        :param box_shape: Shape of background boxes. (default='t', 't'=tight, 'r'=rounded, 'f'=fitted, 'l'=loose)
        :type box_shape: str
        :param distance_algorithm: Distance algorithm. (default=4, 1=v+h, 2=min(v+h,h+v), 4=min(v+h,h+v,v+h+v,h+v+h)
        :type distance_algorithm: int
        :return: Minimum-weight perfect matching of foreground syndrome indices.
        :rtype: frozenset of 2-tuples of 2-tuple of int
        """
        # set grid background
        self.set_background(matched_indices, factor=factor, initial=initial, box_shape=box_shape)
        # prepare graph
        graph = gt.SimpleGraph()
        # create lists of nodes and corresponding vnodes
        # NOTE: encapsulate indices in node objects that implement object reference equality since we may pass multiple
        # virtual plaquettes with the same index for matching.
        nodes, vnodes = [], []
        for index in syndrome_indices:
            nodes.append(_Node(index))
            vnodes.append(_Node(self._code.virtual_plaquette_index(index)))
        # add weighted edges to graph
        for a_node, b_node in itertools.chain(
                itertools.combinations(nodes, 2),  # all nodes to all nodes
                itertools.combinations(vnodes, 2),  # all vnodes to all vnodes
                zip(nodes, vnodes)):  # each node to corresponding vnode
            # find weighted taxi-cab distance between a and b
            distance = self.distance(a_node.index, b_node.index, algorithm=distance_algorithm)
            # add edge with weight=distance
            graph.add_edge(a_node, b_node, distance)
        # find MWPM edges {(a, b), (c, d), ...}
        mates = gt.mwpm(graph)
        # convert to frozenset of sorted 2-tuples {(a_index, b_index), ...}, removing matches if both indices virtual
        matches = frozenset(tuple(sorted((a.index, b.index))) for a, b in mates
                            if self._code.is_in_bounds(a.index) or self._code.is_in_bounds(b.index))
        return matches


class _Node:
    # simple class to contain index and implement object reference equality for mwpm
    __slots__ = ('index',)

    def __init__(self, index):
        self.index = index


@cli_description('Converging MWPM ([factor] FLOAT >=0, ...)')
class PlanarCMWPMDecoder(Decoder):
    """
    Implements a planar Converging Minimum Weight Perfect Matching (CMWPM) decoder.

    Decoding algorithm:

    * Resolve syndrome plaquettes using: :meth:`qecsim.models.planar.PlanarCode.syndrome_to_plaquette_indices`.
    * Separate syndrome plaquettes into primal and dual plaquettes.
    * For max_iterations:

        * Resolve matched_primal_pairs using MWPM with edge weights between primal plaquettes given by the taxi-cab
          distance through a background grid determined by the previous_matched_dual_pairs.
        * Resolve matched_dual_pairs using MWPM with edge weights between dual plaquettes given by the taxi-cab distance
          through a background grid determined by the previous_matched_primal_pairs.
        * Stop if matched_primal_pairs = previous_matched_primal_pairs and matched_dual_pairs =
          previous_matched_dual_pairs.

    * Return recovery operator by applying the shortest path between matching pairs using:
      :meth:`qecsim.models.planar.PlanarPauli.path`.

    Notes on background grid:

    * The grid is initialised with a grid factor (e.g. 3), box-shape (e.g. tight) and distance-algorithm (e.g. 1), and
      each edge is given an initial weight (e.g. 1).
    * The grid background is set such that, for each pair of syndrome indices (e.g. matched Z syndromes), all edges
      outside the chosen box-shape (see below), bounding the pair of indices, is multiplied by the grid factor.
    * The distance between any two syndrome indices (e.g. unmatched X syndromes) is weighted by the taxi-cab path
      through the background according to the chosen distance algorithm (see below).
    * A minimum-weight perfect matching in a graph of syndrome indices (e.g. unmatched X syndromes) with edges weighted
      by distance through the background gives matched pairs (e.g. matched X syndromes) taking into account correlations
      with the background (e.g. matched Z syndromes).
    * Box shape defines area outside of which the background is multiplied by the grid factor:

    Tight::

        X+ + +
        + + + +
         + + +
        + + + +
         + + +X

    Rounded::

           + +
         X+ + +
         + + + +
        + + + + +
         + + + +
          + + +X
           + +

    Fitted::

           + + +
         X+ + + +
         + + + +
        + + + + +
         + + + +
        + + + +X
         + + +

    Loose::

         + + + +
        +X+ + + +
         + + + +
        + + + + +
         + + + +
        + + + +X+
         + + + +

    * Distance algorithm defines how the path sum over the background of weighted edges is calculated:

    Alg. 1::

        X+ + +
        | + + +
         + + +
        | + + +
         - - -X

    Alg. 2::

             X+ + +      X- - -
             | + + +     + + + |
        min(  + + +   ,   + + +  )
             | + + +     + + + |
              - - -X      + + +X

    Alg. 4::

             X+ + +      X- - -      X+ + +      X- | +
             | + + +     + + + |     | + + +     + + + +
        min(  + + +   ,   + + +   ,   - - -   ,   + | +  )
             | + + +     + + + |     + + + |     + + + +
              - - -X      + + +X      + + +X      + - -X

    """

    def __init__(self, factor=3, max_iterations=4, box_shape='t', distance_algorithm=4):
        """
        Initialise new planar CMWPM decoder.

        :param factor: Multiplication factor.
        :type factor: int or float
        :param max_iterations: Maximum number of iterations. (default=4, 0=null, 1=MWPM, 2+=CMWPM)
        :type max_iterations: int
        :param box_shape: Shape of background boxes. (default='t', 't'=tight, 'r'=rounded, 'f'=fitted, 'l'=loose)
        :type box_shape: str
        :param distance_algorithm: Distance algorithm. (default=4, 1=h+v, 2=min(h+v,v+h), 4=min(h+v,v+h,h+v+h,v+h+v)
        :type distance_algorithm: int
        :raises ValueError: if factor is not >= 0.0.
        :raises ValueError: if max_iterations is not >= 0.
        :raises ValueError: if box_shape not in ('t', 'r', 'f', 'l').
        :raises ValueError: if distance_algorithm not in (1, 2, 4).
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI. (operator.index ensures the parameter can be treated as an int)
            if not factor >= 0.0:
                raise ValueError('PlanarCMWPMDecoder valid factor values are number >= 0.0')
            if not operator.index(max_iterations) >= 0:
                raise ValueError('PlanarCMWPMDecoder valid max_iterations values are integer >= 0')
            if box_shape not in ('t', 'r', 'f', 'l'):
                raise ValueError("PlanarCMWPMDecoder valid box_shape values are ('t', 'r', 'f', 'l')")
            if distance_algorithm not in (1, 2, 4):
                raise ValueError("PlanarCMWPMDecoder valid distance_algorithm values are (1, 2, 4)")
        except TypeError as ex:
            raise TypeError('PlanarCMWPMDecoder invalid parameter type') from ex
        self._factor = factor
        self._max_iterations = max_iterations
        self._box_shape = box_shape
        self._distance_algorithm = distance_algorithm
        self._debug_iterations = False

    @classmethod
    def _recovery_pauli(cls, code, *match_sets):
        # prepare recovery
        recovery_pauli = code.new_pauli()
        for matches in match_sets:
            # apply paths
            for a_index, b_index in matches:
                # add path to recover
                recovery_pauli.path(a_index, b_index)
        return recovery_pauli

    def decode(self, code, syndrome, **kwargs):
        """See :meth:`qecsim.model.Decoder.decode`"""
        # get syndrome indices
        syndrome_indices = code.syndrome_to_plaquette_indices(syndrome)
        # split indices into primal and dual
        primal_indices = frozenset(i for i in syndrome_indices if code.is_primal(i))
        dual_indices = frozenset(i for i in syndrome_indices if code.is_dual(i))
        # converge on matching
        grid = StepGrid(code)
        # prepare previous and current matches (in case max_iterations is 0)
        previous_primal_matches = primal_matches = frozenset()
        previous_dual_matches = dual_matches = frozenset()
        # Catch and log floating point errors. This may happen if factor is large/small and there are many matches.
        with np.errstate(all='raise'):
            try:
                for _ in range(self._max_iterations):
                    primal_matches = grid.mwpm(previous_dual_matches, primal_indices, factor=self._factor,
                                               box_shape=self._box_shape, distance_algorithm=self._distance_algorithm)
                    dual_matches = grid.mwpm(previous_primal_matches, dual_indices, factor=self._factor,
                                             box_shape=self._box_shape, distance_algorithm=self._distance_algorithm)
                    if primal_matches == previous_primal_matches and dual_matches == previous_dual_matches:
                        break
                    previous_primal_matches = primal_matches
                    previous_dual_matches = dual_matches
            except FloatingPointError as fpe:
                logger.warning('FPE RAISED FloatingPointError: {}'.format(fpe))
        # prepare recovery
        recovery_pauli = self._recovery_pauli(code, primal_matches, dual_matches)
        # return recover as bsf
        return recovery_pauli.to_bsf()

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        params = [('factor', self._factor), ('max_iterations', self._max_iterations), ('box_shape', self._box_shape),
                  ('distance_algorithm', self._distance_algorithm), ]
        return 'Planar CMWPM ({})'.format(', '.join('{}={}'.format(k, v) for k, v in params if v))

    def __repr__(self):
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            type(self).__name__, self._factor, self._max_iterations, self._box_shape, self._distance_algorithm,
        )
