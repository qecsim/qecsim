import functools
import itertools
import logging
import math
from collections import OrderedDict

import numpy as np

from qecsim import graphtools as gt
from qecsim import paulitools as pt
from qecsim.error import QecsimError
from qecsim.model import Decoder, DecoderFTP, cli_description
from qecsim.models.generic import BiasedDepolarizingErrorModel
from qecsim.models.generic import BitPhaseFlipErrorModel

logger = logging.getLogger(__name__)


@cli_description('Symmetry MWPM ([eta] FLOAT >=0)')
class RotatedPlanarSMWPMDecoder(Decoder, DecoderFTP):
    """
    Implements a rotated planar Symmetry Minimum Weight Perfect Matching (SMWPM) decoder.

    A version of this decoder yielded results reported in https://arxiv.org/abs/1907.02554.

    Note: This decoder handles decoding for both :func:`qecsim.app.run` and :func:`qecsim.app.run_ftp` simulations.

    This decoder is described conceptually in the aforementioned paper. We assume that the noise is highly biased
    towards Y errors. Below we outline the main decoding algorithm with a distance function and an additional algorithm
    to neutralise any Y-defects left by the main decoding algorithm.

    Definitions:

    * Code *plaquettes* are of type X or Z with co-ordinates (x, y).
    * A *syndrome* identifies *syndrome defects* due to qubit/measurement errors, with time and plaquette co-ordinates
      (t, x, y).
    * A *virtual defect* is a defect corresponding to a plaquette adjacent to the boundary of the code but not within
      lattice bounds; for each such plaquette and each time slice there is a corresponding virtual defect.
    * Both syndrome and virtual defects are labeled X or Z depending on their plaquette type,
    * A *row* is a 2D plane with a space direction corresponding to a code row (fixed y) and a time direction.
    * A *column* is a 2D plane with a space direction corresponding to a code column (fixed x) and a time direction.
    * A *parallel* step is a step along a given row or column in space, as induced by a Pauli Y error.
    * A *diagonal* step is a step between rows or columns in space, as induced by a Pauli X or Z error.
    * A *time* steps is a step between time slices, as induced by a measurement error.
    * A *cluster* is a directed path of defects; by construction each cluster is *neutral* or *defective*.
    * A *neutral cluster* is a cluster with an even number of X defects and an even number of Z defects.
    * A *defective cluster* is a non-neutral cluster with a single Y defect (i.e. an extra X and Z defect).

    Main decoding algorithm:

    * Given code (see :class:`qecsim.models.rotatedplanar.RotatedPlanarCode`), syndrome
      (see :func:`qecsim.app.run_once_ftp`), error_model (see :class:`qecsim.model.ErrorModel`), qubit and measurement
      error probabilities *p* and *q*.
    * Derive bias *eta* = p_y / (p_x + p_z), assuming p_x = p_z.
    * Construct *graph* as follows:

        * Add *row* node for each syndrome defect and each virtual defect.
        * Add *column* node for each syndrome defect and each virtual defect.
        * If infinite bias, add edges as follows:

            * Add edge between each pair of nodes in same row, with weighted distance over parallel and time steps.
            * Add edge between each pair of nodes in same column, with weighted distance over parallel and time steps.
            * Add edge between each pair of virtual nodes at same location (in space and time), with zero distance.

        * Else if finite bias, add edges as follows:

            * Add edge between each pair of row nodes (not necessarily belonging to the same row), with weighted
              distance over parallel, diagonal and time steps.
            * Add edge between each pair of column nodes (not necessarily belonging to the same column), with weighted
              distance over parallel, diagonal and time steps.
            * Add edge between each pair of virtual nodes at same location (in space and time), with zero distance.

    * Find *matches* as node pairs using minimum weight perfect matching over graph.
    * Construct *clusters* from matches, where each *cluster* is constructed as follows:

        * Select a pair of column nodes A -> B and remove from matches.
        * Add A to cluster.
        * Select pair of row nodes B -> C and remove from matches.
        * Add B to cluster.
        * Repeat until C is not found in matches.

    * Construct *recovery operator* as follows:

        * For each cluster in clusters:

            * Split cluster into list of X nodes and list of Z nodes.
            * Apply path of Z to recovery operator between each successive pair of X nodes.
            * Apply path of X to recovery operator between each successive pair of Z nodes.

        * Apply paths to recovery operator to neutralise any *defective* clusters.

    * Return recovery operator.


    Distance function between two defect nodes:

    * Number of time steps is smallest number of steps between time slices assuming a periodic time dimension.
    * Number of diagonal steps is smallest number of steps between parallel rows (columns) within lattice.
    * Number of parallel steps is smallest number of steps along a row (column) once then number of diagonal steps has
      been determined.
    * In case of infinite bias (assuming p_x = p_z):

        * Time steps are weighted: -ln(q / (1 - q)).
        * Parallel steps are weighted: -ln(p / (1 - p)).
        * Diagonal steps are undefined.

    * In case of finite bias (assuming p_x = p_z):

        * Time steps are weighted: -ln(q / (1 - q)).
        * Parallel steps are weighted: -(ln(eta / (eta + 1)) + ln(p / (1 - p)))
        * Diagonal steps are weighted: -(ln(1 / (2 * (eta + 1))) + ln(p / (1 - p))).

    * Distance is given by sum of weighted steps.


    Neutralising defective clusters algorithm to update recovery operator:

    * If there are no defective clusters, return without updating recovery operator.

    * Construct *cluster graph* as follows:

        * Add *Y-defect* node for each defective cluster.
        * Add two *neutral* nodes for each neutral cluster if that cluster contains both X and Z defects.
        * Add *corner* node for each lattice corner at each time step.
        * Add edge between each pair of nodes, with distance given by minimum taxi-cab distance between any pairing of
          defects between the nodes.
        * If number of *Y-defect* nodes is odd, then add an *extra* node with zero-distance edges to corner nodes.

    * Find *cluster matches* as cluster node pairs using minimum weight perfect matching over cluster graph.
    * Update recovery operator as follows:

        * For each pair of nodes in cluster matches:

            * If at least one node is not a corner or extra node:

                * Apply path of Z to recovery operator between selected X defect from each pair of nodes in matches.
                * Apply path of X to recovery operator between selected Z defect from each pair of nodes in matches.

    """

    def __init__(self, eta=None):
        """
        Initialise new rotated planar SMWPM decoder.

        :param eta: Bias (default=None, take-from-error-model=None)
        :type eta: float or None
        :raises ValueError: if eta is not None or > 0.0.
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI.
            if not (eta is None or (eta > 0.0 and math.isfinite(eta))):
                raise ValueError('{} valid eta values are None or number > 0.0'.format(type(self).__name__))
        except TypeError as ex:
            raise TypeError('{} invalid parameter type'.format(type(self).__name__)) from ex
        self._eta = eta

    def _bias(self, error_model):
        """Bias of given error model, or eta if specified.

        :param error_model: Error model.
        :type error_model: qecsim.model.ErrorModel
        :return: Bias (a positive finite number or None for infinite bias), i.e. p_y / (p_x + p_z).
        :rtype: float or None
        :raises ValueError: if bias is not a positive finite number or None.
        """
        # let eta override bias
        if self._eta is not None:
            return self._eta
        # deduce bias
        if isinstance(error_model, BiasedDepolarizingErrorModel) and error_model.axis == 'Y':
            bias = error_model.bias
        else:
            p_i, p_x, p_y, p_z = error_model.probability_distribution(1)
            try:
                bias = p_y / (p_x + p_z)
            except ZeroDivisionError:
                bias = None
        if not (bias is None or (bias > 0 and math.isfinite(bias))):
            raise ValueError('Bias for given error model does not resolve to a positive finite number or None: {}'
                             .format(error_model))
        return bias

    def decode(self, code, syndrome,
               error_model=BitPhaseFlipErrorModel(),  # noqa: B008
               error_probability=0.1, **kwargs):
        """
        See :meth:`qecsim.model.Decoder.decode`

        :param code: Rotated planar code.
        :type code: RotatedPlanarCode
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :param error_model: Error model. (default=BitPhaseFlipErrorModel())
        :type error_model: ErrorModel
        :param error_probability: Overall probability of an error on a single qubit. (default=0.1)
        :type error_probability: float
        :return: Recovery operation as binary symplectic vector.
        :rtype: numpy.array (1d)
        """
        # Prepare decode_ftp parameters
        time_steps = 1
        syndrome = np.expand_dims(syndrome, axis=0)  # convert syndrome to 2d
        kwargs['measurement_error_probability'] = 0.0
        return self.decode_ftp(code, time_steps, syndrome, error_model, error_probability, **kwargs)

    def decode_ftp(self, code, time_steps, syndrome,
                   error_model=BitPhaseFlipErrorModel(),  # noqa: B008
                   error_probability=0.1,
                   measurement_error_probability=0.1, **kwargs):
        """
        See :meth:`qecsim.model.DecoderFTP.decode_ftp`

        :param code: Rotated planar code.
        :type code: RotatedPlanarCode
        :param time_steps: Number of time steps.
        :type time_steps: int
        :param syndrome: Syndrome as binary array.
        :type syndrome: numpy.array (2d)
        :param error_model: Error model. (default=BitPhaseFlipErrorModel())
        :type error_model: ErrorModel
        :param error_probability: Overall probability of an error on a single qubit. (default=0.1)
        :type error_probability: float
        :param measurement_error_probability: Overall probability of an error on a single measurement. (default=0.1)
        :type measurement_error_probability: float
        :return: Recovery operation as binary symplectic vector.
        :rtype: numpy.array (1d)
        """
        # deduce bias (potentially overridden by eta)
        bias = self._bias(error_model)

        # IDENTITY RECOVERY
        recovery = code.new_pauli().to_bsf()

        # SYMMETRY MATCHING
        # prepare graph
        graph = _graph(code, time_steps, syndrome, error_probability, measurement_error_probability, bias)
        # minimum weight matching
        matches = _matching(graph)
        del graph  # release heavy object
        # cluster matches
        clusters = _clusters(matches)
        del matches  # release heavy object
        # add recovery from fusing within clusters
        recovery ^= _recovery(code, clusters)

        # RESIDUAL CLUSTER SYNDROME
        cluster_syndrome = np.bitwise_xor.reduce(syndrome) ^ pt.bsp(recovery, code.stabilizers.T)
        # warn if infinite bias and non-null cluster syndrome
        if bias is None and np.any(cluster_syndrome):
            logger.warning('UNEXPECTED CLUSTER SYNDROME WITH INFINITE BIAS')

        # CLUSTER RECOVERY
        # prepare cluster graph
        cluster_graph = _cluster_graph(code, time_steps, clusters)
        del clusters  # release heavy object
        # minimum weight matching
        cluster_matches = _matching(cluster_graph)
        del cluster_graph  # release heavy object
        # resolve cluster recovery from fusing between clusters
        cluster_recovery = _cluster_recovery(code, cluster_matches)
        del cluster_matches  # release heavy object
        # add cluster recovery
        recovery ^= cluster_recovery

        # return recovery
        return recovery

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        # params as (name, value, non-default-falsy-values)
        params = [('eta', self._eta, ()), ]
        params_text = ', '.join('{}={}'.format(k, v) for k, v, f in params if v or v in f)
        return 'Rotated planar SMWPM' + (' ({})'.format(params_text) if params_text else '')

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self._eta)


@functools.lru_cache()
def _plaquette_indices(code):
    """Plaquette indices for entire code lattice including virtual plaquettes adjacent to the boundary.

    :param code: Rotated planar code.
    :type code: RotatedPlanarCode
    :return: Array of plaquette indices as (x, y).
    :rtype: numpy.array (2d) of (int, int) elements.
    """
    max_site_x, max_site_y = code.site_bounds
    row, rows = [], []
    for y in range(max_site_y, -2, -1):
        row = []
        for x in range(-1, max_site_x + 1):
            index = x, y
            row.append(tuple(index))
        rows.append(row)
    # construct empty array of indices then assign elements of rows
    # Note: We cannot construct array directly from rows because numpy will interpret tuples as an extra dimension.
    #       An alternative with (non-hashable) numpy.void types is "np.array(rows, dtype=[('x', int), ('y', int)])"
    indices = np.empty((len(rows), len(row)), dtype=object)
    indices[...] = rows
    return indices


@functools.lru_cache()
def _step_weight_time(q):
    """Time step weight.

    Notes:

    * If q = None, 0 or 1, then step weight is undefined.

    :param q: Measurement error probability.
    :type q: float
    :return: Time step weight.
    :rtype: float
    :raises ValueError: if step weight undefined.
    """
    if q is None or q in (0, 1):
        raise ValueError('Time step weight undefined for measurement error probability {}.'.format(q))
    return -math.log(q / (1 - q))


@functools.lru_cache()
def _step_weight_parallel(eta, p):
    """Parallel step weight.

    Notes:

    * If p = None or 0, then step weight is undefined.
    * If eta is None (infinite bias), then step weight depends only on p.
    * If p = 1, then step weight depends only on bias.

    :param eta: Bias (a positive finite number), i.e. p_y / (p_x + p_z).
    :type eta: float
    :param p: Error probability.
    :type p: float
    :return: Parallel step weight.
    :rtype: float
    :raises ValueError: if step weight undefined.
    """
    if p is None or p == 0:
        raise ValueError('Parallel step weight undefined for error probability {}.'.format(p))
    if eta is None:
        return -math.log(p / (1 - p))
    if p == 1:
        return -math.log(eta / (eta + 1))
    return -(math.log(eta / (eta + 1)) + math.log(p / (1 - p)))  # (assuming p_x == p_z)


@functools.lru_cache()
def _step_weight_diagonal(eta, p):
    """Diagonal step weight.

    Notes:

    * If p = None or 0, then step weight is undefined.
    * If eta is None (infinite bias), then step weight is undefined.
    * If p = 1, then step weight depends only on bias.

    :param eta: Bias (a positive finite number), i.e. p_y / (p_x + p_z).
    :type eta: float
    :param p: Error probability.
    :type p: float
    :return: Parallel step weight.
    :rtype: float
    :raises ValueError: if step weight undefined.
    """
    if p is None or p == 0:
        raise ValueError('Diagonal step weight undefined for error probability {}.'.format(p))
    if eta is None:
        raise ValueError('Diagonal step weight undefined for infinite bias.')
    if p == 1:
        return -math.log(1 / (2 * (eta + 1)))
    return -(math.log(1 / (2 * (eta + 1))) + math.log(p / (1 - p)))  # (assuming p_x == p_z)


def _distance(code, time_steps, a_node, b_node, error_probability=None, measurement_error_probability=None, eta=None):
    """Distance between plaquette nodes.

    Assumptions:

    * Number of time steps is integer >= 1.
    * All indices are within the (virtual) plaquette bounds.
    * Error probability is in [0, 1] or None.
    * Measurement error probability is in [0, 1] or None.
    * Eta is a positive finite number or None.

    Notes:

    * If error probability and measurement error probability are both None then they are assumed to be equal.
    * If eta is None then it is assumed to be infinite.

    Algorithm:

    * Number of time steps is smallest number of steps between time slices assuming a periodic time dimension.
    * Number of diagonal steps is smallest number of steps between parallel rows (columns) within lattice.
    * Number of parallel steps is smallest number of steps along a row (column) once then number of diagonal steps has
      been determined.
    * If infinite bias and equal error and measurement error probabilities:

        * Distance is the sum of steps.
        * Note: if diagonal steps > 0, then distance undefined.

    * Otherwise:

        * Time / Parallel / Diagonal steps are weighted by :func:`_step_weight_time` / :func:`_step_weight_parallel`
          / :func:`_step_weight_diagonal`.
        * Distance is the sum of weighted steps.
        * Note: if time steps > 0 and measurement error probability is None, 0 or 1, then distance is undefined.
        * Note: if parallel steps > 0 and error probability is None or 0, then distance is undefined.
        * Note: if diagonal steps > 0 and error probability is None or 0, then distance is undefined.
        * Note: if diagonal steps > 0 and infinite bias, then distance is undefined.

    :param code: Rotated planar code.
    :type code: RotatedPlanarCode
    :param time_steps: Number of time steps.
    :type time_steps: int
    :param a_node: Node identifying plaquette as ((x, y), is_row).
    :type a_node: ((int, int), bool)
    :param b_node: Node identifying plaquette as ((x, y), is_row).
    :type b_node: ((int, int), bool)
    :param error_probability: Error probability (optional if measurement_error_probability and eta are both None).
    :type error_probability: float or None
    :param measurement_error_probability: Measurement error probability (optional if error_probability and eta are both
        None).
    :type measurement_error_probability: float or None
    :param eta: Bias (a positive finite number or None for infinite bias), i.e. p_y / (p_x + p_z).
    :type eta: float or None
    :return: Distance between nodes.
    :rtype: int
    :raises ValueError: if distance is undefined.
    """
    # unpack nodes
    (a_t, a_x, a_y), a_is_row = a_node
    (b_t, b_x, b_y), b_is_row = b_node

    # assumption checks
    assert int(time_steps) == time_steps and time_steps >= 1
    assert code.is_in_plaquette_bounds((a_x, a_y)) or code.is_virtual_plaquette((a_x, a_y))
    assert code.is_in_plaquette_bounds((b_x, b_y)) or code.is_virtual_plaquette((b_x, b_y))
    assert error_probability is None or (0 <= error_probability <= 1)
    assert measurement_error_probability is None or (0 <= measurement_error_probability <= 1)
    assert eta is None or (eta > 0 and math.isfinite(eta))

    # if not in parallels (i.e. if not both in rows or both in columns)
    if a_is_row != b_is_row:
        # if same index and both virtual then zero distance (even between rows and columns)
        if ((a_t, a_x, a_y) == (b_t, b_x, b_y) and code.is_virtual_plaquette((a_x, a_y))
                and code.is_virtual_plaquette((b_x, b_y))):
            return 0
        # else distance undefined.
        raise ValueError('Distance undefined between orthogonals: {}, {}.'.format(a_node, b_node))

    # NOTE: swap x and y for column case then treat as row case for remainder of method
    a_x, a_y = (a_x, a_y) if a_is_row else reversed((a_x, a_y))
    b_x, b_y = (b_x, b_y) if b_is_row else reversed((b_x, b_y))

    # calculate delta_time
    delta_time = min(abs(a_t - b_t), time_steps - abs(a_t - b_t))
    # calculate delta_diagonal and delta_parallel, where path is between corners of a box
    box_width = abs(a_x - b_x)
    box_height = abs(a_y - b_y)
    if box_width >= box_height:
        # e.g. 0x0 box (wxh)  # e.g. 1x0 box  # e.g. 1x1 box  # e.g. 2x1 box
        # .                   # -             # \             # \_
        delta_parallel = box_width - box_height
        delta_diagonal = box_height
    else:
        # e.g. 0x1 box (wxh)  # e.g. 0x2 box  # e.g. 1x2 box [(2-1) % 2 == 1]  # e.g. 1x3 box [(3-1) % 2 == 0]
        # L                   # \             # \                              # \
        #                     # /             # L                              # /
        #                     #               #                                # \
        delta_parallel = (box_height - box_width) % 2
        delta_diagonal = box_height

    # evaluate distance
    if eta is None and error_probability == measurement_error_probability:
        # exclude case we cannot handle (should not happen)
        if delta_diagonal != 0:
            raise ValueError('Diagonal distance undefined for infinite bias: {}, {}.'.format(a_node, b_node))
        # special case of infinite bias and equal probabilities
        distance = delta_parallel + delta_time
    else:
        # finite bias or distinct probabilities
        distance = 0
        if delta_time:
            distance += delta_time * _step_weight_time(measurement_error_probability)
        if delta_parallel:
            distance += delta_parallel * _step_weight_parallel(eta, error_probability)
        if delta_diagonal:
            distance += delta_diagonal * _step_weight_diagonal(eta, error_probability)
    return distance


def _graph(code, time_steps, syndrome, error_probability=None, measurement_error_probability=None, eta=None):
    """Graph of plaquette nodes and weighted edges consistent with the syndrome.

    Algorithm: see class doc.

    :param code: Rotated planar code.
    :type code: RotatedPlanarCode
    :param time_steps: Number of time steps.
    :type time_steps: int
    :param syndrome: Syndrome as binary array with (t, x, y) dimensions.
    :type syndrome: numpy.array (2d)
    :param error_probability: Error probability
        (optional if equal to measurement_error_probability and eta is None).
    :type error_probability: float or None
    :param measurement_error_probability: Measurement error probability
        (optional if equal to error_probability and eta is None).
    :type measurement_error_probability: float or None
    :param eta: Bias (a positive finite number or None for infinite bias), i.e. p_y / (p_x + p_z).
    :type eta: float or None
    :return: Graph of weighted edges between plaquette nodes, consistent with the syndrome,
        as {((a_t, a_x, a_y), a_is_row), (b_t, b_x, b_y), b_is_row)): weight, ...}.
    :rtype: dict of (((int, int, int), bool), ((int, int, int), bool)) edges to float weights.
    """
    # empty graph
    graph = gt.SimpleGraph()
    # get syndrome indices, as list of set where syndrome_indices[t] corresponds to time t
    syndrome_indices = [code.syndrome_to_plaquette_indices(s) for s in syndrome]
    # all plaquettes as (x, y)
    plaquette_indices = _plaquette_indices(code)

    def _add_edge(a_node, b_node):
        # unpack nodes
        ((a_t, a_x, a_y), a_is_row), ((b_t, b_x, b_y), b_is_row) = a_node, b_node
        # do not add edge between orthogonals
        if a_is_row != b_is_row:
            return
        # do not add edge between time steps if measurement probability is 0 or 1
        if measurement_error_probability in (0, 1) and a_t != b_t:
            return
        # do not add edge between space steps if error_probability is 0
        if error_probability == 0 and (a_x, a_y) != (b_x, b_y):
            return
        # do not add edge between distinct parallels if eta is None
        if eta is None and ((a_is_row and a_y != b_y) or (not a_is_row and a_x != b_x)):
            return
        # add edge to graph
        graph.add_edge(a_node, b_node, _distance(code, time_steps, a_node, b_node, error_probability,
                                                 measurement_error_probability, eta))

    def _add_to_graph(by_row):
        """Loop through lines of plaquette_indices adding nodes consistent with syndrome_indices with edges weighted
        according to distance function. by_row=True/False means process rows/columns."""

        # lattice_nodes (only populated for finite bias, i.e. eta is not None)
        lattice_nodes = []
        # loop through lines (rows if by_row, cols if not by_row)
        for line in plaquette_indices if by_row else plaquette_indices.T:
            # line list of nodes
            line_nodes = []
            # loop through indices on line
            for (x, y) in line:
                # loop through time
                for t in range(time_steps):
                    if code.is_virtual_plaquette((x, y)):
                        # add virtual node to line list
                        v_node = ((t, x, y), by_row)
                        line_nodes.append(v_node)
                        # add virtual node and orthogonal twin to graph with zero distance
                        v_node_twin = ((t, x, y), not by_row)
                        graph.add_edge(v_node, v_node_twin, 0)
                    else:
                        # if index in syndrome
                        if (x, y) in syndrome_indices[t]:
                            # add real node to line list
                            r_node = ((t, x, y), by_row)
                            line_nodes.append(r_node)
            if eta is None:  # if infinite bias
                # add line edges to graph
                for a_node, b_node in itertools.combinations(line_nodes, 2):
                    _add_edge(a_node, b_node)
            else:  # else finite bias
                # add line nodes to lattice nodes
                lattice_nodes.extend(line_nodes)
        # if bias is not infinite and we have some lattice nodes
        if eta and lattice_nodes:
            # add lattice edges to graph (note: lattice_nodes is empty if infinite bias)
            for a_node, b_node in itertools.combinations(lattice_nodes, 2):
                _add_edge(a_node, b_node)

    # add nodes by row
    _add_to_graph(by_row=True)
    # add nodes by column
    _add_to_graph(by_row=False)
    return graph


def _matching(graph):
    """Matching (minimum weight perfect matching) over graph.

    :param graph: Graph of weighted edges between nodes, as {(a_node, b_node): weight, ...}.
    :type graph: dict of (object, object) edges to float weights.
    :return: Matches between nodes as (a_node, b_node).
    :rtype: set of (object, object)
    """
    return gt.mwpm(graph)  # option to switch to networkx or blossom5 here


def _clusters(matches):
    """List of clusters from the given matches where each cluster is a directed path of plaquette indices.

    Notes:

    * Matches between nodes with the same index are removed.
    * For consistency in testing, the clusters are ordered by their SW corner index, and the directed path of each
      cluster starts in the SW corner of the cluster and traverses clockwise (the final index does not repeat the first
      index).

    Algorithm: see class doc.

    :param matches: Matches between index nodes as ((t, x, y), is_row).
    :type matches: set of (((int, int, int), bool), ((int, int, int), bool))
    :return: List of clusters (directed paths of indices) as [[(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)], ...]
    :rtype: list of list of (int, int, int)
    :raises ValueError: If matches are between rows and columns (except virtual nodes at same index).
    """
    # build row and col mates maps
    row_mates, col_mates = {}, {}
    for (a_index, a_is_row), (b_index, b_is_row) in matches:
        # skip if nodes have same index
        if a_index == b_index:
            continue
        # we should not match between rows and columns
        if a_is_row != b_is_row:
            raise ValueError('Matching unsupported between rows and columns (except virtual nodes at same index).')
        # add match and reverse match to appropriate mates map
        mates = row_mates if a_is_row else col_mates
        mates[a_index] = b_index  # add match
        mates[b_index] = a_index  # add reverse match
    # for consistency in testing, loop column sorted column mates so that each cluster begins in sw corner of sw
    # cluster and traverses clockwise.
    col_mates = OrderedDict(sorted(col_mates.items()))
    # build list of clusters
    clusters = []
    # loop until all column mates processed
    while col_mates:
        # single cluster as (x1, y1) -> (x2, y2) ... -> (xn, yn)
        cluster = []
        # pop start_index and next_index (column)
        start_index, next_index = col_mates.popitem(last=False)
        cluster.append(start_index)  # add start_index (column)
        # loop until cluster processed
        while True:
            try:
                cluster.append(next_index)  # add next_index (column)
                del col_mates[next_index]  # delete reverse column match
                next_index = row_mates.pop(next_index)  # find next_index (row)
                cluster.append(next_index)  # add next_index (row)
                del row_mates[next_index]  # delete reverse row match
                next_index = col_mates.pop(next_index)  # find next_index (column)
            except KeyError:
                break  # break when cluster processed
        # sanity: cluster should be closed loop
        if cluster[0] != cluster[-1]:
            raise QecsimError('Cluster is not a closed loop.')
        # remove redundant final index of closed loop
        cluster.pop()
        # sanity: cluster length should be even
        if len(cluster) % 2:
            raise QecsimError('Cluster length is not even.')
        # add cluster to list of clusters
        clusters.append(cluster)
    # sanity: all row_mates should be processed when all col_mates have been processed
    if row_mates:
        raise QecsimError('Some row matches unclustered after all column matches clustered.')
    return clusters


def _cluster_to_paths_and_defect(code, cluster):
    """Splits cluster into paths of X and Z plaquette indices and a non-fusible Y-defect (if present).

    Note:

    * By design, X and Z path lengths have the same parity.
    * If X and Z path lengths are odd then a non-fusible Y-defect is present.
    * If present, the Y-defect is selected to consist of the final X and Z path indices.

    :param code: Rotated planar code.
    :type code: RotatedPlanarCode
    :param cluster: Cluster (directed path of indices) as [(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)].
    :type cluster: list of (int, int, int)
    :return: Path of X-indices, Path of Z-indices, Y-defect as (x_index, z_index) or None.
    :rtype: list of (int, int, int), list of (int, int, int), ((int, int, int), (int, int, int))
    """
    # split into x and z indices
    x_indices = [(t, x, y) for t, x, y in cluster if code.is_x_plaquette((x, y))]
    z_indices = [(t, x, y) for t, x, y in cluster if code.is_z_plaquette((x, y))]
    # sanity: check X and Z path lengths have same parity
    if len(x_indices) % 2 != len(z_indices) % 2:
        raise QecsimError('Cluster has non-fused non-Y defect.')
    # if path lengths odd, choose Y-defect to be final indices
    if len(x_indices) % 2:
        return x_indices[:-1], z_indices[:-1], (x_indices[-1], z_indices[-1])
    else:
        return x_indices, z_indices, None


def _path_operator(code, a_index, b_index):
    """Operator consisting of a path of Pauli operators to fuse the plaquettes indexed by A and B.

    Assumptions:

    * All indices are within the (virtual) plaquette bounds.

    :param code: Rotated planar code.
    :type code: RotatedPlanarCode
    :param a_index: Plaquette index as (x, y).
    :type a_index: (int, int)
    :param b_index: Plaquette index as (x, y).
    :type b_index: (int, int)
    :return: Path operator in binary symplectic form.
    :rtype: numpy.array (1d)
    :raises ValueError: If plaquettes are not of the same type (i.e. X or Z).
    """
    # assumption checks
    assert code.is_in_plaquette_bounds(a_index) or code.is_virtual_plaquette(a_index)
    assert code.is_in_plaquette_bounds(b_index) or code.is_virtual_plaquette(b_index)

    # check both plaquettes are the same type
    if code.is_z_plaquette(a_index) != code.is_z_plaquette(b_index):
        raise ValueError('Path undefined between plaquettes of different types: {}, {}.'.format(a_index, b_index))

    def _start_end_site_coordinate(a_k, b_k):
        """Return start and end coordinates along an axis, where k represents x or y."""
        if a_k < b_k:  # A below B so go from top of A to bottom of B
            start_k = a_k + 1
            end_k = b_k
        elif a_k > b_k:  # A above B so go from bottom of A to top of B
            start_k = a_k
            end_k = b_k + 1
        else:  # A in line with B so go from bottom(top) of A to bottom(top) of B (if k below zero)
            start_k = end_k = max(b_k, 0)
        return start_k, end_k

    # if start and end plaquette indices are the same return identity operator
    if a_index == b_index:
        return code.new_pauli().to_bsf()

    # start and end plaquette indices (Note plaquettes are indexed by their SW corner)
    a_x, a_y = a_index
    b_x, b_y = b_index
    # determine start and end site indices
    start_x, end_x = _start_end_site_coordinate(a_x, b_x)
    start_y, end_y = _start_end_site_coordinate(a_y, b_y)
    # build path (diagonal until inline then straight up/down or left/right)
    path_indices = []
    next_x, next_y = start_x, start_y
    while True:
        # add next_index to path
        path_indices.append((next_x, next_y))
        # test if we got to end
        if (next_x, next_y) == (end_x, end_y):
            break  # we are at the end so stop
        # increment/decrement next_x and/or next_y
        if end_x - next_x > 0:
            next_x += 1
        elif end_x - next_x < 0:
            next_x -= 1
        if end_y - next_y > 0:
            next_y += 1
        elif end_y - next_y < 0:
            next_y -= 1
    # single pauli op
    op = 'X' if code.is_z_plaquette(a_index) else 'Z'
    # full path operator
    path_operator = code.new_pauli().site(op, *path_indices)
    # return as bsf
    return path_operator.to_bsf()


def _recovery(code, clusters):
    """Operator consisting of a paths of Pauli operators to fuse the plaquettes around each cluster.

    Algorithm: see class doc.

    :param code: Rotated planar code.
    :type code: RotatedPlanarCode
    :param clusters: List of clusters (directed paths of indices) as
        [[(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)], ...]
    :type clusters: list of list of (int, int, int)
    :return: Recovery operator in binary symplectic form.
    :rtype: numpy.array (1d)
    """
    # identity recovery
    recovery_operator = code.new_pauli().to_bsf()
    # loop over clusters
    for cluster in clusters:
        # split into x_path, z_path, y_defect
        x_path, z_path, y_defect = _cluster_to_paths_and_defect(code, cluster)
        # loop over x indices in pairs, applying path operator between indices
        for (_, a_x, a_y), (_, b_x, b_y) in zip(x_path[::2], x_path[1::2]):
            recovery_operator ^= _path_operator(code, (a_x, a_y), (b_x, b_y))
        # loop over z indices in pairs, applying path operator between indices
        for (_, a_x, a_y), (_, b_x, b_y) in zip(z_path[::2], z_path[1::2]):
            recovery_operator ^= _path_operator(code, (a_x, a_y), (b_x, b_y))
    return recovery_operator


class _ClusterNode:
    """Simple class containing cluster, X / Z plaquette indices, and virtual flag; with reference equality for MWPM."""

    def __init__(self, cluster=None, x_index=None, z_index=None, is_virtual=False):
        """Initialise new cluster node for MWPM.

        Note: Corner and extra cluster nodes should be flagged as virtual.

        :param cluster: Clusters (directed paths of indices) as [[(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)], ...]
        :type cluster: list of (int, int, int)
        :param x_index: X plaquette index as (t, x, y). Must be one of cluster indices.
        :type x_index: (int, int, int)
        :param z_index: Z plaquette index as (t, x, y). Must be one of cluster indices.
        :type z_index: (int, int, int)
        :param is_virtual: If the cluster node is virtual (i.e. corner or extra).
        :type is_virtual: bool
        """
        assert (cluster is None and x_index is None) or x_index in cluster
        assert (cluster is None and z_index is None) or z_index in cluster
        self.cluster = cluster
        self.x_index = x_index
        self.z_index = z_index
        self.is_virtual = is_virtual


def _cluster_distance(time_steps, a_node, b_node):
    """Distance between cluster nodes.

    Notes:

    * The best distance function between clusters is not clearly defined so a heuristic is used.

    Algorithm:

    * Distance between two virtual nodes (i.e. corner nodes or extra node) is 0.
    * Distance is otherwise defined as shortest Manhattan distance between any pair of plaquette indices in the
      clusters, noting that the time dimension is periodic.

    :param time_steps: Number of time steps.
    :type time_steps: int
    :param a_node: Node identifying cluster.
    :type a_node: _ClusterNode
    :param b_node: Node identifying cluster.
    :type b_node: _ClusterNode
    :return: Cluster distance.
    :rtype: int
    :raises ValueError: if distance is undefined.
    """
    # zero distance between virtual (i.e. corner nodes)
    if a_node.is_virtual and b_node.is_virtual:
        return 0
    # distance undefined if either node does not have a cluster
    if a_node.cluster is None or b_node.cluster is None:
        raise ValueError('Distance undefined between nodes without clusters.')
    # minimum manhattan distance between all pairs of cluster indices (periodic in time dimension)
    return min(min(abs(a_t - b_t), time_steps - abs(a_t - b_t)) + abs(a_x - b_x) + abs(a_y - b_y)
               for (a_t, a_x, a_y), (b_t, b_x, b_y) in itertools.product(a_node.cluster, b_node.cluster))


def _cluster_corner_indices(code):
    """Corner indices of X and Z virtual plaquettes.

    :param code: Rotated planar code.
    :type code: RotatedPlanarCode
    :return: List of corner X and Z type plaquette indices as [((Xx, Xy), (Zx, Zy)), ...].
    :rtype: list of ((int, int), (int, int))
    """
    max_site_x, max_site_y = code.site_bounds
    sw = (0, -1), (-1, -1)
    if max_site_y % 2:  # even number of rows
        nw = (0, max_site_y), (-1, max_site_y)
    else:  # odd number of rows
        nw = (-1, max_site_y), (-1, max_site_y - 1)
    if max_site_x % 2 == max_site_y % 2:  # same parity number of columns and rows
        ne = (max_site_x - 1, max_site_y), (max_site_x, max_site_y)
    else:  # opposite parity number of columns and rows
        ne = (max_site_x, max_site_y), (max_site_x, max_site_y - 1)
    if max_site_x % 2:  # even number of columns
        se = (max_site_x - 1, -1), (max_site_x, -1)
    else:  # odd number of columns
        se = (max_site_x, -1), (max_site_x, 0)
    return [sw, nw, ne, se]


def _cluster_graph(code, time_steps, clusters):
    """Graph of cluster nodes and weighted edges consistent with the syndrome.

    Notes:

    * By construction, each cluster is either neutral (i.e. all defects fused) or defective (i.e. exactly one non-fused
      Y defect).
    * If there are no defective clusters an empty graph is returned.

    Algorithm: see class doc.

    :param code: Rotated planar code.
    :type code: RotatedPlanarCode
    :param time_steps: Number of time steps.
    :type time_steps: int
    :param clusters: List of clusters (directed paths of indices) as
        [[(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)], ...].
    :type clusters: list of list of (int, int)
    :return: Graph of weighted edges between cluster nodes, as {(a_node, b_node): weight, ...}.
    :rtype: dict of (_ClusterNode, _ClusterNode) edges to float weights.
    """
    # empty graph
    graph = gt.SimpleGraph()
    # build cluster nodes
    defective_cluster_nodes = []
    neutral_cluster_nodes = []
    for cluster in clusters:
        # split into x_path, z_path, y_defect
        x_path, z_path, y_defect = _cluster_to_paths_and_defect(code, cluster)
        # add cluster nodes to graph
        if y_defect:  # if cluster has non-fused Y-defect
            # unpack Y-defect indices
            x_defect_index, z_defect_index = y_defect
            # add as defective cluster
            cluster_node = _ClusterNode(cluster, x_defect_index, z_defect_index)
            defective_cluster_nodes.append(cluster_node)
        elif x_path and z_path:  # elif cluster has fused Y-defects
            # add twice as neutral cluster with representative X and Z indices
            neutral_cluster_nodes.append(_ClusterNode(cluster, x_path[0], z_path[0]))
            neutral_cluster_nodes.append(_ClusterNode(cluster, x_path[0], z_path[0]))
        else:  # else cluster has no Y-defects (fused or non-fused) so skip
            pass
    # if no defective cluster nodes then return empty graph
    if not defective_cluster_nodes:
        return graph
    # define extra virtual node (to join with corners) if odd number of cluster nodes
    extra_virtual_node = _ClusterNode(is_virtual=True) if len(defective_cluster_nodes) % 2 else None
    # add defective virtual clusters at corners with edge to extra virtual cluster
    for (x_x, x_y), (z_x, z_y) in _cluster_corner_indices(code):
        # loop through time
        for t in range(time_steps):
            x_index, z_index = (t, x_x, x_y), (t, z_x, z_y)
            corner_virtual_node = _ClusterNode([x_index, z_index], x_index, z_index, is_virtual=True)
            defective_cluster_nodes.append(corner_virtual_node)
            if extra_virtual_node:
                # add virtual corner node and virtual extra node to graph with zero distance
                graph.add_edge(corner_virtual_node, extra_virtual_node, 0)
    # add edges to graph
    for a_node, b_node in itertools.combinations(defective_cluster_nodes + neutral_cluster_nodes, 2):
        graph.add_edge(a_node, b_node, _cluster_distance(time_steps, a_node, b_node))
    return graph


def _cluster_recovery(code, matches):
    """Operator consisting of a paths of Pauli operators to fuse clusters together.

    Algorithm: see class doc.

    :param code: Rotated planar code.
    :type code: RotatedPlanarCode
    :param matches: Matches between cluster nodes.
    :type matches: set of (_ClusterNode, _ClusterNode)
    :return: Recovery operator in binary symplectic form.
    :rtype: numpy.array (1d)
    """
    # identity recovery
    recovery_operator = code.new_pauli().to_bsf()
    # loop over cluster matches
    for a_node, b_node in matches:
        # skip if both nodes are virtual
        if a_node.is_virtual and b_node.is_virtual:
            continue
        # unpack indices
        (_, a_x_x, a_x_y), (_, a_z_x, a_z_y) = a_node.x_index, a_node.z_index
        (_, b_x_x, b_x_y), (_, b_z_x, b_z_y) = b_node.x_index, b_node.z_index
        # apply path between X plaquette indices
        recovery_operator ^= _path_operator(code, (a_x_x, a_x_y), (b_x_x, b_x_y))
        # apply path between Z plaquette indices
        recovery_operator ^= _path_operator(code, (a_z_x, a_z_y), (b_z_x, b_z_y))
    return recovery_operator
