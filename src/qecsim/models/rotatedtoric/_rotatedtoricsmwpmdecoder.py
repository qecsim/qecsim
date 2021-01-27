import functools
import itertools
import logging
import math
from collections import OrderedDict

import numpy as np

from qecsim import graphtools as gt
from qecsim import paulitools as pt
from qecsim.error import QecsimError
from qecsim.model import Decoder, DecoderFTP, DecodeResult, cli_description
from qecsim.models.generic import BiasedDepolarizingErrorModel
from qecsim.models.generic import BitPhaseFlipErrorModel

logger = logging.getLogger(__name__)


@cli_description('Symmetry MWPM ([itp] BOOL, [eta] FLOAT >=0)')
class RotatedToricSMWPMDecoder(Decoder, DecoderFTP):
    """
    Implements a rotated toric Symmetry Minimum Weight Perfect Matching (SMWPM) decoder.

    A version of this decoder yielded results reported in https://arxiv.org/abs/1907.02554.

    Note: This decoder handles decoding for both :func:`qecsim.app.run` and :func:`qecsim.app.run_ftp` simulations.

    This decoder is described conceptually in the aforementioned paper. We assume that the noise is highly biased
    towards Y errors. The decoding algorithm is the same as that of
    :class:`qecsim.models.rotatedplanar.RotatedPlanarSMWPMDecoder`, without the complication of virtual nodes on the
    boundary, but with a test for time-like logical failures.
    """

    def __init__(self, itp=False, eta=None):
        """
        Initialise new rotated toric SMWPM decoder.

        :param itp: Ignore time parity, i.e. decoder should not fail in case of time-like logical. (default=False)
        :type itp: bool
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
        self._itp = itp
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

        Note: The optional keyword parameters ``error_model`` and ``error_probability`` are used to determine the prior
        probability distribution for use in the decoding algorithm. Any provided error model must implement
        :meth:`~qecsim.model.ErrorModel.probability_distribution`.

        :param code: Rotated toric code.
        :type code: RotatedToricCode
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
        kwargs['step_measurement_errors'] = None
        decoding = self.decode_ftp(code, time_steps, syndrome, error_model, error_probability, **kwargs)
        assert decoding.success is None and np.all(decoding.custom_values == 0), (
            'Unexpected time-like logical failure in non-FT decoding')
        return decoding.recovery

    def decode_ftp(self, code, time_steps, syndrome,
                   error_model=BitPhaseFlipErrorModel(),  # noqa: B008
                   error_probability=0.1,
                   measurement_error_probability=0.1,
                   step_measurement_errors=None, **kwargs):
        """
        See :meth:`qecsim.model.DecoderFTP.decode_ftp`

        Note:

        * The optional keyword parameters ``error_model`` and ``error_probability`` are used to determine the prior
          probability distribution for use in the decoding algorithm. Any provided error model must implement
          :meth:`~qecsim.model.ErrorModel.probability_distribution`.
        * This method always returns a ``DecodeResult`` with the following parameters::

            DecodeResult(
                success=None,  # None indicates to be evaluated by app
                               # False indicates time-like logical failure (overrides evaluation by app)
                logical_commutations=None,  # None indicates to be evaluated by app
                recovery=np.array(...),  # recovery operation (used by app to evaluate success and logical_commutations)
                custom_values=np.array([0, 0]),  # [0, 0] no time-like logical failure
                                                 # [1, 0] time-like logical failure through X plaquettes
                                                 # [0, 1] time-like logical failure through Z plaquettes
                                                 # [1, 1] time-like logical failure through both X and Z plaquettes
            )

        :param code: Rotated toric code.
        :type code: RotatedToricCode
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
        :param step_measurement_errors: list of measurement error bits applied to step-syndromes index by time-step.
        :type step_measurement_errors: list of numpy.array (1d)
        :return: Decode result.
        :rtype: DecodeResult
        """
        # deduce bias (potentially overridden by eta)
        bias = self._bias(error_model)

        # IDENTITY RECOVERY AND T-PARITIES
        recovery = code.new_pauli().to_bsf()
        recovery_x_tp = 0
        recovery_z_tp = 0

        # SYMMETRY MATCHING
        # prepare graphs
        graphs = self._graphs(code, time_steps, syndrome, error_probability, measurement_error_probability, bias)
        # minimum weight matching
        matches = self._matching(graphs)
        del graphs  # release heavy object
        # cluster matches
        clusters = self._clusters(matches)
        del matches  # release heavy object
        # resolve symmetry recovery from fusing within clusters
        symmetry_recovery, symmetry_recovery_x_tp, symmetry_recovery_z_tp = self._recovery_tparities(
            code, time_steps, clusters)
        # add symmetry recovery and t-parities
        recovery ^= symmetry_recovery
        recovery_x_tp ^= symmetry_recovery_x_tp
        recovery_z_tp ^= symmetry_recovery_z_tp

        # RESIDUAL CLUSTER SYNDROME
        cluster_syndrome = np.bitwise_xor.reduce(syndrome) ^ pt.bsp(recovery, code.stabilizers.T)
        # warn if infinite bias and non-null cluster syndrome
        if bias is None and np.any(cluster_syndrome):
            logger.warning('UNEXPECTED CLUSTER SYNDROME WITH INFINITE BIAS')

        # CLUSTER RECOVERY
        # prepare cluster graph
        cluster_graph = self._cluster_graph(code, time_steps, clusters)
        del clusters  # release heavy object
        # minimum weight matching
        cluster_matches = self._matching([cluster_graph])
        del cluster_graph  # release heavy object
        # resolve cluster recovery from fusing between clusters
        cluster_recovery, cluster_recovery_x_tp, cluster_recovery_z_tp = self._cluster_recovery_tparities(
            code, time_steps, cluster_matches)
        del cluster_matches  # release heavy object
        # add cluster recovery and t-parities
        recovery ^= cluster_recovery
        recovery_x_tp ^= cluster_recovery_x_tp
        recovery_z_tp ^= cluster_recovery_z_tp

        # TEST T-PARITY
        if self._itp or time_steps == 1:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('decode: ignoring t-parity. itp={}, time_steps={}'.format(self._itp, time_steps))
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('decode: testing t-parity. itp={}, time_steps={}'.format(self._itp, time_steps))
            if not step_measurement_errors:
                raise QecsimError('Failed to test t-parity. step_measurement_errors not provided.')
            # extract t-parity for measurement errors
            measurement_error_tps = self._measurement_error_tparities(code, step_measurement_errors[-1])
            # total t-parity
            total_tps = np.array((recovery_x_tp, recovery_z_tp)) ^ measurement_error_tps
            # return false decode-result if t-parity fails, with time-parity as custom_values
            if np.any(total_tps != 0):
                return DecodeResult(success=False, recovery=recovery, custom_values=total_tps)

        # return recovery with zeros time parity custom values
        return DecodeResult(recovery=recovery, custom_values=np.array((0, 0)))

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        # params as (name, value, non-default-falsy-values)
        params = [('itp', self._itp, ()), ('eta', self._eta, ()), ]
        params_text = ', '.join('{}={}'.format(k, v) for k, v, f in params if v or v in f)
        return 'Rotated toric SMWPM' + (' ({})'.format(params_text) if params_text else '')

    def __repr__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, self._itp, self._eta)

    @classmethod
    @functools.lru_cache()
    def _plaquette_indices(cls, code):
        """Plaquette indices for entire code lattice in an array matching the lattice.

        :param code: Rotated toric code.
        :type code: RotatedToricCode
        :return: Array of plaquette indices as (x, y).
        :rtype: numpy.array (2d) of (int, int) elements.
        """
        max_x, max_y = code.bounds
        row, rows = [], []
        for y in range(max_y, -1, -1):
            row = []
            for x in range(max_x + 1):
                index = x, y
                row.append(tuple(index))
            rows.append(row)
        # construct empty array of indices then assign elements of rows
        # Note: We cannot construct array directly from rows because numpy will interpret tuples as an extra dimension.
        #       An alternative with (non-hashable) numpy.void types is "np.array(rows, dtype=[('x', int), ('y', int)])"
        indices = np.empty((len(rows), len(row)), dtype=object)
        indices[...] = rows
        return indices

    @classmethod
    @functools.lru_cache()
    def _step_weight_time(cls, q):
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

    @classmethod
    @functools.lru_cache()
    def _step_weight_parallel(cls, eta, p):
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

    @classmethod
    @functools.lru_cache()
    def _step_weight_diagonal(cls, eta, p):
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

    @classmethod
    def _distance(cls, code, time_steps, a_node, b_node,
                  error_probability=None, measurement_error_probability=None, eta=None):
        """Distance between plaquette nodes.

        Assumptions:

        * Number of time steps is integer >= 1.
        * All indices are within bounds.
        * Error probability is in [0, 1] or None.
        * Measurement error probability is in [0, 1] or None.
        * Eta is a positive finite number or None.

        Notes:

        * If error probability and measurement error probability are both None then they are assumed to be equal.
        * If eta is None then it is assumed to be infinite.

        Algorithm:

        * Steps between nodes are broken into diagonal steps (between distinct parallels) and parallel steps (along a
          parallel) and time steps.

        * If infinite bias and equal error and measurement error probabilities:

            * Distance is the sum of steps.
            * Note: if diagonal steps > 0, then distance undefined.

        * Otherwise:

            * Time / Parallel / Diagonal steps are weighted by :meth:`_step_weight_time` / :meth:`_step_weight_parallel`
              / :meth:`_step_weight_diagonal`.
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
        :param measurement_error_probability: Measurement error probability (optional if error_probability and eta are
            both None).
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
        assert code.is_in_bounds((a_x, a_y))
        assert code.is_in_bounds((b_x, b_y))
        assert error_probability is None or (0 <= error_probability <= 1)
        assert measurement_error_probability is None or (0 <= measurement_error_probability <= 1)
        assert eta is None or (eta > 0 and math.isfinite(eta))

        # if not in parallels (i.e. if not both in rows or both in columns)
        if a_is_row != b_is_row:
            # distance undefined
            raise ValueError('Distance undefined between orthogonals: {}, {}.'.format(a_node, b_node))

        # NOTE: swap x and y for column case then treat as row case for remainder of method
        a_x, a_y = (a_x, a_y) if a_is_row else reversed((a_x, a_y))
        b_x, b_y = (b_x, b_y) if a_is_row else reversed((b_x, b_y))
        dim_y, dim_x = code.size if a_is_row else reversed(code.size)

        # calculate delta_time
        delta_time = min(abs(a_t - b_t), time_steps - abs(a_t - b_t))
        # calculate delta_diagonal and delta_parallel, where path is between corners of a box
        box_width = min(abs(a_x - b_x), dim_x - abs(a_x - b_x))
        box_height = min(abs(a_y - b_y), dim_y - abs(a_y - b_y))
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
                distance += delta_time * cls._step_weight_time(measurement_error_probability)
            if delta_parallel:
                distance += delta_parallel * cls._step_weight_parallel(eta, error_probability)
            if delta_diagonal:
                distance += delta_diagonal * cls._step_weight_diagonal(eta, error_probability)
        return distance

    @classmethod
    def _graphs(cls, code, time_steps, syndrome, error_probability=None, measurement_error_probability=None, eta=None):
        """Graphs of plaquette nodes and weighted edges consistent with the syndrome.

        Notes:

        * In the case of infinite bias, separate graphs are returned for each row and each column.
        * Nodes are added for all syndrome plaquettes in both "by row" and "by column" passes.
        * Edges are added between nodes on a given row or column; such edges are weighted by the distance function.

        :param code: Rotated toric code.
        :type code: RotatedToricCode
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
        :return: List of graphs weighted edges between plaquette nodes, consistent with the syndrome, as
            {((a_t, a_x, a_y), a_is_row), (b_t, b_x, b_y), b_is_row)): weight, ...}.
        :rtype: generator of dict of (((int, int, int), bool), ((int, int, int), bool)) edges to float weights.
        """
        # list of lattice nodes (or list of list of line nodes in case of infinite bias)
        lattice_nodes = []
        # get syndrome indices, as list of set where syndrome_indices[t] corresponds to time t
        syndrome_indices = [code.syndrome_to_plaquette_indices(s) for s in syndrome]
        # all plaquettes as (x, y)
        plaquette_indices = cls._plaquette_indices(code)

        def _add_edge(graph, a_node, b_node):
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
            graph.add_edge(a_node, b_node, cls._distance(code, time_steps, a_node, b_node, error_probability,
                                                         measurement_error_probability, eta))

        # iterate by rows then by columns
        for by_row in (True, False):
            # loop through lines (rows if by_row, cols if not by_row)
            for line in plaquette_indices if by_row else plaquette_indices.T:
                # line nodes
                line_nodes = []
                # loop through indices on line
                for (x, y) in line:
                    # loop through time
                    for t in range(time_steps):
                        if (x, y) in syndrome_indices[t]:  # if index in syndrome
                            # add node to line nodes
                            node = ((t, x, y), by_row)
                            line_nodes.append(node)
                if line_nodes:  # if any line nodes
                    if eta is None:  # if infinite bias
                        # yield graph for line nodes
                        graph = gt.SimpleGraph()
                        for a_node, b_node in itertools.combinations(line_nodes, 2):
                            _add_edge(graph, a_node, b_node)
                        yield graph
                    else:  # else finite bias
                        # add line nodes to lattice nodes
                        lattice_nodes.extend(line_nodes)

        if lattice_nodes:  # if any lattice nodes
            if eta is not None:  # if finite bias
                # yield graph for lattice nodes
                graph = gt.SimpleGraph()
                for a_node, b_node in itertools.combinations(lattice_nodes, 2):
                    _add_edge(graph, a_node, b_node)
                yield graph

    @classmethod
    def _matching(cls, graphs):
        """Combined matching (minimum weight perfect matching) over given graphs.

        :param graphs: List of graphs of weighted edges between nodes, as {(a_node, b_node): weight, ...}.
        :type graphs: list of dict of (object, object) edges to float weights.
        :return: Matches between nodes as (a_node, b_node).
        :rtype: set of (object, object)
        """
        matches = set()
        for graph in graphs:
            matches.update(gt.mwpm(graph))  # option to switch to networkx or blossom5 here
            del graph  # release heavy object
        return matches

    @classmethod
    def _clusters(cls, matches):
        """List of clusters from the given matches where each cluster is a directed path of plaquette indices.

        Note:

        * Matches between nodes with the same index are removed.
        * For consistency in testing, the clusters are ordered by their SW corner index, and the directed path of each
          cluster starts at the node with the lowest index of the cluster and traverses first along a column (the final
          index does not repeat the first index).

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

    @classmethod
    def _cluster_to_paths_and_defect(cls, code, cluster):
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

    @classmethod
    @functools.lru_cache()
    def _tparity(cls, time_steps, a_t, b_t):
        """Return 0 if a path from a_t to b_t does not cross plane between t=max and t=0, otherwise 1.

        Note:

        * Time indices are modulo time_steps, i.e. over 6 time_steps, -1 indexes the same time as 5.
        * If the shortest distance between a_t and b_t does not cross the plane between t=max and t=0, then 0 is
          returned, otherwise 1 is returned.
        * In case of ambiguity, 0 is returned arbitrarily. (should be right 50% of time)

        :param time_steps: Number of time steps.
        :type time_steps: int
        :param a_t: Time index.
        :type a_t: int
        :param b_t: Time index.
        :type b_t: int
        :return: T-parity, i.e. 0 or 1.
        :rtype: int
        """
        # indices mod time_steps
        a_t, b_t = a_t % time_steps, b_t % time_steps
        # note: a_t and b_t are positive relative to t=0, since taken mod time_steps)
        steps_in_bulk = abs(b_t - a_t)
        # if steps within bulk <= steps across t_max/t_0 boundary
        if steps_in_bulk <= time_steps - steps_in_bulk:
            return 0
        else:
            return 1

    @classmethod
    def _recovery_tparities(cls, code, time_steps, clusters):
        """Operator consisting of a paths of Pauli operators to fuse the plaquettes around each cluster with t-parity
        for X and Z plaquette types.

        :param code: Rotated toric code.
        :type code: RotatedToricCode
        :param time_steps: Number of time steps.
        :type time_steps: int
        :param clusters: List of clusters (directed paths of indices) as
            [[(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)], ...]
        :type clusters: list of list of (int, int, int)
        :return: Recovery operator in binary symplectic form, X-plaquette t-parity, Z-plaquette t-parity
        :rtype: numpy.array (1d), int, int
        """
        # identity recovery
        operator = code.new_pauli().to_bsf()
        # t-parities
        x_tparity, z_tparity = 0, 0
        # loop over clusters
        for cluster in clusters:
            # split into x_path, z_path, y_defect
            x_path, z_path, y_defect = cls._cluster_to_paths_and_defect(code, cluster)
            # loop over x indices in pairs, applying path operator between indices
            for (a_t, a_x, a_y), (b_t, b_x, b_y) in zip(x_path[::2], x_path[1::2]):
                operator ^= code.new_pauli().path((a_x, a_y), (b_x, b_y)).to_bsf()
                x_tparity ^= cls._tparity(time_steps, a_t, b_t)
            # loop over z indices in pairs, applying path operator between indices
            for (a_t, a_x, a_y), (b_t, b_x, b_y) in zip(z_path[::2], z_path[1::2]):
                operator ^= code.new_pauli().path((a_x, a_y), (b_x, b_y)).to_bsf()
                z_tparity ^= cls._tparity(time_steps, a_t, b_t)
        return operator, x_tparity, z_tparity

    @classmethod
    def _measurement_error_tparities(cls, code, measurement_error):
        """T-parity for X and Z plaquette types of the given measurement error.

        :param code: Rotated toric code.
        :type code: RotatedToricCode
        :param measurement_error: Measurement error as binary vector (matches syndrome format).
        :type measurement_error: numpy.array (1d)
        :return: X-plaquette t-parity, Z-plaquette t-parity
        :rtype: int, int
        """
        measurement_error_t_indices = code.syndrome_to_plaquette_indices(measurement_error)
        measurement_error_x_tparity = len([i for i in measurement_error_t_indices if code.is_x_plaquette(i)]) % 2
        measurement_error_z_tparity = (len(measurement_error_t_indices) - measurement_error_x_tparity) % 2
        return measurement_error_x_tparity, measurement_error_z_tparity

    class _ClusterNode:
        """Simple class containing cluster, X / Z plaquette indices; with reference equality for MWPM."""

        def __init__(self, cluster, x_index, z_index):
            """Initialise new cluster node for MWPM.

            :param cluster: Clusters (directed paths of indices) as
                [[(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)], ...]
            :type cluster: list of (int, int, int)
            :param x_index: X plaquette index as (t, x, y). Must be one of cluster indices.
            :type x_index: (int, int, int)
            :param z_index: Z plaquette index as (t, x, y). Must be one of cluster indices.
            :type z_index: (int, int, int)
            """
            assert x_index in cluster
            assert z_index in cluster
            self.cluster = cluster
            self.x_index = x_index
            self.z_index = z_index

    @classmethod
    def _cluster_distance(cls, code, time_steps, a_node, b_node):
        """Distance between cluster nodes.

        Notes:

        * The best distance function between clusters is not clearly defined so a heuristic is used.

        Algorithm:

        * Distance is defined as shortest Manhattan distance between any pair of plaquette indices in the clusters,
          noting that all dimensions are periodic.

        :param code: Rotated toric code.
        :type code: RotatedToricCode
        :param time_steps: Number of time steps.
        :type time_steps: int
        :param a_node: Node identifying cluster.
        :type a_node: _ClusterNode
        :param b_node: Node identifying cluster.
        :type b_node: _ClusterNode
        :return: Cluster distance.
        :rtype: int
        """
        dim_t = time_steps
        dim_y, dim_x = code.size
        # minimum manhattan distance between all pairs of cluster indices (periodic in all dimensions)
        return min(min(abs(a_t - b_t), dim_t - abs(a_t - b_t))
                   + min(abs(a_x - b_x), dim_x - abs(a_x - b_x))
                   + min(abs(a_y - b_y), dim_y - abs(a_y - b_y))
                   for (a_t, a_x, a_y), (b_t, b_x, b_y) in itertools.product(a_node.cluster, b_node.cluster))

    @classmethod
    def _cluster_graph(cls, code, time_steps, clusters):
        """Graph of cluster nodes and weighted edges consistent with the syndrome.

        Notes:

        * By construction, each cluster is either neutral (i.e. all defects fused) or defective (i.e. exactly one
          non-fused Y defect).
        * If there no defective clusters an empty graph is returned.
        * One node is added for each defective cluster.
        * Two nodes are added for each neutral cluster, provided that cluster consists of both X and Z plaquettes.
        * No node is added for any cluster that contains only X or only Z plaquettes.
        * Edges are added between all nodes with weight given by the cluster_distance function.

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
            x_path, z_path, y_defect = cls._cluster_to_paths_and_defect(code, cluster)
            # add cluster nodes to graph
            if y_defect:  # if cluster has non-fused Y-defect
                # unpack Y-defect indices
                x_defect_index, z_defect_index = y_defect
                # add as defective cluster
                cluster_node = cls._ClusterNode(cluster, x_defect_index, z_defect_index)
                defective_cluster_nodes.append(cluster_node)
            elif x_path and z_path:  # elif cluster has fused Y-defects
                # add twice as neutral cluster with representative X and Z indices
                neutral_cluster_nodes.append(cls._ClusterNode(cluster, x_path[0], z_path[0]))
                neutral_cluster_nodes.append(cls._ClusterNode(cluster, x_path[0], z_path[0]))
            else:  # else cluster has no Y-defects (fused or non-fused) so skip
                pass
        # if no defective cluster nodes then return empty graph
        if not defective_cluster_nodes:
            return graph
        # we should have an even number of defective cluster nodes
        assert len(defective_cluster_nodes) % 2 == 0
        # add edges to graph
        for a_node, b_node in itertools.combinations(defective_cluster_nodes + neutral_cluster_nodes, 2):
            graph.add_edge(a_node, b_node, cls._cluster_distance(code, time_steps, a_node, b_node))
        return graph

    @classmethod
    def _cluster_recovery_tparities(cls, code, time_steps, matches):
        """Operator consisting of a paths of Pauli operators to fuse clusters together with t-parity for X and Z
        plaquette types.

        :param code: Rotated toric code.
        :type code: RotatedToricCode
        :param time_steps: Number of time steps.
        :type time_steps: int
        :param matches: Matches between cluster nodes.
        :type matches: set of (_ClusterNode, _ClusterNode)
        :return: Recovery operator in binary symplectic form, X-plaquette t-parity, Z-plaquette t-parity
        :rtype: numpy.array (1d), int, int
        """
        # identity recovery
        operator = code.new_pauli().to_bsf()
        # t-parities
        x_tparity, z_tparity = 0, 0
        # loop over cluster matches
        for a_node, b_node in matches:
            # unpack indices
            (a_x_t, a_x_x, a_x_y), (a_z_t, a_z_x, a_z_y) = a_node.x_index, a_node.z_index
            (b_x_t, b_x_x, b_x_y), (b_z_t, b_z_x, b_z_y) = b_node.x_index, b_node.z_index
            # apply path between X plaquette indices
            operator ^= code.new_pauli().path((a_x_x, a_x_y), (b_x_x, b_x_y)).to_bsf()
            x_tparity ^= cls._tparity(time_steps, a_x_t, b_x_t)
            # apply path between Z plaquette indices
            operator ^= code.new_pauli().path((a_z_x, a_z_y), (b_z_x, b_z_y)).to_bsf()
            z_tparity ^= cls._tparity(time_steps, a_z_t, b_z_t)
        return operator, x_tparity, z_tparity
