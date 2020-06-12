"""
This module is a wrapper for Blossom V, a fast C++ implementation of perfect matching, due to Vladimir Kolmogorov.

BLOSSOM V - implementation of Edmonds' algorithm for computing a minimum cost perfect matching in a graph
Version 2.05
http://pub.ist.ac.at/~vnk/software.html

This wrapper is included to enable high performance matching but some effort is required to build and install the C++
library dependency. When performance is not a concern, the python package NetworkX, https://networkx.github.io/, may be
sufficient and is typically already installed as a qecsim dependency.

The licence for Blossom V does not permit public redistribution of the code (see the original author's site for full
details of the licence). Therefore, Blossom V is not packaged with qecsim.
TODO: link to C++ library build instructions.

The functions in this module assume the presence of a C++ library 'libpypm.so' in one of the locations searched by
:func:`qecsim.util.load_clib`. All functions, except :func:`available` will fail with an OSError if the library cannot
be loaded.
"""
import ctypes
import decimal
import functools
import logging

from qecsim import util

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _libpypm():
    """C++ pypm library (lazy-loaded), with configured function argument and return types.

    :return: C++ pypm library
    :rtype: ctypes.CDLL
    :raises OSError: if Blossom V library cannot be loaded.
    """
    # load pypm library
    lib = util.load_clib('libpypm.so')
    # declare function argument types
    lib.infty.argtypes = None
    lib.infty.restype = ctypes.c_int
    lib.mwpm.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                              ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    lib.mwpm.restype = None
    return lib


@functools.lru_cache(maxsize=1)
def available():
    """
    :return: Availability of Blossom V library.
    :rtype: bool
    """
    try:
        _libpypm()
        return True
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def infty():
    """Integer that represents infinity (Blossom V algorithm).

    Note:

    * This can be useful when converting float to int weights for MWPM.

    :return: Value that represents infinity.
    :rtype: int
    :raises OSError: if Blossom V library cannot be loaded.
    """
    return _libpypm().infty()


def mwpm_ids(edges):
    """Minimum Weight Perfect Matching using node ids (Blossom V algorithm).

    Notes:

    * Node ids are assumed to form a contiguous set of non-negative integers starting at zero, e.g.  {0, 1, ...}.
    * All nodes are assumed to participate in at least one edge.

    :param edges: Edges as [(node_id, node_id, weight), ...].
    :type edges: list of (int, int, int)
    :return: Set of matches as {(node_id, node_id), ...}. (Each tuple is sorted.)
    :rtype: set of (int, int)
    :raises OSError: if Blossom V library cannot be loaded.
    """
    # extract and sort node ids
    node_ids = sorted(set(id for (id_a, id_b, _) in edges for id in (id_a, id_b)))
    # count n_nodes
    n_nodes = len(node_ids)
    # check node ids form contiguous set of non-negative integers starting at zero
    assert n_nodes == 0 or (node_ids[0] == 0 and node_ids[-1] == n_nodes - 1), (
        'Node ids are not a contiguous set of non-negative integers starting at zero.')
    # count n_edges
    n_edges = len(edges)
    # unzip edges
    nodes_a, nodes_b, weights = zip(*edges) if n_edges else ([], [], [])
    # prepare array types
    mates_array_type = ctypes.c_int * n_nodes
    edges_array_type = ctypes.c_int * n_edges
    # prepare empty mates
    mates_array = mates_array_type()
    # call C interface
    _libpypm().mwpm(ctypes.c_int(n_nodes), mates_array, ctypes.c_int(n_edges),
                    edges_array_type(*nodes_a), edges_array_type(*nodes_b), edges_array_type(*weights))
    # structure of mates: mates_array[i] = j means ith node matches jth node
    # convert to more useful format: e.g. convert [1, 0, 3, 2] to {(0, 1), (2, 3)}
    mates = {tuple(sorted((a, b))) for a, b in enumerate(mates_array)}
    return mates


def mwpm(edges):
    """Minimum Weight Perfect Matching using node objects (Blossom V algorithm).

    :param edges: Edges as [(node, node, weight), ...].
    :type edges: list of (object, object, int)
    :return: Set of matches as {(node, node), ...}.
    :rtype: set of (object, object)
    :raises OSError: if Blossom V library cannot be loaded.
    """
    # list of nodes without duplicates
    nodes = list(set(node for (node_a, node_b, _) in edges for node in (node_a, node_b)))
    # dict of node to id
    node_to_id = dict((n, i) for i, n in enumerate(nodes))
    # edges using ids
    edge_ids = [(node_to_id[node_a], node_to_id[node_b], weight) for node_a, node_b, weight in edges]
    # mwpm using ids
    mate_ids = mwpm_ids(edge_ids)
    # matches using objects
    mates = {(nodes[node_id_a], nodes[node_id_b]) for node_id_a, node_id_b in mate_ids}
    return mates


def weight_to_int_fn(weights):
    """Given an iterable of weights, return a function that scales all weights by a multiplicative factor and rounds to
    integers such that the largest (absolute) weight is an order of magnitude smaller than :func:`infty`.

    Notes:

    * The returned function is useful to convert float weights to ints for the Blossom V implementation.
    * If all weights are integers and the largest (absolute) weight is at least an order of magnitude smaller than
      :func:`infty` then the identity function is returned.
    * If the scaling function would scale the smallest (absolute) non-zero weight to zero or less than 3 significant
      figures, then a warning is logged.

    :param weights: Weights (positive and/or negative)
    :type weights: iterable of int or float
    :return: Function with signature weight_to_int(float or int) -> int
    :rtype: function
    :raises OSError: if Blossom V library cannot be loaded.
    """
    # extract absolute non-zero weights
    abs_non_zero_wts = {abs(wt) for wt in weights if wt != 0}

    # if all weights are zero
    if len(abs_non_zero_wts) == 0:
        # return zero (int) function
        return lambda wt: 0

    # extract smallest and largest absolute non-zero weights
    min_abs_non_zero_wt = min(abs_non_zero_wts)
    max_abs_non_zero_wt = max(abs_non_zero_wts)

    # if largest (absolute) weight is less than "infty" and all weight are ints
    if max_abs_non_zero_wt < infty() / 10 and all(isinstance(wt, int) for wt in weights):
        # return identity function
        return lambda wt: wt

    # define scaling so largest (absolute) weight is an order of magnitude smaller than "infty"
    scaling = infty() / 10 / max_abs_non_zero_wt

    # define _weight_to_int using scaling
    def _weight_to_int(weight):
        # multiply weight by scaling (round to nearest with ties going away from zero).
        return int(decimal.Decimal(weight * scaling).to_integral_value(rounding=decimal.ROUND_HALF_UP))

    # warn if smallest (absolute) weight is zero or less than 3 significant figures.
    scaled_min_abs_non_zero_wt = _weight_to_int(min_abs_non_zero_wt)
    if scaled_min_abs_non_zero_wt == 0:
        logger.warning('SCALED MINIMUM ABSOLUTE NON-ZERO WEIGHT IS ZERO')
    elif scaled_min_abs_non_zero_wt < 100:
        logger.warning('SCALED MINIMUM ABSOLUTE NON-ZERO WEIGHT LESS THAN 3 S.F.:{}'.format(scaled_min_abs_non_zero_wt))

    return _weight_to_int
