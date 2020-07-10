"""
Example Python interface for pypm.cpp.

See README-PYW.rst for how to build the C++ library dependency.

To test:
# $ python
# >>> import pypm_example
# >>> pypm_example.INFTY
# 1073741823
# >>>
# >>> edges = [(1, 2, 10), (1, 3, 25), (0, 2, 56), (0, 1, 15), (2, 3, 6)]
# >>> mates = pypm_example.mwpm_ids(edges)
# >>> mates
# {(0, 1), (2, 3)}
# >>>
# >>> edges = [('b', 'c', 10), ('b', 'd', 25), ('a', 'c', 56), ('a', 'b', 15), ('c', 'd', 6)]
# >>> mates = pypm_example.mwpm(edges)
# >>> mates
# {('c', 'd'), ('a', 'b')}
# >>>

"""
import ctypes

# load pypm library
_pypm = ctypes.CDLL('./libpypm.so')
# declare function argument types
_pypm.infty.argtypes = None
_pypm.infty.restype = ctypes.c_int
_pypm.mwpm.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                       ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
_pypm.mwpm.restype = None


# Integer that represents infinity (Blossom V algorithm). (Can be useful when converting float to int weights for MWPM)
INFTY = _pypm.infty()


def mwpm_ids(edges):
    """Minimum Weight Perfect Matching using node ids (Blossom V algorithm).

    Notes:

    * Node ids are assumed to form a contiguous set of non-negative integers starting at zero, e.g.  {0, 1, ...}.
    * All nodes are assumed to participate in at least one edge.

    :param edges: Edges as [(node_id, node_id, weight), ...].
    :type edges: list of (int, int, int)
    :return: Set of matches as {(node_id, node_id), ...}. (Each tuple is sorted.)
    :rtype: set of (int, int)
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
    _pypm.mwpm(ctypes.c_int(n_nodes), mates_array, ctypes.c_int(n_edges),
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
