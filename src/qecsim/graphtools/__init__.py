"""
This module contains a simple graph object and functions for minimum weight perfect matching with a choice of backend.

Possible matching backends are the NetworkX Python library and the Blossom V C++ library.

See also :mod:`qecsim.graphtools.blossom5`
"""
import networkx as nx

from . import blossom5


class SimpleGraph(dict):
    """Extension of dict with :meth:`add_edge` method that prevents duplicate reversed edges as keys."""

    def add_edge(self, node_a, node_b, weight):
        """Add item `(node_a, node_b): weight` and remove the reversed key if it exists."""
        self.pop((node_b, node_a), None)
        self[(node_a, node_b)] = weight


def mwpm(graph):
    """Minimum weight perfect matching over a graph.

    Notes:

    * Attempts to use Blossom V C++ library but, if unavailable, falls back to NetworkX Python library.
    * Behaviour is undefined if the graph contains the same edge in reverse order.

    :param graph: Graph of weighted edges between nodes, as {(a_node, b_node): weight, ...}.
    :type graph: dict of (object, object) edges to float or int weights.
    :return: Matching between nodes as (a_node, b_node).
    :rtype: set of (object, object)
    """
    if blossom5.available():  # clib
        return mwpm_blossom5(graph)
    else:
        return mwpm_networkx(graph)


def mwpm_networkx(graph):
    """Minimum weight perfect matching over a graph (using NetworkX Python library).

    Notes:

    * Behaviour is undefined if the graph contains the same edge in reverse order.

    :param graph: Graph of weighted edges between nodes, as {(a_node, b_node): weight, ...}.
    :type graph: dict of (object, object) edges to float or int weights.
    :return: Matching between nodes as (a_node, b_node).
    :rtype: set of (object, object)
    """
    # if graph has no edges, shortcut to return empty matching
    if not graph:
        return set()
    # networkx.algorithms.max_weight_matching is maximum weight matching so we take negative of all edge weight
    graph_for_matching = nx.Graph()
    graph_for_matching.add_weighted_edges_from((a, b, -w) for (a, b), w in graph.items())
    # matching as a set of tuples (a_node, b_node)
    matching = nx.algorithms.max_weight_matching(graph_for_matching, maxcardinality=True)
    return matching


def mwpm_blossom5(graph):
    """Minimum weight perfect matching over a graph (using Blossom V C++ library).

    Notes:

    * Behaviour is undefined if the graph contains the same edge in reverse order.

    :param graph: Graph of weighted edges between nodes, as {(a_node, b_node): weight, ...}.
    :type graph: dict of (object, object) edges to float or int weights.
    :return: Matching between nodes as (a_node, b_node).
    :rtype: set of (object, object)
    :raises OSError: if Blossom V library cannot be loaded.
    """
    # if graph has no edges, shortcut to return empty matching
    if not graph:
        return set()
    # create weight_to_int function from edge weights
    weight_to_int_fn = blossom5.weight_to_int_fn(list(graph.values()))
    # create edges with integer weights
    edges = list((node_a, node_b, weight_to_int_fn(weight)) for (node_a, node_b), weight in graph.items())
    # matching as a set of tuples (a_node, b_node)
    matching = blossom5.mwpm(edges)
    return matching
