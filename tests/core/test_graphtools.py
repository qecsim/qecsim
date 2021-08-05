import pytest

from qecsim import graphtools as gt


def test_simple_graph():
    graph = gt.SimpleGraph()
    graph.add_edge('a', 'b', 4)
    assert graph == {('a', 'b'): 4}
    graph.add_edge('a', 'b', 5)
    assert graph == {('a', 'b'): 5}
    graph.add_edge('b', 'a', 6)
    assert graph == {('b', 'a'): 6}
    graph.add_edge('a', 'b', 7)
    graph.add_edge('b', 'c', 8)
    assert graph == {('a', 'b'): 7, ('b', 'c'): 8}


def test_mwpm():
    graph = {('b', 'c'): 10, ('b', 'd'): 25, ('a', 'c'): 56, ('a', 'b'): 15, ('c', 'd'): 6}
    mates = gt.mwpm(graph)
    sorted_mates = {tuple(sorted(match)) for match in mates}
    expected = {('a', 'b'), ('c', 'd')}
    assert sorted_mates == expected


def test_mwpm_networkx():
    graph = {('b', 'c'): 10, ('b', 'd'): 25, ('a', 'c'): 56, ('a', 'b'): 15, ('c', 'd'): 6}
    mates = gt.mwpm_networkx(graph)
    sorted_mates = {tuple(sorted(match)) for match in mates}
    expected = {('a', 'b'), ('c', 'd')}
    assert sorted_mates == expected


@pytest.mark.clib
def test_mwpm_blossom5():
    graph = {('b', 'c'): 10, ('b', 'd'): 25, ('a', 'c'): 56, ('a', 'b'): 15, ('c', 'd'): 6}
    mates = gt.mwpm_blossom5(graph)
    sorted_mates = {tuple(sorted(match)) for match in mates}
    expected = {('a', 'b'), ('c', 'd')}
    assert sorted_mates == expected
