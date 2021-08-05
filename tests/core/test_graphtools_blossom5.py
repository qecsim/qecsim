import math

import pytest

from qecsim.graphtools import blossom5


def test_blossom5_available():
    available = blossom5.available()
    print(available)
    assert isinstance(available, bool)


@pytest.mark.clib
def test_blossom5_infty():
    infty = blossom5.infty()
    print(infty)
    assert isinstance(infty, int)
    assert infty > 0


@pytest.mark.clib
def test_blossom5_mwpm_ids():
    edges = [(1, 2, 10), (1, 3, 25), (0, 2, 56), (0, 1, 15), (2, 3, 6)]
    mates = blossom5.mwpm_ids(edges)
    expected = {(0, 1), (2, 3)}
    assert mates == expected


@pytest.mark.clib
def test_blossom5_mwpm_ids_no_edges():
    edges = []
    mates = blossom5.mwpm_ids(edges)
    expected = set()
    assert mates == expected


@pytest.mark.clib
def test_blossom5_mwpm_ids_negative_weights():
    edges = [(1, 2, -90), (1, 3, -75), (0, 2, -44), (0, 1, -85), (2, 3, -94)]
    mates = blossom5.mwpm_ids(edges)
    expected = {(0, 1), (2, 3)}
    assert mates == expected


@pytest.mark.clib
def test_blossom5_mwpm():
    edges = [('b', 'c', 10), ('b', 'd', 25), ('a', 'c', 56), ('a', 'b', 15), ('c', 'd', 6)]
    mates = blossom5.mwpm(edges)
    sorted_mates = {tuple(sorted(match)) for match in mates}
    expected = {('a', 'b'), ('c', 'd')}
    assert sorted_mates == expected


@pytest.mark.clib
def test_blossom5_weight_to_int_fn():
    weights = [0, 0, 0, 0, 4.394449154672438, 8.788898309344876, 10.986122886681095, 4.394449154672438,
               6.591673732008657, 2.197224577336219, 0, 0, 2.197224577336219, 6.591673732008657, 4.394449154672438, 0,
               4.394449154672438, 6.591673732008657, 2.197224577336219, 0, 0, 0, 0, 0, 2.197224577336219,
               6.591673732008657, 10.986122886681095, 4.394449154672438, 8.788898309344876, 4.394449154672438,
               2.197224577336219, 6.591673732008657, 10.986122886681095, 4.394449154672438, 8.788898309344876,
               4.394449154672438, 4.394449154672438, 6.591673732008657, 2.197224577336219, 2.197224577336219,
               6.591673732008657, 4.394449154672438, 4.394449154672438, 8.788898309344876, 10.986122886681095,
               4.394449154672438, 6.591673732008657, 2.197224577336219]
    _weight_to_int = blossom5.weight_to_int_fn(weights)
    scaled_int_weights = [_weight_to_int(wt) for wt in weights]
    assert max(scaled_int_weights) < blossom5.infty() / 10, (
        'Max absolute weight is not smaller than infty by 1 order of magnitude')
    ratio_min_to_max_weight = min(wt for wt in weights if wt != 0) / max(weights)
    ratio_min_to_max_scaled_weight = min(wt for wt in scaled_int_weights if wt != 0) / max(scaled_int_weights)
    assert math.isclose(ratio_min_to_max_weight, ratio_min_to_max_scaled_weight, rel_tol=1e-7), (
        'Ratio of min / max for non-zero weights and scaled weights is not close')


@pytest.mark.clib
def test_blossom5_weight_to_int_fn_warning_zero(caplog):
    weights = [0, 2, blossom5.infty()]
    _weight_to_int = blossom5.weight_to_int_fn(weights)
    scaled_int_weights = [_weight_to_int(wt) for wt in weights]
    print(scaled_int_weights)
    print(caplog.text)
    assert 'SCALED MINIMUM ABSOLUTE NON-ZERO WEIGHT IS ZERO' in caplog.text


@pytest.mark.clib
def test_blossom5_weight_to_int_fn_warning_less_than_3sf(caplog):
    weights = [0, 294, blossom5.infty()]
    _weight_to_int = blossom5.weight_to_int_fn(weights)
    scaled_int_weights = [_weight_to_int(wt) for wt in weights]
    print(scaled_int_weights)
    print(caplog.text)
    assert 'SCALED MINIMUM ABSOLUTE NON-ZERO WEIGHT LESS THAN 3 S.F.' in caplog.text
