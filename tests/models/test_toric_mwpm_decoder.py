import numpy as np
import pytest

from qecsim import paulitools as pt
from qecsim.models.toric import ToricCode, ToricMWPMDecoder


def test_toric_mwpm_decoder_properties():
    decoder = ToricMWPMDecoder()
    assert isinstance(decoder.label, str)
    assert isinstance(repr(decoder), str)
    assert isinstance(str(decoder), str)


@pytest.mark.parametrize('size, a_index, b_index, expected', [
    # 3 x 3
    ((3, 3), (0, 1, 0), (0, 1, 2), 1),  # across boundary horizontally
    ((3, 3), (0, 1, 2), (0, 1, 0), 1),  # ditto reversed
    # 5 x 5
    ((5, 5), (0, 1, 1), (0, 1, 3), 2),  # within lattice horizontally
    ((5, 5), (0, 1, 3), (0, 1, 1), 2),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 1, 4), 2),  # across boundary horizontally
    ((5, 5), (0, 1, 4), (0, 1, 1), 2),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 3, 1), 2),  # within lattice vertically
    ((5, 5), (0, 3, 1), (0, 1, 1), 2),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 4, 1), 2),  # across boundary vertically
    ((5, 5), (0, 4, 1), (0, 1, 1), 2),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 3, 3), 4),  # within lattice diagonally
    ((5, 5), (0, 3, 3), (0, 1, 1), 4),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 4, 4), 4),  # across boundary diagonally
    ((5, 5), (0, 4, 4), (0, 1, 1), 4),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 3, 2), 3),  # within lattice dog-leg
    ((5, 5), (0, 3, 2), (0, 1, 1), 3),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 4, 2), 3),  # across boundary dog-leg
    ((5, 5), (0, 4, 2), (0, 1, 1), 3),  # ditto reversed
    # dual
    ((5, 5), (1, 1, 1), (1, 3, 2), 3),  # within lattice dog-leg (dual)
    ((5, 5), (1, 3, 2), (1, 1, 1), 3),  # ditto reversed
    ((5, 5), (1, 1, 1), (1, 4, 2), 3),  # across boundary dog-leg (dual)
    ((5, 5), (1, 4, 2), (1, 1, 1), 3),  # ditto reversed
    # 3 x 5
    ((3, 5), (0, 1, 0), (0, 1, 2), 2),  # within lattice horizontally
    ((3, 5), (0, 1, 2), (0, 1, 0), 2),  # ditto reversed
    ((3, 5), (0, 1, 0), (0, 1, 3), 2),  # across boundary horizontally
    ((3, 5), (0, 1, 3), (0, 1, 0), 2),  # ditto reversed
    ((3, 5), (0, 1, 1), (0, 2, 1), 1),  # within lattice vertically
    ((3, 5), (0, 2, 1), (0, 1, 1), 1),  # ditto reversed
    ((3, 5), (0, 0, 1), (0, 2, 1), 1),  # across boundary vertically
    ((3, 5), (0, 2, 1), (0, 0, 1), 1),  # ditto reversed
    ((3, 5), (0, 1, 1), (0, 2, 3), 3),  # within lattice dog-leg
    ((3, 5), (0, 2, 3), (0, 1, 1), 3),  # ditto reversed
    ((3, 5), (0, 0, 0), (0, 2, 4), 2),  # across boundary dog-leg
    ((3, 5), (0, 2, 4), (0, 0, 0), 2),  # ditto reversed
    # 5 x 3
    ((5, 3), (0, 0, 1), (0, 2, 1), 2),  # within lattice vertically
    ((5, 3), (0, 2, 1), (0, 0, 1), 2),  # ditto reversed
    ((5, 3), (0, 0, 1), (0, 3, 1), 2),  # across boundary vertically
    ((5, 3), (0, 3, 1), (0, 0, 1), 2),  # ditto reversed
    ((5, 3), (0, 1, 1), (0, 1, 2), 1),  # within lattice horizontally
    ((5, 3), (0, 1, 2), (0, 1, 1), 1),  # ditto reversed
    ((5, 3), (0, 1, 0), (0, 1, 2), 1),  # across boundary horizontally
    ((5, 3), (0, 1, 2), (0, 1, 0), 1),  # ditto reversed
    ((5, 3), (0, 1, 1), (0, 3, 2), 3),  # within lattice dog-leg
    ((5, 3), (0, 3, 2), (0, 1, 1), 3),  # ditto reversed
    ((5, 3), (0, 0, 0), (0, 4, 2), 2),  # across boundary dog-leg
    ((5, 3), (0, 4, 2), (0, 0, 0), 2),  # ditto reversed
    # index modulo shape
    ((3, 3), (2, 7, -3), (0, -2, 5), 1),  # across boundary horizontally
    ((3, 3), (3, -5, 8), (-1, 4, -6), 1),  # ditto reversed
])
def test_toric_mwpm_decoder_distance(size, a_index, b_index, expected):
    assert ToricMWPMDecoder.distance(ToricCode(*size), a_index, b_index) == expected


@pytest.mark.parametrize('size, a_index, b_index', [
    ((5, 5), (0, 1, 1), (1, 2, 2)),  # different lattices
])
def test_toric_mwpm_decoder_invalid_distance(size, a_index, b_index):
    with pytest.raises(IndexError):
        ToricMWPMDecoder.distance(ToricCode(*size), a_index, b_index)


@pytest.mark.parametrize('error_pauli', [
    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1), (0, 2, 1))),
    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1), (0, 2, 1)).site('Z', (0, 3, 2), (0, 1, 0))),
    (ToricCode(5, 5).new_pauli().site('X', (1, 1, 1), (0, 2, 1)).site('Z', (0, 3, 2), (1, 1, 0))),
    (ToricCode(3, 5).new_pauli().site('X', (1, 1, 1), (0, 2, 1)).site('Z', (0, 1, 2), (1, 1, 4))),
    (ToricCode(5, 3).new_pauli().site('X', (1, 1, 1), (0, 2, 1)).site('Z', (0, 4, 2), (1, 2, 0))),
    (ToricCode(5, 3).new_pauli().site('Y', (1, 1, 1), (0, 2, 1)).site('Z', (0, 4, 2), (0, 3, 2), (0, 2, 2))),
    (ToricCode(5, 3).new_pauli().site('Y', (1, 1, 1), (1, 1, 2), (1, 1, 3)).site('Z', (0, 4, 2), (0, 3, 2), (0, 2, 2))),
    (ToricCode(5, 3).new_pauli().site('X', (1, 1, 1), (1, 1, 2), (1, 1, 3), (0, 4, 2), (0, 3, 2), (0, 2, 2))),
    (ToricCode(5, 3).new_pauli().site('Y', (1, 1, 1), (1, 1, 2), (1, 1, 3), (0, 4, 2), (0, 3, 2), (0, 2, 2))),
    (ToricCode(5, 3).new_pauli().site('Z', (1, 1, 1), (1, 1, 2), (1, 1, 3), (0, 4, 2), (0, 3, 2), (0, 2, 2))),
])
def test_toric_mwpm_decoder_decode(error_pauli):
    error = error_pauli.to_bsf()
    code = error_pauli.code
    decoder = ToricMWPMDecoder()
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery = decoder.decode(code, syndrome)
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
