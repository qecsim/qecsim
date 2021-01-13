import numpy as np
import pytest

from qecsim import paulitools as pt
from qecsim.models.planar import PlanarCode, PlanarMWPMDecoder


def test_planar_mwpm_decoder_properties():
    decoder = PlanarMWPMDecoder()
    assert isinstance(decoder.label, str)
    assert isinstance(repr(decoder), str)
    assert isinstance(str(decoder), str)


@pytest.mark.parametrize('size, a_index, b_index, expected', [
    # 3 x 3
    ((3, 3), (-1, 2), (3, 2), 2),  # across north boundary vertically
    ((3, 3), (3, 2), (-1, 2), 2),  # ditto reversed
    ((3, 3), (-1, 2), (1, 4), 2),  # across north boundary diagonally
    ((3, 3), (1, 4), (-1, 2), 2),  # ditto reversed
    ((3, 3), (-1, 2), (3, 4), 3),  # across north boundary dog-leg
    ((3, 3), (3, 4), (-1, 2), 3),  # ditto reversed
    ((3, 3), (-1, 2), (-1, 4), 0),  # outside north boundary
    ((3, 3), (-1, 4), (-1, 2), 0),  # ditto reversed
    ((3, 3), (3, 2), (5, 2), 1),  # across south boundary vertically
    ((3, 3), (5, 2), (3, 2), 1),  # ditto reversed
    ((3, 3), (3, 2), (5, 4), 2),  # across south boundary diagonally
    ((3, 3), (5, 4), (3, 2), 2),  # ditto reversed
    ((3, 3), (3, 0), (5, 4), 3),  # across south boundary dog-leg
    ((3, 3), (5, 4), (3, 0), 3),  # ditto reversed
    ((3, 3), (5, 2), (5, 4), 0),  # outside south boundary
    ((3, 3), (5, 4), (5, 2), 0),  # ditto reversed
    ((3, 3), (-1, 2), (5, 4), 0),  # across north and south boundary
    ((3, 3), (5, 4), (-1, 2), 0),  # ditto reversed
    # 3 x 3 dual
    ((3, 3), (2, -1), (2, 3), 2),  # across west boundary horizontally
    ((3, 3), (2, 3), (2, -1), 2),  # ditto reversed
    ((3, 3), (2, -1), (4, 1), 2),  # across west boundary diagonally
    ((3, 3), (4, 1), (2, -1), 2),  # ditto reversed
    ((3, 3), (2, -1), (4, 3), 3),  # across west boundary dog-leg
    ((3, 3), (4, 3), (2, -1), 3),  # ditto reversed
    ((3, 3), (-2, 3), (4, -1), 0),  # outside west boundary
    ((3, 3), (4, -1), (2, -1), 0),  # ditto reversed
    ((3, 3), (2, 3), (2, 5), 1),  # across east boundary horizontally
    ((3, 3), (2, 5), (2, 3), 1),  # ditto reversed
    ((3, 3), (2, 3), (4, 5), 2),  # across east boundary diagonally
    ((3, 3), (4, 5), (2, 3), 2),  # ditto reversed
    ((3, 3), (0, 3), (4, 5), 3),  # across east boundary dog-leg
    ((3, 3), (4, 5), (0, 3), 3),  # ditto reversed
    ((3, 3), (0, 5), (4, 5), 0),  # outside east boundary
    ((3, 3), (4, 5), (0, 5), 0),  # ditto reversed
    ((3, 3), (-2, 3), (4, 5), 0),  # across west and east boundary
    ((3, 3), (4, 5), (2, -1), 0),  # ditto reversed
    # #  5 x 5
    ((5, 5), (3, 2), (3, 6), 2),  # within lattice horizontally
    ((5, 5), (3, 6), (3, 2), 2),  # ditto reversed
    ((5, 5), (3, 2), (7, 2), 2),  # within lattice vertically
    ((5, 5), (7, 2), (3, 2), 2),  # ditto reversed
    ((5, 5), (3, 2), (7, 6), 4),  # within lattice diagonally
    ((5, 5), (7, 6), (3, 2), 4),  # ditto reversed
    ((5, 5), (3, 2), (7, 4), 3),  # within lattice dog-leg
    ((5, 5), (7, 4), (3, 2), 3),  # ditto reversed
    # 5 x 5 dual
    ((5, 5), (2, 3), (2, 7), 2),  # within lattice horizontally
    ((5, 5), (2, 7), (2, 3), 2),  # ditto reversed
    ((5, 5), (2, 3), (6, 3), 2),  # within lattice vertically
    ((5, 5), (6, 3), (2, 3), 2),  # ditto reversed
    ((5, 5), (2, 3), (6, 7), 4),  # within lattice diagonally
    ((5, 5), (6, 7), (2, 3), 4),  # ditto reversed
    ((5, 5), (2, 3), (6, 5), 3),  # within lattice dog-leg
    ((5, 5), (6, 5), (2, 3), 3),  # ditto reversed
    # 3 x 5
    ((3, 5), (3, 0), (3, 4), 2),  # within lattice horizontally
    ((3, 5), (3, 4), (3, 0), 2),  # ditto reversed
    ((3, 5), (1, 2), (3, 2), 1),  # within lattice vertically
    ((3, 5), (3, 2), (1, 2), 1),  # ditto reversed
    ((3, 5), (1, 2), (3, 6), 3),  # within lattice dog-leg
    ((3, 5), (3, 6), (1, 2), 3),  # ditto reversed
    ((3, 5), (-1, 0), (3, 6), 5),  # across north boundary dog-leg
    ((3, 5), (3, 6), (-1, 0), 5),  # ditto reversed
    ((3, 5), (3, 0), (5, 6), 4),  # across south boundary dog-leg
    ((3, 5), (5, 6), (3, 0), 4),  # ditto reversed
    # 5 x 3
    ((5, 3), (1, 2), (5, 2), 2),  # within lattice vertically
    ((5, 3), (5, 2), (1, 2), 2),  # ditto reversed
    ((5, 3), (3, 2), (3, 4), 1),  # within lattice horizontally
    ((5, 3), (3, 4), (3, 2), 1),  # ditto reversed
    ((5, 3), (3, 2), (7, 4), 3),  # within lattice dog-leg
    ((5, 3), (7, 4), (3, 2), 3),  # ditto reversed
    ((5, 3), (-1, 0), (7, 2), 5),  # across north boundary dog-leg
    ((5, 3), (7, 2), (-1, 0), 5),  # ditto reversed
    ((5, 3), (3, 0), (9, 4), 5),  # across south boundary dog-leg
    ((5, 3), (9, 4), (3, 0), 5),  # ditto reversed
])
def test_planar_mwpm_decoder_distance(size, a_index, b_index, expected):
    assert PlanarMWPMDecoder.distance(PlanarCode(*size), a_index, b_index) == expected


@pytest.mark.parametrize('size, a_index, b_index', [
    ((5, 5), (0, 0), (4, 5)),  # invalid plaquette index
    ((5, 5), (3, 5), (4, 4)),  # invalid plaquette index
    ((5, 5), (3, 2), (4, 5)),  # different lattices
])
def test_planar_mwpm_decoder_invalid_distance(size, a_index, b_index):
    with pytest.raises(IndexError):
        PlanarMWPMDecoder.distance(PlanarCode(*size), a_index, b_index)


@pytest.mark.parametrize('error_pauli', [
    (PlanarCode(5, 5).new_pauli().site('X', (2, 0))),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2), (4, 2))),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2), (4, 2)).site('Z', (6, 4), (2, 0))),
    (PlanarCode(5, 5).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (6, 4), (1, 1))),
    (PlanarCode(3, 5).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (2, 4), (1, 7))),
    (PlanarCode(5, 3).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (8, 4), (3, 1))),
    (PlanarCode(5, 3).new_pauli().site('Y', (1, 3), (4, 2)).site('Z', (8, 4), (6, 4), (4, 4))),
    (PlanarCode(5, 3).new_pauli().site('Y', (1, 3), (3, 3), (5, 3)).site('Z', (8, 4), (6, 4), (4, 4))),
    (PlanarCode(5, 3).new_pauli().site('X', (1, 3), (3, 3), (5, 3), (8, 4), (6, 4), (4, 4))),
    (PlanarCode(5, 3).new_pauli().site('Y', (1, 3), (3, 3), (5, 3), (8, 4), (6, 4), (4, 4))),
    (PlanarCode(5, 3).new_pauli().site('Z', (1, 3), (3, 3), (5, 3), (8, 4), (6, 4), (4, 4))),
])
def test_planar_mwpm_decoder_decode(error_pauli):
    error = error_pauli.to_bsf()
    code = error_pauli.code
    decoder = PlanarMWPMDecoder()
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery = decoder.decode(code, syndrome)
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
