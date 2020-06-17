import numpy as np
import pytest

from qecsim import paulitools as pt
from qecsim.models.rotatedtoric import RotatedToricCode


@pytest.mark.parametrize('size', [
    (2, 2),
    (4, 4),
    (2, 4),
    (4, 2),
    (6, 6),
    (4, 6),
    (6, 4),
])
def test_rotated_toric_pauli_properties(size):
    code = RotatedToricCode(*size)
    pauli = code.new_pauli()
    assert pauli.code == code
    assert isinstance(repr(pauli), str)
    assert isinstance(str(pauli), str)


@pytest.mark.parametrize('pauli', [
    RotatedToricCode(4, 4).new_pauli(),
    RotatedToricCode(4, 6).new_pauli().plaquette((1, 1)).plaquette((3, 2)),
    RotatedToricCode(6, 6).new_pauli().logical_x1().plaquette((-1, 0)).plaquette((4, 1)),
    RotatedToricCode(6, 4).new_pauli().logical_z1().plaquette((1, 5)).plaquette((3, -1)),
    RotatedToricCode(8, 6).new_pauli().logical_x2().plaquette((0, 2)).plaquette((1, 1)),
    RotatedToricCode(6, 8).new_pauli().logical_z2().plaquette((1, -1)).plaquette((2, 4)),
])
def test_rotated_toric_pauli_new_to_bsf(pauli):
    assert pauli.code.new_pauli(pauli.to_bsf()) == pauli, (
        'Conversion to_bsf+from_bsf does not result in equality.')


@pytest.mark.parametrize('pauli', [
    RotatedToricCode(4, 4).new_pauli(),
    RotatedToricCode(4, 6).new_pauli().plaquette((1, 1)).plaquette((3, 2)),
    RotatedToricCode(6, 6).new_pauli().logical_x1().plaquette((-1, 0)).plaquette((4, 1)),
    RotatedToricCode(6, 4).new_pauli().logical_z1().plaquette((1, 5)).plaquette((3, -1)),
    RotatedToricCode(8, 6).new_pauli().logical_x2().plaquette((0, 2)).plaquette((1, 1)),
    RotatedToricCode(6, 8).new_pauli().logical_z2().plaquette((1, -1)).plaquette((2, 4)),
])
def test_rotated_toric_pauli_copy(pauli):
    copy_pauli = pauli.copy()
    assert copy_pauli == pauli, 'Copy Pauli does not equal original Pauli'
    assert copy_pauli.logical_x1() != pauli, 'Modified copy Pauli still equals original Pauli'


@pytest.mark.parametrize('pauli, index, expected', [
    # in-bounds
    (RotatedToricCode(6, 6).new_pauli(), (0, 0), 'I'),
    (RotatedToricCode(6, 6).new_pauli().site('X', (1, 1)), (1, 1), 'X'),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (3, 3)), (3, 3), 'Y'),
    (RotatedToricCode(6, 6).new_pauli().site('Z', (2, 4)), (2, 4), 'Z'),
    (RotatedToricCode(6, 6).new_pauli().site('X', (1, 1)), (3, 3), 'I'),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (3, 3)), (2, 4), 'I'),
    (RotatedToricCode(6, 6).new_pauli().site('Z', (2, 4)), (1, 1), 'I'),
    # out-bounds
    (RotatedToricCode(4, 4).new_pauli(), (-1, -1), 'I'),
    (RotatedToricCode(4, 4).new_pauli().site('X', (4, 2)), (4, 2), 'X'),
    (RotatedToricCode(4, 4).new_pauli().site('Y', (4, 4)), (4, 4), 'Y'),
    (RotatedToricCode(4, 4).new_pauli().site('Z', (-2, 2)), (-2, 2), 'Z'),
    (RotatedToricCode(4, 4).new_pauli().site('X', (4, 2)), (0, 2), 'X'),
    (RotatedToricCode(4, 4).new_pauli().site('Y', (4, 4)), (0, 0), 'Y'),
    (RotatedToricCode(4, 4).new_pauli().site('Z', (-2, 2)), (2, 2), 'Z'),
])
def test_rotated_toric_pauli_operator(pauli, index, expected):
    assert pauli.operator(index) == expected


@pytest.mark.parametrize('toric_pauli, op_counts, message', [
    (RotatedToricCode(4, 4).new_pauli().site('I', (2, 2)),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'site failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('X', (2, 2)),
     {'I': 15, 'X': 1, 'Y': 0, 'Z': 0}, 'site failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('Y', (2, 2)),
     {'I': 15, 'X': 0, 'Y': 1, 'Z': 0}, 'site failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('Z', (2, 2)),
     {'I': 15, 'X': 0, 'Y': 0, 'Z': 1}, 'site failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('X', (2, 2)).site('X', (2, 2)),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('Y', (2, 2)).site('Y', (2, 2)),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('Z', (2, 2)).site('Z', (2, 2)),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('X', (2, 2)).site('Z', (2, 2)),
     {'I': 15, 'X': 0, 'Y': 1, 'Z': 0}, 'site X+Z=Y failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('X', (2, 2)).site('Y', (2, 2)),
     {'I': 15, 'X': 0, 'Y': 0, 'Z': 1}, 'site X+Y=Z failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('Y', (2, 2)).site('Z', (2, 2)),
     {'I': 15, 'X': 1, 'Y': 0, 'Z': 0}, 'site Y+Z=X failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('X', (2, 2)).site('X', (3, 1)),
     {'I': 14, 'X': 2, 'Y': 0, 'Z': 0}, 'multi-site failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('Y', (2, 2)).site('Y', (3, 1)),
     {'I': 14, 'X': 0, 'Y': 2, 'Z': 0}, 'multi-site failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('Z', (2, 2)).site('Z', (3, 1)),
     {'I': 14, 'X': 0, 'Y': 0, 'Z': 2}, 'multi-site failed.'),
    (RotatedToricCode(4, 6).new_pauli().site('X', (0, -1)),
     {'I': 23, 'X': 1, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedToricCode(4, 6).new_pauli().site('X', (-1, 0)),
     {'I': 23, 'X': 1, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedToricCode(4, 6).new_pauli().site('X', (-1, 3)),
     {'I': 23, 'X': 1, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedToricCode(4, 6).new_pauli().site('X', (0, 4)),
     {'I': 23, 'X': 1, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedToricCode(4, 6).new_pauli().site('X', (5, 4)),
     {'I': 23, 'X': 1, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedToricCode(4, 6).new_pauli().site('X', (6, 3)),
     {'I': 23, 'X': 1, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedToricCode(4, 6).new_pauli().site('X', (6, 0)),
     {'I': 23, 'X': 1, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedToricCode(4, 6).new_pauli().site('X', (5, -1)),
     {'I': 23, 'X': 1, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
])
def test_rotated_toric_pauli_site(toric_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(toric_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('toric_pauli, op_counts, message', [
    (RotatedToricCode(4, 4).new_pauli().site('I', (2, 2), (3, 1)),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'sites failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('X', (2, 2), (3, 1)),
     {'I': 14, 'X': 2, 'Y': 0, 'Z': 0}, 'sites failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('Y', (2, 2), (3, 1)),
     {'I': 14, 'X': 0, 'Y': 2, 'Z': 0}, 'sites failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('Z', (2, 2), (3, 1)),
     {'I': 14, 'X': 0, 'Y': 0, 'Z': 2}, 'sites failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('X', (2, 2), (2, 2)),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'sites self-inverse failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('Y', (2, 2), (2, 2)),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'sites self-inverse failed.'),
    (RotatedToricCode(4, 4).new_pauli().site('Z', (2, 2), (2, 2)),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'sites self-inverse failed.'),
])
def test_rotated_toric_pauli_sites(toric_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(toric_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('toric_pauli, op_counts, message', [
    (RotatedToricCode(4, 4).new_pauli().plaquette((1, 1)),
     {'I': 12, 'X': 0, 'Y': 0, 'Z': 4}, 'z-plaquette failed.'),
    (RotatedToricCode(4, 4).new_pauli().plaquette((1, 1)).plaquette((1, 1)),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'z-plaquette self-inverse failed.'),
    (RotatedToricCode(4, 4).new_pauli().plaquette((1, 1)).plaquette((2, 2)),
     {'I': 10, 'X': 0, 'Y': 0, 'Z': 6}, 'adjacent z-plaquettes failed.'),
    (RotatedToricCode(4, 4).new_pauli().plaquette((1, -1)),
     {'I': 12, 'X': 0, 'Y': 0, 'Z': 4}, 'z-boundary plaquette failed.'),
    (RotatedToricCode(4, 4).new_pauli().plaquette((1, 3)),
     {'I': 12, 'X': 0, 'Y': 0, 'Z': 4}, 'z-boundary plaquette failed.'),

    (RotatedToricCode(4, 4).new_pauli().plaquette((1, 2)),
     {'I': 12, 'X': 4, 'Y': 0, 'Z': 0}, 'x-plaquette failed.'),
    (RotatedToricCode(4, 4).new_pauli().plaquette((1, 2)).plaquette((1, 2)),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'x-plaquette self-inverse failed.'),
    (RotatedToricCode(4, 4).new_pauli().plaquette((1, 2)).plaquette((2, 1)),
     {'I': 10, 'X': 6, 'Y': 0, 'Z': 0}, 'adjacent x-plaquettes failed.'),
    (RotatedToricCode(4, 4).new_pauli().plaquette((-1, 0)),
     {'I': 12, 'X': 4, 'Y': 0, 'Z': 0}, 'x-boundary plaquette failed.'),
    (RotatedToricCode(4, 4).new_pauli().plaquette((3, 0)),
     {'I': 12, 'X': 4, 'Y': 0, 'Z': 0}, 'x-boundary plaquette failed.'),

    (RotatedToricCode(4, 4).new_pauli().plaquette((1, 1)).plaquette((1, 2)),
     {'I': 10, 'X': 2, 'Y': 2, 'Z': 2}, 'adjacent z- and x-plaquettes failed.'),
])
def test_rotated_toric_pauli_plaquette(toric_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(toric_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('toric_pauli, op_counts, message', [
    (RotatedToricCode(4, 4).new_pauli().logical_x1(),
     {'I': 12, 'X': 4, 'Y': 0, 'Z': 0}, 'logical_x1 failed.'),
    (RotatedToricCode(4, 4).new_pauli().logical_x1().logical_x1(),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_x1 self-inverse failed.'),
    (RotatedToricCode(4, 4).new_pauli().logical_z1(),
     {'I': 12, 'X': 0, 'Y': 0, 'Z': 4}, 'logical_z1 failed.'),
    (RotatedToricCode(4, 4).new_pauli().logical_z1().logical_z1(),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_z1 self-inverse failed.'),
    (RotatedToricCode(4, 4).new_pauli().logical_x1().logical_z1(),
     {'I': 9, 'X': 3, 'Y': 1, 'Z': 3}, 'logical_x1_z1 failed.'),

    (RotatedToricCode(4, 4).new_pauli().logical_x2(),
     {'I': 12, 'X': 4, 'Y': 0, 'Z': 0}, 'logical_x2 failed.'),
    (RotatedToricCode(4, 4).new_pauli().logical_x2().logical_x2(),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_x2 self-inverse failed.'),
    (RotatedToricCode(4, 4).new_pauli().logical_z2(),
     {'I': 12, 'X': 0, 'Y': 0, 'Z': 4}, 'logical_z2 failed.'),
    (RotatedToricCode(4, 4).new_pauli().logical_z2().logical_z2(),
     {'I': 16, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_z2 self-inverse failed.'),
    (RotatedToricCode(4, 4).new_pauli().logical_x2().logical_z2(),
     {'I': 9, 'X': 3, 'Y': 1, 'Z': 3}, 'logical_x2_z2 failed.'),

    (RotatedToricCode(4, 4).new_pauli().logical_x1().logical_x2(),
     {'I': 10, 'X': 6, 'Y': 0, 'Z': 0}, 'logical_x1_x2 failed.'),
    (RotatedToricCode(4, 4).new_pauli().logical_z1().logical_z2(),
     {'I': 10, 'X': 0, 'Y': 0, 'Z': 6}, 'logical_z1_z2 failed.'),
])
def test_rotated_toric_pauli_logical(toric_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(toric_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('size, a_index, b_index', [
    # 6 x 6
    ((6, 6), (1, 1), (1, 1)),  # same index
    ((6, 6), (1, 1), (3, 1)),  # within lattice horizontally
    ((6, 6), (3, 1), (1, 1)),  # ditto reversed
    ((6, 6), (0, 1), (4, 1)),  # across boundary horizontally
    ((6, 6), (4, 1), (0, 1)),  # ditto reversed
    ((6, 6), (1, 1), (1, 3)),  # within lattice vertically
    ((6, 6), (1, 3), (1, 1)),  # ditto reversed
    ((6, 6), (1, 0), (1, 4)),  # across boundary vertically
    ((6, 6), (1, 4), (1, 0)),  # ditto reversed
    ((6, 6), (1, 1), (2, 2)),  # within lattice dog-leg
    ((6, 6), (2, 2), (1, 1)),  # ditto reversed
    ((6, 6), (2, 0), (3, 5)),  # across boundary dog-leg
    ((6, 6), (3, 5), (2, 0)),  # ditto reversed
    # 4 x 6
    ((4, 6), (1, 1), (2, 2)),  # within lattice dog-leg
    ((4, 6), (2, 2), (1, 1)),  # ditto reversed
    ((4, 6), (2, 0), (3, 3)),  # across boundary dog-leg
    ((4, 6), (3, 3), (2, 0)),  # ditto reversed
    # index modulo shape
    ((6, 6), (0, 1), (-2, 1)),  # across boundary horizontally
    ((6, 6), (-2, 1), (0, 1)),  # ditto reversed
])
def test_rotated_toric_pauli_path(size, a_index, b_index):
    code = RotatedToricCode(*size)
    pauli = code.new_pauli().path(a_index, b_index)
    syndrome = pt.bsp(pauli.to_bsf(), code.stabilizers.T)
    syndrome_indices = code.syndrome_to_plaquette_indices(syndrome)
    if a_index == b_index:
        assert len(syndrome_indices) == 0, 'Unexpected syndrome for null path'
    else:
        dim_y, dim_x = code.size
        a_index = tuple(np.mod(a_index, (dim_x, dim_y)))
        b_index = tuple(np.mod(b_index, (dim_x, dim_y)))
        assert syndrome_indices == {a_index, b_index}, 'Path does not give expected syndrome'


@pytest.mark.parametrize('code, a_index, b_index, expected', [
    # between Z plaquettes
    (RotatedToricCode(8, 8), (2, 2), (2, 2),
     RotatedToricCode(8, 8).new_pauli()),  # same site
    (RotatedToricCode(8, 8), (0, 2), (2, 2),
     RotatedToricCode(8, 8).new_pauli().site('X', (1, 2), (2, 2))),  # along row
    (RotatedToricCode(8, 8), (2, 0), (2, 2),
     RotatedToricCode(8, 8).new_pauli().site('X', (2, 1), (2, 2))),  # along column
    (RotatedToricCode(8, 8), (0, 2), (3, 1),
     RotatedToricCode(8, 8).new_pauli().site('X', (1, 2), (2, 2), (3, 2))),  # dog-leg row
    (RotatedToricCode(8, 8), (0, 2), (3, 3),
     RotatedToricCode(8, 8).new_pauli().site('X', (1, 3), (2, 3), (3, 3))),  # dog-leg row
    (RotatedToricCode(8, 8), (2, 0), (1, 3),
     RotatedToricCode(8, 8).new_pauli().site('X', (2, 1), (2, 2), (2, 3))),  # dog-leg column
    (RotatedToricCode(8, 8), (2, 0), (3, 3),
     RotatedToricCode(8, 8).new_pauli().site('X', (3, 1), (3, 2), (3, 3))),  # dog-leg column
    (RotatedToricCode(8, 8), (0, 2), (6, 2),
     RotatedToricCode(8, 8).new_pauli().site('X', (0, 2), (7, 2))),  # along row across boundary
    (RotatedToricCode(8, 8), (2, 0), (2, 6),
     RotatedToricCode(8, 8).new_pauli().site('X', (2, 0), (2, 7))),  # along column across boundary
    # between X plaquettes
    (RotatedToricCode(8, 8), (2, 1), (2, 1),
     RotatedToricCode(8, 8).new_pauli()),  # same site
    (RotatedToricCode(8, 8), (0, 1), (2, 1),
     RotatedToricCode(8, 8).new_pauli().site('Z', (1, 1), (2, 1))),  # along row
    (RotatedToricCode(8, 8), (2, 3), (2, 1),
     RotatedToricCode(8, 8).new_pauli().site('Z', (2, 2), (2, 3))),  # along column
    (RotatedToricCode(8, 8), (0, 1), (3, 2),
     RotatedToricCode(8, 8).new_pauli().site('Z', (1, 2), (2, 2), (3, 2))),  # dog-leg row
    (RotatedToricCode(8, 8), (0, 1), (3, 0),
     RotatedToricCode(8, 8).new_pauli().site('Z', (1, 1), (2, 1), (3, 1))),  # dog-leg row
    (RotatedToricCode(8, 8), (2, 3), (1, 0),
     RotatedToricCode(8, 8).new_pauli().site('Z', (2, 1), (2, 2), (2, 3))),  # dog-leg column
    (RotatedToricCode(8, 8), (2, 3), (3, 0),
     RotatedToricCode(8, 8).new_pauli().site('Z', (3, 1), (3, 2), (3, 3))),  # dog-leg column
    (RotatedToricCode(8, 8), (0, 1), (6, 1),
     RotatedToricCode(8, 8).new_pauli().site('Z', (0, 1), (7, 1))),  # along row across boundary
    (RotatedToricCode(8, 8), (1, 0), (1, 6),
     RotatedToricCode(8, 8).new_pauli().site('Z', (1, 0), (1, 7))),  # along column across boundary
])
def test_rotated_toric_pauli_exact_path(code, a_index, b_index, expected):
    path_pauli = code.new_pauli().path(a_index, b_index)
    print()
    print('actual:')
    print(path_pauli)
    print()
    print('expected:')
    print(expected)
    assert path_pauli == expected


@pytest.mark.parametrize('pauli_1, pauli_2', [
    (RotatedToricCode(4, 4).new_pauli(), RotatedToricCode(4, 4).new_pauli()),
    (RotatedToricCode(4, 4).new_pauli().plaquette((3, 2)), RotatedToricCode(4, 4).new_pauli().plaquette((3, 2))),
    (RotatedToricCode(4, 4).new_pauli().logical_x1(), RotatedToricCode(4, 4).new_pauli().logical_x1()),
    (RotatedToricCode(4, 4).new_pauli().logical_z1(), RotatedToricCode(4, 4).new_pauli().logical_z1()),
    (RotatedToricCode(4, 4).new_pauli().logical_x2(), RotatedToricCode(4, 4).new_pauli().logical_x2()),
    (RotatedToricCode(4, 4).new_pauli().logical_z2(), RotatedToricCode(4, 4).new_pauli().logical_z2()),
])
def test_rotated_toric_pauli_eq(pauli_1, pauli_2):
    assert pauli_1 == pauli_2
    assert not pauli_1 != pauli_2


@pytest.mark.parametrize('pauli_1, pauli_2', [
    (RotatedToricCode(4, 4).new_pauli(), RotatedToricCode(4, 4).new_pauli().plaquette((3, 2))),
    (RotatedToricCode(4, 4).new_pauli().plaquette((3, 2)), RotatedToricCode(4, 4).new_pauli().plaquette((1, 1))),
    (RotatedToricCode(4, 4).new_pauli().logical_x1(), RotatedToricCode(4, 4).new_pauli().logical_z1()),
    (RotatedToricCode(4, 4).new_pauli().logical_x2(), RotatedToricCode(4, 4).new_pauli().logical_z2()),
    (RotatedToricCode(4, 4).new_pauli().logical_x1(), RotatedToricCode(4, 4).new_pauli().logical_x2()),
    (RotatedToricCode(4, 4).new_pauli().logical_z1(), RotatedToricCode(4, 4).new_pauli().logical_z2()),
    (RotatedToricCode(2, 2).new_pauli(), RotatedToricCode(4, 4).new_pauli()),
    (RotatedToricCode(2, 2).new_pauli(), None),
])
def test_rotated_toric_pauli_ne(pauli_1, pauli_2):
    assert pauli_1 != pauli_2
    assert not pauli_1 == pauli_2
