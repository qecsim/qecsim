import pytest

from qecsim import paulitools as pt
from qecsim.models.rotatedplanar import RotatedPlanarCode


@pytest.mark.parametrize('size', [
    (3, 3),
    (4, 4),
    (3, 5),
    (5, 3),
    (4, 6),
    (6, 4),
    (3, 4),
    (4, 3),
])
def test_rotated_planar_pauli_properties(size):
    code = RotatedPlanarCode(*size)
    pauli = code.new_pauli()
    assert pauli.code == code
    assert isinstance(repr(pauli), str)
    assert isinstance(str(pauli), str)


@pytest.mark.parametrize('planar_pauli', [
    RotatedPlanarCode(5, 5).new_pauli(),
    RotatedPlanarCode(5, 6).new_pauli().plaquette((1, 1)).plaquette((3, 2)),
    RotatedPlanarCode(5, 5).new_pauli().logical_x().plaquette((-1, 0)).plaquette((4, 1)),
    RotatedPlanarCode(6, 5).new_pauli().logical_z().plaquette((1, 5)).plaquette((3, -1)),
    RotatedPlanarCode(7, 5).new_pauli().logical_x().plaquette((0, 2)).plaquette((1, 1)),
    RotatedPlanarCode(5, 7).new_pauli().logical_z().plaquette((1, -1)).plaquette((2, 4)),
])
def test_rotated_planar_pauli_new_to_bsf(planar_pauli):
    assert planar_pauli.code.new_pauli(planar_pauli.to_bsf()) == planar_pauli, (
        'Conversion to_bsf+from_bsf does not result in equality.')


@pytest.mark.parametrize('planar_pauli', [
    RotatedPlanarCode(5, 5).new_pauli(),
    RotatedPlanarCode(5, 6).new_pauli().plaquette((1, 1)).plaquette((3, 2)),
    RotatedPlanarCode(5, 5).new_pauli().logical_x().plaquette((-1, 0)).plaquette((4, 1)),
    RotatedPlanarCode(6, 5).new_pauli().logical_z().plaquette((1, 5)).plaquette((3, -1)),
    RotatedPlanarCode(7, 5).new_pauli().logical_x().plaquette((0, 2)).plaquette((1, 1)),
    RotatedPlanarCode(5, 7).new_pauli().logical_z().plaquette((1, -1)).plaquette((2, 4)),
])
def test_rotated_planar_pauli_copy(planar_pauli):
    copy_pauli = planar_pauli.copy()
    assert copy_pauli == planar_pauli, 'Copy Pauli does not equal original Pauli'
    assert copy_pauli.logical_x() != planar_pauli, 'Modified copy Pauli still equals original Pauli'


@pytest.mark.parametrize('planar_pauli, index, expected', [
    (RotatedPlanarCode(5, 5).new_pauli(), (0, 0), 'I'),
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (1, 1)), (1, 1), 'X'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (3, 3)), (3, 3), 'Y'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (2, 4)), (2, 4), 'Z'),
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (1, 1)), (3, 3), 'I'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (3, 3)), (2, 4), 'I'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (2, 4)), (1, 1), 'I'),
])
def test_rotated_planar_pauli_operator(planar_pauli, index, expected):
    assert planar_pauli.operator(index) == expected


@pytest.mark.parametrize('size, index', [
    ((4, 5), (0, -1)),  # out of bounds
    ((4, 5), (-1, 0)),  # out of bounds
    ((4, 5), (-1, 3)),  # out of bounds
    ((4, 5), (0, 4)),  # out of bounds
    ((4, 5), (4, 4)),  # out of bounds
    ((4, 5), (5, 3)),  # out of bounds
    ((4, 5), (5, 0)),  # out of bounds
    ((4, 5), (4, -1)),  # out of bounds
])
def test_rotated_planar_pauli_operator_invalid_index(size, index):
    pauli = RotatedPlanarCode(*size).new_pauli()
    with pytest.raises(IndexError):
        pauli.operator(index)


@pytest.mark.parametrize('planar_pauli, op_counts, message', [
    (RotatedPlanarCode(5, 5).new_pauli().site('I', (2, 2)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'site failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (2, 2)),
     {'I': 24, 'X': 1, 'Y': 0, 'Z': 0}, 'site failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)),
     {'I': 24, 'X': 0, 'Y': 1, 'Z': 0}, 'site failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (2, 2)),
     {'I': 24, 'X': 0, 'Y': 0, 'Z': 1}, 'site failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (2, 2)).site('X', (2, 2)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)).site('Y', (2, 2)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (2, 2)).site('Z', (2, 2)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (2, 2)).site('Z', (2, 2)),
     {'I': 24, 'X': 0, 'Y': 1, 'Z': 0}, 'site X+Z=Y failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (2, 2)).site('Y', (2, 2)),
     {'I': 24, 'X': 0, 'Y': 0, 'Z': 1}, 'site X+Y=Z failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)).site('Z', (2, 2)),
     {'I': 24, 'X': 1, 'Y': 0, 'Z': 0}, 'site Y+Z=X failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (2, 2)).site('X', (3, 1)),
     {'I': 23, 'X': 2, 'Y': 0, 'Z': 0}, 'multi-site failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)).site('Y', (3, 1)),
     {'I': 23, 'X': 0, 'Y': 2, 'Z': 0}, 'multi-site failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (2, 2)).site('Z', (3, 1)),
     {'I': 23, 'X': 0, 'Y': 0, 'Z': 2}, 'multi-site failed.'),
    (RotatedPlanarCode(4, 5).new_pauli().site('X', (0, -1)),
     {'I': 20, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedPlanarCode(4, 5).new_pauli().site('X', (-1, 0)),
     {'I': 20, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedPlanarCode(4, 5).new_pauli().site('X', (-1, 3)),
     {'I': 20, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedPlanarCode(4, 5).new_pauli().site('X', (0, 4)),
     {'I': 20, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedPlanarCode(4, 5).new_pauli().site('X', (4, 4)),
     {'I': 20, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedPlanarCode(4, 5).new_pauli().site('X', (5, 3)),
     {'I': 20, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedPlanarCode(4, 5).new_pauli().site('X', (5, 0)),
     {'I': 20, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (RotatedPlanarCode(4, 5).new_pauli().site('X', (4, -1)),
     {'I': 20, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
])
def test_rotated_planar_pauli_site(planar_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(planar_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('planar_pauli, op_counts, message', [
    (RotatedPlanarCode(5, 5).new_pauli().site('I', (2, 2), (3, 1)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'sites failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (2, 2), (3, 1)),
     {'I': 23, 'X': 2, 'Y': 0, 'Z': 0}, 'sites failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2), (3, 1)),
     {'I': 23, 'X': 0, 'Y': 2, 'Z': 0}, 'sites failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (2, 2), (3, 1)),
     {'I': 23, 'X': 0, 'Y': 0, 'Z': 2}, 'sites failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (2, 2), (2, 2)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'sites self-inverse failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2), (2, 2)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'sites self-inverse failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (2, 2), (2, 2)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'sites self-inverse failed.'),
])
def test_rotated_planar_pauli_sites(planar_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(planar_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('planar_pauli, op_counts, message', [
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((1, 1)),
     {'I': 21, 'X': 0, 'Y': 0, 'Z': 4}, 'z-plaquette failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((1, 1)).plaquette((1, 1)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'z-plaquette self-inverse failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((1, 1)).plaquette((2, 2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 6}, 'adjacent z-plaquettes failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((1, -1)),
     {'I': 23, 'X': 0, 'Y': 0, 'Z': 2}, 'z-boundary plaquette failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((2, 4)),
     {'I': 23, 'X': 0, 'Y': 0, 'Z': 2}, 'z-boundary plaquette failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((0, -1)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'outside lattice z-plaquette failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((3, 4)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'outside lattice z-plaquette failed.'),

    (RotatedPlanarCode(5, 5).new_pauli().plaquette((1, 2)),
     {'I': 21, 'X': 4, 'Y': 0, 'Z': 0}, 'x-plaquette failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((1, 2)).plaquette((1, 2)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'x-plaquette self-inverse failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((1, 2)).plaquette((2, 1)),
     {'I': 19, 'X': 6, 'Y': 0, 'Z': 0}, 'adjacent x-plaquettes failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((-1, 0)),
     {'I': 23, 'X': 2, 'Y': 0, 'Z': 0}, 'x-boundary plaquette failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((4, 1)),
     {'I': 23, 'X': 2, 'Y': 0, 'Z': 0}, 'x-boundary plaquette failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((-1, 1)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'outside lattice x-plaquette failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((4, 0)),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'outside lattice x-plaquette failed.'),

    (RotatedPlanarCode(5, 5).new_pauli().plaquette((1, 1)).plaquette((1, 2)),
     {'I': 19, 'X': 2, 'Y': 2, 'Z': 2}, 'adjacent z- and x-plaquettes failed.'),
])
def test_rotated_planar_pauli_plaquette(planar_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(planar_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('planar_pauli, op_counts, message', [
    (RotatedPlanarCode(5, 5).new_pauli().logical_x(),
     {'I': 20, 'X': 5, 'Y': 0, 'Z': 0}, 'logical_x failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().logical_x().logical_x(),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_x self-inverse failed.'),

    (RotatedPlanarCode(5, 5).new_pauli().logical_z(),
     {'I': 20, 'X': 0, 'Y': 0, 'Z': 5}, 'logical_z failed.'),
    (RotatedPlanarCode(5, 5).new_pauli().logical_z().logical_z(),
     {'I': 25, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_z self-inverse failed.'),

    (RotatedPlanarCode(5, 5).new_pauli().logical_x().logical_z(),
     {'I': 16, 'X': 4, 'Y': 1, 'Z': 4}, 'logical_x1_z1 failed.'),
])
def test_rotated_planar_pauli_logical(planar_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(planar_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('planar_pauli_1, planar_pauli_2', [
    (RotatedPlanarCode(5, 5).new_pauli(), RotatedPlanarCode(5, 5).new_pauli()),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((3, 2)), RotatedPlanarCode(5, 5).new_pauli().plaquette((3, 2))),
    (RotatedPlanarCode(5, 5).new_pauli().logical_x(), RotatedPlanarCode(5, 5).new_pauli().logical_x()),
    (RotatedPlanarCode(5, 5).new_pauli().logical_z(), RotatedPlanarCode(5, 5).new_pauli().logical_z()),
])
def test_rotated_planar_pauli_eq(planar_pauli_1, planar_pauli_2):
    assert planar_pauli_1 == planar_pauli_2
    assert not planar_pauli_1 != planar_pauli_2


@pytest.mark.parametrize('planar_pauli_1, planar_pauli_2', [
    (RotatedPlanarCode(5, 5).new_pauli(), RotatedPlanarCode(5, 5).new_pauli().plaquette((3, 2))),
    (RotatedPlanarCode(5, 5).new_pauli().plaquette((3, 2)), RotatedPlanarCode(5, 5).new_pauli().plaquette((1, 1))),
    (RotatedPlanarCode(5, 5).new_pauli().logical_x(), RotatedPlanarCode(5, 5).new_pauli().logical_z()),
    (RotatedPlanarCode(3, 3).new_pauli(), RotatedPlanarCode(5, 5).new_pauli()),
    (RotatedPlanarCode(3, 3).new_pauli(), None),
])
def test_rotated_planar_pauli_ne(planar_pauli_1, planar_pauli_2):
    assert planar_pauli_1 != planar_pauli_2
    assert not planar_pauli_1 == planar_pauli_2
