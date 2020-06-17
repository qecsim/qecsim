import pytest

from qecsim import paulitools as pt
from qecsim.models.planar import PlanarCode


@pytest.mark.parametrize('size', [
    (3, 3),
    (5, 5),
    (3, 5),
    (2, 4),
    (7, 4),
    (2, 2),
])
def test_planar_pauli_properties(size):
    lattice = PlanarCode(*size)
    pauli = lattice.new_pauli()
    assert pauli.code == lattice
    assert isinstance(repr(pauli), str)
    assert isinstance(str(pauli), str)


@pytest.mark.parametrize('planar_pauli', [
    PlanarCode(5, 5).new_pauli(),
    PlanarCode(5, 5).new_pauli().plaquette((3, 2)).plaquette((6, 7)),
    PlanarCode(5, 5).new_pauli().logical_x().plaquette((6, 7)).plaquette((4, 5)),
    PlanarCode(5, 5).new_pauli().logical_z().plaquette((9, 8)).plaquette((4, 7)),
    PlanarCode(5, 5).new_pauli().logical_x().plaquette((6, 3)).plaquette((9, 8)),
    PlanarCode(5, 5).new_pauli().logical_z().plaquette((5, 2)).plaquette((6, 5)),
])
def test_planar_pauli_new_to_bsf(planar_pauli):
    assert planar_pauli.code.new_pauli(planar_pauli.to_bsf()) == planar_pauli, (
        'Conversion to_bsf+from_bsf does not result in equality.')


@pytest.mark.parametrize('planar_pauli', [
    PlanarCode(5, 5).new_pauli(),
    PlanarCode(5, 5).new_pauli().plaquette((3, 2)).plaquette((6, 7)),
    PlanarCode(5, 5).new_pauli().logical_x().plaquette((6, 7)).plaquette((4, 5)),
    PlanarCode(5, 5).new_pauli().logical_z().plaquette((9, 8)).plaquette((4, 7)),
    PlanarCode(5, 5).new_pauli().logical_x().plaquette((6, 3)).plaquette((9, 8)),
    PlanarCode(5, 5).new_pauli().logical_z().plaquette((5, 2)).plaquette((6, 5)),
])
def test_planar_pauli_copy(planar_pauli):
    copy_pauli = planar_pauli.copy()
    assert copy_pauli == planar_pauli, 'Copy Pauli does not equal original Pauli'
    assert copy_pauli.logical_x() != planar_pauli, 'Modified copy Pauli still equals original Pauli'


@pytest.mark.parametrize('planar_pauli, index, expected', [
    (PlanarCode(5, 5).new_pauli(), (0, 0), 'I'),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2)), (2, 2), 'X'),
    (PlanarCode(5, 5).new_pauli().site('Y', (5, 5)), (5, 5), 'Y'),
    (PlanarCode(5, 5).new_pauli().site('Z', (4, 8)), (4, 8), 'Z'),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2)), (1, 3), 'I'),
    (PlanarCode(5, 5).new_pauli().site('Y', (5, 5)), (6, 4), 'I'),
    (PlanarCode(5, 5).new_pauli().site('Z', (4, 8)), (3, 7), 'I'),
])
def test_planar_pauli_operator(planar_pauli, index, expected):
    assert planar_pauli.operator(index) == expected


@pytest.mark.parametrize('size, index', [
    ((5, 5), (1, 0)),  # not a site index
    ((5, 5), (0, 1)),  # not a site index
    ((5, 5), (-2, 0)),  # out of bounds
    ((5, 5), (10, 0)),  # out of bounds
    ((5, 5), (0, -2)),  # out of bounds
    ((5, 5), (0, 10)),  # out of bounds
    ((5, 5), (-1, 1)),  # out of bounds
    ((5, 5), (9, 1)),  # out of bounds
    ((5, 5), (-1, -1)),  # out of bounds
    ((5, 5), (-1, 9)),  # out of bounds
])
def test_planar_pauli_operator_invalid_index(size, index):
    pauli = PlanarCode(*size).new_pauli()
    with pytest.raises(IndexError):
        pauli.operator(index)


@pytest.mark.parametrize('planar_pauli, op_counts, message', [
    (PlanarCode(5, 5).new_pauli().site('I', (2, 2)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'site failed.'),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2)),
     {'I': 40, 'X': 1, 'Y': 0, 'Z': 0}, 'site failed.'),
    (PlanarCode(5, 5).new_pauli().site('Y', (2, 2)),
     {'I': 40, 'X': 0, 'Y': 1, 'Z': 0}, 'site failed.'),
    (PlanarCode(5, 5).new_pauli().site('Z', (2, 2)),
     {'I': 40, 'X': 0, 'Y': 0, 'Z': 1}, 'site failed.'),

    (PlanarCode(5, 5).new_pauli().site('I', (1, 3)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'dual site failed.'),
    (PlanarCode(5, 5).new_pauli().site('X', (1, 3)),
     {'I': 40, 'X': 1, 'Y': 0, 'Z': 0}, 'dual site failed.'),
    (PlanarCode(5, 5).new_pauli().site('Y', (1, 3)),
     {'I': 40, 'X': 0, 'Y': 1, 'Z': 0}, 'dual site failed.'),
    (PlanarCode(5, 5).new_pauli().site('Z', (1, 3)),
     {'I': 40, 'X': 0, 'Y': 0, 'Z': 1}, 'dual site failed.'),

    (PlanarCode(5, 5).new_pauli().site('X', (2, 2)).site('X', (2, 2)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),
    (PlanarCode(5, 5).new_pauli().site('Y', (2, 2)).site('Y', (2, 2)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),
    (PlanarCode(5, 5).new_pauli().site('Z', (2, 2)).site('Z', (2, 2)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),

    (PlanarCode(5, 5).new_pauli().site('X', (2, 2)).site('Z', (2, 2)),
     {'I': 40, 'X': 0, 'Y': 1, 'Z': 0}, 'site X+Z=Y failed.'),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2)).site('Y', (2, 2)),
     {'I': 40, 'X': 0, 'Y': 0, 'Z': 1}, 'site X+Y=Z failed.'),
    (PlanarCode(5, 5).new_pauli().site('Y', (2, 2)).site('Z', (2, 2)),
     {'I': 40, 'X': 1, 'Y': 0, 'Z': 0}, 'site Y+Z=X failed.'),

    (PlanarCode(5, 5).new_pauli().site('X', (2, 2)).site('X', (4, 2)),
     {'I': 39, 'X': 2, 'Y': 0, 'Z': 0}, 'multi-site failed.'),
    (PlanarCode(5, 5).new_pauli().site('Y', (2, 2)).site('Y', (4, 2)),
     {'I': 39, 'X': 0, 'Y': 2, 'Z': 0}, 'multi-site failed.'),
    (PlanarCode(5, 5).new_pauli().site('Z', (2, 2)).site('Z', (4, 2)),
     {'I': 39, 'X': 0, 'Y': 0, 'Z': 2}, 'multi-site failed.'),

    (PlanarCode(3, 5).new_pauli().site('X', (0, -2)),
     {'I': 23, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (PlanarCode(3, 5).new_pauli().site('X', (0, 10)),
     {'I': 23, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (PlanarCode(3, 5).new_pauli().site('X', (-2, 0)),
     {'I': 23, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (PlanarCode(3, 5).new_pauli().site('X', (6, 0)),
     {'I': 23, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),

    (PlanarCode(3, 5).new_pauli().site('X', (1, -1)),
     {'I': 23, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (PlanarCode(3, 5).new_pauli().site('X', (1, 9)),
     {'I': 23, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (PlanarCode(3, 5).new_pauli().site('X', (-1, 1)),
     {'I': 23, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (PlanarCode(3, 5).new_pauli().site('X', (9, 1)),
     {'I': 23, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
])
def test_planar_pauli_site(planar_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(planar_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('planar_pauli, op_counts, message', [
    (PlanarCode(5, 5).new_pauli().site('I', (2, 2), (4, 2)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'sites failed.'),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2), (4, 2)),
     {'I': 39, 'X': 2, 'Y': 0, 'Z': 0}, 'sites failed.'),
    (PlanarCode(5, 5).new_pauli().site('Y', (2, 2), (4, 2)),
     {'I': 39, 'X': 0, 'Y': 2, 'Z': 0}, 'sites failed.'),
    (PlanarCode(5, 5).new_pauli().site('Z', (2, 2), (4, 2)),
     {'I': 39, 'X': 0, 'Y': 0, 'Z': 2}, 'sites failed.'),

    (PlanarCode(5, 5).new_pauli().site('X', (2, 2), (2, 2)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'sites self-inverse failed.'),
    (PlanarCode(5, 5).new_pauli().site('Y', (2, 2), (2, 2)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'sites self-inverse failed.'),
    (PlanarCode(5, 5).new_pauli().site('Z', (2, 2), (2, 2)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'sites self-inverse failed.'),
])
def test_planar_pauli_sites(planar_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(planar_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('size, operator, index', [
    ((5, 5), 'Z', (1, 0)),  # not a site index
    ((5, 5), 'Z', (0, 1)),  # not a site index
])
def test_planar_pauli_site_invalid_index(size, operator, index):
    pauli = PlanarCode(*size).new_pauli()
    with pytest.raises(IndexError):
        pauli.site(operator, index)


@pytest.mark.parametrize('planar_pauli, op_counts, message', [
    (PlanarCode(5, 5).new_pauli().plaquette((3, 2)),
     {'I': 37, 'X': 0, 'Y': 0, 'Z': 4}, 'plaquette failed.'),
    (PlanarCode(5, 5).new_pauli().plaquette((3, 2)).plaquette((3, 2)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'plaquette self-inverse failed.'),
    (PlanarCode(5, 5).new_pauli().plaquette((3, 2)).plaquette((5, 2)),
     {'I': 35, 'X': 0, 'Y': 0, 'Z': 6}, 'adjacent plaquettes failed.'),
    (PlanarCode(5, 5).new_pauli().plaquette((3, 0)),
     {'I': 38, 'X': 0, 'Y': 0, 'Z': 3}, '3-site boundary plaquette failed.'),
    (PlanarCode(5, 5).new_pauli().plaquette((-1, 2)),
     {'I': 40, 'X': 0, 'Y': 0, 'Z': 1}, '1-site boundary plaquette failed.'),
    (PlanarCode(5, 5).new_pauli().plaquette((-3, 2)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'outside lattice plaquette failed.'),

    (PlanarCode(5, 5).new_pauli().plaquette((2, 3)),
     {'I': 37, 'X': 4, 'Y': 0, 'Z': 0}, 'vertex failed.'),
    (PlanarCode(5, 5).new_pauli().plaquette((2, 3)).plaquette((2, 3)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'vertex self-inverse failed.'),
    (PlanarCode(5, 5).new_pauli().plaquette((2, 3)).plaquette((4, 3)),
     {'I': 35, 'X': 6, 'Y': 0, 'Z': 0}, 'adjacent vertices failed.'),

    (PlanarCode(5, 5).new_pauli().plaquette((0, 3)),
     {'I': 38, 'X': 3, 'Y': 0, 'Z': 0}, '3-site boundary vertex failed.'),
    (PlanarCode(5, 5).new_pauli().plaquette((2, -1)),
     {'I': 40, 'X': 1, 'Y': 0, 'Z': 0}, '1-site boundary vertex failed.'),
    (PlanarCode(5, 5).new_pauli().plaquette((2, -3)),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'outside lattice vertex failed.'),

    (PlanarCode(5, 5).new_pauli().plaquette((3, 2)).plaquette((2, 3)),
     {'I': 35, 'X': 2, 'Y': 2, 'Z': 2}, 'adjacent plaquette vertex failed.'),
])
def test_planar_pauli_plaquette(planar_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(planar_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('size, index', [
    ((5, 5), (3, 1)),  # not a plaquette index
    ((5, 5), (1, 3)),  # not a plaquette index
])
def test_planar_pauli_invalid_plaquette(size, index):
    pauli = PlanarCode(*size).new_pauli()
    with pytest.raises(IndexError):
        pauli.plaquette(index)


@pytest.mark.parametrize('planar_pauli, op_counts, message', [
    (PlanarCode(5, 5).new_pauli().logical_x(),
     {'I': 36, 'X': 5, 'Y': 0, 'Z': 0}, 'logical_x failed.'),
    (PlanarCode(5, 5).new_pauli().logical_x().logical_x(),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_x self-inverse failed.'),

    (PlanarCode(5, 5).new_pauli().logical_z(),
     {'I': 36, 'X': 0, 'Y': 0, 'Z': 5}, 'logical_z failed.'),
    (PlanarCode(5, 5).new_pauli().logical_z().logical_z(),
     {'I': 41, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_z self-inverse failed.'),

    (PlanarCode(5, 5).new_pauli().logical_x().logical_z(),
     {'I': 32, 'X': 4, 'Y': 1, 'Z': 4}, 'logical_x1_z1 failed.'),
])
def test_planar_pauli_logical(planar_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(planar_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('planar_pauli, op_counts, message', [
    # primal
    (PlanarCode(5, 5).new_pauli().path((3, 2), (7, 4)),
     {'I': 38, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg within lattice failed.'),
    (PlanarCode(5, 5).new_pauli().path((7, 4), (3, 2)),
     {'I': 38, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg reversed within lattice failed.'),
    (PlanarCode(5, 5).new_pauli().path((3, 2), (-1, 2)),
     {'I': 39, 'X': 2, 'Y': 0, 'Z': 0}, 'primal path to boundary failed.'),
    (PlanarCode(5, 5).new_pauli().path((-1, 2), (3, 2)),
     {'I': 39, 'X': 2, 'Y': 0, 'Z': 0}, 'primal path to boundary reversed failed.'),
    (PlanarCode(5, 5).new_pauli().path((3, 2), (9, 4)),
     {'I': 38, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg across boundary failed.'),
    (PlanarCode(5, 5).new_pauli().path((9, 4), (3, 2)),
     {'I': 37, 'X': 4, 'Y': 0, 'Z': 0}, 'primal path dog-leg reversed across boundary failed.'),
    # dual
    (PlanarCode(5, 5).new_pauli().path((2, 3), (6, 5)),
     {'I': 38, 'X': 0, 'Y': 0, 'Z': 3}, 'dual path dog-leg within lattice failed.'),
    (PlanarCode(5, 5).new_pauli().path((6, 5), (2, 3)),
     {'I': 38, 'X': 0, 'Y': 0, 'Z': 3}, 'dual path dog-leg reversed within lattice failed.'),
    (PlanarCode(5, 5).new_pauli().path((2, 3), (2, -1)),
     {'I': 39, 'X': 0, 'Y': 0, 'Z': 2}, 'primal path to boundary failed.'),
    (PlanarCode(5, 5).new_pauli().path((2, -1), (2, 3)),
     {'I': 39, 'X': 0, 'Y': 0, 'Z': 2}, 'primal path to boundary reversed failed.'),
    (PlanarCode(5, 5).new_pauli().path((2, 3), (4, 9)),
     {'I': 37, 'X': 0, 'Y': 0, 'Z': 4}, 'dual path dog-leg across boundary failed.'),
    (PlanarCode(5, 5).new_pauli().path((4, 9), (2, 3)),
     {'I': 38, 'X': 0, 'Y': 0, 'Z': 3}, 'dual path dog-leg reversed across boundary failed.'),
    # 3 x 5
    (PlanarCode(3, 5).new_pauli().path((1, 2), (3, 6)),
     {'I': 20, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg within lattice failed.'),
    (PlanarCode(3, 5).new_pauli().path((3, 6), (1, 2)),
     {'I': 20, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg reversed within lattice failed.'),
    (PlanarCode(3, 5).new_pauli().path((1, 0), (5, 8)),
     {'I': 21, 'X': 2, 'Y': 0, 'Z': 0}, 'primal path dog-leg across boundary failed.'),
    (PlanarCode(3, 5).new_pauli().path((5, 8), (1, 0)),
     {'I': 17, 'X': 6, 'Y': 0, 'Z': 0}, 'primal path dog-leg reversed across boundary failed.'),
    # 5 x 3
    (PlanarCode(5, 3).new_pauli().path((3, 2), (7, 4)),
     {'I': 20, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg within lattice failed.'),
    (PlanarCode(5, 3).new_pauli().path((7, 4), (3, 2)),
     {'I': 20, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg reversed within lattice failed.'),
    (PlanarCode(5, 3).new_pauli().path((1, 0), (9, 4)),
     {'I': 19, 'X': 4, 'Y': 0, 'Z': 0}, 'primal path dog-leg across boundary failed.'),
    (PlanarCode(5, 3).new_pauli().path((9, 4), (1, 0)),
     {'I': 17, 'X': 6, 'Y': 0, 'Z': 0}, 'primal path dog-leg reversed across boundary failed.'),
])
def test_planar_pauli_path(planar_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(planar_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('size, a_index, b_index', [
    ((5, 5), (0, 0), (4, 5)),  # invalid plaquette index
    ((5, 5), (3, 5), (4, 4)),  # invalid plaquette index
    ((5, 5), (3, 2), (4, 5)),  # different lattices
])
def test_planar_pauli_invalid_path(size, a_index, b_index):
    pauli = PlanarCode(*size).new_pauli()
    with pytest.raises(IndexError):
        pauli.path(a_index, b_index)


@pytest.mark.parametrize('planar_pauli_1, planar_pauli_2', [
    (PlanarCode(5, 5).new_pauli(), PlanarCode(5, 5).new_pauli()),
    (PlanarCode(5, 5).new_pauli().plaquette((3, 2)), PlanarCode(5, 5).new_pauli().plaquette((3, 2))),
    (PlanarCode(5, 5).new_pauli().logical_x(), PlanarCode(5, 5).new_pauli().logical_x()),
    (PlanarCode(5, 5).new_pauli().logical_z(), PlanarCode(5, 5).new_pauli().logical_z()),
])
def test_planar_pauli_eq(planar_pauli_1, planar_pauli_2):
    assert planar_pauli_1 == planar_pauli_2
    assert not planar_pauli_1 != planar_pauli_2


@pytest.mark.parametrize('planar_pauli_1, planar_pauli_2', [
    (PlanarCode(5, 5).new_pauli(), PlanarCode(5, 5).new_pauli().plaquette((3, 2))),
    (PlanarCode(5, 5).new_pauli().plaquette((3, 2)), PlanarCode(5, 5).new_pauli().plaquette((5, 4))),
    (PlanarCode(5, 5).new_pauli().logical_x(), PlanarCode(5, 5).new_pauli().logical_z()),
    (PlanarCode(3, 3).new_pauli(), PlanarCode(5, 5).new_pauli()),
    (PlanarCode(3, 3).new_pauli(), None),
])
def test_planar_pauli_ne(planar_pauli_1, planar_pauli_2):
    assert planar_pauli_1 != planar_pauli_2
    assert not planar_pauli_1 == planar_pauli_2
