import pytest

from qecsim import paulitools as pt
from qecsim.models.color import Color666Code


@pytest.mark.parametrize('size', [
    3, 5, 7, 9
])
def test_color666_pauli_properties(size):
    lattice = Color666Code(size)
    pauli = lattice.new_pauli()
    assert pauli.code == lattice
    assert isinstance(repr(pauli), str)
    assert isinstance(str(pauli), str)


@pytest.mark.parametrize('pauli', [
    Color666Code(5).new_pauli(),
    Color666Code(5).new_pauli().plaquette('X', (2, 0)).plaquette('Z', (5, 3)),
    Color666Code(5).new_pauli().logical_x().plaquette('X', (6, 5)).plaquette('Y', (4, 1)),
    Color666Code(5).new_pauli().logical_z().plaquette('Z', (1, 1)).plaquette('Z', (6, 2)),
    Color666Code(5).new_pauli().logical_x().plaquette('X', (5, 3)).plaquette('X', (4, 4)),
    Color666Code(5).new_pauli().logical_z().plaquette('Z', (4, 1)).plaquette('Z', (3, 2)),
])
def test_color666_pauli_new_to_bsf(pauli):
    assert pauli.code.new_pauli(pauli.to_bsf()) == pauli, (
        'Conversion to_bsf+from_bsf does not result in equality.')


@pytest.mark.parametrize('pauli', [
    Color666Code(5).new_pauli(),
    Color666Code(5).new_pauli().plaquette('X', (2, 0)).plaquette('Z', (5, 3)),
    Color666Code(5).new_pauli().logical_x().plaquette('X', (6, 5)).plaquette('Y', (4, 1)),
    Color666Code(5).new_pauli().logical_z().plaquette('Z', (1, 1)).plaquette('Z', (6, 2)),
    Color666Code(5).new_pauli().logical_x().plaquette('X', (5, 3)).plaquette('X', (4, 4)),
    Color666Code(5).new_pauli().logical_z().plaquette('Z', (4, 1)).plaquette('Z', (3, 2)),
])
def test_color666_pauli_copy(pauli):
    copy_pauli = pauli.copy()
    assert copy_pauli == pauli, 'Copy Pauli does not equal original Pauli'
    assert copy_pauli.logical_x() != pauli, 'Modified copy Pauli still equals original Pauli'


@pytest.mark.parametrize('pauli, index, expected', [
    (Color666Code(5).new_pauli(), (0, 0), 'I'),
    (Color666Code(5).new_pauli().site('X', (2, 2)), (2, 2), 'X'),
    (Color666Code(5).new_pauli().site('Y', (5, 5)), (5, 5), 'Y'),
    (Color666Code(5).new_pauli().site('Z', (4, 3)), (4, 3), 'Z'),
    (Color666Code(5).new_pauli().site('X', (2, 2)), (1, 0), 'I'),
    (Color666Code(5).new_pauli().site('Y', (5, 5)), (6, 4), 'I'),
    (Color666Code(5).new_pauli().site('Z', (4, 3)), (3, 1), 'I'),
])
def test_color666_pauli_operator(pauli, index, expected):
    assert pauli.operator(index) == expected


@pytest.mark.parametrize('size, index', [
    (5, (2, 0)),  # not a site index
    (5, (3, 2)),  # not a site index
    (5, (-1, -1)),  # out of bounds
    (5, (2, -1)),  # out of bounds
    (5, (7, 0)),  # out of bounds
    (5, (6, 7)),  # out of bounds
    (5, (0, -1)),  # out of bounds and not a site index
])
def test_color666_pauli_operator_invalid_index(size, index):
    pauli = Color666Code(size).new_pauli()
    with pytest.raises(IndexError):
        pauli.operator(index)


@pytest.mark.parametrize('pauli, op_counts, message', [
    (Color666Code(5).new_pauli().site('I', (2, 2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'site failed.'),
    (Color666Code(5).new_pauli().site('X', (2, 2)),
     {'I': 18, 'X': 1, 'Y': 0, 'Z': 0}, 'site failed.'),
    (Color666Code(5).new_pauli().site('Y', (2, 2)),
     {'I': 18, 'X': 0, 'Y': 1, 'Z': 0}, 'site failed.'),
    (Color666Code(5).new_pauli().site('Z', (2, 2)),
     {'I': 18, 'X': 0, 'Y': 0, 'Z': 1}, 'site failed.'),

    (Color666Code(5).new_pauli().site('X', (2, 2)).site('X', (2, 2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),
    (Color666Code(5).new_pauli().site('Y', (2, 2)).site('Y', (2, 2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),
    (Color666Code(5).new_pauli().site('Z', (2, 2)).site('Z', (2, 2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),

    (Color666Code(5).new_pauli().site('X', (2, 2)).site('Z', (2, 2)),
     {'I': 18, 'X': 0, 'Y': 1, 'Z': 0}, 'site X+Z=Y failed.'),
    (Color666Code(5).new_pauli().site('X', (2, 2)).site('Y', (2, 2)),
     {'I': 18, 'X': 0, 'Y': 0, 'Z': 1}, 'site X+Y=Z failed.'),
    (Color666Code(5).new_pauli().site('Y', (2, 2)).site('Z', (2, 2)),
     {'I': 18, 'X': 1, 'Y': 0, 'Z': 0}, 'site Y+Z=X failed.'),

    (Color666Code(5).new_pauli().site('X', (2, 2)).site('X', (4, 2)),
     {'I': 17, 'X': 2, 'Y': 0, 'Z': 0}, 'multi-site failed.'),
    (Color666Code(5).new_pauli().site('Y', (2, 2)).site('Y', (4, 2)),
     {'I': 17, 'X': 0, 'Y': 2, 'Z': 0}, 'multi-site failed.'),
    (Color666Code(5).new_pauli().site('Z', (2, 2)).site('Z', (4, 2)),
     {'I': 17, 'X': 0, 'Y': 0, 'Z': 2}, 'multi-site failed.'),

    (Color666Code(5).new_pauli().site('X', (0, -2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (Color666Code(5).new_pauli().site('X', (0, 1)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (Color666Code(5).new_pauli().site('X', (7, 0)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
    (Color666Code(5).new_pauli().site('X', (3, 4)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'site outside lattice failed.'),
])
def test_color666_pauli_site(pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('pauli, op_counts, message', [
    (Color666Code(5).new_pauli().site('I', (2, 2), (4, 2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'sites failed.'),
    (Color666Code(5).new_pauli().site('X', (2, 2), (4, 2)),
     {'I': 17, 'X': 2, 'Y': 0, 'Z': 0}, 'sites failed.'),
    (Color666Code(5).new_pauli().site('Y', (2, 2), (4, 2)),
     {'I': 17, 'X': 0, 'Y': 2, 'Z': 0}, 'sites failed.'),
    (Color666Code(5).new_pauli().site('Z', (2, 2), (4, 2)),
     {'I': 17, 'X': 0, 'Y': 0, 'Z': 2}, 'sites failed.'),

    (Color666Code(5).new_pauli().site('X', (2, 2), (2, 2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'sites self-inverse failed.'),
    (Color666Code(5).new_pauli().site('Y', (2, 2), (2, 2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'sites self-inverse failed.'),
    (Color666Code(5).new_pauli().site('Z', (2, 2), (2, 2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'sites self-inverse failed.'),
])
def test_color666_pauli_sites(pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('size, operator, index', [
    (5, 'Z', (1, 1)),  # not a site index
    (5, 'X', (5, 3)),  # not a site index
])
def test_color666_pauli_site_invalid_index(size, operator, index):
    pauli = Color666Code(size).new_pauli()
    with pytest.raises(IndexError):
        pauli.site(operator, index)


@pytest.mark.parametrize('pauli, op_counts, message', [
    (Color666Code(5).new_pauli().plaquette('X', (3, 2)),
     {'I': 13, 'X': 6, 'Y': 0, 'Z': 0}, 'X plaquette failed.'),
    (Color666Code(5).new_pauli().plaquette('X', (3, 2)).plaquette('X', (3, 2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'X plaquette self-inverse failed.'),
    (Color666Code(5).new_pauli().plaquette('X', (3, 2)).plaquette('X', (5, 3)),
     {'I': 11, 'X': 8, 'Y': 0, 'Z': 0}, 'X adjacent plaquettes failed.'),
    (Color666Code(5).new_pauli().plaquette('X', (2, 0)),
     {'I': 15, 'X': 4, 'Y': 0, 'Z': 0}, 'X boundary plaquette failed.'),
    (Color666Code(5).new_pauli().plaquette('X', (4, -2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'X outside lattice plaquette failed.'),

    (Color666Code(5).new_pauli().plaquette('Z', (3, 2)),
     {'I': 13, 'X': 0, 'Y': 0, 'Z': 6}, 'Z plaquette failed.'),
    (Color666Code(5).new_pauli().plaquette('Z', (3, 2)).plaquette('Z', (3, 2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'Z plaquette self-inverse failed.'),
    (Color666Code(5).new_pauli().plaquette('Z', (3, 2)).plaquette('Z', (5, 3)),
     {'I': 11, 'X': 0, 'Y': 0, 'Z': 8}, 'Z adjacent plaquettes failed.'),
    (Color666Code(5).new_pauli().plaquette('Z', (2, 0)),
     {'I': 15, 'X': 0, 'Y': 0, 'Z': 4}, 'Z boundary plaquette failed.'),
    (Color666Code(5).new_pauli().plaquette('Z', (4, -2)),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'Z outside lattice plaquette failed.'),

    (Color666Code(5).new_pauli().plaquette('X', (3, 2)).plaquette('Z', (3, 2)),
     {'I': 13, 'X': 0, 'Y': 6, 'Z': 0}, 'X+Z plaquette failed.'),
    (Color666Code(5).new_pauli().plaquette('X', (3, 2)).plaquette('Z', (5, 3)),
     {'I': 9, 'X': 4, 'Y': 2, 'Z': 4}, 'X+Z adjacent plaquettes failed.'),
])
def test_color666_pauli_plaquette(pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('size, operator, index', [
    (5, 'X', (0, 0)),  # not a plaquette index
    (5, 'Z', (5, 2)),  # not a plaquette index
])
def test_color666_pauli_invalid_plaquette(size, operator, index):
    pauli = Color666Code(size).new_pauli()
    with pytest.raises(IndexError):
        pauli.plaquette(operator, index)


@pytest.mark.parametrize('pauli, op_counts, message', [
    (Color666Code(5).new_pauli().logical_x(),
     {'I': 14, 'X': 5, 'Y': 0, 'Z': 0}, 'logical_x failed.'),
    (Color666Code(5).new_pauli().logical_x().logical_x(),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_x self-inverse failed.'),

    (Color666Code(5).new_pauli().logical_z(),
     {'I': 14, 'X': 0, 'Y': 0, 'Z': 5}, 'logical_z failed.'),
    (Color666Code(5).new_pauli().logical_z().logical_z(),
     {'I': 19, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_z self-inverse failed.'),

    (Color666Code(5).new_pauli().logical_x().logical_z(),
     {'I': 14, 'X': 0, 'Y': 5, 'Z': 0}, 'logical_x_z failed.'),
])
def test_color666_pauli_logical(pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('pauli_1, pauli_2', [
    (Color666Code(5).new_pauli(), Color666Code(5).new_pauli()),
    (Color666Code(5).new_pauli().plaquette('X', (3, 2)), Color666Code(5).new_pauli().plaquette('X', (3, 2))),
    (Color666Code(5).new_pauli().logical_x(), Color666Code(5).new_pauli().logical_x()),
    (Color666Code(5).new_pauli().logical_z(), Color666Code(5).new_pauli().logical_z()),
])
def test_color666_pauli_eq(pauli_1, pauli_2):
    assert pauli_1 == pauli_2
    assert not pauli_1 != pauli_2


@pytest.mark.parametrize('pauli_1, pauli_2', [
    (Color666Code(5).new_pauli(), Color666Code(5).new_pauli().plaquette('X', (3, 2))),
    (Color666Code(5).new_pauli().plaquette('X', (3, 2)), Color666Code(5).new_pauli().plaquette('Z', (3, 2))),
    (Color666Code(5).new_pauli().plaquette('X', (3, 2)), Color666Code(5).new_pauli().plaquette('X', (5, 3))),
    (Color666Code(5).new_pauli().logical_x(), Color666Code(5).new_pauli().logical_z()),
    (Color666Code(3).new_pauli(), Color666Code(5).new_pauli()),
    (Color666Code(3).new_pauli(), None),
])
def test_color666_pauli_ne(pauli_1, pauli_2):
    assert pauli_1 != pauli_2
    assert not pauli_1 == pauli_2
