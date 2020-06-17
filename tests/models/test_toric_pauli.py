import pytest

from qecsim import paulitools as pt
from qecsim.models.toric import ToricCode


@pytest.mark.parametrize('size', [
    (3, 3),
    (5, 5),
    (3, 5),
    (2, 4),
    (7, 4),
    (2, 2),
])
def test_toric_pauli_properties(size):
    lattice = ToricCode(*size)
    pauli = lattice.new_pauli()
    assert pauli.code == lattice
    assert isinstance(repr(pauli), str)
    assert isinstance(str(pauli), str)


@pytest.mark.parametrize('toric_pauli', [
    ToricCode(5, 5).new_pauli(),
    ToricCode(5, 5).new_pauli().plaquette((0, 1, 1)).plaquette((1, 3, 3)),
    ToricCode(5, 5).new_pauli().logical_x1().plaquette((1, 3, 3)).plaquette((1, 2, 2)),
    ToricCode(5, 5).new_pauli().logical_z1().plaquette((0, 4, 4)).plaquette((1, 2, 3)),
    ToricCode(5, 5).new_pauli().logical_x2().plaquette((1, 3, 1)).plaquette((0, 4, 4)),
    ToricCode(5, 5).new_pauli().logical_z2().plaquette((0, 2, 1)).plaquette((1, 3, 2)),
])
def test_toric_pauli_new_to_bsf(toric_pauli):
    assert toric_pauli.code.new_pauli(toric_pauli.to_bsf()) == toric_pauli, (
        'Conversion to_bsf+from_bsf does not result in equality.')


@pytest.mark.parametrize('toric_pauli', [
    ToricCode(5, 5).new_pauli(),
    ToricCode(5, 5).new_pauli().plaquette((0, 1, 1)).plaquette((1, 3, 3)),
    ToricCode(5, 5).new_pauli().logical_x1().plaquette((1, 3, 3)).plaquette((1, 2, 2)),
    ToricCode(5, 5).new_pauli().logical_z1().plaquette((0, 4, 4)).plaquette((1, 2, 3)),
    ToricCode(5, 5).new_pauli().logical_x2().plaquette((1, 3, 1)).plaquette((0, 4, 4)),
    ToricCode(5, 5).new_pauli().logical_z2().plaquette((0, 2, 1)).plaquette((1, 3, 2)),
])
def test_toric_pauli_copy(toric_pauli):
    copy_pauli = toric_pauli.copy()
    assert copy_pauli == toric_pauli, 'Copy Pauli does not equal original Pauli'
    assert copy_pauli.logical_x1() != toric_pauli, 'Modified copy Pauli still equals original Pauli'


@pytest.mark.parametrize('toric_pauli, index, expected', [
    (ToricCode(5, 5).new_pauli(), (0, 0, 0), 'I'),
    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1)), (0, 1, 1), 'X'),
    (ToricCode(5, 5).new_pauli().site('Y', (1, 3, 2)), (1, 3, 2), 'Y'),
    (ToricCode(5, 5).new_pauli().site('Z', (0, 2, 4)), (0, 2, 4), 'Z'),
    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1)), (1, 1, 1), 'I'),
    (ToricCode(5, 5).new_pauli().site('Y', (1, 3, 2)), (0, 3, 2), 'I'),
    (ToricCode(5, 5).new_pauli().site('Z', (0, 2, 4)), (1, 2, 3), 'I'),
])
def test_toric_pauli_operator(toric_pauli, index, expected):
    assert toric_pauli.operator(index) == expected


@pytest.mark.parametrize('toric_pauli, op_counts, message', [
    (ToricCode(5, 5).new_pauli().site('I', (0, 1, 1)),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'site failed.'),
    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1)),
     {'I': 49, 'X': 1, 'Y': 0, 'Z': 0}, 'site failed.'),
    (ToricCode(5, 5).new_pauli().site('Y', (0, 1, 1)),
     {'I': 49, 'X': 0, 'Y': 1, 'Z': 0}, 'site failed.'),
    (ToricCode(5, 5).new_pauli().site('Z', (0, 1, 1)),
     {'I': 49, 'X': 0, 'Y': 0, 'Z': 1}, 'site failed.'),

    (ToricCode(5, 5).new_pauli().site('I', (1, 1, 1)),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'dual site failed.'),
    (ToricCode(5, 5).new_pauli().site('X', (1, 1, 1)),
     {'I': 49, 'X': 1, 'Y': 0, 'Z': 0}, 'dual site failed.'),
    (ToricCode(5, 5).new_pauli().site('Y', (1, 1, 1)),
     {'I': 49, 'X': 0, 'Y': 1, 'Z': 0}, 'dual site failed.'),
    (ToricCode(5, 5).new_pauli().site('Z', (1, 1, 1)),
     {'I': 49, 'X': 0, 'Y': 0, 'Z': 1}, 'dual site failed.'),

    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1)).site('X', (0, 1, 1)),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),
    (ToricCode(5, 5).new_pauli().site('Y', (0, 1, 1)).site('Y', (0, 1, 1)),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),
    (ToricCode(5, 5).new_pauli().site('Z', (0, 1, 1)).site('Z', (0, 1, 1)),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'site self-inverse failed.'),

    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1)).site('Z', (0, 1, 1)),
     {'I': 49, 'X': 0, 'Y': 1, 'Z': 0}, 'site X+Z=Y failed.'),
    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1)).site('Y', (0, 1, 1)),
     {'I': 49, 'X': 0, 'Y': 0, 'Z': 1}, 'site X+Y=Z failed.'),
    (ToricCode(5, 5).new_pauli().site('Y', (0, 1, 1)).site('Z', (0, 1, 1)),
     {'I': 49, 'X': 1, 'Y': 0, 'Z': 0}, 'site Y+Z=X failed.'),

    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1)).site('X', (0, 2, 1)),
     {'I': 48, 'X': 2, 'Y': 0, 'Z': 0}, 'multi-site failed.'),
    (ToricCode(5, 5).new_pauli().site('Y', (0, 1, 1)).site('Y', (0, 2, 1)),
     {'I': 48, 'X': 0, 'Y': 2, 'Z': 0}, 'multi-site failed.'),
    (ToricCode(5, 5).new_pauli().site('Z', (0, 1, 1)).site('Z', (0, 2, 1)),
     {'I': 48, 'X': 0, 'Y': 0, 'Z': 2}, 'multi-site failed.'),
])
def test_toric_pauli_edge(toric_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(toric_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('toric_pauli, op_counts, message', [
    (ToricCode(5, 5).new_pauli().site('I', (0, 1, 1), (0, 2, 1)),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'edges failed.'),
    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1), (0, 2, 1)),
     {'I': 48, 'X': 2, 'Y': 0, 'Z': 0}, 'edges failed.'),
    (ToricCode(5, 5).new_pauli().site('Y', (0, 1, 1), (0, 2, 1)),
     {'I': 48, 'X': 0, 'Y': 2, 'Z': 0}, 'edges failed.'),
    (ToricCode(5, 5).new_pauli().site('Z', (0, 1, 1), (0, 2, 1)),
     {'I': 48, 'X': 0, 'Y': 0, 'Z': 2}, 'edges failed.'),

    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1), (0, 1, 1)),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'edges self-inverse failed.'),
    (ToricCode(5, 5).new_pauli().site('Y', (0, 1, 1), (0, 1, 1)),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'edges self-inverse failed.'),
    (ToricCode(5, 5).new_pauli().site('Z', (0, 1, 1), (0, 1, 1)),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'edges self-inverse failed.'),
])
def test_toric_pauli_edges(toric_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(toric_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('size, a_index, b_index', [
    ((5, 5), (0, 1, 1), (0, 6, 1)),
    ((5, 5), (0, 2, 2), (0, -3, 7)),
    ((3, 4), (0, 2, 2), (0, 5, -2)),
    ((5, 5), (1, 1, 1), (1, 6, 1)),
    ((5, 5), (1, 2, 2), (1, -3, 7)),
    ((3, 4), (1, 2, 2), (1, 5, -2)),
    ((5, 5), (0, 1, 1), (2, 6, 1)),
    ((5, 5), (1, 2, 2), (-1, -3, 7)),
    ((3, 4), (1, 2, 2), (3, 5, -2)),
])
def test_toric_pauli_edge_index_modulo(size, a_index, b_index):
    toric_pauli_1 = ToricCode(*size).new_pauli().site(a_index)
    toric_pauli_2 = ToricCode(*size).new_pauli().site(b_index)
    assert toric_pauli_1 == toric_pauli_2, (
        'Edge indices {} and {} are not equivalent on size {}'.format(a_index, b_index, size))


@pytest.mark.parametrize('toric_pauli, op_counts, message', [
    (ToricCode(5, 5).new_pauli().plaquette((0, 1, 1)),
     {'I': 46, 'X': 0, 'Y': 0, 'Z': 4}, 'plaquette failed.'),
    (ToricCode(5, 5).new_pauli().plaquette((0, 1, 1)).plaquette((0, 1, 1)),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'plaquettes self-inverse failed.'),
    (ToricCode(5, 5).new_pauli().plaquette((0, 1, 1)).plaquette((0, 2, 1)),
     {'I': 44, 'X': 0, 'Y': 0, 'Z': 6}, 'adjacent plaquettes failed.'),
    (ToricCode(5, 5).new_pauli().plaquette((1, 1, 1)),
     {'I': 46, 'X': 4, 'Y': 0, 'Z': 0}, 'vertex failed.'),
    (ToricCode(5, 5).new_pauli().plaquette((1, 1, 1)).plaquette((1, 1, 1)),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'vertex self-inverse failed.'),
    (ToricCode(5, 5).new_pauli().plaquette((1, 1, 1)).plaquette((1, 2, 1)),
     {'I': 44, 'X': 6, 'Y': 0, 'Z': 0}, 'adjacent vertices failed.'),
    (ToricCode(5, 5).new_pauli().plaquette((0, 1, 1)).plaquette((1, 1, 1)),
     {'I': 44, 'X': 2, 'Y': 2, 'Z': 2}, 'adjacent plaquette vertex failed.'),

])
def test_toric_pauli_plaquette(toric_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(toric_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('size, a_index, b_index', [
    ((5, 5), (0, 1, 1), (0, 6, 1)),
    ((5, 5), (0, 2, 2), (0, -3, 7)),
    ((3, 4), (0, 2, 2), (0, 5, -2)),
    ((5, 5), (1, 1, 1), (1, 6, 1)),
    ((5, 5), (1, 2, 2), (1, -3, 7)),
    ((3, 4), (1, 2, 2), (1, 5, -2)),
    ((5, 5), (0, 1, 1), (2, 6, 1)),
    ((5, 5), (1, 2, 2), (-1, -3, 7)),
    ((3, 4), (1, 2, 2), (3, 5, -2)),
])
def test_toric_pauli_plaquette_index_modulo(size, a_index, b_index):
    toric_pauli_1 = ToricCode(*size).new_pauli().plaquette(a_index)
    toric_pauli_2 = ToricCode(*size).new_pauli().plaquette(b_index)
    assert toric_pauli_1 == toric_pauli_2, (
        'Plaquette indices {} and {} are not equivalent on size {}'.format(a_index, b_index, size))


@pytest.mark.parametrize('toric_pauli, op_counts, message', [
    (ToricCode(5, 5).new_pauli().logical_x1(),
     {'I': 45, 'X': 5, 'Y': 0, 'Z': 0}, 'logical_x1 failed.'),
    (ToricCode(5, 5).new_pauli().logical_x1().logical_x1(),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_x1 self-inverse failed.'),
    (ToricCode(5, 5).new_pauli().logical_x2(),
     {'I': 45, 'X': 5, 'Y': 0, 'Z': 0}, 'logical_x2 failed.'),
    (ToricCode(5, 5).new_pauli().logical_x2().logical_x2(),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_x2 self-inverse failed.'),
    (ToricCode(5, 5).new_pauli().logical_x1().logical_x2(),
     {'I': 40, 'X': 10, 'Y': 0, 'Z': 0}, 'logical_x1_x2 failed.'),

    (ToricCode(5, 5).new_pauli().logical_z1(),
     {'I': 45, 'X': 0, 'Y': 0, 'Z': 5}, 'logical_z1 failed.'),
    (ToricCode(5, 5).new_pauli().logical_z1().logical_z1(),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_z1 self-inverse failed.'),
    (ToricCode(5, 5).new_pauli().logical_z2(),
     {'I': 45, 'X': 0, 'Y': 0, 'Z': 5}, 'logical_z2 failed.'),
    (ToricCode(5, 5).new_pauli().logical_z2().logical_z2(),
     {'I': 50, 'X': 0, 'Y': 0, 'Z': 0}, 'logical_z2 self-inverse failed.'),
    (ToricCode(5, 5).new_pauli().logical_z1().logical_z2(),
     {'I': 40, 'X': 0, 'Y': 0, 'Z': 10}, 'logical_z1_z2 failed.'),

    (ToricCode(5, 5).new_pauli().logical_x1().logical_z1(),
     {'I': 41, 'X': 4, 'Y': 1, 'Z': 4}, 'logical_x1_z1 failed.'),
    (ToricCode(5, 5).new_pauli().logical_x2().logical_z2(),
     {'I': 41, 'X': 4, 'Y': 1, 'Z': 4}, 'logical_x2_z2 failed.'),
    (ToricCode(5, 5).new_pauli().logical_x1().logical_z2(),
     {'I': 40, 'X': 5, 'Y': 0, 'Z': 5}, 'logical_x1_z2 failed.'),
    (ToricCode(5, 5).new_pauli().logical_x2().logical_z1(),
     {'I': 40, 'X': 5, 'Y': 0, 'Z': 5}, 'logical_x2_z1 failed.'),
])
def test_toric_pauli_logical(toric_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(toric_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('toric_pauli, op_counts, message', [
    # primal
    (ToricCode(5, 5).new_pauli().path((0, 1, 1), (0, 3, 2)),
     {'I': 47, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg within lattice failed.'),
    (ToricCode(5, 5).new_pauli().path((0, 3, 2), (0, 1, 1)),
     {'I': 47, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg reversed within lattice failed.'),
    (ToricCode(5, 5).new_pauli().path((0, 1, 1), (0, 4, 2)),
     {'I': 47, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg across boundary failed.'),
    (ToricCode(5, 5).new_pauli().path((0, 4, 2), (0, 1, 1)),
     {'I': 47, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg reversed across boundary failed.'),
    # dual
    (ToricCode(5, 5).new_pauli().path((1, 1, 1), (1, 3, 2)),
     {'I': 47, 'X': 0, 'Y': 0, 'Z': 3}, 'dual path dog-leg within lattice failed.'),
    (ToricCode(5, 5).new_pauli().path((1, 3, 2), (1, 1, 1)),
     {'I': 47, 'X': 0, 'Y': 0, 'Z': 3}, 'dual path dog-leg reversed within lattice failed.'),
    (ToricCode(5, 5).new_pauli().path((1, 1, 1), (1, 4, 2)),
     {'I': 47, 'X': 0, 'Y': 0, 'Z': 3}, 'dual path dog-leg across boundary failed.'),
    (ToricCode(5, 5).new_pauli().path((1, 4, 2), (1, 1, 1)),
     {'I': 47, 'X': 0, 'Y': 0, 'Z': 3}, 'dual path dog-leg reversed across boundary failed.'),
    # 3 x 5
    (ToricCode(3, 5).new_pauli().path((0, 1, 1), (0, 2, 3)),
     {'I': 27, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg within lattice failed.'),
    (ToricCode(3, 5).new_pauli().path((0, 2, 3), (0, 1, 1)),
     {'I': 27, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg reversed within lattice failed.'),
    (ToricCode(3, 5).new_pauli().path((0, 0, 0), (0, 2, 4)),
     {'I': 28, 'X': 2, 'Y': 0, 'Z': 0}, 'primal path dog-leg across boundary failed.'),
    (ToricCode(3, 5).new_pauli().path((0, 2, 4), (0, 0, 0)),
     {'I': 28, 'X': 2, 'Y': 0, 'Z': 0}, 'primal path dog-leg reversed across boundary failed.'),
    # 5 x 3
    (ToricCode(5, 3).new_pauli().path((0, 1, 1), (0, 3, 2)),
     {'I': 27, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg within lattice failed.'),
    (ToricCode(5, 3).new_pauli().path((0, 3, 2), (0, 1, 1)),
     {'I': 27, 'X': 3, 'Y': 0, 'Z': 0}, 'primal path dog-leg reversed within lattice failed.'),
    (ToricCode(5, 3).new_pauli().path((0, 0, 0), (0, 4, 2)),
     {'I': 28, 'X': 2, 'Y': 0, 'Z': 0}, 'primal path dog-leg across boundary failed.'),
    (ToricCode(5, 3).new_pauli().path((0, 4, 2), (0, 0, 0)),
     {'I': 28, 'X': 2, 'Y': 0, 'Z': 0}, 'primal path dog-leg reversed across boundary failed.'),
])
def test_toric_pauli_path(toric_pauli, op_counts, message):
    pauli = pt.bsf_to_pauli(toric_pauli.to_bsf())
    for op, count in op_counts.items():
        assert pauli.count(op) == count, message


@pytest.mark.parametrize('size, a_index, b_index', [
    ((5, 5), (0, 1, 1), (1, 2, 2)),  # different lattices
])
def test_toric_pauli_invalid_path(size, a_index, b_index):
    pauli = ToricCode(*size).new_pauli()
    with pytest.raises(IndexError):
        pauli.path(a_index, b_index)


@pytest.mark.parametrize('toric_pauli_1, toric_pauli_2', [
    (ToricCode(5, 5).new_pauli(), ToricCode(5, 5).new_pauli()),
    (ToricCode(5, 5).new_pauli().plaquette((0, 1, 1)), ToricCode(5, 5).new_pauli().plaquette((0, 1, 1))),
    (ToricCode(5, 5).new_pauli().logical_x1(), ToricCode(5, 5).new_pauli().logical_x1()),
    (ToricCode(5, 5).new_pauli().logical_z1(), ToricCode(5, 5).new_pauli().logical_z1()),
    (ToricCode(5, 5).new_pauli().logical_x2(), ToricCode(5, 5).new_pauli().logical_x2()),
    (ToricCode(5, 5).new_pauli().logical_z2(), ToricCode(5, 5).new_pauli().logical_z2()),
])
def test_toric_pauli_eq(toric_pauli_1, toric_pauli_2):
    assert toric_pauli_1 == toric_pauli_2
    assert not toric_pauli_1 != toric_pauli_2


@pytest.mark.parametrize('toric_pauli_1, toric_pauli_2', [
    (ToricCode(5, 5).new_pauli(), ToricCode(5, 5).new_pauli().plaquette((0, 1, 1))),
    (ToricCode(5, 5).new_pauli().plaquette((0, 1, 1)), ToricCode(5, 5).new_pauli().plaquette((0, 2, 2))),
    (ToricCode(5, 5).new_pauli().logical_x1(), ToricCode(5, 5).new_pauli().logical_x2()),
    (ToricCode(5, 5).new_pauli().logical_z1(), ToricCode(5, 5).new_pauli().logical_z2()),
    (ToricCode(5, 5).new_pauli().logical_x2(), ToricCode(5, 5).new_pauli().logical_z2()),
    (ToricCode(5, 5).new_pauli().logical_x1(), ToricCode(5, 5).new_pauli().logical_z1()),
    (ToricCode(3, 3).new_pauli(), ToricCode(5, 5).new_pauli()),
    (ToricCode(3, 3).new_pauli(), None),
])
def test_toric_pauli_ne(toric_pauli_1, toric_pauli_2):
    assert toric_pauli_1 != toric_pauli_2
    assert not toric_pauli_1 == toric_pauli_2
