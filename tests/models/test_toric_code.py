import pytest

from qecsim import paulitools as pt
from qecsim.models.toric import ToricCode


# < Code tests >


@pytest.mark.parametrize('size', [
    (3, 3),
    (5, 5),
    (3, 5),
    (2, 4),
    (7, 4),
    (2, 2),
])
def test_toric_code_properties(size):
    code = ToricCode(*size)
    assert isinstance(code.label, str)
    assert isinstance(repr(code), str)


@pytest.mark.parametrize('rows, columns', [
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (None, 4),
    (4, None),
    ('asdf', 4),
    (4, 'asdf'),
    (4.1, 4),
    (4, 4.1),
])
def test_toric_code_new_invalid_parameters(rows, columns):
    with pytest.raises((ValueError, TypeError), match=r"^ToricCode") as exc_info:
        ToricCode(rows, columns)
    print(exc_info)


@pytest.mark.parametrize('size, expected', [
    ((3, 3), (18, 2, 3)),
    ((5, 5), (50, 2, 5)),
    ((3, 5), (30, 2, 3)),
    ((2, 4), (16, 2, 2)),
    ((7, 4), (56, 2, 4)),
    ((2, 2), (8, 2, 2)),
])
def test_toric_code_n_k_d(size, expected):
    code = ToricCode(*size)
    assert code.n_k_d == expected


@pytest.mark.parametrize('size, expected', [
    ((3, 3), 18),
    ((5, 5), 50),
    ((3, 5), 30),
    ((2, 4), 16),
    ((7, 4), 56),
    ((2, 2), 8),
])
def test_toric_code_stabilizers(size, expected):
    assert len(ToricCode(*size).stabilizers) == expected


def test_toric_code_logical_xs():
    assert len(ToricCode(5, 5).logical_xs) == 2


def test_toric_code_logical_zs():
    assert len(ToricCode(5, 5).logical_zs) == 2


def test_toric_code_logicals():
    assert len(ToricCode(5, 5).logicals) == 4


@pytest.mark.parametrize('size', [
    (3, 3),
    (5, 5),
    (3, 5),
    (2, 4),
    (7, 4),
    (2, 2),
])
def test_toric_code_validate(size):
    code = ToricCode(*size)
    code.validate()  # no error raised


@pytest.mark.parametrize('rows, columns', [
    (3, 3),
    (5, 5),
    (3, 5),
    (2, 4),
    (7, 4),
    (2, 2),
])
def test_toric_code_new(rows, columns):
    code = ToricCode(rows, columns)
    code.validate()  # no error raised


# </ Code tests >
# < Lattice tests >


@pytest.mark.parametrize('size', [
    (3, 3),
    (5, 5),
    (3, 5),
    (2, 4),
    (7, 4),
    (2, 2),
])
def test_toric_lattice_properties(size):
    lattice = ToricCode(*size)
    assert lattice.size == size
    assert lattice.shape == (2, *size)
    assert isinstance(repr(lattice), str)
    assert isinstance(str(lattice), str)


@pytest.mark.parametrize('size, a_index, b_index, expected', [
    # 3 x 3
    ((3, 3), (0, 1, 0), (0, 1, 2), (0, -1)),  # across boundary horizontally
    ((3, 3), (0, 1, 2), (0, 1, 0), (0, 1)),  # ditto reversed
    # 5 x 5
    ((5, 5), (0, 1, 1), (0, 1, 3), (0, 2)),  # within lattice horizontally
    ((5, 5), (0, 1, 3), (0, 1, 1), (0, -2)),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 1, 4), (0, -2)),  # across boundary horizontally
    ((5, 5), (0, 1, 4), (0, 1, 1), (0, 2)),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 3, 1), (2, 0)),  # within lattice vertically
    ((5, 5), (0, 3, 1), (0, 1, 1), (-2, 0)),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 4, 1), (-2, 0)),  # across boundary vertically
    ((5, 5), (0, 4, 1), (0, 1, 1), (2, 0)),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 3, 3), (2, 2)),  # within lattice diagonally
    ((5, 5), (0, 3, 3), (0, 1, 1), (-2, -2)),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 4, 4), (-2, -2)),  # across boundary diagonally
    ((5, 5), (0, 4, 4), (0, 1, 1), (2, 2)),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 3, 2), (2, 1)),  # within lattice dog-leg
    ((5, 5), (0, 3, 2), (0, 1, 1), (-2, -1)),  # ditto reversed
    ((5, 5), (0, 1, 1), (0, 4, 2), (-2, 1)),  # across boundary dog-leg
    ((5, 5), (0, 4, 2), (0, 1, 1), (2, -1)),  # ditto reversed
    # dual
    ((5, 5), (1, 1, 1), (1, 3, 2), (2, 1)),  # within lattice dog-leg (dual)
    ((5, 5), (1, 3, 2), (1, 1, 1), (-2, -1)),  # ditto reversed
    ((5, 5), (1, 1, 1), (1, 4, 2), (-2, 1)),  # across boundary dog-leg (dual)
    ((5, 5), (1, 4, 2), (1, 1, 1), (2, -1)),  # ditto reversed
    # 3 x 5
    ((3, 5), (0, 1, 0), (0, 1, 2), (0, 2)),  # within lattice horizontally
    ((3, 5), (0, 1, 2), (0, 1, 0), (0, -2)),  # ditto reversed
    ((3, 5), (0, 1, 0), (0, 1, 3), (0, -2)),  # across boundary horizontally
    ((3, 5), (0, 1, 3), (0, 1, 0), (0, 2)),  # ditto reversed
    ((3, 5), (0, 1, 1), (0, 2, 1), (1, 0)),  # within lattice vertically
    ((3, 5), (0, 2, 1), (0, 1, 1), (-1, 0)),  # ditto reversed
    ((3, 5), (0, 0, 1), (0, 2, 1), (-1, 0)),  # across boundary vertically
    ((3, 5), (0, 2, 1), (0, 0, 1), (1, 0)),  # ditto reversed
    ((3, 5), (0, 1, 1), (0, 2, 3), (1, 2)),  # within lattice dog-leg
    ((3, 5), (0, 2, 3), (0, 1, 1), (-1, -2)),  # ditto reversed
    ((3, 5), (0, 0, 0), (0, 2, 4), (-1, -1)),  # across boundary dog-leg
    ((3, 5), (0, 2, 4), (0, 0, 0), (1, 1)),  # ditto reversed
    # 5 x 3
    ((5, 3), (0, 0, 1), (0, 2, 1), (2, 0)),  # within lattice vertically
    ((5, 3), (0, 2, 1), (0, 0, 1), (-2, 0)),  # ditto reversed
    ((5, 3), (0, 0, 1), (0, 3, 1), (-2, 0)),  # across boundary vertically
    ((5, 3), (0, 3, 1), (0, 0, 1), (2, 0)),  # ditto reversed
    ((5, 3), (0, 1, 1), (0, 1, 2), (0, 1)),  # within lattice horizontally
    ((5, 3), (0, 1, 2), (0, 1, 1), (0, -1)),  # ditto reversed
    ((5, 3), (0, 1, 0), (0, 1, 2), (0, -1)),  # across boundary horizontally
    ((5, 3), (0, 1, 2), (0, 1, 0), (0, 1)),  # ditto reversed
    ((5, 3), (0, 1, 1), (0, 3, 2), (2, 1)),  # within lattice dog-leg
    ((5, 3), (0, 3, 2), (0, 1, 1), (-2, -1)),  # ditto reversed
    ((5, 3), (0, 0, 0), (0, 4, 2), (-1, -1)),  # across boundary dog-leg
    ((5, 3), (0, 4, 2), (0, 0, 0), (1, 1)),  # ditto reversed
    # index modulo shape
    ((3, 3), (2, 7, -3), (0, -2, 5), (0, -1)),  # across boundary horizontally
    ((3, 3), (3, -5, 8), (-1, 4, -6), (0, 1)),  # ditto reversed
])
def test_toric_lattice_translation(size, a_index, b_index, expected):
    assert ToricCode(*size).translation(a_index, b_index) == expected


@pytest.mark.parametrize('size, a_index, b_index', [
    ((5, 5), (0, 1, 1), (1, 2, 2)),  # different lattices
])
def test_toric_lattice_invalid_translation(size, a_index, b_index):
    code = ToricCode(*size)
    with pytest.raises(IndexError):
        code.translation(a_index, b_index)


@pytest.mark.parametrize('error, expected', [
    (ToricCode(3, 3).new_pauli().site('X', (1, 0, 0), (0, 1, 2)).site('Z', (0, 2, 2), (1, 1, 2), (1, 0, 2)),
     [(0, 0, 0), (0, 1, 2), (1, 1, 0), (1, 2, 2)]),
])
def test_toric_lattice_syndrome_to_plaquette_indices(error, expected):
    code = error.code
    syndrome = pt.bsp(error.to_bsf(), code.stabilizers.T)
    assert set(code.syndrome_to_plaquette_indices(syndrome)) == set(expected)


@pytest.mark.parametrize('pauli', [
    ToricCode(3, 3).new_pauli().site('Y', (0, 0, 0), (1, 2, 2)),
    ToricCode(3, 5).new_pauli().logical_x1(),
    ToricCode(5, 3).new_pauli().logical_z1(),
    ToricCode(3, 5).new_pauli().logical_x2(),
    ToricCode(5, 3).new_pauli().logical_z2(),
])
def test_planar_lattice_ascii_art(pauli):
    code = pauli.code
    syndrome = pt.bsp(pauli.to_bsf(), code.stabilizers.T)
    assert isinstance(code.ascii_art(), str)
    assert isinstance(code.ascii_art(syndrome=syndrome), str)
    assert isinstance(code.ascii_art(pauli=pauli), str)
    assert isinstance(code.ascii_art(syndrome=syndrome, pauli=pauli), str)

# </ Lattice tests >
