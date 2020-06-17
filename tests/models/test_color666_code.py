import pytest

from qecsim import paulitools as pt
from qecsim.models.color import Color666Code


# < Code tests >


@pytest.mark.parametrize('size', [
    3, 5, 7, 9,
])
def test_color666_code_properties(size):
    code = Color666Code(size)
    assert isinstance(code.label, str)
    assert isinstance(repr(code), str)


@pytest.mark.parametrize('size', [
    0, 1, 2, 4, None, 'asdf', 5.1,
])
def test_color666_code_new_invalid_parameters(size):
    with pytest.raises((ValueError, TypeError), match=r"^Color666Code") as exc_info:
        Color666Code(size)
    print(exc_info)


@pytest.mark.parametrize('size, expected', [
    (3, (7, 1, 3)),
    (5, (19, 1, 5)),
    (7, (37, 1, 7)),
    (9, (61, 1, 9)),
])
def test_color666_code_n_k_d(size, expected):
    code = Color666Code(size)
    assert code.n_k_d == expected


@pytest.mark.parametrize('size, expected', [
    (3, 6),
    (5, 18),
    (7, 36),
])
def test_color666_code_stabilizers(size, expected):
    assert len(Color666Code(size).stabilizers) == expected


def test_color666_code_logical_xs():
    assert len(Color666Code(5).logical_xs) == 1


def test_color666_code_logical_zs():
    assert len(Color666Code(5).logical_zs) == 1


def test_color666_code_logicals():
    assert len(Color666Code(5).logicals) == 2


@pytest.mark.parametrize('size', [
    3, 5, 7, 9,
])
def test_color666_code_validate(size):
    code = Color666Code(size)
    code.validate()  # no error raised


# </ Code tests >
# < Lattice tests >


@pytest.mark.parametrize('size', [
    3, 5, 7, 9,
])
def test_color666_lattice_properties(size):
    lattice = Color666Code(size)
    assert lattice.size == size
    assert isinstance(repr(lattice), str)


@pytest.mark.parametrize('index', [
    # in-bounds
    (1, 1),
    (2, 0),
    (3, 2),
    (4, 1),
    (4, 4),
    # out-bounds
    (0, -1),
    (0, 2),
    (1, -2),
    (1, 4),
    (2, -3),
])
def test_color666_lattice_is_plaquette(index):
    assert Color666Code.is_plaquette(index)


@pytest.mark.parametrize('index', [
    # in-bounds
    (0, 0),
    (1, 0),
    (2, 1),
    (2, 2),
    (3, 0),
    # out-bounds
    (0, -2),
    (0, 1),
    (1, -1),
    (1, 2),
    (2, -1),
])
def test_color666_lattice_is_site(index):
    assert Color666Code.is_site(index)


@pytest.mark.parametrize('index', [
    (0, 0),
    (3, 0),
    (3, 3),
    (6, 0),
    (6, 6),
])
def test_color666_lattice_is_in_bounds(index):
    lattice = Color666Code(5)
    assert lattice.is_in_bounds(index)


@pytest.mark.parametrize('index', [
    (-1, 0),
    (3, -1),
    (3, 4),
    (6, -1),
    (6, 7),
    (7, 0),
])
def test_color666_lattice_not_is_in_bounds(index):
    lattice = Color666Code(5)
    assert not lattice.is_in_bounds(index)


@pytest.mark.parametrize('error, expected', [
    (Color666Code(3).new_pauli().site('X', (0, 0)), (set(), {(1, 1)})),
    (Color666Code(3).new_pauli().site('X', (1, 0)), (set(), {(1, 1), (2, 0)})),
    (Color666Code(3).new_pauli().site('X', (2, 1)), (set(), {(1, 1), (2, 0), (3, 2)})),
    (Color666Code(3).new_pauli().site('X', (2, 2)), (set(), {(1, 1), (3, 2)})),
    (Color666Code(3).new_pauli().site('X', (3, 0)), (set(), {(2, 0)})),
    (Color666Code(3).new_pauli().site('X', (3, 1)), (set(), {(2, 0), (3, 2)})),
    (Color666Code(3).new_pauli().site('X', (3, 3)), (set(), {(3, 2)})),

    (Color666Code(3).new_pauli().site('Z', (0, 0)), ({(1, 1)}, set())),
    (Color666Code(3).new_pauli().site('Z', (1, 0)), ({(1, 1), (2, 0)}, set())),
    (Color666Code(3).new_pauli().site('Z', (2, 1)), ({(1, 1), (2, 0), (3, 2)}, set())),
    (Color666Code(3).new_pauli().site('Z', (2, 2)), ({(1, 1), (3, 2)}, set())),
    (Color666Code(3).new_pauli().site('Z', (3, 0)), ({(2, 0)}, set())),
    (Color666Code(3).new_pauli().site('Z', (3, 1)), ({(2, 0), (3, 2)}, set())),
    (Color666Code(3).new_pauli().site('Z', (3, 3)), ({(3, 2)}, set())),

    (Color666Code(3).new_pauli().site('Y', (0, 0)), ({(1, 1)}, {(1, 1)})),
    (Color666Code(3).new_pauli().site('Y', (1, 0)), ({(1, 1), (2, 0)}, {(1, 1), (2, 0)})),
    (Color666Code(3).new_pauli().site('Y', (2, 1)), ({(1, 1), (2, 0), (3, 2)}, {(1, 1), (2, 0), (3, 2)})),
    (Color666Code(3).new_pauli().site('Y', (2, 2)), ({(1, 1), (3, 2)}, {(1, 1), (3, 2)})),
    (Color666Code(3).new_pauli().site('Y', (3, 0)), ({(2, 0)}, {(2, 0)})),
    (Color666Code(3).new_pauli().site('Y', (3, 1)), ({(2, 0), (3, 2)}, {(2, 0), (3, 2)})),
    (Color666Code(3).new_pauli().site('Y', (3, 3)), ({(3, 2)}, {(3, 2)})),
])
def test_color666_lattice_syndrome_to_plaquette_indices(error, expected):
    code = error.code
    syndrome = pt.bsp(error.to_bsf(), code.stabilizers.T)
    assert code.syndrome_to_plaquette_indices(syndrome) == expected


@pytest.mark.parametrize('size, index, expected', [
    # 3
    (3, (2, 0), (2, 3)),  # red to diagonal
    (3, (3, 2), (3, -1)),  # green to left
    (3, (1, 1), (4, 1)),  # blue to lower
    # 5
    (5, (2, 0), (2, 3)),  # red to diagonal
    (5, (5, 3), (5, 6)),  # red to diagonal
    (5, (3, 2), (3, -1)),  # green to left
    (5, (6, 5), (6, -1)),  # green to left
    (5, (1, 1), (7, 1)),  # blue to lower
    (5, (4, 4), (7, 4)),  # blue to lower
])
def test_color666_lattice_virtual_plaquette_index(size, index, expected):
    assert Color666Code(size).virtual_plaquette_index(index) == expected


@pytest.mark.parametrize('size, index', [
    (5, (0, 0)),  # invalid plaquette index
    (5, (5, 2)),  # invalid plaquette index
])
def test_color666_lattice_invalid_virtual_plaquette_index(size, index):
    code = Color666Code(size)
    with pytest.raises(IndexError):
        code.virtual_plaquette_index(index)


@pytest.mark.parametrize('pauli', [
    Color666Code(3).new_pauli().site('Y', (0, 0), (2, 2)),
    Color666Code(5).new_pauli().logical_x().site('X', (4, 3)).site('Z', (3, 3), (6, 6)),
    Color666Code(7).new_pauli().logical_z(),
])
def test_color666_ascii_art(pauli):
    code = pauli.code
    syndrome = pt.bsp(pauli.to_bsf(), code.stabilizers.T)
    assert isinstance(code.ascii_art(), str)
    assert isinstance(code.ascii_art(syndrome=syndrome), str)
    assert isinstance(code.ascii_art(pauli=pauli), str)
    assert isinstance(code.ascii_art(syndrome=syndrome, pauli=pauli), str)

# </ Lattice tests >
