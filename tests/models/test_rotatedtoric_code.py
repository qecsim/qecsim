import pytest

from qecsim import paulitools as pt
from qecsim.models.rotatedtoric import RotatedToricCode


# < Code tests >

@pytest.mark.parametrize('size', [
    (2, 2),
    (4, 4),
    (2, 4),
    (4, 2),
    (6, 6),
    (4, 6),
    (6, 4),
])
def test_rotated_toric_code_properties(size):
    code = RotatedToricCode(*size)
    assert isinstance(code.label, str)
    assert isinstance(repr(code), str)


@pytest.mark.parametrize('rows, columns', [
    (0, 0),
    (1, 1),
    (2, 1),
    (2, 1),
    (3, 3),
    (3, 4),
    (4, 3),
    (None, 4),
    (4, None),
    ('asdf', 4),
    (4, 'asdf'),
    (4.1, 4),
    (4, 4.1),
])
def test_rotated_toric_code_new_invalid_parameters(rows, columns):
    with pytest.raises((ValueError, TypeError), match=r"^RotatedToricCode") as exc_info:
        RotatedToricCode(rows, columns)
    print(exc_info)


@pytest.mark.parametrize('size, expected', [
    ((2, 2), (4, 2, 2)),
    ((4, 4), (16, 2, 4)),
    ((2, 4), (8, 2, 2)),
    ((4, 2), (8, 2, 2)),
    ((6, 6), (36, 2, 6)),
    ((4, 6), (24, 2, 4)),
    ((6, 4), (24, 2, 4)),
])
def test_rotated_toric_code_n_k_d(size, expected):
    code = RotatedToricCode(*size)
    assert code.n_k_d == expected


@pytest.mark.parametrize('size, expected', [
    ((2, 2), 4),
    ((4, 4), 16),
    ((2, 4), 8),
    ((4, 2), 8),
    ((6, 6), 36),
    ((4, 6), 24),
    ((6, 4), 24),
])
def test_rotated_toric_code_stabilizers(size, expected):
    assert len(RotatedToricCode(*size).stabilizers) == expected


def test_rotated_toric_code_logical_xs():
    assert len(RotatedToricCode(4, 4).logical_xs) == 2


def test_rotated_toric_code_logical_zs():
    assert len(RotatedToricCode(4, 4).logical_zs) == 2


def test_rotated_toric_code_logicals():
    assert len(RotatedToricCode(4, 4).logicals) == 4


@pytest.mark.parametrize('size', [
    (2, 2),
    (4, 4),
    (2, 4),
    (4, 2),
    (6, 6),
    (4, 6),
    (6, 4),
])
def test_rotated_toric_code_validate(size):
    code = RotatedToricCode(*size)
    code.validate()  # no error raised


# </ Code tests >
# < Lattice tests >


@pytest.mark.parametrize('size', [
    (2, 2),
    (4, 4),
    (2, 4),
    (4, 2),
    (6, 6),
    (4, 6),
    (6, 4),
])
def test_rotated_toric_lattice_size(size):
    lattice = RotatedToricCode(*size)
    assert lattice.size == size


@pytest.mark.parametrize('index, expected', [
    ((0, 0), True),  # in-bounds z plaquette
    ((1, -1), True),  # boundary z plaquette
    ((4, 0), True),  # out-bounds z plaquette
    ((1, 0), False),  # in-bounds x plaquette
    ((-1, 0), False),  # boundary x plaquette
    ((1, 2), False),  # out-bounds x plaquette
])
def test_rotated_toric_lattice_is_z_plaquette(index, expected):
    lattice = RotatedToricCode(2, 4)
    assert lattice.is_z_plaquette(index) == expected


@pytest.mark.parametrize('index, expected', [
    ((1, 0), True),  # in-bounds x plaquette
    ((3, 0), True),  # boundary x plaquette
    ((3, 2), True),  # out-bounds x plaquette
    ((0, 0), False),  # in-bounds z plaquette
    ((1, 1), False),  # boundary z plaquette
    ((3, 1), False),  # out-bounds z plaquette
])
def test_rotated_toric_lattice_is_x_plaquette(index, expected):
    lattice = RotatedToricCode(2, 4)
    assert lattice.is_x_plaquette(index) == expected


@pytest.mark.parametrize('lattice, expected', [
    # RotatedToricCode(row, cols), (max_site_x, max_site_y)
    (RotatedToricCode(2, 2), (1, 1)),
    (RotatedToricCode(4, 4), (3, 3)),
    (RotatedToricCode(2, 4), (3, 1)),
    (RotatedToricCode(4, 2), (1, 3)),
    (RotatedToricCode(6, 6), (5, 5)),
    (RotatedToricCode(4, 6), (5, 3)),
    (RotatedToricCode(6, 4), (3, 5)),
])
def test_rotated_toric_lattice_bounds(lattice, expected):
    assert lattice.bounds == expected


@pytest.mark.parametrize('index, expected', [
    ((0, 1), True),  # in-bounds left
    ((5, 1), True),  # in-bounds right
    ((2, 3), True),  # in-bounds top
    ((2, 0), True),  # in-bounds bottom
    ((-1, 1), False),  # out-bounds left
    ((6, 1), False),  # out-bounds right
    ((2, 4), False),  # out-bounds top
    ((2, -1), False),  # out-bounds bottom
])
def test_rotated_toric_lattice_is_in_bounds(index, expected):
    lattice = RotatedToricCode(4, 6)
    assert lattice.is_in_bounds(index) == expected


@pytest.mark.parametrize('error, expected', [
    # X
    (RotatedToricCode(4, 4).new_pauli().site('X', (2, 2)), [(1, 1), (2, 2)]),  # center
    (RotatedToricCode(4, 4).new_pauli().site('X', (0, 0)), [(0, 0), (3, 3)]),  # SW
    (RotatedToricCode(4, 4).new_pauli().site('X', (0, 2)), [(0, 2), (3, 1)]),  # W
    (RotatedToricCode(4, 4).new_pauli().site('X', (0, 3)), [(0, 2), (3, 3)]),  # NW
    (RotatedToricCode(4, 4).new_pauli().site('X', (2, 3)), [(2, 2), (1, 3)]),  # N
    (RotatedToricCode(4, 4).new_pauli().site('X', (3, 3)), [(2, 2), (3, 3)]),  # NE
    (RotatedToricCode(4, 4).new_pauli().site('X', (3, 2)), [(3, 1), (2, 2)]),  # E
    (RotatedToricCode(4, 4).new_pauli().site('X', (3, 0)), [(2, 0), (3, 3)]),  # SE
    (RotatedToricCode(4, 4).new_pauli().site('X', (2, 0)), [(2, 0), (1, 3)]),  # S
    # Z
    (RotatedToricCode(4, 4).new_pauli().site('Z', (2, 2)), [(2, 1), (1, 2)]),  # center
    (RotatedToricCode(4, 4).new_pauli().site('Z', (0, 0)), [(3, 0), (0, 3)]),  # SW
    (RotatedToricCode(4, 4).new_pauli().site('Z', (0, 2)), [(0, 1), (3, 2)]),  # W
    (RotatedToricCode(4, 4).new_pauli().site('Z', (0, 3)), [(3, 2), (0, 3)]),  # NW
    (RotatedToricCode(4, 4).new_pauli().site('Z', (2, 3)), [(1, 2), (2, 3)]),  # N
    (RotatedToricCode(4, 4).new_pauli().site('Z', (3, 3)), [(3, 2), (2, 3)]),  # NE
    (RotatedToricCode(4, 4).new_pauli().site('Z', (3, 2)), [(2, 1), (3, 2)]),  # E
    (RotatedToricCode(4, 4).new_pauli().site('Z', (3, 0)), [(3, 0), (2, 3)]),  # SE
    (RotatedToricCode(4, 4).new_pauli().site('Z', (2, 0)), [(1, 0), (2, 3)]),  # S
    # Y
    (RotatedToricCode(4, 4).new_pauli().site('Y', (2, 2)), [(1, 1), (2, 2), (2, 1), (1, 2)]),  # center
    (RotatedToricCode(4, 4).new_pauli().site('Y', (0, 0)), [(0, 0), (3, 3), (3, 0), (0, 3)]),  # SW
    (RotatedToricCode(4, 4).new_pauli().site('Y', (0, 2)), [(3, 1), (0, 2), (0, 1), (3, 2)]),  # W
    (RotatedToricCode(4, 4).new_pauli().site('Y', (0, 3)), [(0, 2), (3, 3), (3, 2), (0, 3)]),  # NW
    (RotatedToricCode(4, 4).new_pauli().site('Y', (2, 3)), [(2, 2), (1, 3), (1, 2), (2, 3)]),  # N
    (RotatedToricCode(4, 4).new_pauli().site('Y', (3, 3)), [(2, 2), (3, 3), (3, 2), (2, 3)]),  # NE
    (RotatedToricCode(4, 4).new_pauli().site('Y', (3, 2)), [(3, 1), (2, 2), (2, 1), (3, 2)]),  # E
    (RotatedToricCode(4, 4).new_pauli().site('Y', (3, 0)), [(2, 0), (3, 3), (3, 0), (2, 3)]),  # SE
    (RotatedToricCode(4, 4).new_pauli().site('Y', (2, 0)), [(2, 0), (1, 3), (1, 0), (2, 3)]),  # S
])
def test_rotated_toric_lattice_syndrome_to_plaquette_indices(error, expected):
    code = error.code
    syndrome = pt.bsp(error.to_bsf(), code.stabilizers.T)
    assert set(code.syndrome_to_plaquette_indices(syndrome)) == set(expected)


@pytest.mark.parametrize('size, a_index, b_index, expected', [
    # 6 x 6
    ((6, 6), (1, 1), (1, 1), (0, 0)),  # same index
    ((6, 6), (1, 1), (3, 1), (2, 0)),  # within lattice horizontally
    ((6, 6), (3, 1), (1, 1), (-2, 0)),  # ditto reversed
    ((6, 6), (0, 1), (4, 1), (-2, 0)),  # across boundary horizontally
    ((6, 6), (4, 1), (0, 1), (2, 0)),  # ditto reversed
    ((6, 6), (1, 1), (1, 3), (0, 2)),  # within lattice vertically
    ((6, 6), (1, 3), (1, 1), (0, -2)),  # ditto reversed
    ((6, 6), (1, 0), (1, 4), (0, -2)),  # across boundary vertically
    ((6, 6), (1, 4), (1, 0), (0, 2)),  # ditto reversed
    ((6, 6), (1, 1), (2, 2), (1, 1)),  # within lattice dog-leg
    ((6, 6), (2, 2), (1, 1), (-1, -1)),  # ditto reversed
    ((6, 6), (2, 0), (3, 5), (1, -1)),  # across boundary dog-leg
    ((6, 6), (3, 5), (2, 0), (-1, 1)),  # ditto reversed
    # 4 x 6
    ((4, 6), (1, 1), (2, 2), (1, 1)),  # within lattice dog-leg
    ((4, 6), (2, 2), (1, 1), (-1, -1)),  # ditto reversed
    ((4, 6), (2, 0), (3, 3), (1, -1)),  # across boundary dog-leg
    ((4, 6), (3, 3), (2, 0), (-1, 1)),  # ditto reversed
    # index modulo shape
    ((6, 6), (0, 1), (-2, 1), (-2, 0)),  # across boundary horizontally
    ((6, 6), (-2, 1), (0, 1), (2, 0)),  # ditto reversed
])
def test_rotated_toric_lattice_translation(size, a_index, b_index, expected):
    assert RotatedToricCode(*size).translation(a_index, b_index) == expected


# @pytest.mark.parametrize('size, a_index, b_index', [
#     ((5, 5), (0, 1, 1), (1, 2, 2)),  # different lattices
# ])
# def test_toric_lattice_invalid_translation(size, a_index, b_index):
#     code = ToricCode(*size)
#     with pytest.raises(IndexError):
#         code.translation(a_index, b_index)

@pytest.mark.parametrize('pauli', [
    RotatedToricCode(2, 2).new_pauli().site('Y', (0, 0), (1, 1)),
    RotatedToricCode(4, 4).new_pauli().site('Y', (0, 0), (1, 1)),
    RotatedToricCode(4, 4).new_pauli().logical_x1(),
    RotatedToricCode(4, 4).new_pauli().logical_z1(),
    RotatedToricCode(4, 4).new_pauli().logical_x2(),
    RotatedToricCode(4, 4).new_pauli().logical_z2(),
    RotatedToricCode(4, 4).new_pauli().logical_x1().logical_z1(),
    RotatedToricCode(4, 4).new_pauli().logical_x2().logical_z2(),
    RotatedToricCode(4, 4).new_pauli().logical_x1().logical_z1().logical_x2().logical_z2(),
    RotatedToricCode(2, 4).new_pauli().logical_x1(),
    RotatedToricCode(4, 2).new_pauli().logical_z1(),
    RotatedToricCode(6, 6).new_pauli().logical_x1(),
    RotatedToricCode(4, 6).new_pauli().logical_z1(),
    RotatedToricCode(6, 4).new_pauli().logical_x1(),
])
def test_rotated_toric_lattice_ascii_art(pauli):
    code = pauli.code
    syndrome = pt.bsp(pauli.to_bsf(), code.stabilizers.T)
    assert isinstance(code.ascii_art(), str)
    assert isinstance(code.ascii_art(syndrome=syndrome), str)
    assert isinstance(code.ascii_art(pauli=pauli), str)
    ascii_art = code.ascii_art(syndrome=syndrome, pauli=pauli)
    print()
    print()
    print(ascii_art)
    assert isinstance(ascii_art, str)


@pytest.mark.parametrize('code, plaquette_labels, site_labels', [
    (RotatedToricCode(4, 4), {(3, 3): '0', (0, 0): '1', (1, 2): '2'}, {(0, 0): 'a', (2, 1): 'b'}),
    (RotatedToricCode(4, 4), {(0, 3): '0', (1, 1): '1', (3, 2): '2'}, {(0, 3): 'a', (3, 0): 'b'}),
])
def test_rotated_toric_lattice_ascii_art_labels(code, plaquette_labels, site_labels):
    ascii_art = code.ascii_art(plaquette_labels=plaquette_labels, site_labels=site_labels)
    print()
    print(ascii_art)
    assert isinstance(ascii_art, str)

# </ Lattice tests >
