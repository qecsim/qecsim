import pytest

from qecsim import paulitools as pt
from qecsim.models.rotatedplanar import RotatedPlanarCode


# < Code tests >

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
def test_rotated_planar_code_properties(size):
    code = RotatedPlanarCode(*size)
    assert isinstance(code.label, str)
    assert isinstance(repr(code), str)


@pytest.mark.parametrize('rows, columns', [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (None, 4),
    (4, None),
    ('asdf', 4),
    (4, 'asdf'),
    (4.1, 4),
    (4, 4.1),
])
def test_rotated_planar_code_new_invalid_parameters(rows, columns):
    with pytest.raises((ValueError, TypeError), match=r"^RotatedPlanarCode") as exc_info:
        RotatedPlanarCode(rows, columns)
    print(exc_info)


@pytest.mark.parametrize('size, expected', [
    ((3, 3), (9, 1, 3)),
    ((4, 4), (16, 1, 4)),
    ((3, 5), (15, 1, 3)),
    ((5, 3), (15, 1, 3)),
    ((4, 6), (24, 1, 4)),
    ((6, 4), (24, 1, 4)),
    ((3, 4), (12, 1, 3)),
    ((4, 3), (12, 1, 3)),
])
def test_rotated_planar_code_n_k_d(size, expected):
    code = RotatedPlanarCode(*size)
    assert code.n_k_d == expected


@pytest.mark.parametrize('size, expected', [
    ((3, 3), 8),
    ((4, 4), 15),
    ((3, 5), 14),
    ((5, 3), 14),
    ((4, 6), 23),
    ((6, 4), 23),
    ((3, 4), 11),
    ((4, 3), 11),
])
def test_rotated_planar_code_stabilizers(size, expected):
    assert len(RotatedPlanarCode(*size).stabilizers) == expected


def test_rotated_planar_code_logical_xs():
    assert len(RotatedPlanarCode(5, 5).logical_xs) == 1


def test_rotated_planar_code_logical_zs():
    assert len(RotatedPlanarCode(5, 5).logical_zs) == 1


def test_rotated_planar_code_logicals():
    assert len(RotatedPlanarCode(5, 5).logicals) == 2


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
def test_rotated_planar_code_validate(size):
    code = RotatedPlanarCode(*size)
    code.validate()  # no error raised


# </ Code tests >
# < Lattice tests >


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
def test_rotated_planar_lattice_size(size):
    lattice = RotatedPlanarCode(*size)
    assert lattice.size == size


@pytest.mark.parametrize('index, expected', [
    ((0, 0), True),  # in-bounds z plaquette
    ((1, -1), True),  # boundary z plaquette
    ((4, 0), True),  # out-bounds z plaquette
    ((1, 0), False),  # in-bounds x plaquette
    ((-1, 0), False),  # boundary x plaquette
    ((1, 2), False),  # out-bounds x plaquette
])
def test_rotated_planar_lattice_is_z_plaquette(index, expected):
    lattice = RotatedPlanarCode(3, 5)
    assert lattice.is_z_plaquette(index) == expected


@pytest.mark.parametrize('index, expected', [
    ((2, 1), True),  # in-bounds x plaquette
    ((4, 1), True),  # boundary x plaquette
    ((2, -1), True),  # out-bounds x plaquette
    ((3, 1), False),  # in-bounds z plaquette
    ((2, 2), False),  # boundary z plaquette
    ((-1, 1), False),  # out-bounds z plaquette
])
def test_rotated_planar_lattice_is_x_plaquette(index, expected):
    lattice = RotatedPlanarCode(3, 5)
    assert lattice.is_x_plaquette(index) == expected


@pytest.mark.parametrize('lattice, expected', [
    # RotatedPlanarCode(row, cols), (max_site_x, max_site_y)
    (RotatedPlanarCode(3, 3), (2, 2)),
    (RotatedPlanarCode(4, 4), (3, 3)),
    (RotatedPlanarCode(3, 5), (4, 2)),
    (RotatedPlanarCode(5, 3), (2, 4)),
    (RotatedPlanarCode(4, 6), (5, 3)),
    (RotatedPlanarCode(6, 4), (3, 5)),
    (RotatedPlanarCode(3, 4), (3, 2)),
    (RotatedPlanarCode(4, 3), (2, 3)),
])
def test_rotated_planar_lattice_site_bounds(lattice, expected):
    assert lattice.site_bounds == expected


@pytest.mark.parametrize('index, expected', [
    ((0, 1), True),  # in-bounds left
    ((4, 1), True),  # in-bounds right
    ((2, 2), True),  # in-bounds top
    ((2, 0), True),  # in-bounds bottom
    ((-1, 1), False),  # out-bounds left
    ((5, 1), False),  # out-bounds right
    ((2, 3), False),  # out-bounds top
    ((2, -1), False),  # out-bounds bottom
])
def test_rotated_planar_lattice_is_in_site_bounds(index, expected):
    lattice = RotatedPlanarCode(3, 5)
    assert lattice.is_in_site_bounds(index) == expected


@pytest.mark.parametrize('lattice, index, expected', [
    # 3x5 around boundary clockwise from (-1, -1)
    (RotatedPlanarCode(3, 5), (-1, 0), True),
    (RotatedPlanarCode(3, 5), (-1, 1), False),
    (RotatedPlanarCode(3, 5), (0, 2), True),
    (RotatedPlanarCode(3, 5), (1, 2), False),
    (RotatedPlanarCode(3, 5), (2, 2), True),
    (RotatedPlanarCode(3, 5), (3, 2), False),
    (RotatedPlanarCode(3, 5), (4, 1), True),
    (RotatedPlanarCode(3, 5), (4, 0), False),
    (RotatedPlanarCode(3, 5), (3, -1), True),
    (RotatedPlanarCode(3, 5), (2, -1), False),
    (RotatedPlanarCode(3, 5), (1, -1), True),
    (RotatedPlanarCode(3, 5), (0, -1), False),
    # 3x5 inside corners
    (RotatedPlanarCode(3, 5), (0, 0), True),
    (RotatedPlanarCode(3, 5), (0, 1), True),
    (RotatedPlanarCode(3, 5), (3, 1), True),
    (RotatedPlanarCode(3, 5), (3, 0), True),
    # 3x5 outside corners
    (RotatedPlanarCode(3, 5), (-1, -1), False),
    (RotatedPlanarCode(3, 5), (-1, 2), False),
    (RotatedPlanarCode(3, 5), (4, 2), False),
    (RotatedPlanarCode(3, 5), (4, -1), False),
    # 4x4 around boundary clockwise from (-1, -1)
    (RotatedPlanarCode(4, 4), (-1, 0), True),
    (RotatedPlanarCode(4, 4), (-1, 1), False),
    (RotatedPlanarCode(4, 4), (-1, 2), True),
    (RotatedPlanarCode(4, 4), (0, 3), False),
    (RotatedPlanarCode(4, 4), (1, 3), True),
    (RotatedPlanarCode(4, 4), (2, 3), False),
    (RotatedPlanarCode(4, 4), (3, 2), True),
    (RotatedPlanarCode(4, 4), (3, 1), False),
    (RotatedPlanarCode(4, 4), (3, 0), True),
    (RotatedPlanarCode(4, 4), (2, -1), False),
    (RotatedPlanarCode(4, 4), (1, -1), True),
    (RotatedPlanarCode(4, 4), (0, -1), False),
    # 4x4 inside corners
    (RotatedPlanarCode(4, 4), (0, 0), True),
    (RotatedPlanarCode(4, 4), (0, 2), True),
    (RotatedPlanarCode(4, 4), (2, 2), True),
    (RotatedPlanarCode(4, 4), (2, 0), True),
    # 4x4 outside corners
    (RotatedPlanarCode(4, 4), (-1, -1), False),
    (RotatedPlanarCode(4, 4), (-1, 3), False),
    (RotatedPlanarCode(4, 4), (3, 3), False),
    (RotatedPlanarCode(4, 4), (3, -1), False),
])
def test_rotated_planar_lattice_is_in_plaquette_bounds(lattice, index, expected):
    assert lattice.is_in_plaquette_bounds(index) == expected


@pytest.mark.parametrize('index, expected', [
    # corners
    ((-1, -1), True),  # virtual sw
    ((-1, 2), True),  # virtual nw
    ((4, 2), True),  # virtual ne
    ((4, -1), True),  # virtual se
    # boundary virtual
    ((-1, 1), True),  # virtual w
    ((4, 0), True),  # virtual e
    ((1, 2), True),  # virtual n
    ((2, -1), True),  # virtual s
    # boundary real
    ((-1, 0), False),  # real w
    ((4, 1), False),  # real e
    ((2, 2), False),  # real n
    ((3, -1), False),  # real s
    # beyond boundary
    ((-2, 1), False),  # out-bounds w
    ((5, 1), False),  # out-bounds e
    ((2, 3), False),  # out-bounds n
    ((2, -2), False),  # out-bounds s
    # bulk
    ((0, 0), False),  # bulk sw
    ((0, 1), False),  # bulk nw
    ((3, 1), False),  # bulk ne
    ((3, 0), False),  # bulk se
])
def test_rotated_planar_lattice_is_virtual_plaquette(index, expected):
    lattice = RotatedPlanarCode(3, 5)
    assert lattice.is_virtual_plaquette(index) == expected


@pytest.mark.parametrize('error, expected', [
    # X
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (2, 2)), [(1, 1), (2, 2)]),  # center
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (0, 0)), [(0, 0)]),  # SW
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (0, 2)), [(0, 2)]),  # W
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (0, 4)), [(0, 4)]),  # NW
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (2, 4)), [(1, 3), (2, 4)]),  # N
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (4, 4)), [(3, 3)]),  # NE
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (4, 2)), [(3, 1)]),  # E
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (4, 0)), [(3, -1)]),  # SE
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (2, 0)), [(1, -1), (2, 0)]),  # S
    # Z
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (2, 2)), [(1, 2), (2, 1)]),  # center
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (0, 0)), [(-1, 0)]),  # SW
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (0, 2)), [(-1, 2), (0, 1)]),  # W
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (0, 4)), [(0, 3)]),  # NW
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (2, 4)), [(2, 3)]),  # N
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (4, 4)), [(4, 3)]),  # NE
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (4, 2)), [(3, 2), (4, 1)]),  # E
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (4, 0)), [(3, 0)]),  # SE
    (RotatedPlanarCode(5, 5).new_pauli().site('Z', (2, 0)), [(1, 0)]),  # S
    # Y
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)), [(1, 1), (2, 2), (1, 2), (2, 1)]),  # center
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0)), [(0, 0), (-1, 0)]),  # SW
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 2)), [(0, 2), (-1, 2), (0, 1)]),  # W
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 4)), [(0, 4), (0, 3)]),  # NW
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 4)), [(1, 3), (2, 4), (2, 3)]),  # N
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (4, 4)), [(3, 3), (4, 3)]),  # NE
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (4, 2)), [(3, 1), (3, 2), (4, 1)]),  # E
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (4, 0)), [(3, -1), (3, 0)]),  # SE
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 0)), [(1, -1), (2, 0), (1, 0)]),  # S
])
def test_rotated_planar_lattice_syndrome_to_plaquette_indices(error, expected):
    code = error.code
    syndrome = pt.bsp(error.to_bsf(), code.stabilizers.T)
    assert set(code.syndrome_to_plaquette_indices(syndrome)) == set(expected)


@pytest.mark.parametrize('pauli', [
    RotatedPlanarCode(3, 3).new_pauli().site('Y', (0, 0), (1, 1)),
    RotatedPlanarCode(4, 4).new_pauli().site('Y', (0, 0), (1, 1)),
    RotatedPlanarCode(3, 5).new_pauli().logical_x(),
    RotatedPlanarCode(5, 3).new_pauli().logical_z(),
    RotatedPlanarCode(4, 6).new_pauli().logical_x(),
    RotatedPlanarCode(6, 4).new_pauli().logical_z(),
    RotatedPlanarCode(3, 4).new_pauli().logical_x(),
    RotatedPlanarCode(4, 3).new_pauli().logical_z(),
])
def test_rotated_planar_lattice_ascii_art(pauli):
    code = pauli.code
    syndrome = pt.bsp(pauli.to_bsf(), code.stabilizers.T)
    assert isinstance(code.ascii_art(), str)
    assert isinstance(code.ascii_art(syndrome=syndrome), str)
    assert isinstance(code.ascii_art(pauli=pauli), str)
    assert isinstance(code.ascii_art(syndrome=syndrome, pauli=pauli), str)


@pytest.mark.parametrize('code, plaquette_labels, site_labels', [
    (RotatedPlanarCode(3, 3), {(-1, -1): '0', (0, 0): '1', (1, 2): '2'}, {(0, 0): 'a', (2, 1): 'b'}),
    (RotatedPlanarCode(4, 4), {(0, 3): '0', (1, 1): '1', (3, 2): '2'}, {(0, 3): 'a', (3, 0): 'b'}),
])
def test_rotated_planar_lattice_ascii_art_labels(code, plaquette_labels, site_labels):
    ascii_art = code.ascii_art(plaquette_labels=plaquette_labels, site_labels=site_labels)
    print()
    print(ascii_art)
    assert isinstance(ascii_art, str)

# </ Lattice tests >
