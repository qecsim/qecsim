import pytest

from qecsim import paulitools as pt
from qecsim.models.planar import PlanarCode


# < Code tests >


@pytest.mark.parametrize('size', [
    (3, 3),
    (5, 5),
    (3, 5),
    (2, 4),
    (7, 4),
    (2, 2),
])
def test_planar_code_properties(size):
    code = PlanarCode(*size)
    assert isinstance(code.label, str)
    assert isinstance(repr(code), str)


@pytest.mark.parametrize('rows, columns', [
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (None, 3),
    (3, None),
    ('asdf', 3),
    (3, 'asdf'),
    (3.1, 3),
    (3, 3.1),
])
def test_planar_code_new_invalid_parameters(rows, columns):
    with pytest.raises((ValueError, TypeError), match=r"^PlanarCode") as exc_info:
        PlanarCode(rows, columns)
    print(exc_info)


@pytest.mark.parametrize('size, expected', [
    ((3, 3), (13, 1, 3)),
    ((5, 5), (41, 1, 5)),
    ((3, 5), (23, 1, 3)),
    ((2, 4), (11, 1, 2)),
    ((7, 4), (46, 1, 4)),
    ((2, 2), (5, 1, 2)),
])
def test_planar_code_n_k_d(size, expected):
    code = PlanarCode(*size)
    assert code.n_k_d == expected


@pytest.mark.parametrize('size, expected', [
    ((3, 3), 12),
    ((5, 5), 40),
    ((3, 5), 22),
    ((2, 4), 10),
    ((7, 4), 45),
    ((2, 2), 4),
])
def test_planar_code_stabilizers(size, expected):
    assert len(PlanarCode(*size).stabilizers) == expected


def test_planar_code_logical_xs():
    assert len(PlanarCode(5, 5).logical_xs) == 1


def test_planar_code_logical_zs():
    assert len(PlanarCode(5, 5).logical_zs) == 1


def test_planar_code_logicals():
    assert len(PlanarCode(5, 5).logicals) == 2


@pytest.mark.parametrize('size', [
    (3, 3),
    (5, 5),
    (3, 5),
    (2, 4),
    (7, 4),
    (2, 2),
])
def test_planar_code_validate(size):
    code = PlanarCode(*size)
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
def test_planar_lattice_properties(size):
    lattice = PlanarCode(*size)
    assert lattice.size == size
    assert isinstance(repr(lattice), str)


@pytest.mark.parametrize('index', [
    (3, 2),  # in-bounds primal plaquette
    (2, 1),  # in-bounds dual plaquette
    (-1, 0),  # out-bounds primal plaquette
    (6, 3),  # out-bounds dual plaquette
])
def test_planar_lattice_is_plaquette(index):
    lattice = PlanarCode(3, 5)
    assert lattice.is_plaquette(index)


@pytest.mark.parametrize('index', [
    (2, 2),  # in-bounds primal site
    (1, 1),  # in-bounds dual site
    (0, -2),  # out-bounds primal site
    (7, 1),  # out-bounds dual site
])
def test_planar_lattice_is_site(index):
    lattice = PlanarCode(3, 5)
    assert lattice.is_site(index)


@pytest.mark.parametrize('index', [
    (3, 2),  # in-bounds primal plaquette
    (-1, 0),  # out-bounds primal plaquette
    (2, 2),  # in-bounds primal site
    (0, -2),  # out-bounds primal site
])
def test_planar_lattice_is_primal(index):
    lattice = PlanarCode(3, 5)
    assert lattice.is_primal(index)


@pytest.mark.parametrize('index', [
    (2, 1),  # in-bounds dual plaquette
    (6, 3),  # out-bounds dual plaquette
    (1, 1),  # in-bounds dual site
    (7, 1),  # out-bounds dual site
])
def test_planar_lattice_is_dual(index):
    lattice = PlanarCode(3, 5)
    assert lattice.is_dual(index)


@pytest.mark.parametrize('lattice, expected', [
    (PlanarCode(3, 3), (4, 4)),
    (PlanarCode(5, 5), (8, 8)),
    (PlanarCode(3, 5), (4, 8)),
    (PlanarCode(5, 3), (8, 4)),
])
def test_planar_lattice_bounds(lattice, expected):
    assert lattice.bounds == expected


@pytest.mark.parametrize('index', [
    (3, 2),  # in-bounds primal plaquette
    (2, 1),  # in-bounds dual plaquette
    (2, 2),  # in-bounds primal site
    (1, 1),  # in-bounds dual site
])
def test_planar_lattice_is_in_bounds(index):
    lattice = PlanarCode(3, 5)
    assert lattice.is_in_bounds(index)


@pytest.mark.parametrize('index', [
    (-1, 0),  # out-bounds primal plaquette
    (6, 3),  # out-bounds dual plaquette
    (0, -2),  # out-bounds primal site
    (7, 1),  # out-bounds dual site
])
def test_planar_lattice_not_is_in_bounds(index):
    lattice = PlanarCode(3, 5)
    assert not lattice.is_in_bounds(index)


@pytest.mark.parametrize('size, a_index, b_index, expected', [
    # 3 x 3
    ((3, 3), (-1, 2), (3, 2), (2, 0)),  # across north boundary vertically
    ((3, 3), (3, 2), (-1, 2), (-2, 0)),  # ditto reversed
    ((3, 3), (-1, 2), (1, 4), (1, 1)),  # across north boundary diagonally
    ((3, 3), (1, 4), (-1, 2), (-1, -1)),  # ditto reversed
    ((3, 3), (-1, 2), (3, 4), (2, 1)),  # across north boundary dog-leg
    ((3, 3), (3, 4), (-1, 2), (-2, -1)),  # ditto reversed
    ((3, 3), (-1, 2), (-1, 4), (0, 0)),  # outside north boundary
    ((3, 3), (-1, 4), (-1, 2), (0, 0)),  # ditto reversed
    ((3, 3), (3, 2), (5, 2), (1, 0)),  # across south boundary vertically
    ((3, 3), (5, 2), (3, 2), (-1, 0)),  # ditto reversed
    ((3, 3), (3, 2), (5, 4), (1, 1)),  # across south boundary diagonally
    ((3, 3), (5, 4), (3, 2), (-1, -1)),  # ditto reversed
    ((3, 3), (3, 0), (5, 4), (1, 2)),  # across south boundary dog-leg
    ((3, 3), (5, 4), (3, 0), (-1, -2)),  # ditto reversed
    ((3, 3), (5, 2), (5, 4), (0, 0)),  # outside south boundary
    ((3, 3), (5, 4), (5, 2), (0, 0)),  # ditto reversed
    ((3, 3), (-1, 2), (5, 4), (0, 0)),  # across north and south boundary
    ((3, 3), (5, 4), (-1, 2), (0, 0)),  # ditto reversed
    # # 3 x 3 dual
    ((3, 3), (2, -1), (2, 3), (0, 2)),  # across west boundary horizontally
    ((3, 3), (2, 3), (2, -1), (0, -2)),  # ditto reversed
    ((3, 3), (2, -1), (4, 1), (1, 1)),  # across west boundary diagonally
    ((3, 3), (4, 1), (2, -1), (-1, -1)),  # ditto reversed
    ((3, 3), (2, -1), (4, 3), (1, 2)),  # across west boundary dog-leg
    ((3, 3), (4, 3), (2, -1), (-1, -2)),  # ditto reversed
    ((3, 3), (2, -1), (4, -1), (0, 0)),  # outside west boundary
    ((3, 3), (4, -1), (2, -1), (0, 0)),  # ditto reversed
    ((3, 3), (2, 3), (2, 5), (0, 1)),  # across east boundary horizontally
    ((3, 3), (2, 5), (2, 3), (0, -1)),  # ditto reversed
    ((3, 3), (2, 3), (4, 5), (1, 1)),  # across east boundary diagonally
    ((3, 3), (4, 5), (2, 3), (-1, -1)),  # ditto reversed
    ((3, 3), (0, 3), (4, 5), (2, 1)),  # across east boundary dog-leg
    ((3, 3), (4, 5), (0, 3), (-2, -1)),  # ditto reversed
    ((3, 3), (0, 5), (4, 5), (0, 0)),  # outside east boundary
    ((3, 3), (4, 5), (0, 5), (0, 0)),  # ditto reversed
    ((3, 3), (2, -1), (4, 5), (0, 0)),  # across west and east boundary
    ((3, 3), (4, 5), (2, -1), (0, 0)),  # ditto reversed
    # 5 x 5
    ((5, 5), (3, 2), (3, 6), (0, 2)),  # within lattice horizontally
    ((5, 5), (3, 6), (3, 2), (0, -2)),  # ditto reversed
    ((5, 5), (3, 2), (7, 2), (2, 0)),  # within lattice vertically
    ((5, 5), (7, 2), (3, 2), (-2, 0)),  # ditto reversed
    ((5, 5), (3, 2), (7, 6), (2, 2)),  # within lattice diagonally
    ((5, 5), (7, 6), (3, 2), (-2, -2)),  # ditto reversed
    ((5, 5), (3, 2), (7, 4), (2, 1)),  # within lattice dog-leg
    ((5, 5), (7, 4), (3, 2), (-2, -1)),  # ditto reversed
    # 5 x 5 dual
    ((5, 5), (2, 3), (2, 7), (0, 2)),  # within lattice horizontally
    ((5, 5), (2, 7), (2, 3), (0, -2)),  # ditto reversed
    ((5, 5), (2, 3), (6, 3), (2, 0)),  # within lattice vertically
    ((5, 5), (6, 3), (2, 3), (-2, 0)),  # ditto reversed
    ((5, 5), (2, 3), (6, 7), (2, 2)),  # within lattice diagonally
    ((5, 5), (6, 7), (2, 3), (-2, -2)),  # ditto reversed
    ((5, 5), (2, 3), (6, 5), (2, 1)),  # within lattice dog-leg
    ((5, 5), (6, 5), (2, 3), (-2, -1)),  # ditto reversed
    # 3 x 5
    ((3, 5), (3, 0), (3, 4), (0, 2)),  # within lattice horizontally
    ((3, 5), (3, 4), (3, 0), (0, -2)),  # ditto reversed
    ((3, 5), (1, 2), (3, 2), (1, 0)),  # within lattice vertically
    ((3, 5), (3, 2), (1, 2), (-1, 0)),  # ditto reversed
    ((3, 5), (1, 2), (3, 6), (1, 2)),  # within lattice dog-leg
    ((3, 5), (3, 6), (1, 2), (-1, -2)),  # ditto reversed
    ((3, 5), (-1, 0), (3, 6), (2, 3)),  # across north boundary dog-leg
    ((3, 5), (3, 6), (-1, 0), (-2, -3)),  # ditto reversed
    ((3, 5), (3, 0), (5, 6), (1, 3)),  # across south boundary dog-leg
    ((3, 5), (5, 6), (3, 0), (-1, -3)),  # ditto reversed
    # 5 x 3
    ((5, 3), (1, 2), (5, 2), (2, 0)),  # within lattice vertically
    ((5, 3), (5, 2), (1, 2), (-2, 0)),  # ditto reversed
    ((5, 3), (3, 2), (3, 4), (0, 1)),  # within lattice horizontally
    ((5, 3), (3, 4), (3, 2), (0, -1)),  # ditto reversed
    ((5, 3), (3, 2), (7, 4), (2, 1)),  # within lattice dog-leg
    ((5, 3), (7, 4), (3, 2), (-2, -1)),  # ditto reversed
    ((5, 3), (-1, 0), (7, 2), (4, 1)),  # across north boundary dog-leg
    ((5, 3), (7, 2), (-1, 0), (-4, -1)),  # ditto reversed
    ((5, 3), (3, 0), (9, 4), (3, 2)),  # across south boundary dog-leg
    ((5, 3), (9, 4), (3, 0), (-3, -2)),  # ditto reversed
])
def test_planar_lattice_translation(size, a_index, b_index, expected):
    assert PlanarCode(*size).translation(a_index, b_index) == expected


@pytest.mark.parametrize('size, a_index, b_index', [
    ((5, 5), (0, 0), (4, 5)),  # invalid plaquette index
    ((5, 5), (1, 0), (4, 4)),  # invalid plaquette index
    ((5, 5), (3, 2), (4, 5)),  # different lattices
])
def test_planar_lattice_invalid_translation(size, a_index, b_index):
    code = PlanarCode(*size)
    with pytest.raises(IndexError):
        code.translation(a_index, b_index)


@pytest.mark.parametrize('error, expected', [
    (PlanarCode(3, 3).new_pauli().site('X', (1, 3)),  # in lattice X
     [(1, 2), (1, 4)]),
    (PlanarCode(3, 3).new_pauli().site('X', (2, 0)),  # dual west site X
     [(1, 0), (3, 0)]),
    (PlanarCode(3, 3).new_pauli().site('X', (2, 4)),  # dual east site X
     [(1, 4), (3, 4)]),
    (PlanarCode(3, 3).new_pauli().site('X', (0, 2)),  # primal north site X
     [(1, 2)]),
    (PlanarCode(3, 3).new_pauli().site('X', (4, 2)),  # primal south site X
     [(3, 2)]),

    (PlanarCode(3, 3).new_pauli().site('Z', (1, 3)),  # in lattice Z
     [(0, 3), (2, 3)]),
    (PlanarCode(3, 3).new_pauli().site('Z', (2, 0)),  # dual west site Z
     [(2, 1)]),
    (PlanarCode(3, 3).new_pauli().site('Z', (2, 4)),  # dual east site Z
     [(2, 3)]),
    (PlanarCode(3, 3).new_pauli().site('Z', (0, 2)),  # primal north site Z
     [(0, 1), (0, 3)]),
    (PlanarCode(3, 3).new_pauli().site('Z', (4, 2)),  # primal south site Z
     [(4, 1), (4, 3)]),

    (PlanarCode(3, 3).new_pauli().site('Y', (1, 3)),  # in lattice Y
     [(1, 2), (1, 4), (0, 3), (2, 3)]),
    (PlanarCode(3, 3).new_pauli().site('Y', (2, 0)),  # dual west site Y
     [(1, 0), (3, 0), (2, 1)]),
    (PlanarCode(3, 3).new_pauli().site('Y', (2, 4)),  # dual east site Y
     [(1, 4), (3, 4), (2, 3)]),
    (PlanarCode(3, 3).new_pauli().site('Y', (0, 2)),  # primal north site Y
     [(1, 2), (0, 1), (0, 3)]),
    (PlanarCode(3, 3).new_pauli().site('Y', (4, 2)),  # primal south site Y
     [(3, 2), (4, 1), (4, 3)]),
])
def test_planar_lattice_syndrome_to_plaquette_indices(error, expected):
    code = error.code
    syndrome = pt.bsp(error.to_bsf(), code.stabilizers.T)
    assert set(code.syndrome_to_plaquette_indices(syndrome)) == set(expected)


@pytest.mark.parametrize('size, index, expected', [
    # 5 x 5
    ((5, 5), (3, 2), (-1, 2)),  # nearest North
    ((5, 5), (7, 2), (9, 2)),  # nearest South
    ((5, 5), (-3, 2), (-1, 2)),  # outside North
    ((5, 5), (11, 2), (9, 2)),  # outside South
    ((5, 5), (-3, -4), (-1, -4)),  # outside North and West
    # 5 x 5 dual
    ((5, 5), (2, 3), (2, -1)),  # nearest West
    ((5, 5), (2, 7), (2, 9)),  # nearest East
    ((5, 5), (2, -3), (2, -1)),  # outside West
    ((5, 5), (2, 11), (2, 9)),  # outside East
    ((5, 5), (12, 11), (12, 9)),  # outside South and East
    # 4 x 4
    ((4, 4), (3, 4), (-1, 4)),  # midway between North and South
    # 4 x 4 dual
    ((4, 4), (4, 3), (4, -1)),  # midway between West and East
    # 3 x 5
    ((3, 5), (1, 8), (-1, 8)),  # nearest North
    # 5 x 3
    ((5, 3), (7, 2), (9, 2)),  # nearest South
])
def test_planar_lattice_virtual_plaquette_index(size, index, expected):
    assert PlanarCode(*size).virtual_plaquette_index(index) == expected


@pytest.mark.parametrize('size, index', [
    ((5, 5), (0, 0)),  # invalid plaquette index
    ((5, 5), (4, 4)),  # invalid plaquette index
])
def test_planar_lattice_invalid_virtual_plaquette_index(size, index):
    code = PlanarCode(*size)
    with pytest.raises(IndexError):
        code.virtual_plaquette_index(index)


@pytest.mark.parametrize('pauli', [
    PlanarCode(2, 2).new_pauli().site('Y', (0, 0), (2, 2)),
    PlanarCode(3, 5).new_pauli().logical_x(),
    PlanarCode(5, 3).new_pauli().logical_z(),
])
def test_planar_lattice_ascii_art(pauli):
    code = pauli.code
    syndrome = pt.bsp(pauli.to_bsf(), code.stabilizers.T)
    assert isinstance(code.ascii_art(), str)
    assert isinstance(code.ascii_art(syndrome=syndrome), str)
    assert isinstance(code.ascii_art(pauli=pauli), str)
    assert isinstance(code.ascii_art(syndrome=syndrome, pauli=pauli), str)

# </ Lattice tests >
