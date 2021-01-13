import numpy as np
import pytest

from qecsim import paulitools as pt
from qecsim.models.planar import PlanarCode, PlanarCMWPMDecoder


def test_planar_cmwpm_decoder_properties():
    decoder = PlanarCMWPMDecoder()
    assert isinstance(decoder.label, str)
    assert isinstance(repr(decoder), str)
    assert isinstance(str(decoder), str)


@pytest.mark.parametrize('factor, max_iterations, box_shape, distance_algorithm', [
    (3, 4, 't', 1),
    (0, 0, 'r', 2),
    (2.5, 10, 'f', 4),
    (3, 4, 'l', 1),
])
def test_planar_mps_decoder_new_valid_parameters(factor, max_iterations, box_shape, distance_algorithm):
    PlanarCMWPMDecoder(factor=factor, max_iterations=max_iterations, box_shape=box_shape,
                       distance_algorithm=distance_algorithm)  # no error raised


@pytest.mark.parametrize('factor, max_iterations, box_shape, distance_algorithm', [
    (-1, 4, 't', 1),  # invalid factor
    (None, 4, 't', 1),  # invalid factor
    (3, -1, 't', 1),  # invalid max_iterations
    (3, 4.1, 't', 1),  # invalid max_iterations
    (3, None, 't', 1),  # invalid max_iterations
    (3, 4, 'z', 1),  # invalid box_shape
    (3, 4, None, 1),  # invalid box_shape
    (3, 4, 't', 3),  # invalid distance_algorithm
    (3, 4, 't', None),  # invalid distance_algorithm
])
def test_planar_mps_decoder_new_invalid_parameters(factor, max_iterations, box_shape, distance_algorithm):
    with pytest.raises((ValueError, TypeError), match=r"^PlanarCMWPMDecoder") as exc_info:
        PlanarCMWPMDecoder(factor=factor, max_iterations=max_iterations, box_shape=box_shape,
                           distance_algorithm=distance_algorithm)
    print(exc_info)


@pytest.mark.parametrize('error_pauli', [
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2), (4, 2))),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2), (4, 2)).site('Z', (6, 4), (2, 0))),
    (PlanarCode(5, 5).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (6, 4), (1, 1))),
    (PlanarCode(3, 5).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (2, 4), (1, 7))),
    (PlanarCode(5, 3).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (8, 4), (3, 1))),
    (PlanarCode(5, 3).new_pauli().site('Y', (1, 3), (4, 2)).site('Z', (8, 4), (6, 4), (4, 4))),
    (PlanarCode(5, 3).new_pauli().site('Y', (1, 3), (3, 3), (5, 3)).site('Z', (8, 4), (6, 4), (4, 4))),
    (PlanarCode(5, 3).new_pauli().site('X', (1, 3), (3, 3), (5, 3), (8, 4), (6, 4), (4, 4))),
    (PlanarCode(5, 3).new_pauli().site('Y', (1, 3), (3, 3), (5, 3), (8, 4), (6, 4), (4, 4))),
    (PlanarCode(5, 3).new_pauli().site('Z', (1, 3), (3, 3), (5, 3), (8, 4), (6, 4), (4, 4))),
])
def test_planar_cmwpm_decoder_decode(error_pauli):
    error = error_pauli.to_bsf()
    code = error_pauli.code
    decoder = PlanarCMWPMDecoder()
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery = decoder.decode(code, syndrome)
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


def test_planar_cmwpm_decoder_null_decoding():
    code = PlanarCode(3, 3)
    error = code.new_pauli().site('Y', (2, 0), (2, 4)).to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    decoder_null = PlanarCMWPMDecoder(max_iterations=0)
    recovery = decoder_null.decode(code, syndrome)
    assert np.all(recovery == 0), 'Null decoder does not return null recovery'


def test_planar_cmwpm_decoder_simple_correlated_error():
    """
    ·─┬─·─┬─·
      ·   ·
    Y─┼─·─┼─Y
      ·   ·
    ·─┴─·─┴─·
    """
    code = PlanarCode(3, 3)
    error = code.new_pauli().site('Y', (2, 0), (2, 4)).to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    decoder_mwpm = PlanarCMWPMDecoder(max_iterations=1)
    decoder_cmwpm = PlanarCMWPMDecoder(factor=3, max_iterations=4, box_shape='t', distance_algorithm=1)
    # show mwpm fails
    recovery = decoder_mwpm.decode(code, syndrome)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'MWPM recovery does not commute with stabilizers.')
    assert not np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'MWPM recovery does commute with logicals.')
    # show cmwpm succeeds
    recovery = decoder_cmwpm.decode(code, syndrome)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'CMWPM recovery does not commute with stabilizers.')
    assert np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'CMWPM recovery does not commute with logicals.')


def test_planar_cmwpm_decoder_odd_diagonal_correlated_error():
    """
    ·─┬─·─┬─·─┬─·
      ·   ·   ·
    ·─┼─·─┼─·─┼─·
      ·   ·   ·
    ·─┼─Y─┼─·─┼─·
      ·   Y   ·
    ·─┼─·─┼─Y─┼─·
      ·   ·   ·
    ·─┼─·─┼─·─┼─·
      ·   ·   ·
    ·─┴─·─┴─·─┴─·
    """
    code = PlanarCode(6, 4)
    error = code.new_pauli().site('Y', (4, 2), (5, 3), (6, 4)).to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    decoder_mwpm = PlanarCMWPMDecoder(max_iterations=1)
    decoder_cmwpm = PlanarCMWPMDecoder(factor=3, max_iterations=4, box_shape='t', distance_algorithm=1)
    # show mwpm fails
    recovery = decoder_mwpm.decode(code, syndrome)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'MWPM recovery does not commute with stabilizers.')
    assert not np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'MWPM recovery does commute with logicals.')
    # show cmwpm succeeds
    recovery = decoder_cmwpm.decode(code, syndrome)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'CMWPM recovery does not commute with stabilizers.')
    assert np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'CMWPM recovery does not commute with logicals.')


def test_planar_cmwpm_decoder_even_diagonal_correlated_error():
    """
    ·─┬─·─┬─·─┬─·─┬─·
      ·   ·   ·   ·
    ·─┼─·─┼─·─┼─·─┼─·
      ·   Y   ·   ·
    ·─┼─·─┼─Y─┼─·─┼─·
      ·   ·   Y   ·
    ·─┼─·─┼─·─┼─Y─┼─·
      ·   ·   ·   ·
    ·─┼─·─┼─·─┼─·─┼─·
      ·   ·   ·   ·
    ·─┼─·─┼─·─┼─·─┼─·
      ·   ·   ·   ·
    ·─┴─·─┴─·─┴─·─┴─·
    """
    code = PlanarCode(7, 5)
    error = code.new_pauli().site('Y', (3, 3), (4, 4), (5, 5), (6, 6)).to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    decoder_mwpm = PlanarCMWPMDecoder(max_iterations=1)
    decoder_cmwpm_t_1 = PlanarCMWPMDecoder(factor=3, max_iterations=4, box_shape='t', distance_algorithm=1)
    decoder_cmwpm_t_2 = PlanarCMWPMDecoder(factor=3, max_iterations=4, box_shape='t', distance_algorithm=2)
    # show mwpm fails
    recovery = decoder_mwpm.decode(code, syndrome)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'MWPM recovery does not commute with stabilizers.')
    assert not np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'MWPM recovery does commute with logicals.')
    # show cmwpm_t_1 fails
    recovery = decoder_cmwpm_t_1.decode(code, syndrome)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'CMWPM (t,1) recovery does not commute with stabilizers.')
    assert not np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'CMWPM (t,1) recovery does commute with logicals.')
    # show cmwpm_t_2 succeeds
    recovery = decoder_cmwpm_t_2.decode(code, syndrome)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'CMWPM (t,2) recovery does not commute with stabilizers.')
    assert np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'CMWPM (t,2) recovery does not commute with logicals.')


def test_planar_cmwpm_decoder_dog_leg_correlated_error():
    """
    ·─┬─·─┬─·─┬─·─┬─·
      ·   ·   ·   ·
    ·─┼─Y─┼─·─┼─·─┼─·
      ·   Y   Y   ·
    ·─┼─·─┼─Y─┼─·─┼─·
      ·   ·   ·   ·
    ·─┴─·─┴─·─┴─·─┴─·
    """
    code = PlanarCode(4, 5)
    error = code.new_pauli().site('Y', (2, 2), (3, 3), (4, 4), (3, 5)).to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    decoder_mwpm = PlanarCMWPMDecoder(max_iterations=1)
    decoder_cmwpm_t_2 = PlanarCMWPMDecoder(factor=3, max_iterations=4, box_shape='t', distance_algorithm=2)
    decoder_cmwpm_t_4 = PlanarCMWPMDecoder(factor=3, max_iterations=4, box_shape='t', distance_algorithm=4)
    decoder_cmwpm_r_1 = PlanarCMWPMDecoder(factor=3, max_iterations=4, box_shape='r', distance_algorithm=1)
    # show mwpm fails
    recovery = decoder_mwpm.decode(code, syndrome)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'MWPM recovery does not commute with stabilizers.')
    assert not np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'MWPM recovery does commute with logicals.')
    # show cmwpm_t_2 fails
    recovery = decoder_cmwpm_t_2.decode(code, syndrome)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'CMWPM (t,2) recovery does not commute with stabilizers.')
    assert not np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'CMWPM (t,2) recovery does commute with logicals.')
    # show cmwpm_t_4 succeeds
    recovery = decoder_cmwpm_t_4.decode(code, syndrome)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'CMWPM (t,4) recovery does not commute with stabilizers.')
    assert np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'CMWPM (t,4) recovery does not commute with logicals.')
    # show cmwpm_r_1 succeeds
    recovery = decoder_cmwpm_r_1.decode(code, syndrome)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'CMWPM (r,1) recovery does not commute with stabilizers.')
    assert np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'CMWPM (r,1) recovery does not commute with logicals.')


def test_planar_cmwpm_step_grid_tight_box():
    """
    Matches {((0, 1), (4, 5))}:
    ──X───┬───┬──
      │   │   │
    ──┼───┼───┼──
      │   │   │
    ──┴───┴───X──
    """
    code = PlanarCode(3, 4)
    # error = code.new_pauli().site('Z', (0, 2), (1, 3), (2, 4), (3, 5)).to_bsf()
    # syndrome = pt.bsp(error, code.stabilizers.T)
    # matches = {tuple(code.syndrome_to_plaquette_indices(syndrome))}  # {((0, 1), (4, 5))}
    # print(code.ascii_art(syndrome=syndrome))
    # print(matches)
    grid = PlanarCMWPMDecoder.StepGrid(code)
    # set background from matches
    grid.set_background({((0, 1), (4, 5))}, factor=3, initial=1, box_shape='t')
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[3., 0., 3., 0., 3., 0., 3., 0., 3.],
         [0., 3., 0., 1., 0., 1., 0., 3., 0.],
         [3., 0., 1., 0., 1., 0., 1., 0., 3.],
         [0., 3., 0., 1., 0., 1., 0., 3., 0.],
         [3., 0., 1., 0., 1., 0., 1., 0., 3.],
         [0., 3., 0., 1., 0., 1., 0., 3., 0.],
         [3., 0., 3., 0., 3., 0., 3., 0., 3.]])
    assert np.array_equal(grid._grid, expected), 'Tight box not expected shape.'
    # test with alternative corners
    grid.set_background({((0, 5), (4, 1))}, factor=3, initial=1, box_shape='t')
    assert np.array_equal(grid._grid, expected), 'Tight box for alternative corners not expected shape.'


def test_planar_cmwpm_step_grid_rounded_box():
    """
    Matches {((0, 1), (4, 5))}:
    ──X───┬───┬──
      │   │   │
    ──┼───┼───┼──
      │   │   │
    ──┴───┴───X──
    """
    code = PlanarCode(3, 4)
    grid = PlanarCMWPMDecoder.StepGrid(code)
    # set background from matches
    grid.set_background({((0, 1), (4, 5))}, factor=3, initial=1, box_shape='r')
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[3., 0., 3., 0., 1., 0., 3., 0., 3.],
         [0., 3., 0., 1., 0., 1., 0., 3., 0.],
         [3., 0., 1., 0., 1., 0., 1., 0., 3.],
         [0., 1., 0., 1., 0., 1., 0., 1., 0.],
         [3., 0., 1., 0., 1., 0., 1., 0., 3.],
         [0., 3., 0., 1., 0., 1., 0., 3., 0.],
         [3., 0., 3., 0., 1., 0., 3., 0., 3.]])
    assert np.array_equal(grid._grid, expected), 'Rounded box not expected shape.'
    # test with alternative corners
    grid.set_background({((0, 5), (4, 1))}, factor=3, initial=1, box_shape='r')
    assert np.array_equal(grid._grid, expected), 'Rounded box for alternative corners not expected shape.'


def test_planar_cmwpm_step_grid_rounded_horizontal_line():
    """
    Matches {((2, 1), (2, 5))}:
    ──┬───┬───┬──
      │   │   │
    ──X───┼───X──
      │   │   │
    ──┴───┴───┴──
    """
    code = PlanarCode(3, 4)
    grid = PlanarCMWPMDecoder.StepGrid(code)
    # set background from matches
    grid.set_background({((2, 1), (2, 5))}, factor=3, initial=1, box_shape='r')
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[3., 0., 3., 0., 3., 0., 3., 0., 3.],
         [0., 3., 0., 3., 0., 3., 0., 3., 0.],
         [3., 0., 3., 0., 1., 0., 3., 0., 3.],
         [0., 3., 0., 1., 0., 1., 0., 3., 0.],
         [3., 0., 3., 0., 1., 0., 3., 0., 3.],
         [0., 3., 0., 3., 0., 3., 0., 3., 0.],
         [3., 0., 3., 0., 3., 0., 3., 0., 3.]])
    assert np.array_equal(grid._grid, expected), 'Rounded box not expected shape.'


def test_planar_cmwpm_step_grid_rounded_vertical_line():
    """
    Matches {((0, 3), (4, 3))}:
    ──┬───X───┬──
      │   │   │
    ──┼───┼───┼──
      │   │   │
    ──┴───X───┴──
    """
    code = PlanarCode(3, 4)
    grid = PlanarCMWPMDecoder.StepGrid(code)
    # set background from matches
    grid.set_background({((0, 3), (4, 3))}, factor=3, initial=1, box_shape='r')
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[3., 0., 3., 0., 3., 0., 3., 0., 3.],
         [0., 3., 0., 3., 0., 3., 0., 3., 0.],
         [3., 0., 3., 0., 1., 0., 3., 0., 3.],
         [0., 3., 0., 1., 0., 1., 0., 3., 0.],
         [3., 0., 3., 0., 1., 0., 3., 0., 3.],
         [0., 3., 0., 3., 0., 3., 0., 3., 0.],
         [3., 0., 3., 0., 3., 0., 3., 0., 3.]])
    assert np.array_equal(grid._grid, expected), 'Rounded box not expected shape.'


def test_planar_cmwpm_step_grid_fitted_box():
    """
    Matches {((0, 1), (4, 5))}:
    ──X───┬───┬──
      │   │   │
    ──┼───┼───┼──
      │   │   │
    ──┴───┴───X──
    """
    code = PlanarCode(3, 4)
    grid = PlanarCMWPMDecoder.StepGrid(code)
    # set background from matches
    grid.set_background({((0, 1), (4, 5))}, factor=3, initial=1, box_shape='f')
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[3., 0., 3., 0., 1., 0., 1., 0., 3.],
         [0., 3., 0., 1., 0., 1., 0., 1., 0.],
         [3., 0., 1., 0., 1., 0., 1., 0., 3.],
         [0., 1., 0., 1., 0., 1., 0., 1., 0.],
         [3., 0., 1., 0., 1., 0., 1., 0., 3.],
         [0., 1., 0., 1., 0., 1., 0., 3., 0.],
         [3., 0., 1., 0., 1., 0., 3., 0., 3.]])
    assert np.array_equal(grid._grid, expected), 'Fitted box not expected shape.'
    # test with alternative corners
    grid.set_background({((0, 5), (4, 1))}, factor=3, initial=1, box_shape='f')
    expected = np.array(
        [[3., 0., 1., 0., 1., 0., 3., 0., 3.],
         [0., 1., 0., 1., 0., 1., 0., 3., 0.],
         [3., 0., 1., 0., 1., 0., 1., 0., 3.],
         [0., 1., 0., 1., 0., 1., 0., 1., 0.],
         [3., 0., 1., 0., 1., 0., 1., 0., 3.],
         [0., 3., 0., 1., 0., 1., 0., 1., 0.],
         [3., 0., 3., 0., 1., 0., 1., 0., 3.]])
    assert np.array_equal(grid._grid, expected), 'Fitted box for alternative corners not expected shape.'


def test_planar_cmwpm_step_grid_fitted_horizontal_line():
    """
    Matches {((2, 1), (2, 5))}:
    ──┬───┬───┬──
      │   │   │
    ──X───┼───X──
      │   │   │
    ──┴───┴───┴──
    """
    code = PlanarCode(3, 4)
    grid = PlanarCMWPMDecoder.StepGrid(code)
    # set background from matches
    grid.set_background({((2, 1), (2, 5))}, factor=3, initial=1, box_shape='f')
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[3., 0., 3., 0., 3., 0., 3., 0., 3.],
         [0., 3., 0., 3., 0., 3., 0., 3., 0.],
         [3., 0., 3., 0., 1., 0., 3., 0., 3.],
         [0., 3., 0., 1., 0., 1., 0., 3., 0.],
         [3., 0., 3., 0., 1., 0., 3., 0., 3.],
         [0., 3., 0., 3., 0., 3., 0., 3., 0.],
         [3., 0., 3., 0., 3., 0., 3., 0., 3.]])
    assert np.array_equal(grid._grid, expected), 'Fitted box not expected shape.'


def test_planar_cmwpm_step_grid_fitted_vertical_line():
    """
    Matches {((0, 3), (4, 3))}:
    ──┬───X───┬──
      │   │   │
    ──┼───┼───┼──
      │   │   │
    ──┴───X───┴──
    """
    code = PlanarCode(3, 4)
    grid = PlanarCMWPMDecoder.StepGrid(code)
    # set background from matches
    grid.set_background({((0, 3), (4, 3))}, factor=3, initial=1, box_shape='f')
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[3., 0., 3., 0., 3., 0., 3., 0., 3.],
         [0., 3., 0., 3., 0., 3., 0., 3., 0.],
         [3., 0., 3., 0., 1., 0., 3., 0., 3.],
         [0., 3., 0., 1., 0., 1., 0., 3., 0.],
         [3., 0., 3., 0., 1., 0., 3., 0., 3.],
         [0., 3., 0., 3., 0., 3., 0., 3., 0.],
         [3., 0., 3., 0., 3., 0., 3., 0., 3.]])
    assert np.array_equal(grid._grid, expected), 'Fitted box not expected shape.'


def test_planar_cmwpm_step_grid_loose_box():
    """
    Matches {((0, 1), (4, 5))}:
    ──X───┬───┬──
      │   │   │
    ──┼───┼───┼──
      │   │   │
    ──┴───┴───X──
    """
    code = PlanarCode(3, 4)
    grid = PlanarCMWPMDecoder.StepGrid(code)
    # set background from matches
    grid.set_background({((0, 1), (4, 5))}, factor=3, initial=1, box_shape='l')
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[3., 0., 1., 0., 1., 0., 1., 0., 3.],
         [0., 1., 0., 1., 0., 1., 0., 1., 0.],
         [3., 0., 1., 0., 1., 0., 1., 0., 3.],
         [0., 1., 0., 1., 0., 1., 0., 1., 0.],
         [3., 0., 1., 0., 1., 0., 1., 0., 3.],
         [0., 1., 0., 1., 0., 1., 0., 1., 0.],
         [3., 0., 1., 0., 1., 0., 1., 0., 3.]])
    assert np.array_equal(grid._grid, expected), 'Loose box not expected shape.'
    # test with alternative corners
    grid.set_background({((0, 5), (4, 1))}, factor=3, initial=1, box_shape='l')
    assert np.array_equal(grid._grid, expected), 'Loose box for alternative corners not expected shape.'


def test_planar_cmwpm_step_grid_loose_box_off_boundaries():
    """
    Matches (various off boundary):
    ──┬───┬───┬──
      │   │   │
    ──┼───┼───┼──
      │   │   │
    ──┴───┴───┴──
    """
    code = PlanarCode(3, 4)
    grid = PlanarCMWPMDecoder.StepGrid(code)
    # set background from matches
    grid.set_background({((0, -1), (2, 3))}, factor=3, initial=1, box_shape='l')  # top-off-left to bulk
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[1., 0., 1., 0., 1., 0., 3., 0., 3.],
         [0., 1., 0., 1., 0., 1., 0., 3., 0.],
         [1., 0., 1., 0., 1., 0., 3., 0., 3.],
         [0., 1., 0., 1., 0., 1., 0., 3., 0.],
         [1., 0., 1., 0., 1., 0., 3., 0., 3.],
         [0., 3., 0., 3., 0., 3., 0., 3., 0.],
         [3., 0., 3., 0., 3., 0., 3., 0., 3.]])
    assert np.array_equal(grid._grid, expected), 'Loose box not expected shape.'
    # set background from matches
    grid.set_background({((2, 3), (4, 7))}, factor=3, initial=1, box_shape='l')  # bulk to bottom-off-right
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[3., 0., 3., 0., 3., 0., 3., 0., 3.],
         [0., 3., 0., 3., 0., 3., 0., 3., 0.],
         [3., 0., 3., 0., 1., 0., 1., 0., 1.],
         [0., 3., 0., 1., 0., 1., 0., 1., 0.],
         [3., 0., 3., 0., 1., 0., 1., 0., 1.],
         [0., 3., 0., 1., 0., 1., 0., 1., 0.],
         [3., 0., 3., 0., 1., 0., 1., 0., 1.]])
    assert np.array_equal(grid._grid, expected), 'Loose box not expected shape.'
    # set background from matches
    grid.set_background({((-1, 0), (3, 2))}, factor=3, initial=1, box_shape='l')  # left-off-top to bulk
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[1., 0., 1., 0., 1., 0., 3., 0., 3.],
         [0., 1., 0., 1., 0., 3., 0., 3., 0.],
         [1., 0., 1., 0., 1., 0., 3., 0., 3.],
         [0., 1., 0., 1., 0., 3., 0., 3., 0.],
         [1., 0., 1., 0., 1., 0., 3., 0., 3.],
         [0., 1., 0., 1., 0., 3., 0., 3., 0.],
         [3., 0., 3., 0., 3., 0., 3., 0., 3.]])
    assert np.array_equal(grid._grid, expected), 'Loose box not expected shape.'
    # set background from matches
    grid.set_background({((1, 4), (5, 6))}, factor=3, initial=1, box_shape='l')  # bulk to right-off-bottom
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[3., 0., 3., 0., 3., 0., 3., 0., 3.],
         [0., 3., 0., 3., 0., 1., 0., 1., 0.],
         [3., 0., 3., 0., 1., 0., 1., 0., 1.],
         [0., 3., 0., 3., 0., 1., 0., 1., 0.],
         [3., 0., 3., 0., 1., 0., 1., 0., 1.],
         [0., 3., 0., 3., 0., 1., 0., 1., 0.],
         [3., 0., 3., 0., 1., 0., 1., 0., 1.]])
    assert np.array_equal(grid._grid, expected), 'Loose box not expected shape.'
    # set background from matches
    grid.set_background({((0, -1), (4, 7))}, factor=3, initial=1, box_shape='l')  # top-off-left to bottom-off-right
    # expected grid. Note: border of virtual indices around grid.
    expected = np.array(
        [[1., 0., 1., 0., 1., 0., 1., 0., 1.],
         [0., 1., 0., 1., 0., 1., 0., 1., 0.],
         [1., 0., 1., 0., 1., 0., 1., 0., 1.],
         [0., 1., 0., 1., 0., 1., 0., 1., 0.],
         [1., 0., 1., 0., 1., 0., 1., 0., 1.],
         [0., 1., 0., 1., 0., 1., 0., 1., 0.],
         [1., 0., 1., 0., 1., 0., 1., 0., 1.]])
    assert np.array_equal(grid._grid, expected), 'Loose box not expected shape.'


def test_planar_cmwpm_step_distance():
    """
    Grid:
    [[3 0 3 0 3 0 3 0 3]
     [0 3 0 1 0 1 0 3 0]
     [3 0 1 0 1 0 1 0 3]
     [0 3 0 1 0 1 0 3 0]
     [3 0 1 0 1 0 1 0 3]
     [0 3 0 1 0 1 0 3 0]
     [3 0 3 0 3 0 3 0 3]
     [0 3 0 3 0 3 0 3 0]
     [3 0 3 0 3 0 3 0 3]]
    """
    f = 3  # factor
    i = 1  # initial
    code = PlanarCode(4, 4)
    # prepare grid distance 1: v+h (i.e. vertical + horizontal)
    grid = PlanarCMWPMDecoder.StepGrid(code)
    grid.set_background({((0, 1), (4, 5))}, factor=f, initial=i, box_shape='t')
    assert grid.distance((1, 0), (5, 4), algorithm=1) == 2 * f + 2 * f, 'Distance 1 left-to-bottom not as expected'
    assert grid.distance((5, 4), (1, 0), algorithm=1) == 2 * i + 2 * i, 'Distance 1 bottom-to-left not as expected'
    assert grid.distance((-1, 2), (3, 6), algorithm=1) == 2 * i + 2 * i, 'Distance 1 top-to-right not as expected'
    assert grid.distance((3, 6), (-1, 2), algorithm=1) == 2 * f + 2 * f, 'Distance 1 right-to-top not as expected'
    assert grid.distance((1, 0), (3, 6), algorithm=1) == 1 * f + 3 * i, 'Distance 1 left-to-right not as expected'
    assert grid.distance((3, 6), (1, 0), algorithm=1) == 1 * f + 3 * i, 'Distance 1 right-to-left not as expected'
    assert grid.distance((-1, 2), (5, 4), algorithm=1) == 3 * i + 1 * f, 'Distance 1 top-to-bottom not as expected'
    assert grid.distance((5, 4), (-1, 2), algorithm=1) == 1 * f + 3 * i, 'Distance 1 bottom-to-top not as expected'
    # prepare grid distance 2: min(v+h, h+v)
    grid = PlanarCMWPMDecoder.StepGrid(code)
    grid.set_background({((0, 1), (4, 5))}, factor=3, initial=1, box_shape='t')
    assert grid.distance((1, 0), (5, 4), algorithm=2) == 2 * i + 2 * i, 'Distance 2 left-to-bottom not as expected'
    assert grid.distance((5, 4), (1, 0), algorithm=2) == 2 * i + 2 * i, 'Distance 2 bottom-to-left not as expected'
    assert grid.distance((-1, 2), (3, 6), algorithm=2) == 2 * i + 2 * i, 'Distance 2 top-to-right not as expected'
    assert grid.distance((3, 6), (-1, 2), algorithm=2) == 2 * i + 2 * i, 'Distance 2 right-to-top not as expected'
    assert grid.distance((1, 0), (3, 6), algorithm=2) == 3 * i + 1 * f, 'Distance 2 left-to-right not as expected'
    assert grid.distance((3, 6), (1, 0), algorithm=2) == 3 * i + 1 * f, 'Distance 2 right-to-left not as expected'
    assert grid.distance((-1, 2), (5, 4), algorithm=2) == 3 * i + 1 * f, 'Distance 2 top-to-bottom not as expected'
    assert grid.distance((5, 4), (-1, 2), algorithm=2) == 3 * i + 1 * f, 'Distance 2 bottom-to-top not as expected'
    # prepare grid distance 4: min(v+h, h+v, v+h+v, h+v+h)
    grid = PlanarCMWPMDecoder.StepGrid(code)
    grid.set_background({((0, 1), (4, 5))}, factor=3, initial=1, box_shape='t')
    assert grid.distance((1, 0), (5, 4), algorithm=4) == 4 * 1, 'Distance 4 left-to-bottom not as expected'
    assert grid.distance((5, 4), (1, 0), algorithm=4) == 4 * i, 'Distance 4 bottom-to-left not as expected'
    assert grid.distance((-1, 2), (3, 6), algorithm=4) == 4 * i, 'Distance 4 top-to-right not as expected'
    assert grid.distance((3, 6), (-1, 2), algorithm=4) == 4 * i, 'Distance 4 right-to-top not as expected'
    assert grid.distance((1, 0), (3, 6), algorithm=4) == 4 * i, 'Distance 4 left-to-right not as expected'
    assert grid.distance((3, 6), (1, 0), algorithm=4) == 4 * i, 'Distance 4 right-to-left not as expected'
    assert grid.distance((-1, 2), (5, 4), algorithm=4) == 4 * i, 'Distance 4 top-to-bottom not as expected'
    assert grid.distance((5, 4), (-1, 2), algorithm=4) == 4 * i, 'Distance 4 bottom-to-top not as expected'


def test_planar_cmwpm_decoder_overflow(caplog):
    """
    Error:
    ·─┬─·─┬─·─┬─·─┬─·
      ·   ·   ·   ·
    ·─┼─Y─┼─·─┼─·─┼─·
      ·   Y   ·   ·
    ·─┼─·─┼─·─┼─·─┼─·
      ·   ·   ·   ·
    ·─┼─·─┼─·─┼─Z─┼─·
      ·   ·   ·   ·
    ·─┴─·─┴─·─┴─·─┴─·

    Syndrome:
    ──┬───┬───┬───┬──
      │ Z │   │   │
    ──X───┼───┼───┼──
      │   │ Z │   │
    ──┼───X───┼───┼──
      │   │   │   │
    ──┼───┼───X───X──
      │   │   │   │
    ──┴───┴───┴───┴──
    """
    factor = 1e+308  # This is just a bit smaller than max float, so it causes overflow with multiple matched indices
    code = PlanarCode(5, 5)
    error = code.new_pauli().site('Y', (2, 2), (3, 3)).site('Z', (6, 6)).to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    decoder_cmwpm = PlanarCMWPMDecoder(factor=factor, max_iterations=4, box_shape='t', distance_algorithm=4)
    # show cmwpm succeeds
    print()
    recovery = decoder_cmwpm.decode(code, syndrome)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'CMWPM recovery does not commute with stabilizers.')
    assert np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'CMWPM recovery does not commute with logicals.')
    assert 'FPE RAISED FloatingPointError' in caplog.text, 'FloatingPointError not logged'
    print(caplog.text)
