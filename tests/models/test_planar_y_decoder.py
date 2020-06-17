import itertools
import math
import time

import numpy as np
import pytest
from mpmath import mp

from qecsim import paulitools as pt
from qecsim.models.planar import PlanarCode, PlanarYDecoder


def _is_close(a, b, rtol=1e-05, atol=1e-08):
    # np.isclose for mp.mpf, i.e. absolute(a - b) <= (atol + rtol * absolute(b))
    try:
        return [mp.almosteq(l, r, rel_eps=rtol, abs_eps=atol) for l, r in itertools.zip_longest(a, b)]
    except TypeError:
        return mp.almosteq(a, b, rel_eps=rtol, abs_eps=atol)


def test_planar_y_decoder_properties():
    decoder = PlanarYDecoder()
    assert isinstance(decoder.label, str)
    assert isinstance(repr(decoder), str)
    assert isinstance(str(decoder), str)


@pytest.mark.parametrize('code, syndrome_index, expected_syndrome_indices', [
    # 4x3 codes: fills down
    # below is for Y at (6, 2)
    (PlanarCode(4, 3), (5, 2), {(5, 2), (6, 1), (6, 3)}),  # P in bulk
    (PlanarCode(4, 3), (6, 1), set()),  # V on lower boundary
    (PlanarCode(4, 3), (6, 3), set()),  # V on lower boundary
    (PlanarCode(4, 3), (7, 2), set()),  # P beyond lower boundary
    # below is for Y at (0, 2)
    (PlanarCode(4, 3), (-1, 2), set()),  # P beyond upper boundary
    (PlanarCode(4, 3), (0, 1), {(0, 1), (6, 3)}),  # V on upper boundary
    (PlanarCode(4, 3), (0, 3), {(0, 3), (6, 1)}),  # V on upper boundary
    (PlanarCode(4, 3), (1, 2), {(1, 2)}),  # P in bulk
    # below is for Y at (2, 2)
    (PlanarCode(4, 3), (1, 2), {(1, 2)}),  # P in bulk
    (PlanarCode(4, 3), (2, 1), {(2, 1), (6, 3)}),  # V in bulk
    (PlanarCode(4, 3), (2, 3), {(2, 3), (6, 1)}),  # V in bulk
    (PlanarCode(4, 3), (3, 2), {(3, 2), (6, 1), (6, 3)}),  # P in bulk
    # below is for Y at (2, 0)
    (PlanarCode(4, 3), (1, 0), {(1, 0)}),  # P on left boundary
    (PlanarCode(4, 3), (2, -1), set()),  # V beyond left boundary
    (PlanarCode(4, 3), (2, 1), {(2, 1), (6, 3)}),  # V in bulk
    (PlanarCode(4, 3), (3, 0), {(3, 0), (6, 3)}),  # P on left boundary
    # below is for Y at (2, 4)
    (PlanarCode(4, 3), (1, 4), {(1, 4)}),  # P on right boundary
    (PlanarCode(4, 3), (2, 3), {(2, 3), (6, 1)}),  # V in bulk
    (PlanarCode(4, 3), (2, 5), set()),  # V beyond right boundary
    (PlanarCode(4, 3), (3, 4), {(3, 4), (6, 1)}),  # P on left boundary
    # 3x4 codes: fills right
    # below is for Y at (2, 6)
    (PlanarCode(3, 4), (2, 5), {(2, 5), (1, 6), (3, 6)}),  # V in bulk
    (PlanarCode(3, 4), (1, 6), set()),  # P on right boundary
    (PlanarCode(3, 4), (3, 6), set()),  # P on right boundary
    (PlanarCode(3, 4), (2, 7), set()),  # V beyond right boundary
    # below is for Y at (2, 0)
    (PlanarCode(3, 4), (2, -1), set()),  # V beyond left boundary
    (PlanarCode(3, 4), (1, 0), {(1, 0), (3, 6)}),  # P on left boundary
    (PlanarCode(3, 4), (3, 0), {(3, 0), (1, 6)}),  # P on left boundary
    (PlanarCode(3, 4), (2, 1), {(2, 1)}),  # V in bulk
    # below is for Y at (2, 2)
    (PlanarCode(3, 4), (2, 1), {(2, 1)}),  # V in bulk
    (PlanarCode(3, 4), (1, 2), {(1, 2), (3, 6)}),  # P in bulk
    (PlanarCode(3, 4), (3, 2), {(3, 2), (1, 6)}),  # P in bulk
    (PlanarCode(3, 4), (2, 3), {(2, 3), (1, 6), (3, 6)}),  # V in bulk
    # below is for Y at (0, 2)
    (PlanarCode(3, 4), (0, 1), {(0, 1)}),  # V on upper boundary
    (PlanarCode(3, 4), (-1, 2), set()),  # P beyond upper boundary
    (PlanarCode(3, 4), (1, 2), {(1, 2), (3, 6)}),  # P in bulk
    (PlanarCode(3, 4), (0, 3), {(0, 3), (3, 6)}),  # V on upper boundary
    # # below is for Y at (4, 2)
    (PlanarCode(3, 4), (4, 1), {(4, 1)}),  # V on lower boundary
    (PlanarCode(3, 4), (3, 2), {(3, 2), (1, 6)}),  # P in bulk
    (PlanarCode(3, 4), (5, 2), set()),  # P beyond lower boundary
    (PlanarCode(3, 4), (4, 3), {(4, 3), (1, 6)}),  # V on lower boundary
])
def test_planar_y_decoder_partial_recovery(code, syndrome_index, expected_syndrome_indices):
    partial_recovery = PlanarYDecoder._partial_recovery(code, syndrome_index)
    print()
    print(code.new_pauli(partial_recovery))
    syndrome = pt.bsp(partial_recovery, code.stabilizers.T)
    syndrome_indices = code.syndrome_to_plaquette_indices(syndrome)
    assert syndrome_indices == expected_syndrome_indices, 'syndrome_indices not as expected'


@pytest.mark.parametrize('code, syndrome_index, expected_syndrome_indices', [
    # 4x3 codes
    # below is for Y at (6, 2)
    (PlanarCode(4, 3), (5, 2), {(5, 2)}),  # P in bulk
    (PlanarCode(4, 3), (6, 1), {(6, 1)}),  # V on lower boundary
    (PlanarCode(4, 3), (6, 3), {(6, 3)}),  # V on lower boundary
    (PlanarCode(4, 3), (7, 2), set()),  # P beyond lower boundary
    # below is for Y at (0, 2)
    (PlanarCode(4, 3), (-1, 2), set()),  # P beyond upper boundary
    (PlanarCode(4, 3), (0, 1), {(0, 1)}),  # V on upper boundary
    (PlanarCode(4, 3), (0, 3), {(0, 3)}),  # V on upper boundary
    (PlanarCode(4, 3), (1, 2), {(1, 2)}),  # P in bulk
    # below is for Y at (2, 2)
    (PlanarCode(4, 3), (1, 2), {(1, 2)}),  # P in bulk
    (PlanarCode(4, 3), (2, 1), {(2, 1)}),  # V in bulk
    (PlanarCode(4, 3), (2, 3), {(2, 3)}),  # V in bulk
    (PlanarCode(4, 3), (3, 2), {(3, 2)}),  # P in bulk
    # below is for Y at (2, 0)
    (PlanarCode(4, 3), (1, 0), {(1, 0)}),  # P on left boundary
    (PlanarCode(4, 3), (2, -1), set()),  # V beyond left boundary
    (PlanarCode(4, 3), (2, 1), {(2, 1)}),  # V in bulk
    (PlanarCode(4, 3), (3, 0), {(3, 0)}),  # P on left boundary
    # # below is for Y at (2, 4)
    (PlanarCode(4, 3), (1, 4), {(1, 4)}),  # P on right boundary
    (PlanarCode(4, 3), (2, 3), {(2, 3)}),  # V in bulk
    (PlanarCode(4, 3), (2, 5), set()),  # V beyond right boundary
    (PlanarCode(4, 3), (3, 4), {(3, 4)}),  # P on left boundary
    # 3x4 codes
    # below is for Y at (2, 6)
    (PlanarCode(3, 4), (2, 5), {(2, 5)}),  # V in bulk
    (PlanarCode(3, 4), (1, 6), {(1, 6)}),  # P on right boundary
    (PlanarCode(3, 4), (3, 6), {(3, 6)}),  # P on right boundary
    (PlanarCode(3, 4), (2, 7), set()),  # V beyond right boundary
    # below is for Y at (2, 0)
    (PlanarCode(3, 4), (2, -1), set()),  # V beyond left boundary
    (PlanarCode(3, 4), (1, 0), {(1, 0)}),  # P on left boundary
    (PlanarCode(3, 4), (3, 0), {(3, 0)}),  # P on left boundary
    (PlanarCode(3, 4), (2, 1), {(2, 1)}),  # V in bulk
    # below is for Y at (2, 2)
    (PlanarCode(3, 4), (2, 1), {(2, 1)}),  # V in bulk
    (PlanarCode(3, 4), (1, 2), {(1, 2)}),  # P in bulk
    (PlanarCode(3, 4), (3, 2), {(3, 2)}),  # P in bulk
    (PlanarCode(3, 4), (2, 3), {(2, 3)}),  # V in bulk
    # below is for Y at (0, 2)
    (PlanarCode(3, 4), (0, 1), {(0, 1)}),  # V on upper boundary
    (PlanarCode(3, 4), (-1, 2), set()),  # P beyond upper boundary
    (PlanarCode(3, 4), (1, 2), {(1, 2)}),  # P in bulk
    (PlanarCode(3, 4), (0, 3), {(0, 3)}),  # V on upper boundary
    # # below is for Y at (4, 2)
    (PlanarCode(3, 4), (4, 1), {(4, 1)}),  # V on lower boundary
    (PlanarCode(3, 4), (3, 2), {(3, 2)}),  # P in bulk
    (PlanarCode(3, 4), (5, 2), set()),  # P beyond lower boundary
    (PlanarCode(3, 4), (4, 3), {(4, 3)}),  # V on lower boundary
])
def test_planar_y_decoder_destabilizer(code, syndrome_index, expected_syndrome_indices):
    destabilizer = PlanarYDecoder._destabilizer(code, syndrome_index)
    print()
    print(code.new_pauli(destabilizer))
    syndrome = pt.bsp(destabilizer, code.stabilizers.T)
    syndrome_indices = code.syndrome_to_plaquette_indices(syndrome)
    print(syndrome_indices)
    assert syndrome_indices == expected_syndrome_indices, 'syndrome_indices not as expected'


@pytest.mark.parametrize('code, syndrome_index', [
    (PlanarCode(4, 4), (5, 2)),  # square code
    (PlanarCode(4, 6), (5, 2)),  # gcd=2 code
])
def test_planar_y_decoder_destabilizer_invalid_code(code, syndrome_index):
    with pytest.raises(ValueError):
        PlanarYDecoder._destabilizer(code, syndrome_index)


@pytest.mark.parametrize('code, syndrome_index', [
    (PlanarCode(4, 3), (-1, 0)),  # syndrome beyond upper boundary
    (PlanarCode(4, 3), (-1, 2)),  # syndrome beyond upper boundary
    (PlanarCode(4, 3), (-1, 4)),  # syndrome beyond upper boundary
    (PlanarCode(4, 3), (7, 2)),  # syndrome beyond lower boundary
])
def test_planar_y_decoder_residual_recovery(code, syndrome_index):
    partial_recovery = PlanarYDecoder._partial_recovery(code, syndrome_index)
    syndrome = pt.bsp(partial_recovery, code.stabilizers.T)
    residual_recovery = PlanarYDecoder._residual_recovery(code, syndrome)
    assert not np.any(partial_recovery ^ residual_recovery), 'Residual recovery does not match partial recovery'


@pytest.mark.parametrize('error_pauli', [
    # co-prime
    (PlanarCode(4, 3).new_pauli().site('Y', (6, 2))),
    (PlanarCode(4, 3).new_pauli().site('Y', (0, 2))),
    (PlanarCode(4, 3).new_pauli().site('Y', (0, 0), (1, 1), (2, 2))),
    (PlanarCode(3, 4).new_pauli().site('Y', (4, 2))),
    (PlanarCode(3, 4).new_pauli().site('Y', (0, 2))),
    (PlanarCode(3, 4).new_pauli().site('Y', (0, 0), (1, 1), (2, 2))),
    # square
    (PlanarCode(4, 4).new_pauli().site('Y', (6, 2))),
    (PlanarCode(4, 4).new_pauli().site('Y', (0, 2))),
    (PlanarCode(4, 4).new_pauli().site('Y', (0, 0), (1, 1), (2, 2))),
    # side multiples
    (PlanarCode(8, 4).new_pauli().site('Y', (14, 2))),
    (PlanarCode(8, 4).new_pauli().site('Y', (0, 2))),
    (PlanarCode(8, 4).new_pauli().site('Y', (0, 0), (1, 1), (2, 2))),
    (PlanarCode(4, 8).new_pauli().site('Y', (6, 2))),
    (PlanarCode(4, 8).new_pauli().site('Y', (0, 2))),
    (PlanarCode(4, 8).new_pauli().site('Y', (0, 0), (1, 1), (2, 2))),

])
def test_planar_y_decoder_sample_recovery_sans_residual(error_pauli):
    def _mock_residual_recovery(code, syndrome):
        assert False, 'Residual recovery called unexpectedly'

    real_residual_recovery = PlanarYDecoder._residual_recovery
    try:
        # mock function
        PlanarYDecoder._residual_recovery = _mock_residual_recovery

        error = error_pauli.to_bsf()
        code = error_pauli.code
        syndrome = pt.bsp(error, code.stabilizers.T)
        recovery = PlanarYDecoder._sample_recovery(code, syndrome)
        print()
        print(code.ascii_art(syndrome, code.new_pauli(recovery)))
        assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
            'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
        assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
            'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
    finally:
        # restore real function
        PlanarYDecoder._residual_recovery = real_residual_recovery


@pytest.mark.parametrize('error_pauli', [
    # gcd=2
    (PlanarCode(4, 6).new_pauli().site('Y', (6, 2))),
    (PlanarCode(4, 6).new_pauli().site('Y', (0, 2))),
    (PlanarCode(4, 6).new_pauli().site('Y', (0, 0), (1, 1), (2, 2))),
    (PlanarCode(6, 4).new_pauli().site('Y', (10, 2))),
    (PlanarCode(6, 4).new_pauli().site('Y', (0, 2))),
    (PlanarCode(6, 4).new_pauli().site('Y', (0, 0), (1, 1), (2, 2))),
])
def test_planar_y_decoder_sample_recovery_with_residual(error_pauli):
    error = error_pauli.to_bsf()
    code = error_pauli.code
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery = PlanarYDecoder._sample_recovery(code, syndrome)
    print()
    print(error_pauli)
    print(code.ascii_art(syndrome, code.new_pauli(recovery)))
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


@pytest.mark.parametrize('code', [
    PlanarCode(3, 4),
    PlanarCode(2, 4),
    PlanarCode(3, 6),
    PlanarCode(4, 4),
    PlanarCode(4, 3),
    PlanarCode(4, 2),
    PlanarCode(6, 3),
])
def test_planar_y_decoder_y_stabilizers(code):
    y_stabilizers = PlanarYDecoder._y_stabilizers(code)
    code_gcd = math.gcd(*code.size)
    assert len(y_stabilizers) == 2 ** (code_gcd - 1), 'There are not 2^(gcd(p,q)-1) y-stabilizers'
    assert not np.any(pt.bsp(y_stabilizers, code.stabilizers.T)), 'Y-stabilizers do not commute with code stabilizers'
    assert not np.any(pt.bsp(y_stabilizers, code.logicals.T)), 'Y-stabilizers do not commute with code logicals'


@pytest.mark.parametrize('code', [
    PlanarCode(3, 4),
    PlanarCode(2, 4),
    PlanarCode(3, 6),
    PlanarCode(4, 4),
    PlanarCode(4, 3),
    PlanarCode(4, 2),
    PlanarCode(6, 3),
])
def test_planar_y_decoder_y_logical(code):
    y_logical = PlanarYDecoder._y_logical(code)
    assert not np.any(pt.bsp(y_logical, code.stabilizers.T)), 'Y-logical does not commute with code stabilizers'
    assert np.any(pt.bsp(y_logical, code.logicals.T)), 'Y-logical does commute with code logicals'


@pytest.mark.parametrize('prob_dist, pauli, expected_probability_text', [
    # worked by hand
    ((0.9, 0.0, 0.1, 0.0), PlanarCode(4, 3).new_pauli().site('Y', (6, 2)),
     '0.016677181699666576920472863220495089151524560126694432'),  # value from good run to prevent regression
    ((0.9, 0.0, 0.1, 0.0), PlanarCode(4, 3).new_pauli().site('Y', (0, 2)),
     '0.016677181699666576920472863220495089151524560126694432'),
    ((0.9, 0.0, 0.1, 0.0), PlanarCode(4, 3).new_pauli().site('Y', (0, 0), (1, 1), (2, 2)),
     '0.00020589113209464911048278645184914092536454811793011564'),
    # half horizontal edges
    ((0.9, 0.0, 0.1, 0.0), PlanarCode(2, 3).new_pauli().site('Y', (0, 0), (0, 2), (2, 4)),
     '0.00059049000000000017117807171729284463400321296383646319'),
    # square
    ((0.9, 0.0, 0.1, 0.0), PlanarCode(3, 3).new_pauli().site('Y', (0, 0)),
     '0.02824306059240000992945027840086584809960952996235049'),
    ((0.9, 0.0, 0.1, 0.0), PlanarCode(5, 5).new_pauli().site('Y', (0, 2), (1, 3), (2, 4)),
     '0.000018248003636636630722305203424093067093845450849262954'),
    ((0.9, 0.0, 0.1, 0.0), PlanarCode(5, 5).new_pauli().site('Y', (0, 0), (1, 1), (2, 2), (3, 3)),
     '0.0000020275560061461695431500176557622487264372635511740519'),
    # gcd = 3
    ((0.9, 0.0, 0.1, 0.0), PlanarCode(3, 6).new_pauli().site('Y', (0, 0)),
     '0.0058149737003045216766571418843054710209596058424128056'),
    ((0.9, 0.0, 0.1, 0.0), PlanarCode(6, 9).new_pauli().site('Y', (0, 2), (1, 3), (2, 4)),
     '0.000000068559613241279918447229206765612507675623798486356152'),
])
def test_planar_y_decoder_coset_probability(prob_dist, pauli, expected_probability_text):
    with mp.workdps(50):
        expected_probability = mp.mpf(expected_probability_text)
        # calculate coset probabilities
        y_stabilizers = PlanarYDecoder._y_stabilizers(pauli.code)
        coset = y_stabilizers ^ pauli.to_bsf()  # numpy broadcasting applies recovery to all stabilizers
        coset_probability = PlanarYDecoder._coset_probability(prob_dist, coset)
        print(repr(coset_probability), repr(expected_probability))
        assert _is_close(expected_probability, coset_probability, rtol=1e-50, atol=0), (
            'Coset probability not as expected')


def test_planar_y_decoder_coset_probability_performance():
    print()
    with mp.workdps(50):
        n_run = 20
        code = PlanarCode(16, 16)  # 16, 16
        prob_dist = (0.9, 0.0, 0.1, 0.0)
        # preload stabilizer cache
        PlanarYDecoder._y_stabilizers(code)
        # time runs
        start_time = time.time()
        for run in range(n_run):
            coset = PlanarYDecoder._y_stabilizers(code)
            coset_probability = PlanarYDecoder._coset_probability(prob_dist, coset)
        print(repr(coset_probability))
        run_time = time.time() - start_time
        print('run_time = {}'.format(run_time))
        # test to avoid regression
        assert run_time < 6  # 5.423123121261597


@pytest.mark.parametrize('error_pauli', [
    # co-prime
    (PlanarCode(4, 3).new_pauli().site('Y', (6, 2))),
    (PlanarCode(4, 3).new_pauli().site('Y', (0, 2))),
    (PlanarCode(4, 3).new_pauli().site('Y', (0, 0), (1, 1), (2, 2))),
    (PlanarCode(3, 4).new_pauli().site('Y', (4, 2))),
    (PlanarCode(3, 4).new_pauli().site('Y', (0, 2))),
    (PlanarCode(3, 4).new_pauli().site('Y', (0, 0), (1, 1), (2, 2))),
    # square
    (PlanarCode(3, 3).new_pauli().site('Y', (0, 0))),
    (PlanarCode(5, 5).new_pauli().site('Y', (0, 2), (1, 3), (2, 4))),
    (PlanarCode(5, 5).new_pauli().site('Y', (0, 0), (1, 1), (2, 2), (3, 3))),
    # gcd=2
    (PlanarCode(4, 6).new_pauli().site('Y', (6, 2))),
    (PlanarCode(4, 6).new_pauli().site('Y', (0, 2))),
    (PlanarCode(4, 6).new_pauli().site('Y', (0, 0), (1, 1), (2, 2))),
    (PlanarCode(6, 4).new_pauli().site('Y', (10, 2))),
    (PlanarCode(6, 4).new_pauli().site('Y', (0, 2))),
    (PlanarCode(6, 4).new_pauli().site('Y', (0, 0), (1, 1), (2, 2))),
])
def test_planar_y_decoder_decode(error_pauli):
    error = error_pauli.to_bsf()
    code = error_pauli.code
    decoder = PlanarYDecoder()
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery = decoder.decode(code, syndrome)
    print()
    print(code.new_pauli(recovery))
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with logicals.'.format(recovery, error))


def test_planar_y_decoder_decode_equal_coset_probabilities():
    code = PlanarCode(2, 3)
    decoder = PlanarYDecoder()
    # The following error Pauli gives identical coset probabilities:
    # I-+-I-+-Y
    #   I   I
    # Y-+-Y-+-I
    # So we expect approximately equal success and failure
    error_pauli = PlanarCode(2, 3).new_pauli().site('Y', (2, 0), (2, 2), (0, 4))
    # count success and fail
    success, fail = 0, 0
    # run simulations
    error = error_pauli.to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    for i in range(2000):
        recovery = decoder.decode(code, syndrome)
        assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
            'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
        assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
            'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
        if np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0):
            success += 1
        else:
            fail += 1
    assert _is_close(success, fail, rtol=0.2,
                     atol=0), 'Success and fail not equally likely with equal coset probabilities'


def test_planar_y_decoder_partial_recovery_idempotence():
    # tests for bug where destabilizer modified cached return value of _snake_fill
    code = PlanarCode(4, 3)
    syndrome_index = (2, 1)
    # create partial_recovery1, destabilzer and partial_recovery2, copying to test for changes
    partial_recovery1 = np.copy(PlanarYDecoder._partial_recovery(code, syndrome_index))
    destabilizer = np.copy(PlanarYDecoder._destabilizer(code, syndrome_index))
    partial_recovery2 = np.copy(PlanarYDecoder._partial_recovery(code, syndrome_index))
    print(code.new_pauli(partial_recovery1))
    print(code.new_pauli(destabilizer))
    print(code.new_pauli(partial_recovery2))
    assert np.array_equal(partial_recovery1, partial_recovery2), '_partial_recovery is not idempotent'
    assert not np.array_equal(partial_recovery2, destabilizer), '_partial_recovery == _destabilizer'
