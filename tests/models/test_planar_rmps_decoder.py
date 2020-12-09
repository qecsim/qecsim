import itertools
import logging
import time

import numpy as np
import pytest
from mpmath import mp

from qecsim import paulitools as pt
from qecsim.models.generic import DepolarizingErrorModel, BiasedDepolarizingErrorModel
from qecsim.models.planar import PlanarCode, PlanarRMPSDecoder, PlanarMWPMDecoder, PlanarMPSDecoder


def _is_close(a, b, rtol=1e-05, atol=1e-08):
    # np.isclose for mp.mpf, i.e. absolute(a - b) <= (atol + rtol * absolute(b))
    try:
        return [mp.almosteq(le, ri, rel_eps=rtol, abs_eps=atol) for le, ri in itertools.zip_longest(a, b)]
    except TypeError:
        return mp.almosteq(a, b, rel_eps=rtol, abs_eps=atol)


# @pytest.mark.parametrize('error_pauli, chi', [
#     (PlanarCode(29, 29).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (6, 4), (1, 1)), 8),
# ])
# def test_planar_rmps_perf(error_pauli, chi):
#     with CliRunner().isolated_filesystem():
#         error = error_pauli.to_bsf()
#         code = error_pauli.code
#         decoder = PlanarRMPSDecoder(chi=chi)
#         syndrome = pt.bsp(error, code.stabilizers.T)
#         for i in range(5):
#             print('# decode ', i)
#             recovery = decoder.decode(code, syndrome)
#             assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
#                 'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
#             assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
#                 'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


def test_planar_rmps_decoder_properties():
    decoder = PlanarRMPSDecoder(chi=8, mode='r', stp=0.5, tol=1e-14)
    assert isinstance(decoder.label, str)
    assert isinstance(repr(decoder), str)
    assert isinstance(str(decoder), str)


@pytest.mark.parametrize('chi, mode, stp, tol', [
    (None, 'c', None, None),
    (6, 'c', None, None),
    (None, 'r', None, None),
    (None, 'a', None, None),
    (None, 'c', 0.5, None),
    (None, 'c', None, 0.1),
    (None, 'c', None, 1),
])
def test_planar_rmps_decoder_new_valid_parameters(chi, mode, stp, tol):
    PlanarRMPSDecoder(chi=chi, mode=mode, stp=stp, tol=tol)  # no error raised


@pytest.mark.parametrize('chi, mode, stp, tol', [
    (-1, 'c', None, None),  # invalid chi
    (0.1, 'c', None, None),  # invalid chi
    ('asdf', 'c', None, None),  # invalid chi
    (None, None, None, None),  # invalid mode
    (None, 't', None, None),  # invalid mode
    (None, 2, None, None),  # invalid mode
    (None, 'c', -0.1, None),  # invalid stp
    (None, 'c', 1.1, None),  # invalid stp
    (None, 'c', 'asdf', None),  # invalid stp
    (None, 'c', None, -1),  # invalid tol
    (None, 'c', None, 'asdf'),  # invalid tol
])
def test_planar_rmps_decoder_new_invalid_parameters(chi, mode, stp, tol):
    with pytest.raises((ValueError, TypeError), match=r"^PlanarRMPSDecoder") as exc_info:
        PlanarRMPSDecoder(chi=chi, mode=mode, stp=stp, tol=tol)
    print(exc_info)


@pytest.mark.parametrize('error_pauli', [
    PlanarCode(3, 3).new_pauli().site('X', (2, 0)).site('Y', (3, 3)),
    PlanarCode(5, 5).new_pauli().site('X', (3, 1)).site('Y', (2, 2)).site('Z', (6, 4)),
    PlanarCode(7, 7).new_pauli().site('X', (4, 2)).site('Y', (3, 3)).site('Z', (8, 4), (7, 3)),
])
def test_planar_rmps_decoder_sample_recovery(error_pauli):
    error = error_pauli.to_bsf()
    code = error_pauli.code
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery_pauli = PlanarRMPSDecoder.sample_recovery(code, syndrome)
    recovery = recovery_pauli.to_bsf()
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


@pytest.mark.parametrize('mode, rtol', [
    ('c', 1e-4),  # contract by column, tolerance is O(0.0001). Tolerance is better than with Bravyi results O(1).
    ('r', 1e-4),  # contract by row.
    ('a', 1e-4),  # averaged. Tolerance unchanged because symmetry is same for row and column.
])
def test_planar_rmps_decoder_cosets_probability_inequality(mode, rtol):
    code = PlanarCode(25, 25)
    decoder = PlanarRMPSDecoder(chi=5, mode=mode)
    # probabilities
    prob_dist = DepolarizingErrorModel().probability_distribution(0.1)
    # coset probabilities for null Pauli
    coset_ps, _ = decoder._coset_probabilities(prob_dist, code.new_pauli())
    coset_i_p, coset_x_p, coset_y_p, coset_z_p = coset_ps
    # expect Pr(IG) > Pr(XG) ~= Pr(ZG) > Pr(YG)
    print('{} > {} ~= {} > {}. rtol={}'.format(
        coset_i_p, coset_x_p, coset_z_p, coset_y_p, abs(coset_x_p - coset_z_p) / abs(coset_z_p)))
    print('types: Pr(IG):{}, Pr(XG):{}, Pr(ZG):{}, Pr(YG):{}'.format(
        type(coset_i_p), type(coset_x_p), type(coset_z_p), type(coset_y_p)))
    assert coset_i_p > coset_x_p, 'Coset probabilites do not satisfy Pr(IG) > Pr(XG)'
    assert coset_i_p > coset_z_p, 'Coset probabilites do not satisfy Pr(IG) > Pr(ZG)'
    assert _is_close(coset_x_p, coset_z_p, rtol=rtol, atol=0), 'Coset probabilites do not satisfy Pr(XG) ~= Pr(ZG)'
    assert coset_x_p > coset_y_p, 'Coset probabilites do not satisfy Pr(XG) > Pr(YG)'
    assert coset_z_p > coset_y_p, 'Coset probabilites do not satisfy Pr(ZG) > Pr(YG)'


@pytest.mark.parametrize('shape, mode', [
    ((4, 4), 'c'),
    ((3, 4), 'c'),
    ((4, 3), 'c'),
    ((4, 4), 'r'),
    ((3, 4), 'r'),
    ((4, 3), 'r'),
])
def test_planar_rmps_decoder_cosets_probability_pair_optimisation(shape, mode):
    code = PlanarCode(*shape)
    decoder = PlanarRMPSDecoder(mode=mode)
    # probabilities
    prob_dist = BiasedDepolarizingErrorModel(bias=10).probability_distribution(0.1)
    # coset probabilities for null Pauli
    coset_i_ps, _ = decoder._coset_probabilities(prob_dist, code.new_pauli())
    # X
    coset_x_ps, _ = decoder._coset_probabilities(prob_dist, code.new_pauli().logical_x())
    # expect Pr(iIG) ~= Pr(xXG)
    assert _is_close(coset_i_ps[0], coset_x_ps[1], rtol=1e-15, atol=0), (
        'Coset probabilites do not satisfy Pr(iIG) ~= Pr(xXG)')
    # expect Pr(iXG) ~= Pr(xIG)
    assert _is_close(coset_i_ps[1], coset_x_ps[0], rtol=1e-15, atol=0), (
        'Coset probabilites do not satisfy Pr(iXG) ~= Pr(xIG)')
    # expect Pr(iYG) ~= Pr(xZG)
    assert _is_close(coset_i_ps[2], coset_x_ps[3], rtol=1e-15, atol=0), (
        'Coset probabilites do not satisfy Pr(iYG) ~= Pr(xZG)')
    # expect Pr(iZG) ~= Pr(xYG)
    assert _is_close(coset_i_ps[3], coset_x_ps[2], rtol=1e-15, atol=0), (
        'Coset probabilites do not satisfy Pr(iZG) ~= Pr(xYG)')
    # Y
    coset_y_ps, _ = decoder._coset_probabilities(prob_dist, code.new_pauli().logical_x().logical_z())
    # expect Pr(iIG) ~= Pr(yYG)
    assert _is_close(coset_i_ps[0], coset_y_ps[2], rtol=1e-15, atol=0), (
        'Coset probabilites do not satisfy Pr(iIG) ~= Pr(yYG)')
    # expect Pr(iXG) ~= Pr(yZG)
    assert _is_close(coset_i_ps[1], coset_y_ps[3], rtol=1e-15, atol=0), (
        'Coset probabilites do not satisfy Pr(iXG) ~= Pr(yZG)')
    # expect Pr(iYG) ~= Pr(yIG)
    assert _is_close(coset_i_ps[2], coset_y_ps[0], rtol=1e-15, atol=0), (
        'Coset probabilites do not satisfy Pr(iYG) ~= Pr(yIG)')
    # expect Pr(iZG) ~= Pr(yXG)
    assert _is_close(coset_i_ps[3], coset_y_ps[1], rtol=1e-15, atol=0), (
        'Coset probabilites do not satisfy Pr(iZG) ~= Pr(yXG)')
    # Z
    coset_z_ps, _ = decoder._coset_probabilities(prob_dist, code.new_pauli().logical_z())
    # expect Pr(iIG) ~= Pr(zZG)
    assert _is_close(coset_i_ps[0], coset_z_ps[3], rtol=1e-15, atol=0), (
        'Coset probabilites do not satisfy Pr(iIG) ~= Pr(zZG)')
    # expect Pr(iXG) ~= Pr(zYG)
    assert _is_close(coset_i_ps[1], coset_z_ps[2], rtol=1e-15, atol=0), (
        'Coset probabilites do not satisfy Pr(iXG) ~= Pr(zYG)')
    # expect Pr(iYG) ~= Pr(zXG)
    assert _is_close(coset_i_ps[2], coset_z_ps[1], rtol=1e-15, atol=0), (
        'Coset probabilites do not satisfy Pr(iYG) ~= Pr(zXG)')
    # expect Pr(iZG) ~= Pr(zIG)
    assert _is_close(coset_i_ps[3], coset_z_ps[0], rtol=1e-15, atol=0), (
        'Coset probabilites do not satisfy Pr(iZG) ~= Pr(zIG)')


@pytest.mark.parametrize('sample_pauli_f, sample_pauli_g', [
    (PlanarCode(5, 5).new_pauli(), PlanarCode(5, 5).new_pauli()),
    (PlanarCode(5, 5).new_pauli(), PlanarCode(5, 5).new_pauli().plaquette((1, 4)).plaquette((4, 5))),
    (PlanarCode(5, 5).new_pauli().logical_x(),
     PlanarCode(5, 5).new_pauli().logical_x().plaquette((0, 5)).plaquette((2, 5)).plaquette((4, 5))),
    (PlanarCode(5, 5).new_pauli().logical_z(),
     PlanarCode(5, 5).new_pauli().logical_z().plaquette((3, 0)).plaquette((3, 2)).plaquette((3, 4))),
])
def test_planar_rmps_decoder_cosets_probability_equivalence(sample_pauli_f, sample_pauli_g):
    decoder = PlanarRMPSDecoder(chi=8)
    # probabilities
    prob_dist = DepolarizingErrorModel().probability_distribution(0.1)
    # coset probabilities
    coset_f_ps, _ = decoder._coset_probabilities(prob_dist, sample_pauli_f)
    coset_g_ps, _ = decoder._coset_probabilities(prob_dist, sample_pauli_g)
    print('#Pr(fG)=', coset_f_ps)
    print('#Pr(gG)=', coset_g_ps)
    assert all(_is_close(coset_f_ps, coset_g_ps, rtol=1e-9, atol=0)), (
        'Coset probabilites do not satisfy Pr(fG) ~= Pr(gG)')


@pytest.mark.parametrize('error_pauli, chi', [
    (PlanarCode(2, 2).new_pauli().site('X', (0, 0)), None),
    (PlanarCode(4, 4).new_pauli().site('X', (2, 2), (4, 2)), None),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2), (4, 2)), 4),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2), (4, 2)).site('Z', (6, 4), (2, 0)), 6),
    (PlanarCode(5, 5).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (6, 4), (1, 1)), 8),
    (PlanarCode(3, 5).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (2, 4), (1, 7)), 6),
    (PlanarCode(5, 3).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (8, 4), (3, 1)), 6),
    (PlanarCode(5, 3).new_pauli().site('Y', (1, 3), (4, 2)).site('Z', (8, 4), (6, 4), (4, 4)), 6),
    (PlanarCode(5, 3).new_pauli()
     .site('Y', (1, 3), (3, 3), (5, 3))
     .site('Z', (8, 4), (6, 4), (4, 4)), 6),
    (PlanarCode(5, 3).new_pauli().site('X', (1, 3), (3, 3), (5, 3), (8, 4), (6, 4), (4, 4)), 6),
    (PlanarCode(5, 3).new_pauli().site('Y', (1, 3), (3, 3), (5, 3), (8, 4), (6, 4), (4, 4)), 6),
    (PlanarCode(5, 3).new_pauli().site('Z', (1, 3), (3, 3), (5, 3), (8, 4), (6, 4), (4, 4)), 6),
])
def test_planar_rmps_decoder_decode(error_pauli, chi, caplog):
    with caplog.at_level(logging.WARN):
        error = error_pauli.to_bsf()
        code = error_pauli.code
        syndrome = pt.bsp(error, code.stabilizers.T)
        decoder = PlanarRMPSDecoder(chi=chi)
        recovery = decoder.decode(code, syndrome)
        assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
            'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
        assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
            'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
        assert len(caplog.records) == 0, 'Unexpected log messages: {}'.format(caplog.text)


def test_planar_rmps_decoder_small_codes_exact_approx():
    code = PlanarCode(4, 4)
    exact_decoder = PlanarRMPSDecoder()
    approx_decoder = PlanarRMPSDecoder(chi=8)
    identity = code.new_pauli()
    # probabilities
    prob_dist = BiasedDepolarizingErrorModel(bias=10).probability_distribution(probability=0.1)
    # coset probabilities
    exact_coset_ps, _ = exact_decoder._coset_probabilities(prob_dist, identity)
    approx_coset_ps, _ = approx_decoder._coset_probabilities(prob_dist, identity)
    print('#exact Pr(G)=', exact_coset_ps)
    print('#approx Pr(G)=', approx_coset_ps)
    assert all(_is_close(exact_coset_ps, approx_coset_ps, rtol=1e-11, atol=0)), (
        'Coset probabilites do not satisfy exact Pr(G) ~= approx Pr(G)')


def test_planar_rmps_decoder_correlated_errors():
    # check MPS decoder successfully decodes for error
    # I--+--I--+--I
    #    I     I
    # Y--+--I--+--Y
    #    I     I
    # I--+--I--+--I
    # and MWPM decoder fails as expected
    code = PlanarCode(3, 3)
    error = code.new_pauli().site('Y', (2, 0), (2, 4)).to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    # MPS decoder
    decoder = PlanarRMPSDecoder()
    recovery = decoder.decode(code, syndrome)
    # check recovery ^ error commutes with stabilizers (by construction)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers for MPS decoder.'.format(recovery, error))
    # check recovery ^ error commutes with logicals (we expect this to succeed for MPS)
    assert np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with logicals for MPS decoder.'.format(recovery, error))
    # MWPM decoder
    decoder = PlanarMWPMDecoder()
    recovery = decoder.decode(code, syndrome)
    # check recovery ^ error commutes with stabilizers (by construction)
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers for MWPM decoder.'.format(recovery, error))
    # check recovery ^ error commutes with logicals (we expect this to fail for MWPM)
    assert not np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0), (
        'recovery ^ error ({} ^ {}) does commute with logicals for MWPM decoder.'.format(recovery, error))


def test_planar_rmps_decoder_cosets_probability_stp():
    # parameters
    sample = PlanarCode(3, 4).new_pauli().site('Y', (2, 0), (2, 4))
    prob_dist = DepolarizingErrorModel().probability_distribution(0.1)

    # coset probabilities exact
    exact_coset_ps, _ = PlanarRMPSDecoder(mode='a')._coset_probabilities(prob_dist, sample)
    print('#exact_coset_ps=', exact_coset_ps)

    # coset probabilities approx (chi=6)
    approx_coset_ps, _ = PlanarRMPSDecoder(chi=6, mode='a')._coset_probabilities(prob_dist, sample)
    print('#approx_coset_ps=', approx_coset_ps)
    assert all(_is_close(exact_coset_ps, approx_coset_ps, rtol=1e-14, atol=0)), (
        'approx_coset_ps not close to exact_coset_ps')

    # coset probabilities approx (chi=6, stp=0)
    coset_ps, _ = PlanarRMPSDecoder(chi=6, mode='a', stp=0)._coset_probabilities(prob_dist, sample)
    print('#coset_ps (chi=6, stp=0)=', coset_ps)
    assert all(_is_close(approx_coset_ps, coset_ps, rtol=0, atol=0)), (
        'coset_ps (chi=6, stp=0) not equal to approx_coset_ps')

    # coset probabilities approx (chi=6, stp=1)
    coset_ps, _ = PlanarRMPSDecoder(chi=6, mode='a', stp=1)._coset_probabilities(prob_dist, sample)
    print('#coset_ps (chi=6, stp=1)=', coset_ps)
    assert all(_is_close(exact_coset_ps, coset_ps, rtol=0, atol=0)), (
        'coset_ps (chi=6, stp=1) not equal to exact_coset_ps')

    # coset probabilities approx (chi=6, stp=0.5)
    coset_ps, _ = PlanarRMPSDecoder(chi=6, mode='a', stp=0.5)._coset_probabilities(prob_dist, sample)
    print('#coset_ps (chi=6, stp=0.5)=', coset_ps)
    assert all(_is_close(exact_coset_ps, coset_ps, rtol=1e-10, atol=0)), (
        'coset_ps (chi=6, stp=0.5) not close to exact_coset_ps')
    assert all(_is_close(approx_coset_ps, coset_ps, rtol=1e-10, atol=0)), (
        'coset_ps (chi=6, stp=0.5) not close to approx_coset_ps')


@pytest.mark.parametrize('error_pauli', [
    PlanarCode(3, 3).new_pauli().site('X', (2, 0)).site('Y', (3, 3)),
    PlanarCode(5, 5).new_pauli().site('X', (3, 1)).site('Y', (2, 2)).site('Z', (6, 4)),
    PlanarCode(7, 7).new_pauli().site('X', (4, 2)).site('Y', (3, 3)).site('Z', (8, 4), (7, 3)),
])
def test_planar_rmps_mps_accuracy(error_pauli):
    error = error_pauli.to_bsf()
    code = error_pauli.code
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery_pauli = PlanarRMPSDecoder.sample_recovery(code, syndrome)
    prob_dist = DepolarizingErrorModel().probability_distribution(0.1)
    rmps_coset_ps, _ = PlanarRMPSDecoder(chi=8)._coset_probabilities(prob_dist, recovery_pauli)
    print('#rmps_coset_ps (chi=8)=', rmps_coset_ps)
    mps_coset_ps, _ = PlanarMPSDecoder(chi=8)._coset_probabilities(prob_dist, recovery_pauli)
    print('#mps_coset_ps (chi=8)=', mps_coset_ps)
    assert all(_is_close(rmps_coset_ps, mps_coset_ps, rtol=1e-1, atol=0)), (
        'rmps_coset_ps (chi=8) not close to mps_coset_ps (chi=8)')


def test_planar_rmps_mwpm_performance():
    n_run = 5
    code = PlanarCode(25, 25)
    error_model = DepolarizingErrorModel()
    error_probability = 0.4

    def _timed_runs(decoder):
        start_time = time.time()
        for _ in range(n_run):
            error = error_model.generate(code, error_probability)
            syndrome = pt.bsp(error, code.stabilizers.T)
            recovery = decoder.decode(code, syndrome)
            assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
                'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
        return time.time() - start_time

    rmps_time = _timed_runs(PlanarRMPSDecoder(chi=8))
    mwpm_time = _timed_runs(PlanarMWPMDecoder())
    # expect mps_time < mwpm_time
    print('rmps_time = {} < {} = mwpm_time'.format(rmps_time, mwpm_time))
    assert rmps_time < mwpm_time, 'RMPS decoder slower than MWPM decoder'


def test_planar_rmps_mps_performance():
    n_run = 5
    code = PlanarCode(21, 21)
    error_model = DepolarizingErrorModel()
    error_probability = 0.2

    def _timed_runs(decoder):
        start_time = time.time()
        for _ in range(n_run):
            error = error_model.generate(code, error_probability)
            syndrome = pt.bsp(error, code.stabilizers.T)
            recovery = decoder.decode(code, syndrome)
            assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
                'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
        return time.time() - start_time

    rmps_time = _timed_runs(PlanarRMPSDecoder(chi=8))
    mps_time = _timed_runs(PlanarMPSDecoder(chi=8))
    # expect rmps_time < mps_time
    print('rmps_time = {} < {} = mps_time'.format(rmps_time, mps_time))
    assert rmps_time < mps_time, 'RMPS decoder slower than MPS decoder'
