import itertools
import logging
import time

import numpy as np
import pytest
from mpmath import mp
from qecsim import paulitools as pt
from qecsim.models.generic import BitFlipErrorModel
from qecsim.models.generic import DepolarizingErrorModel, BiasedDepolarizingErrorModel
from qecsim.models.planar import PlanarCode, PlanarMPSDecoder, PlanarMWPMDecoder


def _is_close(a, b, rtol=1e-05, atol=1e-08):
    # np.isclose for mp.mpf, i.e. absolute(a - b) <= (atol + rtol * absolute(b))
    try:
        return [mp.almosteq(le, ri, rel_eps=rtol, abs_eps=atol) for le, ri in itertools.zip_longest(a, b)]
    except TypeError:
        return mp.almosteq(a, b, rel_eps=rtol, abs_eps=atol)


# @pytest.mark.parametrize('error_pauli, chi', [
#     (PlanarCode(29, 29).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (6, 4), (1, 1)), 8),
# ])
# def test_planar_mps_perf(error_pauli, chi):
#     error = error_pauli.to_bsf()
#     code = error_pauli.code
#     decoder = PlanarMPSDecoder(chi=chi)
#     syndrome = pt.bsp(error, code.stabilizers.T)
#     for i in range(5):
#         print('# decode ', i)
#         recovery = decoder.decode(code, syndrome)
#         assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
#             'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
#         assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
#             'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


def test_planar_mps_decoder_properties():
    decoder = PlanarMPSDecoder(chi=8, mode='r', tol=1e-14)
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
    (None, 'c', None, None),
])
def test_planar_mps_decoder_new_valid_parameters(chi, mode, stp, tol):
    PlanarMPSDecoder(chi=chi, mode=mode, stp=stp, tol=tol)  # no error raised


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
def test_planar_mps_decoder_new_invalid_parameters(chi, mode, stp, tol):
    with pytest.raises((ValueError, TypeError), match=r"^PlanarMPSDecoder") as exc_info:
        PlanarMPSDecoder(chi=chi, mode=mode, stp=stp, tol=tol)
    print(exc_info)


@pytest.mark.parametrize('error_pauli', [
    PlanarCode(3, 3).new_pauli().site('X', (2, 0)).site('Y', (3, 3)),
    PlanarCode(5, 5).new_pauli().site('X', (3, 1)).site('Y', (2, 2)).site('Z', (6, 4)),
    PlanarCode(7, 7).new_pauli().site('X', (4, 2)).site('Y', (3, 3)).site('Z', (8, 4), (7, 3)),
])
def test_planar_mps_decoder_sample_recovery(error_pauli):
    error = error_pauli.to_bsf()
    code = error_pauli.code
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery_pauli = PlanarMPSDecoder._sample_recovery(code, syndrome)
    recovery = recovery_pauli.to_bsf()
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


@pytest.mark.parametrize('mode, rtol', [
    ('c', 1e-3),  # contract by column.
    ('r', 1e-3),  # contract by row.
    ('a', 0),  # averaged, zero tolerance. Tolerance should be zero because of symmetry between 'r' and 'c' modes.
])
def test_planar_mps_decoder_cosets_probability_inequality(mode, rtol):
    code = PlanarCode(13, 13)
    decoder = PlanarMPSDecoder(chi=16, mode=mode)
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


@pytest.mark.parametrize('mode, rtol', [
    ('c', 1e-0),  # contract by column, tolerance is O(1). Tolerance is consistent with BSV results.
    ('r', 1e-0),  # contract by row, tolerance is O(1). Tolerance is consistent with BSV results.
    ('a', 0),  # averaged, zero tolerance. Tolerance should be zero because of symmetry between 'r' and 'c' modes.
])
def test_planar_mps_decoder_cosets_probability_inequality_bsv(mode, rtol):
    code = PlanarCode(25, 25)
    decoder = PlanarMPSDecoder(chi=5, mode=mode)
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


@pytest.mark.parametrize('mode', [
    'c',
    'r',
])
def test_planar_mps_decoder_cosets_probability_pair_optimisation(mode):
    code = PlanarCode(5, 5)
    decoder = PlanarMPSDecoder(mode=mode)
    # probabilities
    prob_dist = BiasedDepolarizingErrorModel(bias=10).probability_distribution(probability=0.1)
    # coset probabilities for null Pauli
    coset_i_ps, _ = decoder._coset_probabilities(prob_dist, code.new_pauli())
    coset_x_ps, _ = decoder._coset_probabilities(prob_dist, code.new_pauli().logical_x())
    # expect Pr(iIG) ~= Pr(xXG)
    assert _is_close(coset_i_ps[0], coset_x_ps[1], rtol=0, atol=0), (
        'Coset probabilites do not satisfy Pr(iIG) ~= Pr(xXG)')
    # expect Pr(iXG) ~= Pr(xIG)
    assert _is_close(coset_i_ps[1], coset_x_ps[0], rtol=0, atol=0), (
        'Coset probabilites do not satisfy Pr(iXG) ~= Pr(xIG)')
    # expect Pr(iZG) ~= Pr(xYG)
    assert _is_close(coset_i_ps[3], coset_x_ps[2], rtol=0, atol=0), (
        'Coset probabilites do not satisfy Pr(iZG) ~= Pr(xYG)')
    # expect Pr(iYG) ~= Pr(xZG)
    assert _is_close(coset_i_ps[2], coset_x_ps[3], rtol=0, atol=0), (
        'Coset probabilites do not satisfy Pr(iYG) ~= Pr(xZG)')


@pytest.mark.parametrize('sample_pauli_f, sample_pauli_g', [
    (PlanarCode(5, 5).new_pauli(), PlanarCode(5, 5).new_pauli()),
    (PlanarCode(5, 5).new_pauli(), PlanarCode(5, 5).new_pauli().plaquette((1, 4)).plaquette((4, 5))),
    (PlanarCode(5, 5).new_pauli().logical_x(),
     PlanarCode(5, 5).new_pauli().logical_x().plaquette((0, 5)).plaquette((2, 5)).plaquette((4, 5))),
    (PlanarCode(5, 5).new_pauli().logical_z(),
     PlanarCode(5, 5).new_pauli().logical_z().plaquette((3, 0)).plaquette((3, 2)).plaquette((3, 4))),
])
def test_planar_mps_decoder_cosets_probability_equivalence(sample_pauli_f, sample_pauli_g):
    decoder = PlanarMPSDecoder(chi=8)
    # probabilities
    prob_dist = BiasedDepolarizingErrorModel(bias=10).probability_distribution(probability=0.1)
    # coset probabilities
    coset_f_ps, _ = decoder._coset_probabilities(prob_dist, sample_pauli_f)
    coset_g_ps, _ = decoder._coset_probabilities(prob_dist, sample_pauli_g)
    print('#Pr(fG)=', coset_f_ps)
    print('#Pr(gG)=', coset_g_ps)
    assert all(_is_close(coset_f_ps, coset_g_ps, rtol=1e-12, atol=0)), (
        'Coset probabilites do not satisfy Pr(fG) ~= Pr(gG)')


@pytest.mark.parametrize('error_pauli, chi', [
    (PlanarCode(2, 2).new_pauli().site('X', (0, 0)), None),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2), (4, 2)), None),
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
def test_planar_mps_decoder_decode(error_pauli, chi, caplog):
    with caplog.at_level(logging.WARN):
        error = error_pauli.to_bsf()
        code = error_pauli.code
        syndrome = pt.bsp(error, code.stabilizers.T)
        decoder = PlanarMPSDecoder(chi=chi)
        recovery = decoder.decode(code, syndrome)
        assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
            'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
        assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
            'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
        assert len(caplog.records) == 0, 'Unexpected log messages: {}'.format(caplog.text)


def test_planar_mps_decoder_small_codes_exact_approx():
    code = PlanarCode(4, 4)
    exact_decoder = PlanarMPSDecoder()
    approx_decoder = PlanarMPSDecoder(chi=8)
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


def test_planar_mps_decoder_correlated_errors():
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
    decoder = PlanarMPSDecoder()
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


def test_planar_mps_decoder_cosets_probability_stp():
    # parameters
    sample = PlanarCode(3, 4).new_pauli().site('Y', (2, 0), (2, 4))
    prob_dist = DepolarizingErrorModel().probability_distribution(0.1)

    # coset probabilities exact
    exact_coset_ps, _ = PlanarMPSDecoder(mode='a')._coset_probabilities(prob_dist, sample)
    print('#exact_coset_ps=', exact_coset_ps)

    # coset probabilities approx (chi=6)
    approx_coset_ps, _ = PlanarMPSDecoder(chi=6, mode='a')._coset_probabilities(prob_dist, sample)
    print('#approx_coset_ps=', approx_coset_ps)
    assert all(_is_close(exact_coset_ps, approx_coset_ps, rtol=1e-14, atol=0)), (
        'approx_coset_ps not close to exact_coset_ps')

    # coset probabilities approx (chi=6, stp=0)
    coset_ps, _ = PlanarMPSDecoder(chi=6, mode='a', stp=0)._coset_probabilities(prob_dist, sample)
    print('#coset_ps (chi=6, stp=0)=', coset_ps)
    assert all(_is_close(approx_coset_ps, coset_ps, rtol=0, atol=0)), (
        'coset_ps (chi=6, stp=0) not equal to approx_coset_ps')

    # coset probabilities approx (chi=6, stp=1)
    coset_ps, _ = PlanarMPSDecoder(chi=6, mode='a', stp=1)._coset_probabilities(prob_dist, sample)
    print('#coset_ps (chi=6, stp=1)=', coset_ps)
    assert all(_is_close(exact_coset_ps, coset_ps, rtol=0, atol=0)), (
        'coset_ps (chi=6, stp=1) not equal to exact_coset_ps')

    # coset probabilities approx (chi=6, stp=0.5)
    coset_ps, _ = PlanarMPSDecoder(chi=6, mode='a', stp=0.5)._coset_probabilities(prob_dist, sample)
    print('#coset_ps (chi=6, stp=0.5)=', coset_ps)
    assert all(_is_close(exact_coset_ps, coset_ps, rtol=1e-10, atol=0)), (
        'coset_ps (chi=6, stp=0.5) not close to exact_coset_ps')
    assert all(_is_close(approx_coset_ps, coset_ps, rtol=1e-10, atol=0)), (
        'coset_ps (chi=6, stp=0.5) not close to approx_coset_ps')


def test_planar_mps_mwpm_performance():
    n_run = 5
    code = PlanarCode(25, 25)
    error_model = DepolarizingErrorModel()
    error_probability = 0.4
    rng = np.random.default_rng(13)

    def _timed_runs(decoder):
        start_time = time.time()
        for _ in range(n_run):
            error = error_model.generate(code, error_probability, rng)
            syndrome = pt.bsp(error, code.stabilizers.T)
            recovery = decoder.decode(code, syndrome)
            assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
                'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
        return time.time() - start_time

    mps_time = _timed_runs(PlanarMPSDecoder(chi=6))
    mwpm_time = _timed_runs(PlanarMWPMDecoder())
    # expect mps_time < mwpm_time
    print('mps_time = {} < {} = mwpm_time'.format(mps_time, mwpm_time))
    assert mps_time < mwpm_time, 'MPS decoder slower than MWPM decoder'


def test_planar_mps_decoder_decode_logging_nonpositivefinite_max_coset_probability(caplog):
    # taken from corner case mode='a' of test_planar_mps_decoder_positive_max_coset_probability
    code = PlanarCode(9, 9)
    decoder = PlanarMPSDecoder(chi=48, mode='a')
    error_model = BiasedDepolarizingErrorModel(bias=100)
    error_probability = 0.41
    error = pt.unpack(["c96aa012210dc2254031f15d9ce80c871fb864b510c91086e112a018f8aece7406638fdc00", 290])
    syndrome = pt.unpack(["8f59cd273bd1c027b3b925085af85f2aaf22", 144])
    assert np.array_equal(syndrome, pt.bsp(error, code.stabilizers.T))
    decoder.decode(code, syndrome, error_model=error_model, error_probability=error_probability)
    assert 'NON-POSITIVE-FINITE MAX COSET PROBABILITY' in caplog.text, (
        'Non-positive-finite max coset probability not logged')


# < Corner case tests >

# Used to fail if lcf method invoked with normalise=True due to divide by zero leaving NaN in arrays
def test_planar_mps_decoder_svd_does_not_converge():
    code = PlanarCode(21, 21)
    decoder = PlanarMPSDecoder(chi=4)
    error = pt.unpack((
        '001281500200080080000000000080001000000c0000002012000000801040004000000100000000004000002100000800800000000000'
        '02000100028022001000002044841000080080008110020000400801200000801040112008010004400000000000000002000000402201'
        '10040000000000000481000200000601000080080000000820200020000000008820000100000010045000004000010000000000000000'
        '40010000840010200008000400024000880000000004000000004000200890040001082000000000000002000000', 1682))
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery = decoder.decode(code, syndrome)  # no error raised
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


# Used to fail using float instead of mpmath, with all zero probabilities, for codes=25 and 29
@pytest.mark.parametrize('code, chi', [
    (PlanarCode(17, 17), 8),
    (PlanarCode(21, 21), 8),
    (PlanarCode(25, 25), 8),
    (PlanarCode(29, 29), 8),
])
def test_planar_mps_decoder_zero_max_coset_probability(code, chi):
    decoder = PlanarMPSDecoder(chi=chi, mode='c')
    error_model = BiasedDepolarizingErrorModel(bias=1000)
    random_seed = 69
    # probabilities
    probability = 0.4
    prob_dist = error_model.probability_distribution(probability)
    # error
    error = error_model.generate(code, probability, np.random.default_rng(random_seed))
    # syndrome
    syndrome = pt.bsp(error, code.stabilizers.T)
    # any_recovery
    any_recovery = decoder._sample_recovery(code, syndrome)
    # coset probabilities
    coset_ps, _ = decoder._coset_probabilities(prob_dist, any_recovery)
    print(coset_ps)
    max_coset_p = max(coset_ps)
    assert mp.isfinite(max_coset_p) and max_coset_p > 0, 'Max coset probability out of bounds {}'.format(coset_ps)


# Results in a zero valued norm of the final tensor in a MPS truncate to cover tt_mps.zeros_like in context.
def test_planar_mps_decoder_zero_norm_in_left_canonical_form():
    # parameters
    random_seed = 13
    code = PlanarCode(7, 7)
    error_model = BitFlipErrorModel()
    decoder = PlanarMPSDecoder(chi=6, mode='c')
    error_probability = 0.1
    # single run
    error = error_model.generate(code, error_probability, np.random.default_rng(random_seed))
    syndrome = pt.bsp(error, code.stabilizers.T)
    decoder.decode(code, syndrome, error_model=error_model, error_probability=error_probability)
    # No exception raised!


# TO_DEBUG: Log gives negative max coset probability for mode=a
@pytest.mark.parametrize('mode', [
    'c',
    'r',
    # 'a',  # (enable for debugging)
])
def test_planar_mps_decoder_positive_max_coset_probability(mode):
    # parameters
    code = PlanarCode(9, 9)
    decoder = PlanarMPSDecoder(chi=48, mode=mode)
    error_model = BiasedDepolarizingErrorModel(bias=100)
    error_probability = 0.41
    # logged run values
    error = pt.unpack(["c96aa012210dc2254031f15d9ce80c871fb864b510c91086e112a018f8aece7406638fdc00", 290])
    syndrome = pt.unpack(["8f59cd273bd1c027b3b925085af85f2aaf22", 144])
    assert np.array_equal(syndrome, pt.bsp(error, code.stabilizers.T))
    # debug
    # print(code.ascii_art(syndrome, code.new_pauli(error)))
    # decode
    prob_dist = error_model.probability_distribution(error_probability)
    any_recovery = decoder._sample_recovery(code, syndrome)
    # coset probabilities
    coset_ps, recoveries = decoder._coset_probabilities(prob_dist, any_recovery)
    print('mode={}, coset_ps={}'.format(mode, coset_ps))
    max_coset_p, max_recovery = max(zip(coset_ps, recoveries), key=lambda coset_p_recovery: coset_p_recovery[0])
    success = np.all(pt.bsp(max_recovery.to_bsf() ^ error, code.logicals.T) == 0)
    print('### success=', success)
    assert mp.isfinite(max_coset_p) and max_coset_p > 0, 'Max coset probability not as expected'


# TO_DEBUG: Log gives negative coset probability for mode=r
@pytest.mark.parametrize('chi, mode', [
    (2, 'c'),
    (None, 'c'),
    # (2, 'r'),  # (enable for debugging)
    (None, 'r'),
    (2, 'a'),
    (None, 'a'),
])
def test_planar_mps_decoder_small_code_negative_coset_probability(chi, mode):
    # parameters
    code = PlanarCode(3, 3)
    decoder = PlanarMPSDecoder(chi=chi, mode=mode)
    error_model = DepolarizingErrorModel()
    error_probability = 0.1
    # logged run values
    error = pt.unpack(["e0048000", 26])
    syndrome = pt.bsp(error, code.stabilizers.T)
    # debug
    print()
    print(code.ascii_art(syndrome, code.new_pauli(error)))
    # decode
    prob_dist = error_model.probability_distribution(error_probability)
    any_recovery = decoder._sample_recovery(code, syndrome)
    # coset probabilities
    coset_ps, recoveries = decoder._coset_probabilities(prob_dist, any_recovery)
    print('chi={}, mode={}, coset_ps={}'.format(chi, mode, coset_ps))
    max_coset_p, max_recovery = max(zip(coset_ps, recoveries), key=lambda coset_p_recovery: coset_p_recovery[0])
    success = np.all(pt.bsp(max_recovery.to_bsf() ^ error, code.logicals.T) == 0)
    print('### success=', success)
    assert mp.isfinite(max_coset_p) and max_coset_p > 0, 'Max coset probability not as expected'
    assert np.all(np.array(coset_ps) >= 0), 'At least one coset probability is negative'

# </ Corner case tests >
