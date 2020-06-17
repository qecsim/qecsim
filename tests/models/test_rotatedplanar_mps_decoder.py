import itertools
import logging

import numpy as np
import pytest
from mpmath import mp

from qecsim import paulitools as pt
from qecsim.models.generic import DepolarizingErrorModel, BiasedDepolarizingErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarMPSDecoder


def _is_close(a, b, rtol=1e-05, atol=1e-08):
    # np.isclose for mp.mpf, i.e. absolute(a - b) <= (atol + rtol * absolute(b))
    try:
        return [mp.almosteq(l, r, rel_eps=rtol, abs_eps=atol) for l, r in itertools.zip_longest(a, b)]
    except TypeError:
        return mp.almosteq(a, b, rel_eps=rtol, abs_eps=atol)


# @pytest.mark.parametrize('error_pauli, chi', [
#     (RotatedPlanarCode(29, 29).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (6, 4), (1, 1)), 8),
# ])
# def test_rotated_planar_mps_perf(error_pauli, chi):
#     from click.testing import CliRunner
#     with CliRunner().isolated_filesystem():
#         error = error_pauli.to_bsf()
#         code = error_pauli.code
#         decoder = RotatedPlanarMPSDecoder(chi=chi)
#         syndrome = pt.bsp(error, code.stabilizers.T)
#         for i in range(5):
#             print('# decode ', i)
#             recovery = decoder.decode(code, syndrome)
#             assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
#                 'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
#             assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
#                 'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


def test_rotated_planar_mps_decoder_properties():
    decoder = RotatedPlanarMPSDecoder(chi=8, mode='r', tol=1e-14)
    assert isinstance(decoder.label, str)
    assert isinstance(repr(decoder), str)
    assert isinstance(str(decoder), str)


@pytest.mark.parametrize('chi, mode, tol', [
    (None, 'c', None),
    (6, 'c', None),
    (None, 'r', None),
    (None, 'a', None),
    (None, 'c', 0.1),
    (None, 'c', 1),
])
def test_rotated_planar_mps_decoder_new_valid_parameters(chi, mode, tol):
    RotatedPlanarMPSDecoder(chi=chi, mode=mode, tol=tol)  # no error raised


@pytest.mark.parametrize('chi, mode, tol', [
    (-1, 'c', None),  # invalid chi
    (0.1, 'c', None),  # invalid chi
    ('asdf', 'c', None),  # invalid chi
    (None, None, None),  # invalid mode
    (None, 't', None),  # invalid mode
    (None, 2, None),  # invalid mode
    (None, 'c', -1),  # invalid tol
    (None, 'c', 'asdf'),  # invalid tol
])
def test_rotated_planar_mps_decoder_new_invalid_parameters(chi, mode, tol):
    with pytest.raises((ValueError, TypeError), match=r"^RotatedPlanarMPSDecoder") as exc_info:
        RotatedPlanarMPSDecoder(chi=chi, mode=mode, tol=tol)
    print(exc_info)


@pytest.mark.parametrize('error_pauli', [
    RotatedPlanarCode(3, 3).new_pauli().site('X', (1, 0)).site('Y', (2, 2)),
    RotatedPlanarCode(5, 5).new_pauli().site('X', (1, 1)).site('Y', (2, 2)).site('Z', (3, 2)),
    RotatedPlanarCode(7, 7).new_pauli().site('X', (4, 2)).site('Y', (3, 3)).site('Z', (6, 4), (6, 3)),
])
def test_rotated_planar_mps_decoder_sample_recovery(error_pauli):
    print('ERROR:')
    print(error_pauli)
    error = error_pauli.to_bsf()
    code = error_pauli.code
    syndrome = pt.bsp(error, code.stabilizers.T)
    print('SYNDROME:')
    print(code.ascii_art(syndrome=syndrome))
    recovery_pauli = RotatedPlanarMPSDecoder._sample_recovery(code, syndrome)
    print('SAMPLE_RECOVERY:')
    print(recovery_pauli)
    recovery = recovery_pauli.to_bsf()
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


@pytest.mark.parametrize('mode, rtol', [
    ('c', 1e-2),  # contract by column, tolerance is O(0.01). Tolerance is better than with Bravyi results O(1).
    ('r', 1e-2),  # contract by row.
    ('a', 1e-3),  # averaged.
])
def test_rotated_planar_mps_decoder_cosets_probability_inequality(mode, rtol):
    code = RotatedPlanarCode(25, 25)
    decoder = RotatedPlanarMPSDecoder(chi=5, mode=mode)
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


@pytest.mark.parametrize('sample_pauli_f, sample_pauli_g', [
    (RotatedPlanarCode(5, 5).new_pauli(), RotatedPlanarCode(5, 5).new_pauli()),
    (RotatedPlanarCode(5, 5).new_pauli(), RotatedPlanarCode(5, 5).new_pauli().plaquette((0, 2)).plaquette((2, 3))),
    (RotatedPlanarCode(5, 5).new_pauli().logical_x(),
     RotatedPlanarCode(5, 5).new_pauli().logical_x().plaquette((-1, 0)).plaquette((2, 2)).plaquette((2, 4))),
    (RotatedPlanarCode(5, 5).new_pauli().logical_z(),
     RotatedPlanarCode(5, 5).new_pauli().logical_z().plaquette((1, -1)).plaquette((1, 2)).plaquette((4, 3))),
])
def test_rotated_planar_mps_decoder_cosets_probability_equivalence(sample_pauli_f, sample_pauli_g):
    decoder = RotatedPlanarMPSDecoder(chi=8)
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
    (RotatedPlanarCode(3, 3).new_pauli().site('X', (0, 0)), None),
    (RotatedPlanarCode(4, 4).new_pauli().site('X', (1, 1), (2, 1)), None),
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (1, 1), (2, 1)), 4),
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (1, 1), (2, 1)).site('Z', (3, 2), (1, 0)), 6),
    (RotatedPlanarCode(5, 5).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (3, 4), (1, 1)), 8),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (1, 3), (4, 2), (3, 0)).site('Z', (3, 4), (1, 1)), 8),
    (RotatedPlanarCode(4, 5).new_pauli().site('X', (1, 3), (4, 2)), 6),
    (RotatedPlanarCode(4, 5).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (4, 3), (3, 1)), 6),
    (RotatedPlanarCode(4, 5).new_pauli().site('Y', (1, 3), (4, 2)).site('Z', (2, 3), (3, 2), (2, 2)), 6),
    (RotatedPlanarCode(4, 5).new_pauli().site('Y', (1, 3), (3, 3), (4, 3)).site('Z', (2, 3), (3, 2), (2, 2)), 6),
    (RotatedPlanarCode(5, 4).new_pauli().site('X', (3, 1), (2, 4)), 6),
    (RotatedPlanarCode(5, 4).new_pauli().site('X', (3, 1), (2, 4)).site('Z', (3, 4), (1, 2)), 6),
    (RotatedPlanarCode(5, 4).new_pauli().site('Y', (3, 1), (2, 4)).site('Z', (3, 2), (2, 3), (2, 2)), 6),
    (RotatedPlanarCode(5, 4).new_pauli().site('Y', (3, 1), (2, 4), (3, 4)).site('Z', (3, 2), (2, 3), (2, 2)), 6),
])
def test_rotated_planar_mps_decoder_decode(error_pauli, chi, caplog):
    with caplog.at_level(logging.WARN):
        error = error_pauli.to_bsf()
        code = error_pauli.code
        syndrome = pt.bsp(error, code.stabilizers.T)
        decoder = RotatedPlanarMPSDecoder(chi=chi)
        recovery = decoder.decode(code, syndrome)
        assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
            'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
        assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
            'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
        assert len(caplog.records) == 0, 'Unexpected log messages: {}'.format(caplog.text)


def test_rotated_planar_mps_decoder_small_codes_exact_approx():
    code = RotatedPlanarCode(5, 5)
    exact_decoder = RotatedPlanarMPSDecoder()
    approx_decoder = RotatedPlanarMPSDecoder(chi=8)
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

# <Corner case tests >

# # rp31x31_y_c4mct1e-12_p0.450.o3056314:2019-03-13 21:13:19,297 qecsim.rotatedplanar._rotatedplanarmpsdecoder WARNING
# # NON-POSITIVE-FINITE MAX COSET PROBABILITY: {"code": "RotatedPlanarCode(31, 31)",
# # "coset_ps": ["mpf('0.0')", "mpf('0.0')", "mpf('0.0')", "mpf('0.0')"],
# # "decoder": "RotatedPlanarMPSDecoder(4, 'c', 1e-12)",
# # "error": ["85edea48133571273d264a8ea15ce769273c047c83e6c9c1c901aec223ab8c4c1fc22c5d103bb310c2cf804f9917942834289ecd4ed30bff046d12f83a9a5d3279497714e61bc279603b4e1f40a69518e12eb4b4f314f978a5c055c0cbf80b906d750eee9e467950c8014868c6076bd25703f38dae03acdac2f6f524099ab8939e93254750ae73b4939e023e41f364e0e480d76111d5c6260fe1162e881dd9886167c027cc8bca141a144f66a76985ff8236897c1d4d2e993ca4bb8a730de13cb01da70fa0534a8c70975a5a798a7cbc52e02ae065fc05c836ba87774f233ca86400a4346303b5e92b81f9c6d701d66d40", 1922],
# # "error_model": "BitPhaseFlipErrorModel()", "error_probability": 0.45, "prob_dist": [0.55, 0, 0.45, 0],
# # "syndrome": ["266d15ae7fb368c88bf7864ef5c66ec97fbc85e72086af4034e4a10798741a62f477dbcfec3160084f622d24e0184330cbc6ced7b08cadeb417450c7b2fd34701926becf23a2420299e52ef21a437ad2f52ba8cbfaf27fe43318a1088c35a6618d610df31b6378b1fe18ac535b8d6748a6cfcc80c2ce0543", 960]}
#
# def test_rotated_planar_mps_decoder_corner_case_non_positive_finite_max_coset_probability():
#     code = RotatedPlanarCode(31, 31)
#     error_model = BitPhaseFlipErrorModel()
#     error_probability = 0.45
#     prob_dist = error_model.probability_distribution(error_probability)
#     # decoder = RotatedPlanarMPSDecoder(4, 'c', 1e-12)
#     decoder = RotatedPlanarMPSDecoder(4, 'c')
#     error = pt.unpack(["85edea48133571273d264a8ea15ce769273c047c83e6c9c1c901aec223ab8c4c1fc22c5d103bb310c2cf804f9917942834289ecd4ed30bff046d12f83a9a5d3279497714e61bc279603b4e1f40a69518e12eb4b4f314f978a5c055c0cbf80b906d750eee9e467950c8014868c6076bd25703f38dae03acdac2f6f524099ab8939e93254750ae73b4939e023e41f364e0e480d76111d5c6260fe1162e881dd9886167c027cc8bca141a144f66a76985ff8236897c1d4d2e993ca4bb8a730de13cb01da70fa0534a8c70975a5a798a7cbc52e02ae065fc05c836ba87774f233ca86400a4346303b5e92b81f9c6d701d66d40", 1922])
#     syndrome = pt.bsp(error, code.stabilizers.T)
#     sample_recovery = decoder._sample_recovery(code, syndrome)
#     # coset_ps, recoveries = decoder._coset_probabilities(prob_dist, sample_recovery)
#     # print(coset_ps)
#     recovery = decoder.decode(code, syndrome, error_model=error_model, error_probability=error_probability)
#     print(pt.bsp(recovery ^ error, code.stabilizers.T))
#     print(pt.bsp(recovery ^ error, code.logicals.T))

# </ Corner case tests >
