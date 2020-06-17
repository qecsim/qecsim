import numpy as np
import pytest

from qecsim.models.generic import BitFlipErrorModel, DepolarizingErrorModel
from qecsim.models.generic import CenterSliceErrorModel, BitPhaseFlipErrorModel, PhaseFlipErrorModel
from qecsim.models.basic import FiveQubitCode


# center-slice
def test_center_slice_error_model_properties():
    lim = (0, 0, 1)
    pos = 0.5
    error_model = CenterSliceErrorModel(lim, pos)
    assert isinstance(error_model.label, str)
    assert isinstance(repr(error_model), str)
    assert error_model.lim == lim
    assert error_model.pos == pos
    assert error_model.neg_lim == (0.5, 0.5, 0)
    assert error_model.ratio == (1 / 6, 1 / 6, 2 / 3)


@pytest.mark.parametrize('lim, pos', [
    ((0, 0, 1), 1),
    ((0, 0, 1), 0.5),
    ((0, 0, 1), 0),
    ((0, 0, 1), -0.5),
    ((0, 0, 1), -1),
    ((0, 0.1, 0.9), 1),
    ((0.5, 0.5, 0), 1),
    ((1, 0, 0), 1),
    ((1, 1, 0), 1),  # unnormalized is ok too
])
def test_center_slice_error_model_valid_parameters(lim, pos):
    CenterSliceErrorModel(lim, pos)  # no error raised


@pytest.mark.parametrize('lim, pos', [
    # invalid lim
    ((1, 1, 1), 1),  # not on boundary
    ((0, 0, 0), 1),  # not normalizable
    ((1, 1), 1),  # not 3-tuple
    (None, 1),  # type error
    ('x', 1),  # type error
    # invalid pos
    ((0, 0, 1), 1.1),  # out of range
    ((0, 0, 1), -1.1),  # out of range
    ((0, 0, 1), None),  # type error
    ((0, 0, 1), 'x'),  # type error
])
def test_center_slice_error_model_invalid_parameters(lim, pos):
    with pytest.raises((ValueError, TypeError), match=r"^CenterSliceErrorModel") as exc_info:
        CenterSliceErrorModel(lim, pos)
    print(exc_info)


@pytest.mark.parametrize('lim', [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0.8, 0.2, 0),
    (0, 0.8, 0.2),
    (0.2, 0, 0.8),
    (0.8, 0, 0.2),
    (0.2, 0.8, 0),
    (0, 0.2, 0.8),
    (1, 1, 0),  # unnormalized is ok too
])
def test_center_slice_error_model_probability_distribution(lim):
    # floating point tolerances (better than default)
    rtol, atol = 0.0, 1e-15
    # probability 10%
    p = 0.1
    for pos in (1, 0.5, 0, -0.5, -1):
        error_model = CenterSliceErrorModel(lim, pos)
        prob_dist = error_model.probability_distribution(p)
        print('Lim={!r}: (Pr(I), Pr(X), Pr(Y), Pr(Z))={}'.format(lim, prob_dist))
        p_i, p_x, p_y, p_z = prob_dist
        # check Pr(I) = 1 - p
        print('Lim={!r}: Pr(I) = {!r} ~= {!r} = 1 - p'.format(lim, p_i, 1 - p))
        assert np.isclose(p_i, 1 - p, rtol=rtol, atol=atol), 'Pr(I) != 1 - p.'
        # check Pr(X) + Pr(Y) + Pr(Z) = p
        print('Lim={!r}: Pr(X) + Pr(Y) + Pr(Z) = {!r} ~= {!r} = p'.format(lim, p_x + p_y + p_z, p))
        assert np.isclose(p_x + p_y + p_z, p, rtol=rtol, atol=atol), 'Pr(X) + Pr(Y) + Pr(Z) != p'
        # check Pr(i) + Pr(X) + Pr(Y) + Pr(Z) = 1
        print('Lim={!r}: Pr(I) + Pr(X) + Pr(Y) + Pr(Z) = {!r} ~= 1'.format(lim, sum(prob_dist)))
        assert np.isclose(np.sum(prob_dist), 1, rtol=rtol, atol=atol), 'sum(prob_dist) != 1'
    # check center-slice specific properties
    csem_lim_zero = CenterSliceErrorModel(lim, 0)
    csem_lim_plus1 = CenterSliceErrorModel(lim, 1)
    csem_lim_minus1 = CenterSliceErrorModel(lim, -1)
    csem_neg_lim_zero = CenterSliceErrorModel(csem_lim_zero.neg_lim, 0)
    csem_neg_lim_plus1 = CenterSliceErrorModel(csem_lim_zero.neg_lim, 1)
    csem_neg_lim_minus1 = CenterSliceErrorModel(csem_lim_zero.neg_lim, -1)
    # check lim
    assert np.all(np.isclose(csem_lim_zero.lim, csem_neg_lim_zero.neg_lim, rtol, atol))
    assert np.all(np.isclose(csem_lim_zero.neg_lim, csem_neg_lim_zero.lim, rtol, atol))
    # check ratio
    assert csem_lim_zero.ratio == csem_neg_lim_zero.ratio == (1 / 3, 1 / 3, 1 / 3)
    assert np.all(np.isclose(csem_lim_plus1.ratio, csem_neg_lim_minus1.ratio, rtol, atol))
    assert np.all(np.isclose(csem_lim_minus1.ratio, csem_neg_lim_plus1.ratio, rtol, atol))
    # check probability distribution
    assert np.all(np.isclose(csem_lim_zero.probability_distribution(p),
                             csem_neg_lim_zero.probability_distribution(p)))
    assert np.all(np.isclose(csem_lim_plus1.probability_distribution(p),
                             csem_neg_lim_minus1.probability_distribution(p)))
    assert np.all(np.isclose(csem_lim_minus1.probability_distribution(p),
                             csem_neg_lim_plus1.probability_distribution(p)))


@pytest.mark.parametrize('csem, sem', [
    (CenterSliceErrorModel((1, 0, 0), 0), DepolarizingErrorModel()),
    (CenterSliceErrorModel((0, 1, 0), 0), DepolarizingErrorModel()),
    (CenterSliceErrorModel((0, 0, 1), 0), DepolarizingErrorModel()),
    (CenterSliceErrorModel((1, 0, 0), 1), BitFlipErrorModel()),
    (CenterSliceErrorModel((0, 1, 0), 1), BitPhaseFlipErrorModel()),
    (CenterSliceErrorModel((0, 0, 1), 1), PhaseFlipErrorModel()),
])
def test_center_slice_standard_error_models(csem, sem):
    p = 0.1
    assert csem.probability_distribution(p) == sem.probability_distribution(p)


def test_center_slice_error_model_generate():
    code = FiveQubitCode()
    lim = (0, 0, 1)
    pos = 0.5
    error_model = CenterSliceErrorModel(lim, pos)
    probability = 0.1
    error = error_model.generate(code, probability)
    assert len(error) == code.n_k_d[0] * 2, 'Error length is not twice number of physical qubits.'
    assert issubclass(error.dtype.type, np.integer), 'Error is not integer array.'
    assert set(np.unique(error)).issubset({0, 1}), 'Error is not binary.'


def test_center_slice_error_model_generate_seeded():
    code = FiveQubitCode()
    lim = (0, 0, 1)
    pos = 0.5
    error_model = CenterSliceErrorModel(lim, pos)
    probability = 0.1
    error1 = error_model.generate(code, probability, np.random.default_rng(5))
    error2 = error_model.generate(code, probability, np.random.default_rng(5))
    assert np.array_equal(error1, error2), 'Identically seeded errors are not the same.'
