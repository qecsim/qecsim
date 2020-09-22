import numpy as np
import pytest

from qecsim.models.basic import FiveQubitCode
from qecsim.models.generic import BiasedDepolarizingErrorModel, BiasedYXErrorModel
from qecsim.models.generic import BitFlipErrorModel, DepolarizingErrorModel


# biased-depolarizing

def test_biased_depolarizing_error_model_properties():
    bias = 100
    axis = 'Z'
    error_model = BiasedDepolarizingErrorModel(bias, axis)
    assert isinstance(error_model.label, str)
    assert isinstance(repr(error_model), str)
    assert error_model.bias == bias
    assert error_model.axis == axis


@pytest.mark.parametrize('bias, axis', [
    (0.1, 'X'),
    (10, 'Y'),
    (100, 'Z'),
    (100, 'z'),
])
def test_biased_depolarizing_error_model_valid_parameters(bias, axis):
    BiasedDepolarizingErrorModel(bias, axis)  # no error raised


@pytest.mark.parametrize('bias, axis', [
    # invalid bias
    (0, 'X'),
    ('asdf', 'X'),
    (None, 'X'),
    (-1, 'X'),
    # invalid axis
    (10, 'a'),
    (10, None),
    (10, 2),
])
def test_biased_depolarizing_error_model_invalid_parameters(bias, axis):
    with pytest.raises((ValueError, TypeError), match=r"^BiasedDepolarizingErrorModel") as exc_info:
        BiasedDepolarizingErrorModel(bias, axis)
    print(exc_info)


@pytest.mark.parametrize('bias', [
    1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
])
def test_biased_depolarising_error_model_probability_distribution(bias):
    for axis in 'XYZ':
        # floating point tolerances (better than default)
        rtol, atol = 0.0, 1e-15
        error_model = BiasedDepolarizingErrorModel(bias, axis)
        p = 0.1
        prob_dist = error_model.probability_distribution(p)
        print('Bias={!r}: (Pr(I), Pr(X), Pr(Y), Pr(Z))={}'.format(bias, prob_dist))
        p_i, p_x, p_y, p_z = prob_dist
        # check Pr(I) = 1 - p
        print('Bias={!r}: Pr(I) = {!r} ~= {!r} = 1 - p'.format(bias, p_i, 1 - p))
        assert np.isclose(p_i, 1 - p, rtol=rtol, atol=atol), 'Pr(I) != 1 - p.'
        # check Pr(X) + Pr(Y) + Pr(Z) = p
        print('Bias={!r}: Pr(X) + Pr(Y) + Pr(Z) = {!r} ~= {!r} = p'.format(bias, p_x + p_y + p_z, p))
        assert np.isclose(p_x + p_y + p_z, p, rtol=rtol, atol=atol), 'Pr(X) + Pr(Y) + Pr(Z) != p'
        # check Pr(i) + Pr(X) + Pr(Y) + Pr(Z) = 1
        print('Bias={!r}: Pr(I) + Pr(X) + Pr(Y) + Pr(Z) = {!r} ~= 1'.format(bias, sum(prob_dist)))
        assert np.isclose(np.sum(prob_dist), 1, rtol=rtol, atol=atol), 'sum(prob_dist) != 1'
        # check biased-depolarizing specific properties
        if axis == 'X':
            p_hr, p_lr1, p_lr2 = p_x, p_y, p_z
        elif axis == 'Y':
            p_hr, p_lr1, p_lr2 = p_y, p_x, p_z
        elif axis == 'Z':
            p_hr, p_lr1, p_lr2 = p_z, p_x, p_y
        assert p_lr1 == p_lr2, 'Pr(low-rate-1) != Pr(low-rate-2).'
        print('Bias={!r}: Pr(high-rate) = {!r} ~= {!r} = bias * (Pr(low-rate-1) + Pr(low-rate-2))'.format(
            bias, p_hr, bias * (p_lr1 + p_lr2)))
        assert np.isclose(p_hr, bias * (p_lr1 + p_lr2), rtol=rtol, atol=atol), (
            'Pr(high-rate) != bias * (Pr(low-rate-1) + Pr(low-rate-2)).')
        print()


def test_biased_depolarising_error_model_probability_distribution_bias_half():
    for axis in 'XYZ':
        bias = 0.5
        error_model = BiasedDepolarizingErrorModel(bias, axis)
        depolarizing_error_model = DepolarizingErrorModel()
        p = 0.1
        assert error_model.probability_distribution(p) == depolarizing_error_model.probability_distribution(p), (
            'Biased-depolarizing probability (bias=0.5) does not match standard depolarizing probability distribution.')


def test_biased_depolarising_error_model_generate():
    code = FiveQubitCode()
    bias = 100
    error_model = BiasedDepolarizingErrorModel(bias)
    probability = 0.1
    error = error_model.generate(code, probability)
    assert len(error) == code.n_k_d[0] * 2, 'Error length is not twice number of physical qubits.'
    assert issubclass(error.dtype.type, np.integer), 'Error is not integer array.'
    assert set(np.unique(error)).issubset({0, 1}), 'Error is not binary.'


def test_biased_depolarising_error_model_generate_seeded():
    code = FiveQubitCode()
    bias = 100
    error_model = BiasedDepolarizingErrorModel(bias)
    probability = 0.1
    error1 = error_model.generate(code, probability, np.random.default_rng(5))
    error2 = error_model.generate(code, probability, np.random.default_rng(5))
    assert np.array_equal(error1, error2), 'Identically seeded errors are not the same.'


# biased-Y-X

def test_biased_y_x_error_model_properties():
    bias = 100
    error_model = BiasedYXErrorModel(bias)
    assert isinstance(error_model.label, str)
    assert isinstance(repr(error_model), str)
    assert isinstance(error_model.bias, (int, float))


@pytest.mark.parametrize('bias', [
    0, 0.1, 10, 100
])
def test_biased_y_x_error_model_valid_parameters(bias):
    BiasedYXErrorModel(bias)  # no error raised


@pytest.mark.parametrize('bias', [
    'asdf', None, -1,
])
def test_biased_y_x_error_model_invalid_parameters(bias):
    with pytest.raises((ValueError, TypeError), match=r"^BiasedYXErrorModel") as exc_info:
        BiasedYXErrorModel(bias)
    print(exc_info)


@pytest.mark.parametrize('bias', [
    0,
    1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,  # 1e-9 fails with default isclose tolerances
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    1e1, 1e2, 1e3, 1e4, 1e5,  # 1e6 fails with the default isclose tolerances
])
def test_biased_y_x_error_model_probability_distribution(bias):
    # floating point tolerances (default)
    rtol, atol = 1e-5, 1e-8
    error_model = BiasedYXErrorModel(bias)
    p = 0.1
    prob_dist = error_model.probability_distribution(p)
    print('Bias={!r}: (Pr(I), Pr(X), Pr(Y), Pr(Z))={}'.format(bias, prob_dist))
    p_i, p_x, p_y, p_z = prob_dist
    # check Pr(I) = 1 - p
    print('Bias={!r}: Pr(I) = {} ~= {} = 1 - p'.format(bias, p_i, 1 - p))
    assert np.isclose(p_i, 1 - p, rtol=rtol, atol=atol), 'Pr(I) != 1 - p.'
    # check Pr(X) + Pr(Y) + Pr(Z) = p
    print('Bias={!r}: Pr(X) + Pr(Y) + Pr(Z) = {!r} ~= {!r} = p'.format(bias, p_x + p_y + p_z, p))
    assert np.isclose(p_x + p_y + p_z, p, rtol=rtol, atol=atol), 'Pr(X) + Pr(Y) + Pr(Z) != p'
    # check Pr(i) + Pr(X) + Pr(Y) + Pr(Z) = 1
    print('Bias={!r}: Pr(I) + Pr(X) + Pr(Y) + Pr(Z) = {!r} ~= 1'.format(bias, sum(prob_dist)))
    assert np.isclose(np.sum(prob_dist), 1, rtol=rtol, atol=atol), 'sum(prob_dist) != 1'
    # check biased-Y-X specific properties
    print('Bias={!r}: bias = {!r} ~= {!r} = Pr(Y) / Pr(X)'.format(bias, bias, p_y / p_x))
    assert np.isclose(bias, p_y / p_x, rtol=rtol, atol=atol), 'bias != Pr(Y) / Pr(X)'


def test_biased_y_x_error_model_probability_distribution_bias_zero():
    bias = 0
    error_model = BiasedYXErrorModel(bias)
    bit_flip_error_model = BitFlipErrorModel()
    p = 0.1
    assert error_model.probability_distribution(p) == bit_flip_error_model.probability_distribution(p), (
        'Biased-Y-X probability distribution (bias=0) does not match bit-flip probability distribution.')


def test_biased_y_x_error_model_generate():
    code = FiveQubitCode()
    bias = 100
    error_model = BiasedYXErrorModel(bias)
    probability = 0.1
    error = error_model.generate(code, probability)
    assert len(error) == code.n_k_d[0] * 2, 'Error length is not twice number of physical qubits.'
    assert issubclass(error.dtype.type, np.integer), 'Error is not integer array.'
    assert set(np.unique(error)).issubset({0, 1}), 'Error is not binary.'


def test_biased_y_x_error_model_generate_seeded():
    code = FiveQubitCode()
    bias = 100
    error_model = BiasedYXErrorModel(bias)
    probability = 0.1
    error1 = error_model.generate(code, probability, np.random.default_rng(5))
    error2 = error_model.generate(code, probability, np.random.default_rng(5))
    assert np.array_equal(error1, error2), 'Identically seeded errors are not the same.'
