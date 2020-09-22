import numpy as np

from qecsim import paulitools as pt
from qecsim.models.basic import FiveQubitCode
from qecsim.models.generic import BitFlipErrorModel
from qecsim.models.generic import BitPhaseFlipErrorModel
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.generic import PhaseFlipErrorModel
from qecsim.models.generic import SimpleErrorModel


def test_generic_error_model_properties():
    error_model = DepolarizingErrorModel()
    assert isinstance(error_model.label, str)
    assert isinstance(repr(error_model), str)


def test_simple_error_model_generate():
    # input
    xyz_rates = np.array([0.2, 0.3, 0.5])
    error_probability = 0.1
    rng = np.random.default_rng(12)
    n_runs = 10000

    class CustomErrorModel(SimpleErrorModel):

        def probability_distribution(self, probability):
            p_x, p_y, p_z = xyz_rates * probability
            p_i = 1 - sum((p_x, p_y, p_z))
            return p_i, p_x, p_y, p_z

        def label(self):
            return 'Custom rates (x, y, z)=({}, {}, {})'.format(*xyz_rates)

    # models
    code = FiveQubitCode()
    error_model = CustomErrorModel()

    # output
    ixyz_counts = np.array([0, 0, 0, 0])
    for _ in range(n_runs):
        error = error_model.generate(code, error_probability, rng)
        pauli_error = pt.bsf_to_pauli(error)
        ixyz_counts += (pauli_error.count('I'), pauli_error.count('X'), pauli_error.count('Y'), pauli_error.count('Z'))

    print()
    print('#expected actual')
    print(xyz_rates[0], ixyz_counts[1] / sum(ixyz_counts[1:]))
    print(xyz_rates[1], ixyz_counts[2] / sum(ixyz_counts[1:]))
    print(xyz_rates[2], ixyz_counts[3] / sum(ixyz_counts[1:]))
    print(error_probability, sum(ixyz_counts[1:]) / sum(ixyz_counts))

    # check error counts match error rates to 2 d.p.
    assert np.isclose(xyz_rates[0], ixyz_counts[1] / sum(ixyz_counts[1:]), rtol=0, atol=0.01)
    assert np.isclose(xyz_rates[1], ixyz_counts[2] / sum(ixyz_counts[1:]), rtol=0, atol=0.01)
    assert np.isclose(xyz_rates[2], ixyz_counts[3] / sum(ixyz_counts[1:]), rtol=0, atol=0.01)
    # check error counts match error probability to 2 d.p.
    assert np.isclose(error_probability, sum(ixyz_counts[1:]) / sum(ixyz_counts), rtol=0, atol=0.01)


def test_depolarising_error_model_probability_distribution():
    p = 1 / 4
    expected = (1 - p, 1 / 3 * p, 1 / 3 * p, 1 / 3 * p)
    assert DepolarizingErrorModel().probability_distribution(p) == expected, 'Incorrect probability distribution.'


def test_depolarising_error_model_generate():
    code = FiveQubitCode()
    probability = 0.1
    error_model = DepolarizingErrorModel()
    error = error_model.generate(code, probability)
    assert len(error) == code.n_k_d[0] * 2, 'Error length is not twice number of physical qubits.'
    assert issubclass(error.dtype.type, np.integer), 'Error is not integer array.'
    assert set(np.unique(error)).issubset({0, 1}), 'Error is not binary.'


def test_depolarising_error_model_generate_seeded():
    code = FiveQubitCode()
    probability = 0.1
    error_model = DepolarizingErrorModel()
    error1 = error_model.generate(code, probability, np.random.default_rng(5))
    error2 = error_model.generate(code, probability, np.random.default_rng(5))
    assert np.array_equal(error1, error2), 'Identically seeded errors are not the same.'


def test_bit_flip_error_model_probability_distribution():
    p = 1 / 4
    expected = (1 - p, p, 0, 0)
    assert BitFlipErrorModel().probability_distribution(p) == expected, 'Incorrect probability distribution.'


def test_bit_flip_error_model_generate():
    code = FiveQubitCode()
    probability = 0.1
    error_model = BitFlipErrorModel()
    error = error_model.generate(code, probability)
    error_pauli = pt.bsf_to_pauli(error)
    assert len(error_pauli) == code.n_k_d[0], 'Error length is number of physical qubits.'
    assert error_pauli.count('Y') == error_pauli.count('Z') == 0, 'Bit-flip error contains only phase-flips.'


def test_phase_flip_error_model_probability_distribution():
    p = 1 / 4
    expected = (1 - p, 0, 0, p)
    assert PhaseFlipErrorModel().probability_distribution(p) == expected, 'Incorrect probability distribution.'


def test_phase_flip_error_model_generate():
    code = FiveQubitCode()
    probability = 0.1
    error_model = PhaseFlipErrorModel()
    error = error_model.generate(code, probability)
    error_pauli = pt.bsf_to_pauli(error)
    assert len(error_pauli) == code.n_k_d[0], 'Error length is number of physical qubits.'
    assert error_pauli.count('X') == error_pauli.count('Y') == 0, 'Phase-flip error contains only bit-flips.'


def test_bit_phase_flip_error_model_probability_distribution():
    p = 1 / 4
    expected = (1 - p, 0, p, 0)
    assert BitPhaseFlipErrorModel().probability_distribution(p) == expected, 'Incorrect probability distribution.'


def test_bit_phase_flip_error_model_generate():
    code = FiveQubitCode()
    probability = 0.1
    error_model = BitPhaseFlipErrorModel()
    error = error_model.generate(code, probability)
    error_pauli = pt.bsf_to_pauli(error)
    assert len(error_pauli) == code.n_k_d[0], 'Error length is number of physical qubits.'
    assert error_pauli.count('X') == error_pauli.count('Z') == 0, 'Bit-phase-flip error contains only bit-phase-flips.'
