import numpy as np
import pytest
from qecsim import paulitools as pt
from qecsim.models.basic import FiveQubitCode, SteaneCode
from qecsim.models.generic import NaiveDecoder


def test_naive_decoder_properties():
    decoder = NaiveDecoder()
    assert isinstance(decoder.label, str)
    assert isinstance(repr(decoder), str)
    assert isinstance(str(decoder), str)


@pytest.mark.parametrize('max_qubits', [
    (None),
    (6),
])
def test_naive_decoder_new_valid_parameters(max_qubits):
    NaiveDecoder(max_qubits)  # no error raised


@pytest.mark.parametrize('max_qubits', [
    (-1),
    (12.5),
    ('asdf'),
])
def test_naive_decoder_new_invalid_parameters(max_qubits):
    with pytest.raises((ValueError, TypeError), match=r"^NaiveDecoder") as exc_info:
        NaiveDecoder(max_qubits)
    print(exc_info)


@pytest.mark.parametrize('error', [
    pt.pauli_to_bsf('IIIII'),
    pt.pauli_to_bsf('IXIII'),
    pt.pauli_to_bsf('IIYII'),
    pt.pauli_to_bsf('IIIIZ'),
    pt.pauli_to_bsf('IZYXI'),
    pt.pauli_to_bsf('YZYXX'),
])
def test_naive_decoder_decode(error):
    code = FiveQubitCode()
    decoder = NaiveDecoder()
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery = decoder.decode(code, syndrome)
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


def test_naive_decoder_max_qubits_limit():
    code = SteaneCode()
    decoder = NaiveDecoder(max_qubits=5)
    with pytest.raises(ValueError):
        decoder.decode(code, None)


def test_naive_decoder_max_qubits_override():
    code = SteaneCode()
    decoder = NaiveDecoder(max_qubits=None)
    decoder.decode(code, [])  # no error raised
