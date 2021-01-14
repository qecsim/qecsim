import pytest

from qecsim import paulitools as pt
from qecsim.models.basic import FiveQubitCode, BasicCode, SteaneCode


@pytest.mark.parametrize('stabilizers, logical_xs, logical_zs, n_k_d, label', [
    (('XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'), ('XXXXX',), ('ZZZZZ',), (5, 1, 3), '5-qubit'),
])
def test_basic_code_init_properties(stabilizers, logical_xs, logical_zs, n_k_d, label):
    code = BasicCode(stabilizers, logical_xs, logical_zs, n_k_d, label)
    assert code._pauli_stabilizers == stabilizers
    assert code._pauli_logical_xs == logical_xs
    assert code._pauli_logical_zs == logical_zs
    assert code.n_k_d == n_k_d
    assert code.label == label


@pytest.mark.parametrize('stabilizers, logical_xs, logical_zs, n_k_d', [
    (('XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'), ('XXXXX',), ('ZZZZZ',), (5, 1, None)),
])
def test_basic_code_calculated_properties(stabilizers, logical_xs, logical_zs, n_k_d):
    code = BasicCode(stabilizers, logical_xs, logical_zs)
    assert tuple(pt.bsf_to_pauli(code.stabilizers)) == stabilizers
    assert tuple(pt.bsf_to_pauli(code.logical_xs)) == logical_xs
    assert tuple(pt.bsf_to_pauli(code.logical_zs)) == logical_zs
    assert tuple(pt.bsf_to_pauli(code.logicals)) == logical_xs + logical_zs
    assert code.n_k_d == n_k_d
    assert isinstance(code.label, str)
    assert isinstance(repr(code), str)


def test_basic_five_qubit_code_validate():
    code = FiveQubitCode()
    code.validate()  # no error raised


def test_basic_steane_code_validate():
    code = SteaneCode()
    code.validate()  # no error raised
