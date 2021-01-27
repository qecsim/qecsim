import numpy as np
import pytest

from qecsim.error import QecsimError
from qecsim.model import ErrorModel, DecodeResult
from qecsim.models.basic import BasicCode


def test_code_validate():
    code = BasicCode(('XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'), ('XXXXX',), ('ZZZZZ',))
    code.validate()  # no error raised


def test_code_validate_with_non_commuting_stabilizers():
    code = BasicCode(('XXXXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'), ('XXXXX',), ('ZZZZZ',))
    with pytest.raises(QecsimError) as exc_info:
        code.validate()
    assert 'Stabilizers do not mutually commute.' == str(exc_info.value)


def test_code_validate_with_non_commuting_stabilizers_with_logicals():
    code = BasicCode(('XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'), ('XXXII',), ('IIZZZ',))
    with pytest.raises(QecsimError) as exc_info:
        code.validate()
    assert 'Stabilizers do not commute with logicals.' == str(exc_info.value)


def test_code_validate_with_commuting_logicals():
    code = BasicCode(('XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'), ('IIIII',), ('IIIII',))
    with pytest.raises(QecsimError) as exc_info:
        code.validate()
    assert 'Logicals do not commute as expected.' == str(exc_info.value)


def test_error_model_probability_distribution_notimplemented():

    class MyErrorModel(ErrorModel):

        def generate(self, code, probability, rng=None):
            pass

        def label(self):
            pass

    with pytest.raises(NotImplementedError):
        MyErrorModel().probability_distribution(0.1)  # raises expected error


def test_decode_result():
    # valid options no error raised
    DecodeResult(success=True)
    DecodeResult(success=True, logical_commutations=np.array([0, 0]))
    DecodeResult(recovery=np.array([0, 0, 0, 0]))
    DecodeResult(success=True, recovery=np.array([0, 0, 0, 0]))
    DecodeResult(logical_commutations=np.array([0, 0]), recovery=np.array([0, 0, 0, 0]))
    DecodeResult(success=True, logical_commutations=np.array([0, 0]), recovery=np.array([0, 0, 0, 0]))
    with pytest.raises(QecsimError):
        # at least one of success and recovery must be specified
        DecodeResult()  # raises expected error
    with pytest.raises(QecsimError):
        # at least one of success and recovery must be specified
        DecodeResult(logical_commutations=np.array([0, 0]))  # raises expected error
