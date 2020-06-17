import numpy as np
import pytest

from qecsim import paulitools as pt
from qecsim.models.toric import ToricCode, ToricMWPMDecoder


def test_toric_mwpm_decoder_properties():
    decoder = ToricMWPMDecoder()
    assert isinstance(decoder.label, str)
    assert isinstance(repr(decoder), str)
    assert isinstance(str(decoder), str)


@pytest.mark.parametrize('error_pauli', [
    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1), (0, 2, 1))),
    (ToricCode(5, 5).new_pauli().site('X', (0, 1, 1), (0, 2, 1)).site('Z', (0, 3, 2), (0, 1, 0))),
    (ToricCode(5, 5).new_pauli().site('X', (1, 1, 1), (0, 2, 1)).site('Z', (0, 3, 2), (1, 1, 0))),
    (ToricCode(3, 5).new_pauli().site('X', (1, 1, 1), (0, 2, 1)).site('Z', (0, 1, 2), (1, 1, 4))),
    (ToricCode(5, 3).new_pauli().site('X', (1, 1, 1), (0, 2, 1)).site('Z', (0, 4, 2), (1, 2, 0))),
    (ToricCode(5, 3).new_pauli().site('Y', (1, 1, 1), (0, 2, 1)).site('Z', (0, 4, 2), (0, 3, 2), (0, 2, 2))),
    (ToricCode(5, 3).new_pauli().site('Y', (1, 1, 1), (1, 1, 2), (1, 1, 3)).site('Z', (0, 4, 2), (0, 3, 2), (0, 2, 2))),
    (ToricCode(5, 3).new_pauli().site('X', (1, 1, 1), (1, 1, 2), (1, 1, 3), (0, 4, 2), (0, 3, 2), (0, 2, 2))),
    (ToricCode(5, 3).new_pauli().site('Y', (1, 1, 1), (1, 1, 2), (1, 1, 3), (0, 4, 2), (0, 3, 2), (0, 2, 2))),
    (ToricCode(5, 3).new_pauli().site('Z', (1, 1, 1), (1, 1, 2), (1, 1, 3), (0, 4, 2), (0, 3, 2), (0, 2, 2))),
])
def test_toric_mwpm_decoder_decode(error_pauli):
    error = error_pauli.to_bsf()
    code = error_pauli.code
    decoder = ToricMWPMDecoder()
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery = decoder.decode(code, syndrome)
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
