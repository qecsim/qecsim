import numpy as np
import pytest

from qecsim import paulitools as pt
from qecsim.models.planar import PlanarCode, PlanarMWPMDecoder


def test_planar_mwpm_decoder_properties():
    decoder = PlanarMWPMDecoder()
    assert isinstance(decoder.label, str)
    assert isinstance(repr(decoder), str)
    assert isinstance(str(decoder), str)


@pytest.mark.parametrize('error_pauli', [
    (PlanarCode(5, 5).new_pauli().site('X', (2, 0))),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2), (4, 2))),
    (PlanarCode(5, 5).new_pauli().site('X', (2, 2), (4, 2)).site('Z', (6, 4), (2, 0))),
    (PlanarCode(5, 5).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (6, 4), (1, 1))),
    (PlanarCode(3, 5).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (2, 4), (1, 7))),
    (PlanarCode(5, 3).new_pauli().site('X', (1, 3), (4, 2)).site('Z', (8, 4), (3, 1))),
    (PlanarCode(5, 3).new_pauli().site('Y', (1, 3), (4, 2)).site('Z', (8, 4), (6, 4), (4, 4))),
    (PlanarCode(5, 3).new_pauli().site('Y', (1, 3), (3, 3), (5, 3)).site('Z', (8, 4), (6, 4), (4, 4))),
    (PlanarCode(5, 3).new_pauli().site('X', (1, 3), (3, 3), (5, 3), (8, 4), (6, 4), (4, 4))),
    (PlanarCode(5, 3).new_pauli().site('Y', (1, 3), (3, 3), (5, 3), (8, 4), (6, 4), (4, 4))),
    (PlanarCode(5, 3).new_pauli().site('Z', (1, 3), (3, 3), (5, 3), (8, 4), (6, 4), (4, 4))),
])
def test_planar_mwpm_decoder_decode(error_pauli):
    error = error_pauli.to_bsf()
    code = error_pauli.code
    decoder = PlanarMWPMDecoder()
    syndrome = pt.bsp(error, code.stabilizers.T)
    recovery = decoder.decode(code, syndrome)
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
