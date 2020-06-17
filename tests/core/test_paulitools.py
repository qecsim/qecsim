import itertools

import numpy as np
import pytest
from qecsim import paulitools as pt


@pytest.mark.parametrize('p, expected', [
    ('XIZIY', np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1])),
    ('IIIII', np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ('XXXXX', np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])),
    ('ZZZZZ', np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])),
    ('YYYYY', np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])),
    (['XIZIY', 'IXZYI'], np.array([
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 1, 1, 0]
    ])),
    (['XXXXX', 'ZZZZZ', 'YYYYY'], np.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])),
    ('X', np.array([1, 0])),
    ('IXZY', np.array([0, 1, 0, 1, 0, 0, 1, 1])),
])
def test_pauli_to_bsf(p, expected):
    assert np.array_equal(pt.pauli_to_bsf(p), expected)


@pytest.mark.parametrize('p, expected', [
    ('IIIII', 0),
    ('XIZIY', 3),
    ('XXXXX', 5),
    ('ZZZZZ', 5),
    ('ZZZZZ', 5),
    (['XIZIY', 'XXXXX'], 8),
])
def test_pauli_weight(p, expected):
    assert pt.pauli_wt(p) == expected


@pytest.mark.parametrize('b, expected', [
    (np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1]), 'XIZIY'),
    (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'IIIII'),
    (np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), 'XXXXX'),
    (np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), 'ZZZZZ'),
    (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'YYYYY'),
    (np.array([
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 1, 1, 0]
    ]), ['XIZIY', 'IXZYI']),
    (np.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]), ['XXXXX', 'ZZZZZ', 'YYYYY']),
    (np.array([1, 0]), 'X'),
    (np.array([0, 1, 0, 1, 0, 0, 1, 1]), 'IXZY'),
])
def test_bsf_to_pauli(b, expected):
    assert pt.bsf_to_pauli(b) == expected


@pytest.mark.parametrize('b, expected', [
    (pt.pauli_to_bsf('IIIII'), 0),
    (pt.pauli_to_bsf('XIZIY'), 3),
    (pt.pauli_to_bsf('XXXXX'), 5),
    (pt.pauli_to_bsf('ZZZZZ'), 5),
    (pt.pauli_to_bsf('ZZZZZ'), 5),
    (np.array([
        pt.pauli_to_bsf('XIZIY'),
        pt.pauli_to_bsf('XXXXX')
    ]), 8)
])
def test_bsf_weight(b, expected):
    assert pt.bsf_wt(b) == expected


@pytest.mark.parametrize('a, b, expected', [
    (np.array([0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0]), 0),  # III bsp III commute
    (np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0]), 0),  # XII bsp XII commute
    (np.array([1, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 1, 1]), 0),  # XII bsp IZZ commute
    (np.array([1, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 1, 0, 0]), 1),  # XII bsp ZII do not commute
    (np.array([1, 1, 0, 0, 0, 0]), np.array([0, 0, 0, 1, 1, 0]), 0),  # XXI bsp ZZI commute
    (np.array([1, 1, 1, 0, 0, 0]), np.array([0, 0, 0, 1, 1, 1]), 1),  # XXX bsp ZZZ do not commute

    (np.array([1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 1, 0, 0]), 1),  # XII bsp YII do not commute
    (np.array([1, 0, 0, 0, 0, 0]), np.array([0, 1, 1, 0, 1, 1]), 0),  # XII bsp IYY commute
    (np.array([0, 0, 0, 1, 0, 0]), np.array([1, 0, 0, 1, 0, 0]), 1),  # ZII bsp YII do not commute
    (np.array([1, 1, 0, 0, 0, 0]), np.array([1, 1, 0, 1, 1, 0]), 0),  # XXI bsp YYI commute
    (np.array([1, 1, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1]), 1),  # YYY bsp ZZZ do not commute

    (pt.pauli_to_bsf('III'), pt.pauli_to_bsf('III'), 0),
    (pt.pauli_to_bsf('XII'), pt.pauli_to_bsf('XII'), 0),
    (pt.pauli_to_bsf('XII'), pt.pauli_to_bsf('IZZ'), 0),
    (pt.pauli_to_bsf('XII'), pt.pauli_to_bsf('ZII'), 1),
    (pt.pauli_to_bsf('XXI'), pt.pauli_to_bsf('ZZI'), 0),
    (pt.pauli_to_bsf('XXX'), pt.pauli_to_bsf('ZZZ'), 1),

    (pt.pauli_to_bsf('XII'), pt.pauli_to_bsf('YII'), 1),
    (pt.pauli_to_bsf('XII'), pt.pauli_to_bsf('IYY'), 0),
    (pt.pauli_to_bsf('ZII'), pt.pauli_to_bsf('YII'), 1),
    (pt.pauli_to_bsf('XXI'), pt.pauli_to_bsf('YYI'), 0),
    (pt.pauli_to_bsf('YYY'), pt.pauli_to_bsf('ZZZ'), 1),

    (pt.pauli_to_bsf('XIIII'), pt.pauli_to_bsf(['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']).T, [0, 0, 0, 1]),
    (pt.pauli_to_bsf('IIZII'), pt.pauli_to_bsf(['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']).T, [0, 0, 1, 0]),
    (pt.pauli_to_bsf('IIIIY'), pt.pauli_to_bsf(['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']).T, [0, 1, 1, 1]),
    (pt.pauli_to_bsf('IZXYI'), pt.pauli_to_bsf(['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']).T, [0, 1, 1, 0]),

    (pt.pauli_to_bsf(['XIIII', 'IIZII', 'IIIIY', 'IZXYI']),
     pt.pauli_to_bsf(['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']).T,
     [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 1, 1], [0, 1, 1, 0]]),

    (pt.pauli_to_bsf(['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']), pt.pauli_to_bsf('IZXYI'), [0, 1, 1, 0]),
])
def test_bsp(a, b, expected):
    # Note: numpy scalars have same methods as numpy array. So .all() is valid here.
    assert (pt.bsp(a, b) == expected).all()


@pytest.mark.parametrize('n_qubits, min_weight, max_weight', [
    (5, 0, 5),
    (3, 1, 2),
    (10, 0, 5),
])
def test_ipauli(n_qubits, min_weight, max_weight):
    previous_weight = None
    weight_counts = {}  # map of weight to count for that weight

    for pauli in pt.ipauli(n_qubits, min_weight, max_weight):
        weight = pt.pauli_wt(pauli)
        if previous_weight is None:
            assert weight == min_weight, (
                'Initial Pauli {} found with (weight={}) > (min_weight={}).'.format(pauli, weight, min_weight))
        else:
            assert weight >= previous_weight, (
                'Pauli {} found with (weight={}) < (previous_weight={}).'.format(pauli, weight, previous_weight))
        weight_counts[weight] = weight_counts.get(weight, 0) + 1
        previous_weight = weight

    def n_choose_r(n, r):
        f = np.math.factorial
        return f(n) / f(r) / f(n - r)

    for weight, count in weight_counts.items():
        expected_count = 3 ** weight * n_choose_r(n_qubits, weight)
        assert count == expected_count, 'Pauli count for weight {} not as expected {}.'.format(count, expected_count)


@pytest.mark.parametrize('n_qubits, min_weight, max_weight', [
    (4, 0, 4),
    (3, 1, 2),
    (3, 0, None),
])
def test_ibsf(n_qubits, min_weight, max_weight):
    pauli_iter = pt.ipauli(n_qubits, min_weight, max_weight)
    bsf_iter = pt.ibsf(n_qubits, min_weight, max_weight)
    assert all(p == pt.bsf_to_pauli(b) for p, b in itertools.zip_longest(pauli_iter, bsf_iter)), (
        'BSF iterator does not correspond to equivalent Pauli iterator.')


def test_pack_unpack_random():
    rng = np.random.default_rng()
    for length in range(0, 5000):
        binary_array = rng.choice(2, length)
        packed_binary_array = pt.pack(binary_array)
        unpacked_binary_array = pt.unpack(packed_binary_array)
        assert np.array_equal(binary_array, unpacked_binary_array), (
            'Unpacked binary array {} does not equal expected {}.'.format(unpacked_binary_array, binary_array))


def test_pack_unpack_65bit():
    binary_array = np.array([0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
                             1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                             1, 0, 0, 0, 1])
    print(len(binary_array))
    packed_binary_array = pt.pack(binary_array)
    unpacked_binary_array = pt.unpack(packed_binary_array)
    assert np.array_equal(binary_array, unpacked_binary_array), (
        'Unpacked binary array {} does not equal expected {}.'.format(unpacked_binary_array, binary_array))
