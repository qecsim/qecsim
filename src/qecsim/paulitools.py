"""
This module contains functions for Pauli strings and binary symplectic vectors / matrices.

In qecsim, a Pauli operator on N qubits is represented, without phase, by either of the following:

* a string of I, X, Y, Z of length N, such as ``'XIZIY'``, where the nth element operates on the nth qubit.
* a binary 1d numpy.array of length 2N, such as ``np.array([1,0,0,0,1,0,0,1,0,1])``, where the nth element operates
  as X on the nth qubit, the N+nth element operates as Z on the nth qubit, and XZ operates as Y (since phase is
  ignored).

A group of Pauli operators is specified as a list of string, or a 2d numpy.array, respectively.
"""

import itertools

import numpy as np


def pauli_to_bsf(pauli):
    """
    Convert the given Pauli operator(s) to binary symplectic form.

    XIZIY -> (1 0 0 0 1 | 0 0 1 0 1)

    Assumptions:

    * pauli is a string of I, X, Y, Z such as 'XIZIY' or a list of such strings of the same length.

    :param pauli: A single or list of Pauli operators.
    :type pauli: str or list of str
    :return: Binary symplectic representation of Pauli.
    :rtype: numpy.array (1d or 2d)
    """

    def _to_bsf(p):
        ps = np.array(list(p))
        xs = (ps == 'X') + (ps == 'Y')
        zs = (ps == 'Z') + (ps == 'Y')
        return np.hstack((xs, zs)).astype(int)

    if isinstance(pauli, str):
        return _to_bsf(pauli)
    else:
        return np.vstack([_to_bsf(p) for p in pauli])


def pauli_wt(pauli):
    """
    Return weight of given Pauli operator(s).

    :param pauli: A single or list of Pauli operators.
    :type pauli: str or list of str
    :return: Weight
    :rtype: int
    """

    def _wt(p):
        return p.count('X') + p.count('Y') + p.count('Z')

    if isinstance(pauli, str):
        return _wt(pauli)
    else:
        return sum(_wt(p) for p in pauli)


def bsf_to_pauli(bsf):
    """
    Convert the given binary symplectic form to Pauli operator(s).

    (1 0 0 0 1 | 0 0 1 0 1) -> XIZIY

    Assumptions:

    * bsf is a numpy.array (1d or 2d) in binary symplectic form.

    :param bsf: Binary symplectic vector or matrix.
    :type bsf: numpy.array (1d or 2d)
    :return: Pauli operators.
    :rtype: str or list of str
    """
    assert np.array_equal(bsf % 2, bsf), 'BSF {} is not in binary form'.format(bsf)

    def _to_pauli(b, t=str.maketrans('0123', 'IXZY')):  # assign t here so it is only created once
        xs, zs = np.hsplit(b, 2)
        ps = (xs + zs * 2).astype(str)  # 0=I, 1=X, 2=Z, 3=Y
        return ''.join(ps).translate(t)

    if bsf.ndim == 1:
        return _to_pauli(bsf)
    else:
        return [_to_pauli(b) for b in bsf]


def bsf_wt(bsf):
    """
    Return weight of given binary symplectic form.

    :param bsf: Binary symplectic vector or matrix.
    :type bsf: numpy.array (1d or 2d)
    :return: Weight
    :rtype: int
    """
    assert np.array_equal(bsf % 2, bsf), 'BSF {} is not in binary form'.format(bsf)
    return np.count_nonzero(sum(np.hsplit(bsf, 2)))


def bsp(a, b):
    r"""
    Return the binary symplectic product of A with B.

    The binary symplectic product :math:`\odot` is defined as :math:`A \odot B \equiv A \Lambda B \bmod 2` where
    :math:`\Lambda = \left[\begin{matrix} 0 & I \\ I & 0 \end{matrix}\right]`.

    Assumptions:

    * A and B should be 1d (vector) or 2d (matrix) numpy arrays with elements 0 or 1.
    * A should have an even number of columns (or elements if 1d).
    * B should have an even number of rows (or elements if 1d).
    * A and B should have compatible dimensions for a dot product, as per numpy requirements.

    :param a: LHS binary symplectic vector or matrix.
    :type a: numpy.array (1d or 2d)
    :param b: RHS binary symplectic vector or matrix.
    :type b: numpy.array (1d or 2d)
    :return: Binary symplectic product of A with B.
    :rtype: int if A and B vectors; numpy.array (1d if A or B vector, 2d if A and B matrices)
    """
    assert np.array_equal(a % 2, a), 'BSF {} is not in binary form'.format(a)
    assert np.array_equal(b % 2, b), 'BSF {} is not in binary form'.format(b)
    # let A = (A1|A2) and B = (B1|B2) return (A2|A1).(B1|B2)
    a1, a2 = np.hsplit(a, 2)
    return np.hstack((a2, a1)).dot(b) % 2


def ipauli(n_qubits, min_weight=0, max_weight=None):
    """
    Return an iterator of Paulis in ascending weight.

    Notes:

    * Each Pauli is a string of I, X, Y, Z such as 'XIZIY'

    :param n_qubits: Number of qubits.
    :type n_qubits: int
    :param min_weight: Minimum weight. (default=0)
    :type min_weight: int
    :param max_weight: Maximum weight. (default=n_qubits)
    :type max_weight: int
    :return: Iterator of Paulis.
    :rtype: iterator of str
    """
    if max_weight is None:
        max_weight = n_qubits
    assert min_weight <= max_weight <= n_qubits, (
        'Weights must satisfy (min_wt={}) <= (max_wt={}) <= (n_qubits={}).'.format(min_weight, max_weight, n_qubits))

    for weight in range(min_weight, max_weight + 1):
        for selected_qubits in itertools.combinations(range(n_qubits), weight):
            for selected_xyzs in itertools.product('XZY', repeat=weight):
                pauli = ['I'] * n_qubits
                for selected_qubit, selected_xyz in zip(selected_qubits, selected_xyzs):
                    pauli[selected_qubit] = selected_xyz
                yield ''.join(pauli)


def ibsf(n_qubits, min_weight=0, max_weight=None):
    """
    Return an iterator of binary symplectic representations of Paulis in ascending weight.

    :param n_qubits: Number of physical qubits.
    :type n_qubits: int
    :param min_weight: Minimum weight. (default=0)
    :type min_weight: int
    :param max_weight: Maximum weight. (default=n_qubits)
    :type max_weight: int
    :return: Iterator of binary symplectic representation of Pauli.
    :rtype: iterator of numpy.array (1d)
    """
    for pauli in ipauli(n_qubits, min_weight, max_weight):
        yield pauli_to_bsf(pauli)


def pack(binary_array):
    """
    Return packed representation of the given binary array (e.g. binary symplectic form of Pauli).

    :param binary_array: Binary array.
    :type binary_array: numpy.array (1d)
    :return: Packed binary array as (integer value of binary array, length of binary array)
    :rtype: (str, int)
    """
    assert np.array_equal(binary_array % 2, binary_array), 'Binary array {} is not in binary form'.format(binary_array)
    hex_value = bytearray(np.packbits(binary_array).tolist()).hex()
    length = len(binary_array)
    return hex_value, length


def unpack(packed_binary_array):
    """
    Return a binary array corresponding to the packed representation.

    :param packed_binary_array: Packed binary array as (hex string of binary array, length of binary array)
    :type packed_binary_array: (str, int)
    :return: Binary array
    :rtype: numpy.array (1d)
    """
    hex_value, length = packed_binary_array
    return np.array(np.unpackbits(bytearray.fromhex(hex_value))[:length], dtype=int)
