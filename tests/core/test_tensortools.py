import numpy as np
import pytest
from mpmath import mp
from scipy import linalg as sp_linalg

from qecsim import tensortools as tt
from qecsim.models.color import Color666Code
from qecsim.models.color import Color666MPSDecoder
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.planar import PlanarCode
from qecsim.models.planar import PlanarMPSDecoder
from qecsim.models.rotatedplanar import RotatedPlanarCode
from qecsim.models.rotatedplanar import _rotatedplanarrmpsdecoder


def _is_close(a, b, rtol=1e-05, atol=1e-08):
    # np.isclose for mp.mpf, i.e. absolute(a - b) <= (atol + rtol * absolute(b))
    return mp.almosteq(a, b, rel_eps=rtol, abs_eps=atol)


@pytest.mark.parametrize('shape, expected', [
    ((2,), np.array([1, 1])),
    ((2, 2), np.identity(2)),
    ((2, 2, 2), np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])),
    ((1,), np.array([1])),
    ((1, 1), np.array([[1]])),
    ((1, 1, 1), np.array([[[1]]])),
    ((2, 1), np.array([[1], [1]])),
    ((2, 2, 1), np.array([[[1], [0]], [[0], [1]]])),
])
def test_tsr_delta(shape, expected):
    delta = tt.tsr.delta(shape)
    assert np.array_equal(delta, expected), 'Delta not as expected'


@pytest.mark.parametrize('tsr, expected', [
    (np.array(3), 3),
    (np.array([2.4]), 2.4),
    (np.array([[[[6]]]]), 6),
])
def test_tsr_as_scalar(tsr, expected):
    scalar = tt.tsr.as_scalar(tsr)
    assert scalar == expected, 'Scalar not as expected'


@pytest.mark.parametrize('tsr', [
    (np.array([1, 1])),
])
def test_tsr_as_scalar_invalid(tsr):
    with pytest.raises(ValueError):
        tt.tsr.as_scalar(tsr)


@pytest.mark.parametrize('mps', [
    ([  # MPS
        np.full((1, 1, 4, 8), 1),  # N, E, S, W
        np.full((4, 1, 4, 8), 2),  # N, E, S, W
        np.full((4, 1, 4, 8), 3.0),  # N, E, S, W
        np.full((4, 1, 4, 8), 4),  # N, E, S, W
        np.full((4, 1, 4, 8), 5.0),  # N, E, S, W
        np.full((4, 1, 1, 8), 6),  # N, E, S, W
    ]),
    ([  # Sparse MPS
        None,  # N, E, S, W
        np.full((1, 1, 3, 6), 2),  # N, E, S, W
        np.full((3, 1, 3, 6), 3.0),  # N, E, S, W
        np.full((3, 1, 1, 6), 4),  # N, E, S, W
        None,  # N, E, S, W
        None,  # N, E, S, W
    ]),
    ([  # MPO
        np.full((1, 4, 2, 8), 1),  # N, E, S, W
        np.full((2, 4, 2, 8), 2.0),  # N, E, S, W
        np.full((2, 4, 2, 8), 3),  # N, E, S, W
        np.full((2, 4, 2, 8), 4.0),  # N, E, S, W
        np.full((2, 4, 2, 8), 5),  # N, E, S, W
        np.full((2, 4, 1, 8), 6),  # N, E, S, W
    ]),
])
def test_mps_zeros_like(mps):
    # zeros_mps from mps
    zeros_mps = tt.mps.zeros_like(mps)
    # zeros_mps tensors should have same shape (except 1 dimensional N, S indices), dtype, and be filled with zeros.
    assert len(zeros_mps) == len(mps), 'Zeros MPS is not the same length as given MPS.'
    for zeros_tsr, tsr in zip(zeros_mps, mps):
        if zeros_tsr is None or tsr is None:
            assert zeros_tsr is None and tsr is None, 'Mismatched None tensor.'
        else:
            assert len(zeros_tsr.shape) == len(tsr.shape) == 4, 'Mismatched tensor dimension.'
            assert zeros_tsr.shape == (1, tsr.shape[1], 1, tsr.shape[3]), 'Mismatched tensor index dimensions.'
            assert zeros_tsr.dtype == tsr.dtype, 'Mismatched tensor dtypes.'
            assert np.count_nonzero(zeros_tsr) == 0, 'Non-zero values in zeros MPS.'


def test_planar_mps_functions():
    prob_dist = DepolarizingErrorModel().probability_distribution(0.1)
    rng = np.random.default_rng()
    tnc = PlanarMPSDecoder.TNC()
    # column bra, mpo and ket
    bra = [
        tnc.create_h_node(prob_dist, 'X', 'nw'),
        tnc.create_s_node('w'),
        tnc.create_h_node(prob_dist, 'Z', 'sw'),
    ]
    mpo = [
        tnc.create_s_node('n'),
        tnc.create_v_node(prob_dist, 'Y'),
        tnc.create_s_node('s'),
    ]
    ket = [
        tnc.create_h_node(prob_dist, 'Z', 'ne'),
        tnc.create_s_node('e'),
        tnc.create_h_node(prob_dist, 'I', 'se'),
    ]
    # tensor network corresponding to column bra, mpo and ket
    tn = np.array([bra, mpo, ket], dtype=object).transpose()
    expected = 0.00096790123456790152

    # exact contraction for expected
    result = tt.mps.contract_pairwise(bra, mpo)
    result = tt.mps.inner_product(result, ket)
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction not as expected'

    # exact contraction with contract_ladder
    result = tt.mps.contract_pairwise(bra, mpo)
    result = tt.mps.contract_pairwise(result, ket)
    result = tt.mps.contract_ladder(result)[0, 0, 0, 0]
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction not as expected'

    # exact contraction with lcf
    result = tt.mps.left_canonical_form(bra)
    result = tt.mps.contract_pairwise(result, mpo)
    result = tt.mps.left_canonical_form(result)
    result = tt.mps.inner_product(result, ket)
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction with lcf not as expected'

    # exact contraction with rcf
    result = tt.mps.right_canonical_form(bra)
    result = tt.mps.contract_pairwise(result, mpo)
    result = tt.mps.right_canonical_form(result)
    result = tt.mps.inner_product(result, ket)
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction with rcf not as expected'

    # exact contraction with truncate
    result = tt.mps.contract_pairwise(bra, mpo)
    result, norm = tt.mps.truncate(result)
    result = tt.mps.inner_product(result, ket)
    result *= norm
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction with truncate not as expected'

    # exact contraction with normalise lcf
    result, norm1 = tt.mps.left_canonical_form(bra, normalise=True)
    result = tt.mps.contract_pairwise(result, mpo)
    result, norm2 = tt.mps.left_canonical_form(result, normalise=True)
    result = tt.mps.inner_product(result, ket)
    result *= norm1 * norm2
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction with normalise lcf not as expected'

    # exact contraction with normalise rcf
    result, norm1 = tt.mps.right_canonical_form(bra, normalise=True)
    result = tt.mps.contract_pairwise(result, mpo)
    result, norm2 = tt.mps.right_canonical_form(result, normalise=True)
    result = tt.mps.inner_product(result, ket)
    result *= norm1 * norm2
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction with normalise rcf not as expected'

    # approximate contraction with truncate (chi)
    result = tt.mps.contract_pairwise(bra, mpo)
    result, norm = tt.mps.truncate(result, chi=2)
    result = tt.mps.inner_product(result, ket)
    result *= norm
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Approx. contraction with truncate (chi) not as expected'

    # approximate contraction with truncate (tol)
    result = tt.mps.contract_pairwise(bra, mpo)
    result, norm = tt.mps.truncate(result, tol=1e-8)
    result = tt.mps.inner_product(result, ket)
    result *= norm
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Approx. contraction with truncate (tol) not as expected'

    # tn_contract
    result = tt.mps2d.contract(tn)
    assert isinstance(result, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact tn contraction not as expected'

    # tn_contract with truncate (chi)
    result = tt.mps2d.contract(tn, chi=2)
    assert isinstance(result, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Approx. tn contraction with truncate (chi) not as expected'

    # tn_contract with truncate (tol)
    result = tt.mps2d.contract(tn, tol=1e-8)
    assert isinstance(result, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Approx. tn contraction with truncate (tol) not as expected'

    # tn_contract with truncate (chi, mask)
    stp = 0.2  # skip truncate probability
    mask = rng.choice((True, False), size=tn.shape, p=(1 - stp, stp))
    result = tt.mps2d.contract(tn, chi=2, mask=mask)
    assert isinstance(result, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), (
        'Approx. tn contraction with truncate (chi, mask) not as expected')


def test_planar_mps2d_contract():
    code = PlanarCode(3, 3)
    sample = code.new_pauli().site('Y', (2, 0), (2, 4))
    prob_dist = DepolarizingErrorModel().probability_distribution(0.1)
    tnc = PlanarMPSDecoder.TNC()
    tn = tnc.create_tn(prob_dist, sample)
    result = tt.mps2d.contract(tn)
    assert isinstance(result, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    assert 0 <= result <= 1, 'Contracted tensor network not within bounds'


def test_planar_mps2d_transpose():
    code = PlanarCode(3, 3)
    sample = code.new_pauli().site('Y', (2, 0), (2, 4))
    prob_dist = DepolarizingErrorModel().probability_distribution(0.1)
    tnc = PlanarMPSDecoder.TNC()
    tn = tnc.create_tn(prob_dist, sample)
    result = tt.mps2d.contract(tn)
    tn_transpose = tt.mps2d.transpose(tn)
    result_transpose = tt.mps2d.contract(tn_transpose)
    print(result, result_transpose)
    assert isinstance(result, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    assert 0 <= result <= 1, 'Contracted tensor network not within bounds'
    assert isinstance(result_transpose, mp.mpf), 'Contracted transposed tensor network is not an mp.mpf'
    assert 0 <= result_transpose <= 1, 'Contracted transposed tensor network not within bounds'
    assert result == result_transpose, 'Contracted tensor network != Contracted transposed tensor network'


def test_planar_mps2d_contract_mask():
    code = PlanarCode(3, 4)
    sample = code.new_pauli().site('Y', (2, 0), (2, 4))
    prob_dist = DepolarizingErrorModel().probability_distribution(0.1)
    tnc = PlanarMPSDecoder.TNC()
    tn = tnc.create_tn(prob_dist, sample)
    rng = np.random.default_rng()

    # tn_contract exact
    exact_result = tt.mps2d.contract(tn)
    assert isinstance(exact_result, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    assert 0 <= exact_result <= 1, 'Contracted tensor network not within bounds'
    print('#exact_result=', exact_result)

    # tn_contract approx
    approx_result = tt.mps2d.contract(tn, chi=2)
    assert isinstance(approx_result, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    assert 0 <= approx_result <= 1, 'Contracted tensor network not within bounds'
    print('#approx_result=', approx_result, '#rtol=', abs(approx_result - exact_result) / abs(exact_result))
    assert _is_close(exact_result, approx_result, rtol=1e-4, atol=0), 'tn_contract(chi=2) not as expected'

    # tn_contract with truncate (chi, mask=0)
    stp = 0  # skip truncate probability
    mask = rng.choice((True, False), size=tn.shape, p=(1 - stp, stp))
    result = tt.mps2d.contract(tn, chi=2, mask=mask)
    assert isinstance(result, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    assert 0 <= result <= 1, 'Contracted tensor network not within bounds'
    print('#result (chi=2, mask=0)=', result, '#rtol=', abs(result - approx_result) / abs(approx_result))
    assert approx_result == result, 'tn_contract(chi=2, mask=0) not same as approx_result'

    # tn_contract with truncate (chi, mask=1)
    stp = 1  # skip truncate probability
    mask = rng.choice((True, False), size=tn.shape, p=(1 - stp, stp))
    result = tt.mps2d.contract(tn, chi=2, mask=mask)
    assert isinstance(result, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    assert 0 <= result <= 1, 'Contracted tensor network not within bounds'
    print('#result (chi=2, mask=1)=', result, '#rtol=', abs(result - exact_result) / abs(exact_result))
    assert exact_result == result, 'tn_contract(chi=2, mask=1) not same as exact_result'

    # tn_contract with truncate (chi, mask=0.5)
    stp = 0.5  # skip truncate probability
    mask = rng.choice((True, False), size=tn.shape, p=(1 - stp, stp))
    result = tt.mps2d.contract(tn, chi=2, mask=mask)
    assert isinstance(result, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    assert 0 <= result <= 1, 'Contracted tensor network not within bounds'
    print('#result (chi=2, mask=0.5)=', result, '#rtol=', abs(result - exact_result) / abs(exact_result))
    assert exact_result != result, 'tn_contract(chi=2, mask=0.5) should not equal exact_result'
    assert approx_result != result, 'tn_contract(chi=2, mask=0.5) should not equal approx_result'
    assert _is_close(exact_result, result, rtol=1e-4, atol=0), 'tn_contract(chi=2, mask=0.5) not close to exact_result'
    print('#result (chi=2, mask=0.5)=', result, '#rtol=', abs(result - approx_result) / abs(approx_result))
    assert _is_close(approx_result, result, rtol=1e-4, atol=0), (
        'tn_contract(chi=2, mask=0.5) not close to approx_result')


def test_svd_sign_flip_from_truncation():
    # Demonstrate how truncated svd can lead to a sign flip
    a = np.array([[-6.95233550e-02, -3.81431985e-02, 3.62756491e-02, -1.14542331e-17],
                  [-1.70566980e-01, -3.81431985e-02, 9.79271665e-01, -1.15081833e-17],
                  [2.23441823e-03, 3.77718139e-02, -1.16586681e-03, 3.68128773e-19],
                  [1.74010158e-03, 3.77718139e-02, 3.44737462e-03, 1.13961329e-17]])
    print()
    print('A =', a)
    print()
    u, s, v = sp_linalg.svd(a, full_matrices=False, lapack_driver='gesvd')
    # print('U =', u)
    # print('s =', s)
    # print('Vh =', v)
    print()
    print('U @ s @ Vh =', u @ np.diag(s) @ v)
    print()

    # truncate s to chi values
    chi = 2
    s = s[:chi]
    # truncate U, V in accordance with S
    u = u[:, :len(s)]
    v = v[:len(s), :]

    a_chi = u @ np.diag(s) @ v
    print('A_chi = trunc(U @ s @ Vh) =', a_chi)
    print()

    b = np.array([0, 0, 1, 0])
    print('B =', b)
    print()

    print('(A @ B).T @ B     =', (a @ b).T @ b)
    print('(A_chi @ B).T @ B =', (a_chi @ b).T @ b)


def test_svd_sign_flip_from_matrix_product_associativity():
    # Demonstrate how full svd can lead to a sign flip
    a = np.array([[-0.99927361, -0.0010349, 0.02150859, 0.],
                  [-0.0025061, -0.02293941, 0.00155779, -0.02129864]])
    print()
    print('A =', a)
    print()
    u, s, v = sp_linalg.svd(a, full_matrices=False, lapack_driver='gesvd')

    print('U =', u)
    print('s =', s)
    print('Vh =', v)
    print()

    print('U @ s @ Vh =', u @ np.diag(s) @ v)
    print()
    print('(U @ s) @ Vh =', (u @ np.diag(s)) @ v)
    print()
    print('NOTE: Sign change despite associativity of matrix multiplication!')
    print('U @ (s @ Vh) =', u @ (np.diag(s) @ v))
    print()

    # truncate s to chi values
    chi = 2
    s = s[:chi]
    # truncate U, V in accordance with S
    u = u[:, :len(s)]
    v = v[:len(s), :]

    print('U =', u)
    print('s =', s)
    print('Vh =', v)
    print()

    # a_chi = np.dot(u, np.dot(np.diag(s), v))
    a_chi = u @ np.diag(s) @ v
    print('A_chi = trunc(U @ s @ Vh) =', a_chi)
    print()

    b = np.array([0, 0, 0, 1])
    print('B =', b)
    c = np.array([1, 0])
    print('C =', c)
    print()

    print('(A @ B).T      =', (a @ b).T)
    print('(A_chi @ B).T  =', (a_chi @ b).T)

    print('(A @ B).T @ C     =', (a @ b).T @ c)
    print('(A_chi @ B).T @ C =', (a_chi @ b).T @ c)


def test_color666_mps_functions():
    prob_dist = DepolarizingErrorModel().probability_distribution(0.1)
    tnc = Color666MPSDecoder.TNC()
    bra = [
        tnc.create_q_node(prob_dist, ('I', 'I'), 'nw'),
        tnc.create_s_node('w'),
        tnc.create_q_node(prob_dist, ('I', None), 'sw'),
    ]
    mpo1 = [
        tnc.create_s_node('n'),
        tnc.create_q_node(prob_dist, ('I', 'I'), 's'),
        None,
    ]
    mpo2 = [
        tnc.create_q_node(prob_dist, (None, 'I'), 'ne'),
        tnc.create_s_node('s'),
        None,
    ]
    ket = [
        None,
        tnc.create_q_node(prob_dist, ('I', None), 'e'),
        None,
    ]
    expected = 0.47831585185185249

    # exact contraction for expected
    result = tt.mps.contract_pairwise(bra, mpo1)
    result = tt.mps.contract_pairwise(result, mpo2)
    result = tt.mps.inner_product(result, ket)
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction not as expected'

    # exact contraction with contract_ladder
    result = tt.mps.contract_pairwise(bra, mpo1)
    result = tt.mps.contract_pairwise(result, mpo2)
    result = tt.mps.contract_pairwise(result, ket)
    result = tt.mps.contract_ladder(result)[0, 0, 0, 0]
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction not as expected'

    # exact contraction with lcf
    result = tt.mps.left_canonical_form(bra)
    result = tt.mps.contract_pairwise(result, mpo1)
    result = tt.mps.left_canonical_form(result)
    result = tt.mps.contract_pairwise(result, mpo2)
    result = tt.mps.left_canonical_form(result)
    result = tt.mps.inner_product(result, ket)
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction with lcf not as expected'

    # exact contraction with rcf
    result = tt.mps.right_canonical_form(bra)
    result = tt.mps.contract_pairwise(result, mpo1)
    result = tt.mps.right_canonical_form(result)
    result = tt.mps.contract_pairwise(result, mpo2)
    result = tt.mps.right_canonical_form(result)
    result = tt.mps.inner_product(result, ket)
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction with rcf not as expected'

    # exact contraction with truncate
    result = tt.mps.contract_pairwise(bra, mpo1)
    result, norm1 = tt.mps.truncate(result)
    result = tt.mps.contract_pairwise(result, mpo2)
    result, norm2 = tt.mps.truncate(result)
    result = tt.mps.inner_product(result, ket)
    result *= norm1 * norm2
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction with truncate not as expected'

    # exact contraction with normalise lcf
    result, norm1 = tt.mps.left_canonical_form(bra, normalise=True)
    result = tt.mps.contract_pairwise(result, mpo1)
    result, norm2 = tt.mps.left_canonical_form(result, normalise=True)
    result = tt.mps.contract_pairwise(result, mpo2)
    result, norm3 = tt.mps.left_canonical_form(result, normalise=True)
    result = tt.mps.inner_product(result, ket)
    result *= norm1 * norm2 * norm3
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction with normalise lcf not as expected'

    # exact contraction with normalise rcf
    result, norm1 = tt.mps.right_canonical_form(bra, normalise=True)
    result = tt.mps.contract_pairwise(result, mpo1)
    result, norm2 = tt.mps.right_canonical_form(result, normalise=True)
    result = tt.mps.contract_pairwise(result, mpo2)
    result, norm3 = tt.mps.right_canonical_form(result, normalise=True)
    result = tt.mps.inner_product(result, ket)
    result *= norm1 * norm2 * norm3
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Exact contraction with normalise rcf not as expected'

    # approximate contraction with truncate (chi)
    result = tt.mps.contract_pairwise(bra, mpo1)
    result, norm1 = tt.mps.truncate(result, chi=4)
    result = tt.mps.contract_pairwise(result, mpo2)
    result, norm2 = tt.mps.truncate(result, chi=4)
    result = tt.mps.inner_product(result, ket)
    result *= norm1 * norm2
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Approx. contraction with truncate (chi) not as expected'

    # approximate contraction with truncate (tol)
    result = tt.mps.contract_pairwise(bra, mpo1)
    result, norm1 = tt.mps.truncate(result, tol=1e-8)
    result = tt.mps.contract_pairwise(result, mpo2)
    result, norm2 = tt.mps.truncate(result, tol=1e-8)
    result = tt.mps.inner_product(result, ket)
    result *= norm1 * norm2
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Approx. contraction with truncate (tol) not as expected'

    # reversed exact contraction for expected
    result = tt.mps.contract_pairwise(mpo2, ket)
    result = tt.mps.contract_pairwise(mpo1, result)
    result = tt.mps.inner_product(bra, result)
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), 'Reversed exact contraction not as expected'

    # reversed approximate contraction with truncate (chi)
    result = tt.mps.contract_pairwise(mpo2, ket)
    result, norm1 = tt.mps.truncate(result, chi=4)
    result = tt.mps.contract_pairwise(mpo1, result)
    result, norm2 = tt.mps.truncate(result, chi=4)
    result = tt.mps.inner_product(bra, result)
    result *= norm1 * norm2
    print('#result=', result, '#rtol=', abs(result - expected) / abs(expected))
    assert _is_close(expected, result, rtol=1e-14, atol=0), (
        'Reversed approx. contraction with truncate (chi) not as expected')


def test_color666_mps2d_contract():
    code = Color666Code(5)
    sample = code.new_pauli().site('X', (3, 1)).site('Y', (2, 2)).site('Z', (6, 4))
    print()
    print(code.ascii_art(pauli=sample))
    prob_dist = DepolarizingErrorModel().probability_distribution(0.1)
    tnc = Color666MPSDecoder.TNC()
    tn = tnc.create_tn(prob_dist, sample)
    result_forwards = tt.mps2d.contract(tn)
    print('Forward contraction result: ', repr(result_forwards))
    assert isinstance(result_forwards, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    assert 0 <= result_forwards <= 1, 'Contracted tensor network not within bounds'
    result_backwards = tt.mps2d.contract(tn, start=-1, step=-1)
    print('Backward contraction result:', repr(result_backwards))
    assert isinstance(result_backwards, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    assert 0 <= result_backwards <= 1, 'Contracted tensor network not within bounds'
    assert _is_close(result_forwards, result_backwards, rtol=1e-14, atol=0), (
        'Contracting forwards does not give the same result as contracting backwards')


@pytest.mark.parametrize('pauli', [
    (RotatedPlanarCode(3, 3).new_pauli().site('Y', (2, 0), (2, 4))),
    (RotatedPlanarCode(4, 4).new_pauli().site('Y', (2, 0), (2, 4))),
    (RotatedPlanarCode(3, 4).new_pauli().site('Y', (2, 0), (2, 4))),
    (RotatedPlanarCode(4, 3).new_pauli().site('Y', (2, 0), (2, 4))),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 0), (2, 4))),
    (RotatedPlanarCode(4, 5).new_pauli().site('Y', (2, 0), (2, 4))),
    (RotatedPlanarCode(5, 4).new_pauli().site('Y', (2, 0), (2, 4))),
    (RotatedPlanarCode(6, 6).new_pauli().site('Y', (2, 0), (2, 4))),
])
def test_rotatedplanar_mps2d_contract(pauli):
    prob_dist = DepolarizingErrorModel().probability_distribution(0.1)
    tn = _rotatedplanarrmpsdecoder._create_tn(prob_dist, pauli)
    print('tn.shape=', tn.shape)
    result = tt.mps2d.contract(tn)
    assert isinstance(result, mp.mpf), 'Contracted tensor network is not an mp.mpf'
    assert 0 <= result <= 1, 'Contracted tensor network not within bounds'
    # partial contraction
    # note: for optimization we contract tn from left to left_stop as bra common to all cosets
    left_stop = ((tn.shape[1] - 1) // 2) - 1
    bra, bra_mult = tt.mps2d.contract(tn, stop=left_stop)
    # empty bra if None
    bra = np.full(tn.shape[0], None, dtype=object) if bra is None else bra
    print('len(bra)=', len(bra))
    assert isinstance(bra, np.ndarray), 'Partially contracted tensor network is not an np.array'
    assert isinstance(bra_mult, mp.mpf), 'Partially contracted tensor network multiplier is not an mp.mpf'
    # note: for optimization we contract tn from right to right_stop as ket common to all cosets
    right_stop = left_stop + 2
    ket, ket_mult = tt.mps2d.contract(tn, start=-1, stop=right_stop, step=-1)
    # empty ket if None
    ket = np.full(tn.shape[0], None, dtype=object) if ket is None else ket
    print('len(ket)=', len(ket))
    assert isinstance(ket, np.ndarray), 'Partially contracted tensor network is not an np.array'
    assert isinstance(ket_mult, mp.mpf), 'Partially contracted tensor network multiplier is not an mp.mpf'
    # stack bra, remaining tn and ket
    middle = tn[:, left_stop:right_stop + 1]
    print('middle.shape=', middle.shape)
    partial_tn = np.column_stack((bra, middle, ket))
    result_from_partial = tt.mps2d.contract(partial_tn)
    assert isinstance(result_from_partial, mp.mpf), 'Contracted tensor network (from partial) is not an mp.mpf'
    assert 0 <= result_from_partial <= 1, 'Contracted tensor network (from partial) not within bounds'
    assert result == result_from_partial, 'Results contracted (one go) and (from partial) differ'
    print(result, result_from_partial)
