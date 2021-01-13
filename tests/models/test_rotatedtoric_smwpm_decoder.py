import numpy as np
import pytest

from qecsim import paulitools as pt
from qecsim.models.generic import BiasedDepolarizingErrorModel, BitPhaseFlipErrorModel, DepolarizingErrorModel
from qecsim.models.generic import BitFlipErrorModel
from qecsim.models.rotatedtoric import RotatedToricCode, RotatedToricSMWPMDecoder
from qecsim.models.rotatedtoric import _rotatedtoricsmwpmdecoder as _rtsd


# utility functions
def _print_clusters(code, clusters, debug=False):
    # all clusters on single lattice with clusters distinguished alphabetically
    plaquette_labels = {}
    import string
    for cluster, label in zip(clusters, string.ascii_uppercase + '*' * len(clusters)):
        plaquette_labels.update(((x, y), label) for t, x, y in cluster)
    print(code.ascii_art(plaquette_labels=plaquette_labels))
    if debug:
        # lattice for each cluster with order plaquettes in cluster given alphabetically
        for cluster in clusters:
            plaquette_labels = {(x, y): la for (t, x, y), la in zip(cluster, string.ascii_letters + '*' * len(cluster))}
            print(code.ascii_art(plaquette_labels=plaquette_labels))


def _code_error_syndrome(code, error_dicts, measurement_error_indices):
    """
    Resolve to code, error and periodic FT syndrome.

    :param code: Rotated toric code
    :type code: RotatedToricCode
    :param error_dicts: List of error dicts, e.g. [{'X': [(0, 0)]}, {'Y': [(1, 1), (1, 2)]}, ...]
    :type error_dicts: list of dict
    :param measurement_error_lists: List of measurement error indices, e.g. [[(1, 1)], [(1, 1), (2, 1), ...] ]
    :type measurement_error_lists: list of list
    :return: Code, Error, Periodic syndrome, Step errors, Step measurement errors
    :rtype: RotatedToricCode, np.array (1d), np.array (2d), list of np.array (1d), list of np.array (1d)
    """
    assert len(error_dicts) == len(measurement_error_indices)
    step_errors = []
    step_syndromes = []
    for error_dict in error_dicts:
        step_error_pauli = code.new_pauli()
        for op, indices in error_dict.items():
            step_error_pauli.site(op, *indices)
        step_error = step_error_pauli.to_bsf()
        step_errors.append(step_error)
        step_syndromes.append(pt.bsp(step_error, code.stabilizers.T))
    step_measurement_errors = []
    for indices in measurement_error_indices:
        step_measurement_error = []
        for index in code._plaquette_indices:
            step_measurement_error.append(1 if index in indices else 0)
        step_measurement_errors.append(np.array(step_measurement_error))
    error = np.bitwise_xor.reduce(step_errors)
    syndrome = []
    for t in range(len(step_syndromes)):
        syndrome.append(step_measurement_errors[t - 1] ^ step_syndromes[t] ^ step_measurement_errors[t])
    syndrome = np.array(syndrome)
    return code, error, syndrome, step_errors, step_measurement_errors


# @pytest.mark.perf
# def test_rotated_toric_smwpm_perf():
#     from click.testing import CliRunner
#     from qecsim import app
#     with CliRunner().isolated_filesystem():
#         code = RotatedToricCode(20, 20)
#         error_model = BiasedDepolarizingErrorModel(bias=100)
#         decoder = RotatedToricSMWPMDecoder()
#         error_probability = 0.2
#         app.run(code, error_model, decoder, error_probability, max_runs=5)


# @pytest.mark.perf
# def test_rotated_toric_smwpm_perf_ftp():
#     from click.testing import CliRunner
#     from qecsim import app
#     with CliRunner().isolated_filesystem():
#         code = RotatedToricCode(12, 12)
#         time_steps = 12
#         error_model = BiasedDepolarizingErrorModel(bias=10)
#         # error_model = BitPhaseFlipErrorModel()
#         decoder = RotatedToricSMWPMDecoder()
#         error_probability = 0.05
#         app.run_ftp(code, time_steps, error_model, decoder, error_probability, max_runs=5)


def test_rotated_toric_smwpm_decoder_properties():
    decoder = RotatedToricSMWPMDecoder(itp=True, eta=10)
    assert isinstance(decoder.label, str)
    assert isinstance(repr(decoder), str)
    assert isinstance(str(decoder), str)


@pytest.mark.parametrize('eta', [
    None,
    0.5,
    10,
])
def test_rotated_toric_smwpm_decoder_new_valid_parameters(eta):
    RotatedToricSMWPMDecoder(eta=eta)  # no error raised


@pytest.mark.parametrize('eta', [
    0,
    -1,
    'c',
])
def test_rotated_toric_smwpm_decoder_new_invalid_parameters(eta):
    with pytest.raises((ValueError, TypeError), match=r"^RotatedToricSMWPMDecoder") as exc_info:
        RotatedToricSMWPMDecoder(eta=eta)
    print(exc_info)


@pytest.mark.parametrize('error_model, expected', [
    (BitPhaseFlipErrorModel(), None),
    (DepolarizingErrorModel(), 0.5),
    (BiasedDepolarizingErrorModel(0.5), 0.5),
    (BiasedDepolarizingErrorModel(10), 10),
    # this model should not be used with the decoder but we just test the bias method works here
    (BiasedDepolarizingErrorModel(10, axis='Z'), 0.04761904761904762),
])
def test_rotated_toric_smwpm_decoder_bias(error_model, expected):
    bias = RotatedToricSMWPMDecoder()._bias(error_model)
    assert bias == expected, 'Bias not as expected'


@pytest.mark.parametrize('error_model', [
    BitFlipErrorModel(),
])
def test_rotated_toric_smwpm_decoder_bias_invalid(error_model):
    with pytest.raises(ValueError):
        RotatedToricSMWPMDecoder()._bias(error_model)


@pytest.mark.parametrize('error_model', [
    BitPhaseFlipErrorModel(),
    DepolarizingErrorModel(),
    BiasedDepolarizingErrorModel(0.5),
    BiasedDepolarizingErrorModel(10),
    BitFlipErrorModel(),
])
def test_rotated_toric_smwpm_decoder_bias_override(error_model):
    eta = 10
    bias = RotatedToricSMWPMDecoder(eta=eta)._bias(error_model)
    assert bias == eta, 'Bias not overridden by eta'


@pytest.mark.parametrize('code', [
    RotatedToricCode(2, 2),
    RotatedToricCode(2, 4),
    RotatedToricCode(4, 2),
    RotatedToricCode(4, 4),
])
def test_rotated_toric_smwpm_decoder_plaquette_indices(code):
    plaquette_indices = _rtsd._plaquette_indices(code)
    # check size
    n_rows, n_cols = code.size
    assert plaquette_indices.shape == (n_rows, n_cols), 'plaquette_indices wrong shape'
    # unpack into set to check uniqueness
    plaquette_indices_set = set((x, y) for x, y in plaquette_indices.flatten())
    assert len(plaquette_indices_set) == n_rows * n_cols, 'plaquette_indices not unique'


@pytest.mark.parametrize('code, a, b, expected', [
    # code, ((t, x, y), is_row), ((t, x, y), is_row), expected_distance

    # in bulk cases
    (RotatedToricCode(4, 6), ((0, 0, 0), True), ((0, 2, 0), True), 2),  # row: in bulk
    (RotatedToricCode(4, 6), ((0, 0, 0), False), ((0, 0, 1), False), 1),  # col: in bulk

    # around bulk cases
    (RotatedToricCode(4, 6), ((0, 0, 0), True), ((0, 5, 0), True), 1),  # row: in bulk
    (RotatedToricCode(4, 6), ((0, 0, 0), False), ((0, 0, 3), False), 1),  # col: in bulk
])
def test_rotated_toric_smwpm_decoder_distance(code, a, b, expected):
    assert _rtsd._distance(code, 1, a, b) == expected, 'Distance not as expected'


@pytest.mark.parametrize('code, time_steps, a, b, error_probability, measurement_error_probability, eta', [
    # code, ((t, x, y), is_row), ((t, x, y), is_row)
    (RotatedToricCode(6, 6), 1, ((0, 0, 0), True), ((0, 2, 0), False), None, None, None),  # orthogonals
    (RotatedToricCode(6, 6), 1, ((0, 0, 0), True), ((0, 0, 2), True), None, None, None),  # distinct rows inf bias
    (RotatedToricCode(6, 6), 1, ((0, 0, 0), False), ((0, 2, 0), False), None, None, None),  # distinct cols inf bias
    (RotatedToricCode(6, 6), 5, ((0, 0, 0), True), ((2, 0, 0), True), None, None, 10),  # invalid q for time steps
    (RotatedToricCode(6, 6), 5, ((0, 0, 0), True), ((2, 0, 0), True), None, 0, 10),  # invalid q for time steps
    (RotatedToricCode(6, 6), 5, ((0, 0, 0), True), ((2, 0, 0), True), None, 1, 10),  # invalid q for time steps
    (RotatedToricCode(6, 6), 5, ((0, 0, 0), True), ((0, 2, 0), True), None, None, 10),  # invalid p for parallel steps
    (RotatedToricCode(6, 6), 5, ((0, 0, 0), True), ((0, 2, 0), True), 0, None, 10),  # invalid p for parallel steps
    (RotatedToricCode(6, 6), 5, ((0, 0, 0), False), ((0, 2, 0), False), None, None, 10),  # invalid p for diagonal stps
    (RotatedToricCode(6, 6), 5, ((0, 0, 0), False), ((0, 2, 0), False), 0, None, 10),  # invalid p for diagonal steps
    (RotatedToricCode(6, 6), 5, ((0, 0, 0), False), ((0, 2, 0), False), 0.2, 0.1, None),  # inf eta for diagonal steps

])
def test_rotated_toric_smwpm_decoder_distance_invalid(code, time_steps, a, b, error_probability,
                                                      measurement_error_probability, eta):
    with pytest.raises(ValueError):
        _rtsd._distance(code, time_steps, a, b, error_probability, measurement_error_probability, eta)


def test_rotated_toric_smwpm_decoder_space_step_weights():
    p = 0.1
    eta = 0.5
    parallel_step_wt_half = _rtsd._step_weight_parallel(eta, p)
    diagonal_step_wt_half = _rtsd._step_weight_diagonal(eta, p)
    print('eta=', eta, 'p=', p, 'parallel_wt=', parallel_step_wt_half, 'diagonal_wt=', diagonal_step_wt_half)
    assert 0 < parallel_step_wt_half == diagonal_step_wt_half
    eta = 10
    parallel_step_wt_10 = _rtsd._step_weight_parallel(eta, p)
    diagonal_step_wt_10 = _rtsd._step_weight_diagonal(eta, p)
    print('eta=', eta, 'p=', p, 'parallel_wt=', parallel_step_wt_10, 'diagonal_wt=', diagonal_step_wt_10)
    assert 0 < parallel_step_wt_10 < diagonal_step_wt_10
    eta = 100
    parallel_step_wt_100 = _rtsd._step_weight_parallel(eta, p)
    diagonal_step_wt_100 = _rtsd._step_weight_diagonal(eta, p)
    print('eta=', eta, 'p=', p, 'parallel_wt=', parallel_step_wt_100, 'diagonal_wt=', diagonal_step_wt_100)
    assert 0 < parallel_step_wt_100 < diagonal_step_wt_100
    assert 0 < parallel_step_wt_100 < parallel_step_wt_10
    assert 0 < diagonal_step_wt_10 < diagonal_step_wt_100
    eta = 1000
    parallel_step_wt_1000 = _rtsd._step_weight_parallel(eta, p)
    diagonal_step_wt_1000 = _rtsd._step_weight_diagonal(eta, p)
    print('eta=', eta, 'p=', p, 'parallel_wt=', parallel_step_wt_1000, 'diagonal_wt=', diagonal_step_wt_1000)
    assert 0 < parallel_step_wt_1000 < diagonal_step_wt_1000
    assert 0 < parallel_step_wt_1000 < parallel_step_wt_100
    assert 0 < diagonal_step_wt_100 < diagonal_step_wt_1000


def test_rotated_toric_smwpm_decoder_time_step_weights_ftp():
    # infinite bias
    eta = None
    time_step_wt_10pc = _rtsd._step_weight_time(0.1)
    parallel_step_wt_10pc = _rtsd._step_weight_parallel(eta, 0.1)
    time_step_wt_20pc = _rtsd._step_weight_time(0.2)
    parallel_step_wt_20pc = _rtsd._step_weight_parallel(eta, 0.2)
    assert 0 < time_step_wt_20pc < time_step_wt_10pc
    assert 0 < parallel_step_wt_20pc < parallel_step_wt_10pc
    assert time_step_wt_10pc == parallel_step_wt_10pc
    assert time_step_wt_20pc == parallel_step_wt_20pc
    # finite bias
    eta = 100
    parallel_step_wt_10pc = _rtsd._step_weight_parallel(eta, 0.1)
    parallel_step_wt_20pc = _rtsd._step_weight_parallel(eta, 0.2)
    assert 0 < time_step_wt_10pc < parallel_step_wt_10pc
    assert 0 < time_step_wt_20pc < parallel_step_wt_20pc


def test_rotated_toric_smwpm_decoder_step_weights_invalid_ftp():
    with pytest.raises(ValueError):
        _rtsd._step_weight_time(q=None)
    with pytest.raises(ValueError):
        _rtsd._step_weight_time(q=0)
    with pytest.raises(ValueError):
        _rtsd._step_weight_time(q=1)
    with pytest.raises(ValueError):
        _rtsd._step_weight_parallel(eta=100, p=None)
    with pytest.raises(ValueError):
        _rtsd._step_weight_parallel(eta=100, p=0)
    with pytest.raises(ValueError):
        _rtsd._step_weight_diagonal(eta=100, p=None)
    with pytest.raises(ValueError):
        _rtsd._step_weight_diagonal(eta=100, p=0)
    with pytest.raises(ValueError):
        _rtsd._step_weight_diagonal(eta=None, p=0.1)


@pytest.mark.parametrize('code, time_steps, a, b, eta, exp_delta_time, exp_delta_parallel, exp_delta_diagonal', [

    # code, time_steps, ((t, x, y), is_row), ((t, x, y), is_row), eta, expected_deltas

    # eta = 10
    # between rows
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 0, 0), True), 10, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 0, 1), True), 10, 0, 1, 1),  # row: 0x1 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 1, 0), True), 10, 0, 1, 0),  # row: 1x0 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 0, 2), True), 10, 0, 0, 2),  # row: 0x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 2, 0), True), 10, 0, 2, 0),  # row: 2x0 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 1, 1), True), 10, 0, 0, 1),  # row: 1x1 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 2, 2), True), 10, 0, 0, 2),  # row: 2x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 3, 2), True), 10, 0, 1, 2),  # row: 3x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 2, 3), True), 10, 0, 1, 3),  # row: 2x3 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 4, 2), True), 10, 0, 2, 2),  # row: 4x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 2, 4), True), 10, 0, 0, 4),  # row: 2x4 box
    # between rows (across boundary)
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 0, 0), True), 10, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedToricCode(12, 12), 1, ((0, 0, 11), True), ((0, 0, 0), True), 10, 0, 1, 1),  # row: 0x1 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 0), True), ((0, 0, 0), True), 10, 0, 1, 0),  # row: 1x0 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 11), True), ((0, 0, 1), True), 10, 0, 0, 2),  # row: 0x2 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 0), True), ((0, 1, 0), True), 10, 0, 2, 0),  # row: 2x0 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 11), True), ((0, 0, 0), True), 10, 0, 0, 1),  # row: 1x1 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 11), True), ((0, 1, 1), True), 10, 0, 0, 2),  # row: 2x2 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 11), True), ((0, 2, 1), True), 10, 0, 1, 2),  # row: 3x2 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 11), True), ((0, 1, 2), True), 10, 0, 1, 3),  # row: 2x3 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 11), True), ((0, 3, 1), True), 10, 0, 2, 2),  # row: 4x2 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 11), True), ((0, 1, 3), True), 10, 0, 0, 4),  # row: 2x4 box
    # between columns
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 0, 0), False), 10, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 0, 1), False), 10, 0, 1, 0),  # row: 0x1 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 1, 0), False), 10, 0, 1, 1),  # row: 1x0 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 0, 2), False), 10, 0, 2, 0),  # row: 0x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 2, 0), False), 10, 0, 0, 2),  # row: 2x0 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 1, 1), False), 10, 0, 0, 1),  # row: 1x1 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 2, 2), False), 10, 0, 0, 2),  # row: 2x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 3, 2), False), 10, 0, 1, 3),  # row: 3x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 2, 3), False), 10, 0, 1, 2),  # row: 2x3 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 4, 2), False), 10, 0, 0, 4),  # row: 4x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 2, 4), False), 10, 0, 2, 2),  # row: 2x4 box
    # between columns (across boundary)
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 0, 0), False), 10, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedToricCode(12, 12), 1, ((0, 0, 11), False), ((0, 0, 0), False), 10, 0, 1, 0),  # row: 0x1 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 0), False), ((0, 0, 0), False), 10, 0, 1, 1),  # row: 1x0 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 11), False), ((0, 0, 1), False), 10, 0, 2, 0),  # row: 0x2 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 0), False), ((0, 1, 0), False), 10, 0, 0, 2),  # row: 2x0 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 11), False), ((0, 0, 0), False), 10, 0, 0, 1),  # row: 1x1 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 11), False), ((0, 1, 1), False), 10, 0, 0, 2),  # row: 2x2 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 11), False), ((0, 2, 1), False), 10, 0, 1, 3),  # row: 3x2 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 11), False), ((0, 1, 2), False), 10, 0, 1, 2),  # row: 2x3 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 11), False), ((0, 3, 1), False), 10, 0, 0, 4),  # row: 4x2 box
    (RotatedToricCode(12, 12), 1, ((0, 11, 11), False), ((0, 1, 3), False), 10, 0, 2, 2),  # row: 2x4 box
    # between columns and times
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((0, 0, 0), False), 10, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((1, 0, 1), False), 10, 1, 1, 0),  # row: 0x1 box
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((2, 1, 0), False), 10, 2, 1, 1),  # row: 1x0 box
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((3, 0, 2), False), 10, 3, 2, 0),  # row: 0x2 box
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((4, 2, 0), False), 10, 4, 0, 2),  # row: 2x0 box
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((5, 1, 1), False), 10, 5, 0, 1),  # row: 1x1 box
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((6, 2, 2), False), 10, 6, 0, 2),  # row: 2x2 box
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((7, 3, 2), False), 10, 5, 1, 3),  # row: 3x2 box
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((8, 2, 3), False), 10, 4, 1, 2),  # row: 2x3 box
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((9, 4, 2), False), 10, 3, 0, 4),  # row: 4x2 box
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((10, 2, 4), False), 10, 2, 2, 2),  # row: 2x4 box
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((11, 2, 4), False), 10, 1, 2, 2),  # row: 2x4 box

    # eta = 100
    # between rows
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 0, 0), True), 100, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 0, 1), True), 100, 0, 1, 1),  # row: 0x1 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 1, 0), True), 100, 0, 1, 0),  # row: 1x0 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 0, 2), True), 100, 0, 0, 2),  # row: 0x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 2, 0), True), 100, 0, 2, 0),  # row: 2x0 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 1, 1), True), 100, 0, 0, 1),  # row: 1x1 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 2, 2), True), 100, 0, 0, 2),  # row: 2x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 3, 2), True), 100, 0, 1, 2),  # row: 3x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 2, 3), True), 100, 0, 1, 3),  # row: 2x3 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 4, 2), True), 100, 0, 2, 2),  # row: 4x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 2, 4), True), 100, 0, 0, 4),  # row: 2x4 box
    # between columns
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 0, 0), False), 100, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 0, 1), False), 100, 0, 1, 0),  # row: 0x1 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 1, 0), False), 100, 0, 1, 1),  # row: 1x0 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 0, 2), False), 100, 0, 2, 0),  # row: 0x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 2, 0), False), 100, 0, 0, 2),  # row: 2x0 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 1, 1), False), 100, 0, 0, 1),  # row: 1x1 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 2, 2), False), 100, 0, 0, 2),  # row: 2x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 3, 2), False), 100, 0, 1, 3),  # row: 3x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 2, 3), False), 100, 0, 1, 2),  # row: 2x3 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 4, 2), False), 100, 0, 0, 4),  # row: 4x2 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 2, 4), False), 100, 0, 2, 2),  # row: 2x4 box

    # eta = None
    # between rows
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 0, 0), True), None, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 1, 0), True), None, 0, 1, 0),  # row: 1x0 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), True), ((0, 2, 0), True), None, 0, 2, 0),  # row: 2x0 box
    # between columns
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 0, 0), False), None, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 0, 1), False), None, 0, 1, 0),  # row: 0x1 box
    (RotatedToricCode(12, 12), 1, ((0, 0, 0), False), ((0, 0, 2), False), None, 0, 2, 0),  # row: 0x2 box
    # between columns and time
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((0, 0, 0), False), None, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((4, 0, 1), False), None, 4, 1, 0),  # row: 0x1 box
    (RotatedToricCode(12, 12), 12, ((0, 0, 0), False), ((9, 0, 2), False), None, 3, 2, 0),  # row: 0x2 box

])
def test_rotated_toric_smwpm_decoder_distance_ftp(code, time_steps, a, b, eta, exp_delta_time, exp_delta_parallel,
                                                  exp_delta_diagonal):
    p, q = 0.2, 0.1
    expected_distance = 0
    if exp_delta_time:
        expected_distance += exp_delta_time * _rtsd._step_weight_time(q)
    if exp_delta_parallel:
        expected_distance += exp_delta_parallel * _rtsd._step_weight_parallel(eta, p)
    if exp_delta_diagonal:
        expected_distance += exp_delta_diagonal * _rtsd._step_weight_diagonal(eta, p)
    assert _rtsd._distance(code, time_steps, a, b, eta=eta,
                           error_probability=p, measurement_error_probability=q) == expected_distance, (
        'Distance with bias not as expected')


@pytest.mark.parametrize('error_pauli, expected', [
    # nodes (within edges) are sorted for comparison (automatically in test).
    # edges are in an unsorted set.
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2)),  # bulk
     {(((0, 1, 1), True), ((0, 2, 1), True), 1),  # bottom edge
      (((0, 1, 2), True), ((0, 2, 2), True), 1),  # top edge
      (((0, 1, 2), False), ((0, 1, 1), False), 1),  # left edge
      (((0, 2, 2), False), ((0, 2, 1), False), 1),  # right edge
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2), (3, 2)),  # bulk
     {(((0, 1, 1), True), ((0, 3, 1), True), 2),  # bottom edge
      (((0, 1, 2), True), ((0, 3, 2), True), 2),  # top edge
      (((0, 1, 2), False), ((0, 1, 1), False), 1),  # left edge
      (((0, 3, 2), False), ((0, 3, 1), False), 1),  # right edge
      }),
])
def test_rotated_toric_smwpm_decoder_graph(error_pauli, expected):
    # parameters
    code = error_pauli.code
    time_steps = 1
    error = error_pauli.to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    # ensure syndrome is 2d
    syndrome = np.expand_dims(syndrome, axis=0)
    # call
    graphs = _rtsd._graphs(code, time_steps, syndrome)
    # prepare actual (sort nodes within edges, and extract a_node, b_node, weight)
    actual = {(*sorted((a_node, b_node)), wt) for graph in graphs for (a_node, b_node), wt in graph.items()}
    # prepare expected (sort nodes within edges)
    expected = set((*sorted((a_node, b_node)), weight) for a_node, b_node, weight in expected)
    # check
    assert actual == expected, 'Edges between plaquettes not as expected.'


@pytest.mark.parametrize('error_pauli, eta, error_probability, expected', [
    # nodes (within edges) are sorted for comparison (manually).
    # edges are sorted by weight, a_node, b_node, where True > False (manually).
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2)).site('X', (3, 3)),  # bulk
     100, 0.1,
     # nodes: (1, 1), (1, 2), (2, 1), (3, 3)
     ((((0, 1, 1), False), ((0, 1, 2), False)),  # 0 diagonal + 1 parallel
      (((0, 1, 1), True), ((0, 2, 1), True)),
      (((0, 1, 2), False), ((0, 2, 1), False)),  # 1 diagonal + 0 parallel
      (((0, 1, 2), True), ((0, 2, 1), True)),
      (((0, 1, 1), False), ((0, 2, 1), False)),  # 1 diagonal + 1 parallel
      (((0, 1, 1), True), ((0, 1, 2), True)),
      (((0, 1, 2), True), ((0, 3, 3), True)),
      (((0, 2, 1), False), ((0, 3, 3), False)),
      (((0, 1, 1), False), ((0, 3, 3), False)),  # 2 diagonal + 0 parallel
      (((0, 1, 1), True), ((0, 3, 3), True)),
      (((0, 1, 2), False), ((0, 3, 3), False)),  # 2 diagonal + 1 parallel
      (((0, 2, 1), True), ((0, 3, 3), True)),
      )),
])
def test_rotated_toric_smwpm_decoder_graph_with_bias(error_pauli, eta, error_probability, expected):
    # parameters
    code = error_pauli.code
    time_steps = 1
    error = error_pauli.to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    # ensure syndrome is 2d
    syndrome = np.expand_dims(syndrome, axis=0)
    # call
    graphs = _rtsd._graphs(code, time_steps, syndrome, error_probability, eta=eta)
    # prepare actual
    # i.e. sort edges by weight, a_node, b_node where nodes are sorted within edges, and extract a_node, b_node
    actual = tuple((a_node, b_node) for wt, a_node, b_node in
                   sorted((wt, *sorted((a_node, b_node))) for graph in graphs
                          for (a_node, b_node), wt in graph.items()))
    # check
    assert actual == expected, 'Edges between plaquettes not as expected.'


@pytest.mark.parametrize('code, error, syndrome, step_errors, step_measurement_errors, expected', [
    # nodes (within edges) are sorted for comparison (automatically in test).
    # edges are in an unsorted set.

    (*_code_error_syndrome(  # 2 time-steps, 1 Y in bulk, 1 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [],
        ]),
     {(((0, 1, 1), True), ((0, 2, 1), True), 1),  # bottom edge
      (((0, 1, 2), True), ((1, 2, 2), True), 2),  # top edge
      (((0, 1, 2), False), ((0, 1, 1), False), 1),  # left edge
      (((1, 2, 2), False), ((0, 2, 1), False), 2),  # right edge
      }),
])
def test_rotated_toric_smwpm_decoder_graph_ftp(code, error, syndrome, step_errors, step_measurement_errors, expected):
    # call
    graphs = _rtsd._graphs(code, len(syndrome), syndrome)
    # prepare actual (sort nodes within edges, and extract a_node, b_node, weight)
    actual = set((*sorted((a_node, b_node)), wt) for graph in graphs for (a_node, b_node), wt in graph.items())
    # prepare expected (sort nodes within edges)
    expected = set((*sorted((a_node, b_node)), weight) for a_node, b_node, weight in expected)
    # check
    assert actual == expected, 'Edges between plaquettes not as expected.'


@pytest.mark.parametrize('error_pauli, expected', [
    # edges starting up from bottom left and then clockwise

    # BULK
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2)),  # . bulk
     {frozenset({((0, 1, 1), False), ((0, 1, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((0, 2, 2), True)}),  # right
      frozenset({((0, 2, 2), False), ((0, 2, 1), False)}),  # down
      frozenset({((0, 2, 1), True), ((0, 1, 1), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2), (3, 2)),  # .. bulk
     {frozenset({((0, 1, 1), False), ((0, 1, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((0, 3, 2), True)}),  # right
      frozenset({((0, 3, 2), False), ((0, 3, 1), False)}),  # down
      frozenset({((0, 3, 1), True), ((0, 1, 1), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2), (2, 3)),  # : bulk
     {frozenset({((0, 1, 1), False), ((0, 1, 3), False)}),  # up
      frozenset({((0, 1, 3), True), ((0, 2, 3), True)}),  # right
      frozenset({((0, 2, 3), False), ((0, 2, 1), False)}),  # down
      frozenset({((0, 2, 1), True), ((0, 1, 1), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2), (2, 3), (3, 3), (3, 2)),  # :: bulk
     {frozenset({((0, 1, 1), False), ((0, 1, 3), False)}),  # up
      frozenset({((0, 1, 3), True), ((0, 3, 3), True)}),  # right
      frozenset({((0, 3, 3), False), ((0, 3, 1), False)}),  # down
      frozenset({((0, 3, 1), True), ((0, 1, 1), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2), (3, 3)),  # / in bulk
     {frozenset({((0, 1, 1), False), ((0, 1, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((0, 3, 2), True)}),  # right
      frozenset({((0, 3, 2), False), ((0, 3, 3), False)}),  # up
      frozenset({((0, 3, 3), True), ((0, 2, 3), True)}),  # left
      frozenset({((0, 2, 3), False), ((0, 2, 1), False)}),  # down
      frozenset({((0, 2, 1), True), ((0, 1, 1), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 3), (3, 2)),  # \ in bulk
     {frozenset({((0, 1, 2), False), ((0, 1, 3), False)}),  # up
      frozenset({((0, 1, 3), True), ((0, 2, 3), True)}),  # right
      frozenset({((0, 2, 3), False), ((0, 2, 1), False)}),  # down
      frozenset({((0, 2, 1), True), ((0, 3, 1), True)}),  # right
      frozenset({((0, 3, 1), False), ((0, 3, 2), False)}),  # up
      frozenset({((0, 3, 2), True), ((0, 1, 2), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2), (2, 3), (3, 2)),  # :. in bulk
     {frozenset({((0, 1, 1), False), ((0, 1, 3), False)}),  # up
      frozenset({((0, 1, 3), True), ((0, 2, 3), True)}),  # right
      frozenset({((0, 2, 3), False), ((0, 2, 2), False)}),  # down
      frozenset({((0, 2, 2), True), ((0, 3, 2), True)}),  # right
      frozenset({((0, 3, 2), False), ((0, 3, 1), False)}),  # down
      frozenset({((0, 3, 1), True), ((0, 1, 1), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 3), (3, 3), (3, 2)),  # ': in bulk
     {frozenset({((0, 1, 2), False), ((0, 1, 3), False)}),  # up
      frozenset({((0, 1, 3), True), ((0, 3, 3), True)}),  # right
      frozenset({((0, 3, 3), False), ((0, 3, 1), False)}),  # down
      frozenset({((0, 3, 1), True), ((0, 2, 1), True)}),  # left
      frozenset({((0, 2, 1), False), ((0, 2, 2), False)}),  # up
      frozenset({((0, 2, 2), True), ((0, 1, 2), True)}),  # left
      }),

    # CORNER SW
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0)),  # . in sw corner
     {frozenset({((0, 5, 5), False), ((0, 5, 0), False)}),  # up
      frozenset({((0, 5, 0), True), ((0, 0, 0), True)}),  # right
      frozenset({((0, 0, 0), False), ((0, 0, 5), False)}),  # down
      frozenset({((0, 0, 5), True), ((0, 5, 5), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (1, 0)),  # .. in sw corner
     {frozenset({((0, 5, 5), False), ((0, 5, 0), False)}),  # up
      frozenset({((0, 5, 0), True), ((0, 1, 0), True)}),  # right
      frozenset({((0, 1, 0), False), ((0, 1, 5), False)}),  # down
      frozenset({((0, 1, 5), True), ((0, 5, 5), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (0, 1)),  # : in sw corner
     {frozenset({((0, 5, 5), False), ((0, 5, 1), False)}),  # up
      frozenset({((0, 5, 1), True), ((0, 0, 1), True)}),  # right
      frozenset({((0, 0, 1), False), ((0, 0, 5), False)}),  # down
      frozenset({((0, 0, 5), True), ((0, 5, 5), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (0, 1), (1, 1), (1, 0)),  # :: in sw corner
     {frozenset({((0, 5, 5), False), ((0, 5, 1), False)}),  # up
      frozenset({((0, 5, 1), True), ((0, 1, 1), True)}),  # right
      frozenset({((0, 1, 1), False), ((0, 1, 5), False)}),  # down
      frozenset({((0, 1, 5), True), ((0, 5, 5), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (1, 0), (2, 0)),  # ... in sw corner
     {frozenset({((0, 5, 5), False), ((0, 5, 0), False)}),  # up
      frozenset({((0, 5, 0), True), ((0, 2, 0), True)}),  # right
      frozenset({((0, 2, 0), False), ((0, 2, 5), False)}),  # down
      frozenset({((0, 2, 5), True), ((0, 5, 5), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (0, 1), (0, 2)),  # ! in sw corner
     {frozenset({((0, 5, 5), False), ((0, 5, 2), False)}),  # up
      frozenset({((0, 5, 2), True), ((0, 0, 2), True)}),  # right
      frozenset({((0, 0, 2), False), ((0, 0, 5), False)}),  # down
      frozenset({((0, 0, 5), True), ((0, 5, 5), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (1, 1)),  # / in sw corner
     {frozenset({((0, 5, 5), False), ((0, 5, 0), False)}),  # up
      frozenset({((0, 5, 0), True), ((0, 1, 0), True)}),  # right
      frozenset({((0, 1, 0), False), ((0, 1, 1), False)}),  # up
      frozenset({((0, 1, 1), True), ((0, 0, 1), True)}),  # left
      frozenset({((0, 0, 1), False), ((0, 0, 5), False)}),  # down
      frozenset({((0, 0, 5), True), ((0, 5, 5), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 1), (1, 0)),  # \ in sw corner
     {frozenset({((0, 5, 0), False), ((0, 5, 1), False)}),  # up
      frozenset({((0, 5, 1), True), ((0, 0, 1), True)}),  # right
      frozenset({((0, 0, 1), False), ((0, 0, 5), False)}),  # down
      frozenset({((0, 0, 5), True), ((0, 1, 5), True)}),  # right
      frozenset({((0, 1, 5), False), ((0, 1, 0), False)}),  # up
      frozenset({((0, 1, 0), True), ((0, 5, 0), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (0, 1), (1, 0)),  # :. in sw corner
     {frozenset({((0, 5, 5), False), ((0, 5, 1), False)}),  # up
      frozenset({((0, 5, 1), True), ((0, 0, 1), True)}),  # right
      frozenset({((0, 0, 1), False), ((0, 0, 0), False)}),  # down
      frozenset({((0, 0, 0), True), ((0, 1, 0), True)}),  # right
      frozenset({((0, 1, 0), False), ((0, 1, 5), False)}),  # down
      frozenset({((0, 1, 5), True), ((0, 5, 5), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 1), (1, 1), (1, 0)),  # ': in sw corner
     {frozenset({((0, 5, 0), False), ((0, 5, 1), False)}),  # up
      frozenset({((0, 5, 1), True), ((0, 1, 1), True)}),  # right
      frozenset({((0, 1, 1), False), ((0, 1, 5), False)}),  # down
      frozenset({((0, 1, 5), True), ((0, 0, 5), True)}),  # left
      frozenset({((0, 0, 5), False), ((0, 0, 0), False)}),  # up
      frozenset({((0, 0, 0), True), ((0, 5, 0), True)}),  # left
      }),

    # CORNERS (NW, NE SE)
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 5)),  # . in nw corner
     {frozenset({((0, 5, 4), False), ((0, 5, 5), False)}),  # up
      frozenset({((0, 5, 5), True), ((0, 0, 5), True)}),  # right
      frozenset({((0, 0, 5), False), ((0, 0, 4), False)}),  # down
      frozenset({((0, 0, 4), True), ((0, 5, 4), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (5, 5)),  # . in ne corner
     {frozenset({((0, 4, 4), False), ((0, 4, 5), False)}),  # up
      frozenset({((0, 4, 5), True), ((0, 5, 5), True)}),  # right
      frozenset({((0, 5, 5), False), ((0, 5, 4), False)}),  # down
      frozenset({((0, 5, 4), True), ((0, 4, 4), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (5, 0)),  # . in se corner
     {frozenset({((0, 4, 5), False), ((0, 4, 0), False)}),  # up
      frozenset({((0, 4, 0), True), ((0, 5, 0), True)}),  # right
      frozenset({((0, 5, 0), False), ((0, 5, 5), False)}),  # down
      frozenset({((0, 5, 5), True), ((0, 4, 5), True)}),  # left
      }),

    # BOUNDARIES (N, E, S, W)
    (RotatedToricCode(6, 6).new_pauli().site('Y', (1, 5)),  # . on n boundary
     {frozenset({((0, 0, 4), False), ((0, 0, 5), False)}),  # up
      frozenset({((0, 0, 5), True), ((0, 1, 5), True)}),  # right
      frozenset({((0, 1, 5), False), ((0, 1, 4), False)}),  # down
      frozenset({((0, 1, 4), True), ((0, 0, 4), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (1, 5), (2, 5)),  # .. on n boundary
     {frozenset({((0, 0, 4), False), ((0, 0, 5), False)}),  # up
      frozenset({((0, 0, 5), True), ((0, 2, 5), True)}),  # right
      frozenset({((0, 2, 5), False), ((0, 2, 4), False)}),  # down
      frozenset({((0, 2, 4), True), ((0, 0, 4), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (5, 2)),  # . on e boundary
     {frozenset({((0, 4, 1), False), ((0, 4, 2), False)}),  # up
      frozenset({((0, 4, 2), True), ((0, 5, 2), True)}),  # right
      frozenset({((0, 5, 2), False), ((0, 5, 1), False)}),  # down
      frozenset({((0, 5, 1), True), ((0, 4, 1), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (5, 2), (5, 3)),  # : on e boundary
     {frozenset({((0, 4, 1), False), ((0, 4, 3), False)}),  # up
      frozenset({((0, 4, 3), True), ((0, 5, 3), True)}),  # right
      frozenset({((0, 5, 3), False), ((0, 5, 1), False)}),  # down
      frozenset({((0, 5, 1), True), ((0, 4, 1), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (1, 0)),  # . on s boundary
     {frozenset({((0, 0, 5), False), ((0, 0, 0), False)}),  # up
      frozenset({((0, 0, 0), True), ((0, 1, 0), True)}),  # right
      frozenset({((0, 1, 0), False), ((0, 1, 5), False)}),  # down
      frozenset({((0, 1, 5), True), ((0, 0, 5), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (1, 0), (2, 0)),  # .. on s boundary
     {frozenset({((0, 0, 5), False), ((0, 0, 0), False)}),  # up
      frozenset({((0, 0, 0), True), ((0, 2, 0), True)}),  # right
      frozenset({((0, 2, 0), False), ((0, 2, 5), False)}),  # down
      frozenset({((0, 2, 5), True), ((0, 0, 5), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 2)),  # . on w boundary
     {frozenset({((0, 5, 1), False), ((0, 5, 2), False)}),  # up
      frozenset({((0, 5, 2), True), ((0, 0, 2), True)}),  # right
      frozenset({((0, 0, 2), False), ((0, 0, 1), False)}),  # down
      frozenset({((0, 0, 1), True), ((0, 5, 1), True)}),  # left
      }),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 2), (0, 3)),  # : on w boundary
     {frozenset({((0, 5, 1), False), ((0, 5, 3), False)}),  # up
      frozenset({((0, 5, 3), True), ((0, 0, 3), True)}),  # right
      frozenset({((0, 0, 3), False), ((0, 0, 1), False)}),  # down
      frozenset({((0, 0, 1), True), ((0, 5, 1), True)}),  # left
      }),
])
def test_rotated_toric_smwpm_decoder_matching(error_pauli, expected):
    # parameters
    code = error_pauli.code
    time_steps = 1
    error = error_pauli.to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    # ensure syndrome is 2d
    syndrome = np.expand_dims(syndrome, axis=0)
    # calls
    graphs = _rtsd._graphs(code, time_steps, syndrome)
    matches = _rtsd._matching(graphs)
    # prepare actual (filter out same index mates and convert mates to frozenset)
    actual = set()
    for ((a_index, a_is_row), (b_index, b_is_row)) in matches:
        if a_index != b_index:
            actual.add(frozenset(((a_index, a_is_row), (b_index, b_is_row))))
    # check
    assert actual == expected, 'Matches not as expected.'


@pytest.mark.parametrize('error_pauli, eta, error_probability, expected', [
    # edges starting up from bottom left and then clockwise

    # . bulk, eta=None (infinite)
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2)),
     None, 0.1,
     {frozenset({((0, 1, 1), False), ((0, 1, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((0, 2, 2), True)}),  # right
      frozenset({((0, 2, 2), False), ((0, 2, 1), False)}),  # down
      frozenset({((0, 2, 1), True), ((0, 1, 1), True)}),  # left
      }),
    # / bulk, eta=3
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2)).site('X', (3, 3)),  # / in center of bulk, eta=10
     10, 0.1,
     {frozenset({((0, 1, 1), False), ((0, 1, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((0, 3, 3), True)}),  # right (and up)
      frozenset({((0, 3, 3), False), ((0, 2, 1), False)}),  # down (and left)
      frozenset({((0, 2, 1), True), ((0, 1, 1), True)}),  # left
      }),
])
def test_rotated_toric_smwpm_decoder_matching_with_bias(error_pauli, eta, error_probability, expected):
    # parameters
    code = error_pauli.code
    time_steps = 1
    error = error_pauli.to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    # ensure syndrome is 2d
    syndrome = np.expand_dims(syndrome, axis=0)
    # calls
    graphs = _rtsd._graphs(code, time_steps, syndrome, error_probability, eta=eta)
    matches = _rtsd._matching(graphs)
    # prepare actual (convert mates to frozenset)
    actual = set()
    for ((a_index, a_is_row), (b_index, b_is_row)) in matches:
        if a_index != b_index:
            actual.add(frozenset(((a_index, a_is_row), (b_index, b_is_row))))
    # check
    assert actual == expected, 'Matches not as expected.'


@pytest.mark.parametrize('code, error, syndrome, step_errors, step_measurement_errors, p, q, eta, expected', [
    # edges starting up from bottom left and then clockwise

    (*_code_error_syndrome(  # 2 time-steps, Y in bulk, 1 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [],
        ]),
     None, None, None,
     {frozenset({((0, 1, 1), False), ((0, 1, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((1, 2, 2), True)}),  # right
      frozenset({((1, 2, 2), False), ((0, 2, 1), False)}),  # down
      frozenset({((0, 2, 1), True), ((0, 1, 1), True)}),  # left
      }),

    (*_code_error_syndrome(  # 2 time-steps, Y in bulk, 2 measurement errors
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(1, 1), (2, 2)],
            [],
        ]),
     0.1, 0.1, 10,
     {frozenset({((1, 1, 1), False), ((0, 1, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((1, 2, 2), True)}),  # right
      frozenset({((1, 2, 2), False), ((0, 2, 1), False)}),  # down
      frozenset({((0, 2, 1), True), ((1, 1, 1), True)}),  # left
      }),

    (*_code_error_syndrome(  # 2 time-steps, Y in bulk, 2 measurement errors (low q so no matches between time steps)
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(1, 1), (2, 2)],
            [],
        ]),
     0.1, 0.01, 10,
     {frozenset({((1, 1, 1), False), ((1, 2, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((0, 2, 1), True)}),  # right
      frozenset({((0, 1, 2), False), ((0, 2, 1), False)}),  # down
      frozenset({((1, 2, 2), True), ((1, 1, 1), True)}),  # left
      }),

])
def test_rotated_toric_smwpm_decoder_matching_ftp(code, error, syndrome, step_errors, step_measurement_errors,
                                                  p, q, eta, expected):
    # parameters
    # calls
    graphs = _rtsd._graphs(code, len(syndrome), syndrome, p, q, eta)
    matches = _rtsd._matching(graphs)
    # prepare actual (filter out same index mates and convert mates to frozenset)
    actual = set()
    for ((a_index, a_is_row), (b_index, b_is_row)) in matches:
        if a_index != b_index:
            actual.add(frozenset(((a_index, a_is_row), (b_index, b_is_row))))
    # check
    assert actual == expected, 'Matches not as expected.'


@pytest.mark.parametrize('code, error, syndrome, step_errors, step_measurement_errors, p, q, eta, expected', [
    # edges starting up from bottom left and then clockwise

    (*_code_error_syndrome(  # 6 time-steps, 0 qubit errors, 1 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {}, {}, {}, {}, {}, {},
        ],
        [  # step_measurement_errors
            [(2, 2)], [], [], [], [], [],
        ]),
     0, 0.04, None,
     {frozenset({((1, 2, 2), False), ((0, 2, 2), False)}),
      frozenset({((1, 2, 2), True), ((0, 2, 2), True)}),
      }),
    (*_code_error_syndrome(  # 6 time-steps, 0 qubit errors, 1 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {}, {}, {}, {}, {}, {},
        ],
        [  # step_measurement_errors
            [], [(2, 2)], [], [], [], [],
        ]),
     0, 0.04, None,
     {frozenset({((1, 2, 2), False), ((2, 2, 2), False)}),
      frozenset({((1, 2, 2), True), ((2, 2, 2), True)}),
      }),
    (*_code_error_syndrome(  # 6 time-steps, 0 qubit errors, 3 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {}, {}, {}, {}, {}, {},
        ],
        [  # step_measurement_errors
            [], [(2, 2)], [(2, 2)], [(2, 2)], [], [],
        ]),
     0, 0.04, None,
     {frozenset({((1, 2, 2), False), ((4, 2, 2), False)}),
      frozenset({((1, 2, 2), True), ((4, 2, 2), True)}),
      }),
    (*_code_error_syndrome(  # 6 time-steps, 0 qubit errors, 3 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {}, {}, {}, {}, {}, {},
        ],
        [  # step_measurement_errors
            [(2, 2)], [], [], [], [(2, 2)], [(2, 2)],
        ]),
     0, 0.04, None,
     {frozenset({((1, 2, 2), False), ((4, 2, 2), False)}),
      frozenset({((1, 2, 2), True), ((4, 2, 2), True)}),
      }),
    (*_code_error_syndrome(  # 6 time-steps, 0 qubit errors, 1 measurement errors
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {}, {}, {}, {}, {}, {},
        ],
        [  # step_measurement_errors
            [], [], [], [], [], [(2, 2)],
        ]),
     0, 0.04, None,
     {frozenset({((0, 2, 2), False), ((5, 2, 2), False)}),
      frozenset({((0, 2, 2), True), ((5, 2, 2), True)}),
      }),
    (*_code_error_syndrome(  # 6 time-steps, 0 qubit errors, 2 measurement errors
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {}, {}, {}, {}, {}, {},
        ],
        [  # step_measurement_errors
            [(2, 2)], [], [], [], [], [(2, 2)],
        ]),
     0, 0.04, None,
     {frozenset({((1, 2, 2), False), ((5, 2, 2), False)}),
      frozenset({((1, 2, 2), True), ((5, 2, 2), True)}),
      }),
])
def test_rotated_toric_smwpm_decoder_matching_tparity(code, error, syndrome, step_errors, step_measurement_errors,
                                                      p, q, eta, expected):
    # parameters
    # calls
    graphs = _rtsd._graphs(code, len(syndrome), syndrome, p, q, eta)
    matches = _rtsd._matching(graphs)
    # prepare actual (filter out same index mates and convert mates to frozenset)
    actual = set()
    for ((a_index, a_is_row), (b_index, b_is_row)) in matches:
        if a_index != b_index:
            actual.add(frozenset(((a_index, a_is_row), (b_index, b_is_row))))
    # check
    assert actual == expected, 'Matches not as expected.'


@pytest.mark.parametrize('error_pauli, expected', [
    # edges starting up from bottom left and then clockwise

    # BULK
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2)),  # . bulk
     [[(0, 1, 1), (0, 1, 2), (0, 2, 2), (0, 2, 1)],
      ]),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2), (3, 3)),  # / in bulk
     [[(0, 1, 1), (0, 1, 2), (0, 3, 2), (0, 3, 3), (0, 2, 3), (0, 2, 1)],
      ]),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 3), (3, 2)),  # \ in bulk
     [[(0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 2, 1), (0, 3, 1), (0, 3, 2)],
      ]),

    # CORNER SW
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0)),  # . in sw corner
     [[(0, 0, 0), (0, 0, 5), (0, 5, 5), (0, 5, 0)],
      ]),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (1, 1)),  # / in sw corner
     [[(0, 0, 1), (0, 0, 5), (0, 5, 5), (0, 5, 0), (0, 1, 0), (0, 1, 1)],
      ]),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 1), (1, 0)),  # \ in sw corner
     [[(0, 0, 1), (0, 0, 5), (0, 1, 5), (0, 1, 0), (0, 5, 0), (0, 5, 1)],
      ]),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (1, 1), (2, 2)),  # .*' in sw corner
     [[(0, 0, 1), (0, 0, 5), (0, 5, 5), (0, 5, 0), (0, 1, 0), (0, 1, 2), (0, 2, 2), (0, 2, 1)],
      ]),

    # CORNERS (NW, NE SE)
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 5)),  # . in nw corner
     [[(0, 0, 4), (0, 0, 5), (0, 5, 5), (0, 5, 4)],
      ]),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (5, 5)),  # . in ne corner
     [[(0, 4, 4), (0, 4, 5), (0, 5, 5), (0, 5, 4)],
      ]),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (5, 0)),  # . in se corner
     [[(0, 4, 0), (0, 4, 5), (0, 5, 5), (0, 5, 0)],
      ]),

    # TWO CLUSTERS
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (2, 2)),  # . in sw corner and . in bulk
     [[(0, 0, 0), (0, 0, 5), (0, 5, 5), (0, 5, 0)],
      [(0, 1, 1), (0, 1, 2), (0, 2, 2), (0, 2, 1)],
      ]),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (1, 1), (4, 4)),  # / in sw corner and . near ne corner
     [[(0, 0, 1), (0, 0, 5), (0, 5, 5), (0, 5, 0), (0, 1, 0), (0, 1, 1)],
      [(0, 3, 3), (0, 3, 4), (0, 4, 4), (0, 4, 3)],
      ]),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (1, 1), (5, 5)),  # / in sw corner and . in ne corner
     [[(0, 0, 1), (0, 0, 5), (0, 4, 5), (0, 4, 4), (0, 5, 4), (0, 5, 0), (0, 1, 0), (0, 1, 1)],
      ]),
])
def test_rotated_toric_smwpm_decoder_clusters(error_pauli, expected):
    print()
    print('error:')
    print(error_pauli)
    # parameters
    code = error_pauli.code
    time_steps = 1
    error = error_pauli.to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    print()
    print('syndrome:')
    print(code.ascii_art(syndrome=syndrome))
    # ensure syndrome is 2d
    syndrome = np.expand_dims(syndrome, axis=0)
    # calls
    graphs = _rtsd._graphs(code, time_steps, syndrome)
    matches = _rtsd._matching(graphs)
    clusters = _rtsd._clusters(matches)
    print('### clusters=', clusters)
    print('### expected=', expected)
    _print_clusters(code, clusters)
    # check
    assert clusters == expected, 'Clusters not as expected.'


@pytest.mark.parametrize('error_pauli, eta, error_probability, expected', [
    # edges starting up from bottom left and then clockwise

    # BULK
    # . in bulk, eta=None (infinite)
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2)),
     None, 0.1,
     [[(0, 1, 1), (0, 1, 2), (0, 2, 2), (0, 2, 1)],
      ]),
    # / in bulk, eta=10
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2)).site('X', (3, 3)),
     10, 0.1,
     [[(0, 1, 1), (0, 1, 2), (0, 3, 3), (0, 2, 1)],
      ]),

    # TWO CLUSTERS
    # . in sw corner and . in bulk, eta=10
    (RotatedToricCode(8, 8).new_pauli().site('Y', (0, 0), (3, 3)).site('X', (4, 4)),
     10, 0.1,
     [[(0, 0, 0), (0, 0, 7), (0, 7, 7), (0, 7, 0)],
      [(0, 2, 2), (0, 2, 3), (0, 4, 4), (0, 3, 2)],
      ]),

    # ISOLATED Y DEFECTS
    # Y defects in SW and NE corners, eta=10
    (RotatedToricCode(8, 8).new_pauli()
     .site('X', (0, 1), (1, 1), (7, 1), (6, 0), (6, 7))
     .site('Z', (2, 1), (2, 0), (2, 7), (1, 7), (0, 7), (7, 7), (6, 7)),
     10, 0.1,
     [[(0, 1, 1), (0, 2, 1)],
      [(0, 5, 6), (0, 6, 6)],
      ]),
    # Y defects in SW and NE corners, eta=10
    (RotatedToricCode(10, 10).new_pauli()
     .site('X', (0, 1), (1, 1), (9, 1), (8, 0), (8, 9))
     .site('Z', (2, 1), (2, 0), (2, 9), (1, 9), (0, 9), (9, 9), (8, 9)),
     10, 0.1,
     [[(0, 1, 1), (0, 2, 1)],
      [(0, 7, 8), (0, 8, 8)],
      ]),
    # Y defects in SW and NE corners and neutral cluster in center, eta=10
    (RotatedToricCode(10, 10).new_pauli()
     .site('X', (0, 1), (1, 1), (9, 1), (8, 0), (8, 9))
     .site('Z', (2, 1), (2, 0), (2, 9), (1, 9), (0, 9), (9, 9), (8, 9))
     .site('Y', (4, 5), (5, 5), (6, 5)),
     10, 0.1,
     [[(0, 1, 1), (0, 2, 1)],
      [(0, 3, 4), (0, 3, 5), (0, 6, 5), (0, 6, 4)],
      [(0, 7, 8), (0, 8, 8)],
      ]),
    # Y defects in SW and NE corners and neutral cluster in center, eta=10
    (RotatedToricCode(12, 12).new_pauli()
     .site('X', (0, 1), (1, 1), (11, 1), (10, 0), (10, 11))
     .site('Z', (2, 1), (2, 0), (2, 11), (1, 11), (0, 11), (11, 11), (10, 11))
     .site('Y', (5, 6), (6, 6), (7, 6)),
     10, 0.1,
     [[(0, 1, 1), (0, 2, 1)],
      [(0, 4, 5), (0, 4, 6), (0, 7, 6), (0, 7, 5)],
      [(0, 9, 10), (0, 10, 10)],
      ]),
    # Y defects in SW and NE corners and 2 neutral clusters in center, eta=10
    (RotatedToricCode(12, 12).new_pauli()
     .site('X', (0, 1), (1, 1), (11, 1), (10, 0), (10, 11))
     .site('Z', (2, 1), (2, 0), (2, 11), (1, 11), (0, 11), (11, 11), (10, 11))
     .site('Y', (4, 5), (8, 7)),
     10, 0.1,
     [[(0, 1, 1), (0, 2, 1)],
      [(0, 3, 4), (0, 3, 5), (0, 4, 5), (0, 4, 4)],
      [(0, 7, 6), (0, 7, 7), (0, 8, 7), (0, 8, 6)],
      [(0, 9, 10), (0, 10, 10)],
      ]),
])
def test_rotated_toric_smwpm_decoder_clusters_with_bias(error_pauli, eta, error_probability, expected):
    print()
    print('error:')
    print(error_pauli)
    # parameters
    code = error_pauli.code
    time_steps = 1
    error = error_pauli.to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    print()
    print('syndrome:')
    print(code.ascii_art(syndrome=syndrome))
    # ensure syndrome is 2d
    syndrome = np.expand_dims(syndrome, axis=0)
    # calls
    graphs = _rtsd._graphs(code, time_steps, syndrome, error_probability, eta=eta)
    matches = _rtsd._matching(graphs)
    clusters = _rtsd._clusters(matches)
    print('### clusters=', clusters)
    print('### expected=', expected)
    _print_clusters(code, clusters)
    # check
    assert clusters == expected, 'Clusters not as expected.'


@pytest.mark.parametrize('code, error, syndrome, step_errors, step_measurement_errors, expected', [
    # edges starting up from bottom left and then clockwise

    (*_code_error_syndrome(  # 2 time-steps, Y in bulk, 1 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [],
        ]),
     [[(0, 1, 1), (0, 1, 2), (1, 2, 2), (0, 2, 1)],
      ]),
])
def test_rotated_toric_smwpm_decoder_clusters_ftp(code, error, syndrome, step_errors, step_measurement_errors,
                                                  expected):
    # calls
    graphs = _rtsd._graphs(code, len(syndrome), syndrome)
    matches = _rtsd._matching(graphs)
    clusters = _rtsd._clusters(matches)
    # check
    assert clusters == expected, 'Clusters not as expected.'


@pytest.mark.parametrize('code, error, syndrome, step_errors, step_measurement_errors, p, q, eta, expected', [
    # edges starting up from bottom left and then clockwise

    (*_code_error_syndrome(  # 6 time-steps, 0 qubit errors, 1 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {}, {}, {}, {}, {}, {},
        ],
        [  # step_measurement_errors
            [(2, 2)], [], [], [], [], [],
        ]),
     0, 0.04, None,
     [[(0, 2, 2), (1, 2, 2)],
      ]),
    (*_code_error_syndrome(  # 6 time-steps, 0 qubit errors, 1 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {}, {}, {}, {}, {}, {},
        ],
        [  # step_measurement_errors
            [], [(2, 2)], [], [], [], [],
        ]),
     0, 0.04, None,
     [[(1, 2, 2), (2, 2, 2)],
      ]),
    (*_code_error_syndrome(  # 6 time-steps, 0 qubit errors, 3 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {}, {}, {}, {}, {}, {},
        ],
        [  # step_measurement_errors
            [], [(2, 2)], [(2, 2)], [(2, 2)], [], [],
        ]),
     0, 0.04, None,
     [[(1, 2, 2), (4, 2, 2)],
      ]),
    (*_code_error_syndrome(  # 6 time-steps, 0 qubit errors, 3 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {}, {}, {}, {}, {}, {},
        ],
        [  # step_measurement_errors
            [(2, 2)], [], [], [], [(2, 2)], [(2, 2)],
        ]),
     0, 0.04, None,
     [[(1, 2, 2), (4, 2, 2)],
      ]),
    (*_code_error_syndrome(  # 6 time-steps, 0 qubit errors, 1 measurement errors
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {}, {}, {}, {}, {}, {},
        ],
        [  # step_measurement_errors
            [], [], [], [], [], [(2, 2)],
        ]),
     0, 0.04, None,
     [[(0, 2, 2), (5, 2, 2)],
      ]),
    (*_code_error_syndrome(  # 6 time-steps, 0 qubit errors, 2 measurement errors
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {}, {}, {}, {}, {}, {},
        ],
        [  # step_measurement_errors
            [(2, 2)], [], [], [], [], [(2, 2)],
        ]),
     0, 0.04, None,
     [[(1, 2, 2), (5, 2, 2)],
      ]),
])
def test_rotated_toric_smwpm_decoder_clusters_tparity(code, error, syndrome, step_errors, step_measurement_errors,
                                                      p, q, eta, expected):
    # calls
    graphs = _rtsd._graphs(code, len(syndrome), syndrome, p, q, eta)
    matches = _rtsd._matching(graphs)
    clusters = _rtsd._clusters(matches)
    # check
    assert clusters == expected, 'Clusters not as expected.'


@pytest.mark.parametrize('code, cluster, expected', [
    # On-lattice (no Y-defect)
    (RotatedToricCode(6, 6),
     [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],  # ZXZX
     ([(0, 0, 1), (0, 1, 0)], [(0, 0, 0), (0, 1, 1)], None)
     ),
    # On-lattice (no Y-defect, 1 measurement error)
    (RotatedToricCode(6, 6),
     [(0, 0, 0), (0, 0, 1), (1, 1, 1), (0, 1, 0)],  # ZXZX
     ([(0, 0, 1), (0, 1, 0)], [(0, 0, 0), (1, 1, 1)], None)
     ),
    # On-lattice (Y-defect)
    (RotatedToricCode(6, 6),
     [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 2, 0)],  # ZXZZ
     ([], [(0, 0, 0), (0, 1, 1)], ((0, 0, 1), (0, 2, 0)))
     ),
    # On-lattice (Y-defect, 1 measurement error)
    (RotatedToricCode(6, 6),
     [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 2, 0)],  # ZXZZ
     ([], [(0, 0, 0), (0, 1, 1)], ((0, 0, 1), (1, 2, 0)))
     ),
    # Off-lattice (no Y-defect)
    (RotatedToricCode(6, 6),
     [(0, 5, 5), (0, 5, 0), (0, 0, 0), (0, 0, 5)],  # ZXZX
     ([(0, 5, 0), (0, 0, 5)], [(0, 5, 5), (0, 0, 0)], None)
     ),
    # Off-lattice (no Y-defect, 1 measurement error)
    (RotatedToricCode(6, 6),
     [(0, 5, 5), (1, 5, 0), (0, 0, 0), (0, 0, 5)],  # ZXZX
     ([(1, 5, 0), (0, 0, 5)], [(0, 5, 5), (0, 0, 0)], None)
     ),
    # Off-lattice (Y-defect)
    (RotatedToricCode(6, 6),
     [(0, 5, 5), (0, 5, 0), (0, 1, 0), (0, 2, 5)],  # ZXXX
     ([(0, 5, 0), (0, 1, 0)], [], ((0, 2, 5), (0, 5, 5)))
     ),
    # Off-lattice (Y-defect, 1 measurment error)
    (RotatedToricCode(6, 6),
     [(1, 5, 5), (0, 5, 0), (0, 1, 0), (0, 2, 5)],  # ZXXX
     ([(0, 5, 0), (0, 1, 0)], [], ((0, 2, 5), (1, 5, 5)))
     ),
    # Partially off-lattice (Y-defect)
    (RotatedToricCode(6, 6),
     [(0, 0, 5), (0, 5, 0), (0, 1, 0), (0, 1, 5)],  # XXXZ
     ([(0, 0, 5), (0, 5, 0)], [], ((0, 1, 0), (0, 1, 5)))
     ),
    # Partially off-lattice (Y-defect, 1 measurement error)
    (RotatedToricCode(6, 6),
     [(0, 0, 5), (0, 5, 0), (0, 1, 0), (1, 1, 5)],  # XXXZ
     ([(0, 0, 5), (0, 5, 0)], [], ((0, 1, 0), (1, 1, 5)))
     ),
])
def test_rotated_toric_smwpm_decoder_cluster_to_paths_and_defect_ftp(code, cluster, expected):
    x_path, z_path, y_defect = _rtsd._cluster_to_paths_and_defect(code, cluster)
    print()
    print('actual:')
    print(x_path, z_path, y_defect)
    print()
    print('expected:')
    print(expected)
    assert (x_path, z_path, y_defect) == expected


@pytest.mark.parametrize('error_pauli', [
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (1, 1))),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 0), (4, 4))),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (0, 2), (1, 2))),
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 0), (2, 1))),
])
def test_rotated_toric_smwpm_decoder_decode(error_pauli):
    print()
    print('error:')
    print(error_pauli)
    error = error_pauli.to_bsf()
    code = error_pauli.code
    decoder = RotatedToricSMWPMDecoder()
    syndrome = pt.bsp(error, code.stabilizers.T)
    print()
    print('syndrome:')
    print(code.ascii_art(syndrome=syndrome))
    recovery = decoder.decode(code, syndrome)
    print()
    print('recovery:')
    print(code.new_pauli(recovery))
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


@pytest.mark.parametrize('code, error, syndrome, step_errors, step_measurement_errors', [

    _code_error_syndrome(  # 2 time-steps, Y in bulk, 1 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [],
        ]),
    _code_error_syndrome(  # 2 time-steps, Y in bulk, 2 measurement errors
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [(0, 0)],
        ]),
    _code_error_syndrome(  # 2 time-steps, 2 Y in bulk, 1 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(1, 1), (2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [],
        ]),
    _code_error_syndrome(  # 2 time-steps, 2 Y in bulk, 2 measurement errors
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(1, 1), (2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [(2, 3)],
        ]),
])
def test_rotated_toric_smwpm_decoder_decode_ftp(code, error, syndrome, step_errors, step_measurement_errors):
    print()
    print('error:')
    print(error)
    decoder = RotatedToricSMWPMDecoder(itp=True)
    print()
    print('syndrome:')
    print(syndrome)
    recovery = decoder.decode_ftp(code, len(syndrome), syndrome)
    print()
    print('recovery:')
    print(code.new_pauli(recovery))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


@pytest.mark.parametrize('time_steps, a_t, b_t, expected', [

    (1, 0, 0, 0),  # single time_step
    (1, 1, 0, 0),  # single time_step
    (1, 0, 1, 0),  # single time_step

    (6, 0, 0, 0),  # 0 to 0 skip -1
    (6, 0, 1, 0),  # 0 to 1 skip -1
    (6, 0, 2, 0),  # 0 to 2 skip -1
    (6, 0, 3, 0),  # 0 to 3 skip -1 (ambiguous)
    (6, 0, 4, 1),  # 0 to 4 thru -1
    (6, 0, 5, 1),  # 0 to 5 thru -1

    (6, 1, 0, 0),  # 1 to 0 skip -1
    (6, 1, 1, 0),  # 1 to 1 skip -1
    (6, 1, 2, 0),  # 1 to 2 skip -1
    (6, 1, 3, 0),  # 1 to 3 skip -1
    (6, 1, 4, 0),  # 1 to 4 skip -1 (ambiguous)
    (6, 1, 5, 1),  # 1 to 5 thru -1

    (6, 2, 0, 0),  # 2 to 0 skip -1
    (6, 2, 1, 0),  # 2 to 1 skip -1
    (6, 2, 2, 0),  # 2 to 2 skip -1
    (6, 2, 3, 0),  # 2 to 3 skip -1
    (6, 2, 4, 0),  # 2 to 4 skip -1
    (6, 2, 5, 0),  # 2 to 5 skip -1 (ambiguous)

    (6, 3, 0, 0),  # 3 to 0 thru -1 (ambiguous)
    (6, 3, 1, 0),  # 3 to 1 skip -1
    (6, 3, 2, 0),  # 3 to 2 skip -1
    (6, 3, 3, 0),  # 3 to 3 skip -1
    (6, 3, 4, 0),  # 3 to 4 skip -1
    (6, 3, 5, 0),  # 3 to 5 skip -1

    (6, 4, 0, 1),  # 4 to 0 skip -1
    (6, 4, 1, 0),  # 4 to 1 thru -1 (ambiguous)
    (6, 4, 2, 0),  # 4 to 2 skip -1
    (6, 4, 3, 0),  # 4 to 3 skip -1
    (6, 4, 4, 0),  # 4 to 4 skip -1
    (6, 4, 5, 0),  # 4 to 5 skip -1

    (6, 5, 0, 1),  # 5 to 0 skip -1
    (6, 5, 1, 1),  # 5 to 1 thru -1
    (6, 5, 2, 0),  # 5 to 2 thru -1 (ambiguous)
    (6, 5, 3, 0),  # 5 to 3 skip -1
    (6, 5, 4, 0),  # 5 to 4 skip -1
    (6, 5, 5, 0),  # 5 to 5 skip -1

    (7, 0, 0, 0),  # 0 to 0 skip -1
    (7, 0, 1, 0),  # 0 to 1 thru -1
    (7, 0, 2, 0),  # 0 to 2 thru -1
    (7, 0, 3, 0),  # 0 to 3 thru -1
    (7, 0, 4, 1),  # 0 to 4 skip -1
    (7, 0, 5, 1),  # 0 to 5 skip -1
    (7, 0, 6, 1),  # 0 to 5 skip -1

    (7, 6, 0, 1),  # 6 to 0 thru -1
    (7, 6, 1, 1),  # 6 to 1 skip -1
    (7, 6, 2, 1),  # 6 to 2 skip -1
    (7, 6, 3, 0),  # 6 to 3 skip -1
    (7, 6, 4, 0),  # 6 to 4 skip -1
    (7, 6, 5, 0),  # 6 to 5 thru -1
    (7, 6, 6, 0),  # 6 to 5 thru -1

    # modulo
    (7, -1, 0, 1),  # 6 to 0 thru -1
    (7, -1, -6, 1),  # 6 to 1 skip -1
    (7, -1, -5, 1),  # 6 to 2 skip -1
    (7, -1, -4, 0),  # 6 to 3 skip -1
    (7, -1, -3, 0),  # 6 to 4 skip -1
    (7, -1, -2, 0),  # 6 to 5 thru -1
    (7, -1, -1, 0),  # 6 to 5 thru -1
])
def test_rotated_toric_smwpm_decoder_tparity(time_steps, a_t, b_t, expected):
    tparity = _rtsd._tparity(time_steps, a_t, b_t)
    assert tparity == expected


def test_rotated_toric_smwpm_decoder_measurement_error_tparities():
    code = RotatedToricCode(6, 6)
    measurement_error_indices = {
        (0, 1),  # X-plaquettes
        (1, 1), (3, 1),  # Z-plaquettes
    }
    expected = 1, 0
    # build measurement error
    measurement_error = []
    for index in code._plaquette_indices:
        measurement_error.append(1 if index in measurement_error_indices else 0)
    measurement_error = np.array(measurement_error)
    # call
    measurement_error_tps = _rtsd._measurement_error_tparities(code, measurement_error)
    assert measurement_error_tps == expected


@pytest.mark.parametrize('code, error, syndrome, step_errors, step_measurement_errors', [

    _code_error_syndrome(  # 2 time-steps, Y in bulk, 1 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [],
        ]),
    _code_error_syndrome(  # 2 time-steps, Y in bulk, 2 measurement errors
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [(0, 0)],
        ]),
    _code_error_syndrome(  # 2 time-steps, 2 Y in bulk, 1 measurement error
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(1, 1), (2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [],
        ]),
    _code_error_syndrome(  # 2 time-steps, 2 Y in bulk, 2 measurement errors
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(1, 1), (2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [(2, 3)],
        ]),
])
def test_rotated_toric_smwpm_decoder_decode_ftp_tparity(code, error, syndrome, step_errors, step_measurement_errors):
    print()
    print('error:')
    print(error)
    decoder = RotatedToricSMWPMDecoder(itp=False)
    print()
    print('syndrome:')
    print(syndrome)
    decoding = decoder.decode_ftp(code, len(syndrome), syndrome, step_measurement_errors=step_measurement_errors)
    print()
    print('decoding:', decoding)
    if not isinstance(decoding, bool):
        recovery = decoding
        print()
        print('recovery:')
        print(code.new_pauli(recovery))
        assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
            'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


@pytest.mark.parametrize('code, time_steps, a_node, b_node, expected', [
    (RotatedToricCode(6, 6), 1,
     _rtsd._ClusterNode([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], (0, 0, 1), (0, 0, 0)),
     _rtsd._ClusterNode([(0, 2, 3), (0, 2, 4), (0, 3, 4), (0, 3, 3)], (0, 2, 3), (0, 2, 4)),
     3  # manhattan distance between (0, 1, 1) and (0, 2, 3)
     ),
    (RotatedToricCode(6, 6), 1,
     _rtsd._ClusterNode([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], (0, 0, 1), (0, 0, 0)),
     _rtsd._ClusterNode([(0, 1, 4), (0, 1, 5), (0, 2, 5), (0, 2, 4)], (0, 1, 4), (0, 1, 5)),
     1  # manhattan distance between (0, 1, 0) and (0, 1, 5) (periodic in y)
     ),
    (RotatedToricCode(6, 6), 1,
     _rtsd._ClusterNode([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], (0, 0, 1), (0, 0, 0)),
     _rtsd._ClusterNode([(0, 4, 1), (0, 4, 2)], (0, 4, 1), (0, 4, 2)),
     2  # manhattan distance between (0, 0, 1) and (0, 4, 1) (periodic in x)
     ),
    (RotatedToricCode(6, 6), 6,
     _rtsd._ClusterNode([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], (0, 0, 1), (0, 0, 0)),
     _rtsd._ClusterNode([(2, 2, 3), (2, 2, 4), (2, 3, 4), (2, 3, 3)], (2, 2, 3), (2, 2, 4)),
     5  # manhattan distance between (0, 1, 1) and (2, 2, 3)
     ),
    (RotatedToricCode(6, 6), 6,
     _rtsd._ClusterNode([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], (0, 0, 1), (0, 0, 0)),
     _rtsd._ClusterNode([(5, 2, 3), (5, 2, 4), (5, 3, 4), (5, 3, 3)], (5, 2, 3), (5, 2, 4)),
     4  # manhattan distance between (0, 1, 1) and (5, 2, 3) (periodic in t)
     ),
])
def test_rotated_toric_smwpm_decoder_cluster_distance_ftp(code, time_steps, a_node, b_node, expected):
    distance = _rtsd._cluster_distance(code, time_steps, a_node, b_node)
    assert distance == expected, 'Cluster distance not as expected'


def test_rotated_toric_smwpm_decoder_cluster_distance_twins_ftp():
    code = RotatedToricCode(6, 6)
    time_steps = 6
    a_node = _rtsd._ClusterNode([(0, 0, 0), (1, 0, 1), (2, 1, 1), (3, 1, 0)], (1, 0, 1), (0, 0, 0))
    b_node = _rtsd._ClusterNode([(0, 0, 0), (1, 0, 1), (2, 1, 1), (3, 1, 0)], (1, 0, 1), (0, 0, 0))
    distance = _rtsd._cluster_distance(code, time_steps, a_node, b_node)
    expected = 0  # manhattan distance between twin clusters
    assert distance == expected, 'Cluster distance not as expected'


def test_rotated_toric_smwpm_decoder_cluster_graph_and_matching():
    code = RotatedToricCode(8, 8)

    # TWO ISOLATED Y DEFECTS
    # Y defect in SW corner: [(0, 2, 1), (0, 1, 1)]
    # Y defect in NE corner: [(0, 5, 6), (0, 6, 6)]
    error = code.new_pauli()
    error = error.site('X', (0, 1), (1, 1), (7, 1), (6, 0), (6, 7))
    error = error.site('Z', (2, 1), (2, 0), (2, 7), (1, 7), (0, 7), (7, 7), (6, 7))
    error = error.to_bsf()

    syndrome = pt.bsp(error, code.stabilizers.T)
    # ensure syndrome is 2d (it would be 1d if time_steps == 1)
    syndrome = np.expand_dims(syndrome, axis=0) if syndrome.ndim == 1 else syndrome
    time_steps = len(syndrome)
    eta = 10
    error_probability = 0.1
    measurement_error_probability = 0.0
    # identity recovery
    recovery = code.new_pauli().to_bsf()
    recovery_x_tp = 0
    recovery_z_tp = 0
    # SYMMETRY DECODING
    # prepare graph
    graphs = _rtsd._graphs(code, time_steps, syndrome, error_probability, measurement_error_probability, eta)
    # minimum weight matching
    matches = _rtsd._matching(graphs)
    # cluster matches
    clusters = _rtsd._clusters(matches)
    # resolve symmetry recovery from fusing within clusters
    symmetry_recovery, symmetry_recovery_x_tp, symmetry_recovery_z_tp = _rtsd._recovery_tparities(
        code, time_steps, clusters)
    # add symmetry recovery and t-parities
    recovery ^= symmetry_recovery
    recovery_x_tp ^= symmetry_recovery_x_tp
    recovery_z_tp ^= symmetry_recovery_z_tp

    # RESIDUAL CLUSTER SYNDROME
    cluster_syndrome = np.bitwise_xor.reduce(syndrome) ^ pt.bsp(recovery, code.stabilizers.T)
    assert np.any(cluster_syndrome), 'There should be two isolated Y defects'

    # CLUSTER GRAPH
    cluster_graph = _rtsd._cluster_graph(code, time_steps, clusters)
    # prepare actual as {((x_index, z_index), (x_index, z_index), weight), ...}
    actual = set((*sorted(((a.x_index, a.z_index), (b.x_index, b.z_index))), weight)
                 for (a, b), weight in cluster_graph.items())
    # expected
    expected = {
        (((0, 2, 1), (0, 1, 1)), ((0, 5, 6), (0, 6, 6)), 6)
    }
    assert actual == expected, 'Cluster graph not as expected'

    # CLUSTER MATCHING
    cluster_matches = _rtsd._matching([cluster_graph])
    # prepare actual as {((x_index, z_index), (x_index, z_index)), ...} excluding matches between virtual nodes
    actual = set((*sorted(((a.x_index, a.z_index), (b.x_index, b.z_index))),) for a, b in cluster_matches)
    # expected
    expected = {(((0, 2, 1), (0, 1, 1)), ((0, 5, 6), (0, 6, 6)))}
    assert actual == expected, 'Cluster matches not as expected'


@pytest.mark.parametrize('error_pauli, bias, error_probability', [
    # BULK
    # . in bulk, bias=None (infinite)
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2)), None, 0.1),
    # / in bulk, bias=10
    (RotatedToricCode(6, 6).new_pauli().site('Y', (2, 2)).site('X', (3, 3)), 10, 0.1),
    # TWO CLUSTERS
    # . in sw corner and . in bulk, bias=10
    (RotatedToricCode(8, 8).new_pauli().site('Y', (0, 0), (3, 3)).site('X', (4, 4)), 10, 0.1),
    # ISOLATED Y DEFECTS
    # Y defects in SW and NE corners, eta=10
    (RotatedToricCode(8, 8).new_pauli()
     .site('X', (0, 1), (1, 1), (7, 1), (6, 0), (6, 7))
     .site('Z', (2, 1), (2, 0), (2, 7), (1, 7), (0, 7), (7, 7), (6, 7)),
     10, 0.1),
    # Y defects in SW and NE corners, eta=10
    (RotatedToricCode(10, 10).new_pauli()
     .site('X', (0, 1), (1, 1), (9, 1), (8, 0), (8, 9))
     .site('Z', (2, 1), (2, 0), (2, 9), (1, 9), (0, 9), (9, 9), (8, 9)),
     10, 0.1),
    # Y defects in SW and NE corners and neutral cluster in center, eta=10
    (RotatedToricCode(10, 10).new_pauli()
     .site('X', (0, 1), (1, 1), (9, 1), (8, 0), (8, 9))
     .site('Z', (2, 1), (2, 0), (2, 9), (1, 9), (0, 9), (9, 9), (8, 9))
     .site('Y', (4, 5), (5, 5), (6, 5)),
     10, 0.1),
    # Y defects in SW and NE corners and neutral cluster in center, eta=10
    (RotatedToricCode(12, 12).new_pauli()
     .site('X', (0, 1), (1, 1), (11, 1), (10, 0), (10, 11))
     .site('Z', (2, 1), (2, 0), (2, 11), (1, 11), (0, 11), (11, 11), (10, 11))
     .site('Y', (5, 6), (6, 6), (7, 6)),
     10, 0.1),
    # Y defects in SW and NE corners and 2 neutral clusters in center, eta=10
    (RotatedToricCode(12, 12).new_pauli()
     .site('X', (0, 1), (1, 1), (11, 1), (10, 0), (10, 11))
     .site('Z', (2, 1), (2, 0), (2, 11), (1, 11), (0, 11), (11, 11), (10, 11))
     .site('Y', (4, 5), (8, 7)),
     10, 0.1),
])
def test_rotated_toric_smwpm_decoder_decode_with_bias(error_pauli, bias, error_probability):
    print()
    print('error:')
    print(error_pauli)
    error = error_pauli.to_bsf()
    code = error_pauli.code
    decoder = RotatedToricSMWPMDecoder()
    error_model = BitPhaseFlipErrorModel() if bias is None else BiasedDepolarizingErrorModel(bias)
    syndrome = pt.bsp(error, code.stabilizers.T)
    print()
    print('syndrome:')
    print(code.ascii_art(syndrome=syndrome))
    recovery = decoder.decode(code, syndrome, error_model=error_model, error_probability=error_probability)
    print()
    print('recovery:')
    print(code.new_pauli(recovery))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


@pytest.mark.parametrize('code, error, syndrome, step_errors, step_measurement_errors, p, q, eta', [

    # 2 time-steps, . in bulk, bias=None (infinite), 1 measurement error
    (*_code_error_syndrome(
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            []
        ]),
     0.1, 0.1, None
     ),

    # 2 time-steps, / in bulk, bias=10, 1 measurement error
    (*_code_error_syndrome(
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)], 'X': [(3, 3)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            []
        ]),
     0.1, 0.05, 10
     ),
])
def test_rotated_toric_smwpm_decoder_decode_with_bias_ftp(code, error, syndrome, step_errors, step_measurement_errors,
                                                          p, q, eta):
    print()
    print('error:')
    print(error)
    decoder = RotatedToricSMWPMDecoder()
    error_model = BitPhaseFlipErrorModel() if eta is None else BiasedDepolarizingErrorModel(eta)
    print()
    print('syndrome:')
    print(syndrome)
    recovery = decoder.decode_ftp(code, len(syndrome), syndrome, error_model=error_model, error_probability=p,
                                  measurement_error_probability=q, step_measurement_errors=step_measurement_errors)
    print()
    print('recovery:')
    print(code.new_pauli(recovery))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


@pytest.mark.parametrize('code, error, syndrome, step_errors, step_measurement_errors, p, q, eta', [

    # 2 time-steps, / in bulk, bias=10, 1 measurement error
    (*_code_error_syndrome(
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)], 'X': [(3, 3)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            []
        ]),
     0.1, 0.05, 10
     ),

    # 2 time-steps, / in bulk, bias=10, 1 measurement error
    (*_code_error_syndrome(
        RotatedToricCode(6, 6),  # code
        [  # step_errors
            {'Y': [(2, 2)], 'X': [(3, 3)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [(2, 3)]
        ]),
     0.1, 0.05, 10
     ),
])
def test_rotated_toric_smwpm_decoder_decode_with_bias_ftp_tparity(code, error, syndrome, step_errors,
                                                                  step_measurement_errors, p, q, eta):
    print()
    print('error:')
    print(error)
    decoder = RotatedToricSMWPMDecoder(itp=False)
    error_model = BitPhaseFlipErrorModel() if eta is None else BiasedDepolarizingErrorModel(eta)
    print()
    print('syndrome:')
    print(syndrome)
    decoding = decoder.decode_ftp(code, len(syndrome), syndrome, error_model=error_model, error_probability=p,
                                  measurement_error_probability=q, step_measurement_errors=step_measurement_errors)
    print()
    print('decoding:', decoding)
    if not isinstance(decoding, bool):
        recovery = decoding
        print()
        print('recovery:')
        print(code.new_pauli(recovery))
        assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
            'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
