import numpy as np
import pytest

from qecsim import paulitools as pt
from qecsim.models.generic import BiasedDepolarizingErrorModel, BitPhaseFlipErrorModel, DepolarizingErrorModel
from qecsim.models.generic import BitFlipErrorModel
from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarSMWPMDecoder
from qecsim.models.rotatedplanar import _rotatedplanarsmwpmdecoder as _rpsd


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

    :param code: Rotated planar code
    :type code: RotatedPlanarCode
    :param error_dicts: List of error dicts, e.g. [{'X': [(0, 0)]}, {'Y': [(1, 1), (1, 2)]}, ...]
    :type error_dicts: list of dict
    :param measurement_error_lists: List of measurement error indices, e.g. [[(1, 1)], [(1, 1), (2, 1), ...] ]
    :type measurement_error_lists: list of list
    :return: Code, Error, Periodic syndrome
    :rtype: RotatedPlanarCode, np.array (1d), np.array (2d)
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
    return code, error, syndrome


# def test_rotated_planar_smwpm_perf():
#     from click.testing import CliRunner
#     from qecsim import app
#     with CliRunner().isolated_filesystem():
#         code = RotatedPlanarCode(21, 21)
#         error_model = BiasedDepolarizingErrorModel(bias=100)
#         decoder = RotatedPlanarSMWPMDecoder()
#         error_probability = 0.2
#         app.run(code, error_model, decoder, error_probability, max_runs=5)
#
#
# def test_rotated_planar_smwpm_perf_ftp():
#     from click.testing import CliRunner
#     from qecsim import app
#     with CliRunner().isolated_filesystem():
#         code = RotatedPlanarCode(11, 11)
#         time_steps = 11
#         error_model = BiasedDepolarizingErrorModel(bias=100)
#         decoder = RotatedPlanarSMWPMDecoder()
#         error_probability = 0.05
#         app.run_ftp(code, time_steps, error_model, decoder, error_probability, max_runs=5)


def test_rotated_planar_smwpm_decoder_properties():
    decoder = RotatedPlanarSMWPMDecoder(eta=10)
    assert isinstance(decoder.label, str)
    assert isinstance(repr(decoder), str)
    assert isinstance(str(decoder), str)


@pytest.mark.parametrize('eta', [
    None,
    0.5,
    10,
])
def test_rotated_planar_smwpm_decoder_new_valid_parameters(eta):
    RotatedPlanarSMWPMDecoder(eta=eta)  # no error raised


@pytest.mark.parametrize('eta', [
    0,
    -1,
    'c',
])
def test_rotated_planar_smwpm_decoder_new_invalid_parameters(eta):
    with pytest.raises((ValueError, TypeError), match=r"^RotatedPlanarSMWPMDecoder") as exc_info:
        RotatedPlanarSMWPMDecoder(eta=eta)
    print(exc_info)


@pytest.mark.parametrize('error_model, expected', [
    (BitPhaseFlipErrorModel(), None),
    (DepolarizingErrorModel(), 0.5),
    (BiasedDepolarizingErrorModel(0.5), 0.5),
    (BiasedDepolarizingErrorModel(10), 10),
    # this model should not be used with the decoder but we just test the bias method works here
    (BiasedDepolarizingErrorModel(10, axis='Z'), 0.04761904761904762),
])
def test_rotated_planar_smwpm_decoder_bias(error_model, expected):
    bias = RotatedPlanarSMWPMDecoder()._bias(error_model)
    assert bias == expected, 'Bias not as expected'


@pytest.mark.parametrize('error_model', [
    BitFlipErrorModel(),
])
def test_rotated_planar_smwpm_decoder_bias_invalid(error_model):
    with pytest.raises(ValueError):
        RotatedPlanarSMWPMDecoder()._bias(error_model)


@pytest.mark.parametrize('error_model', [
    BitPhaseFlipErrorModel(),
    DepolarizingErrorModel(),
    BiasedDepolarizingErrorModel(0.5),
    BiasedDepolarizingErrorModel(10),
    BitFlipErrorModel(),
])
def test_rotated_planar_smwpm_decoder_bias_override(error_model):
    eta = 10
    bias = RotatedPlanarSMWPMDecoder(eta=eta)._bias(error_model)
    assert bias == eta, 'Bias not overridden by eta'


@pytest.mark.parametrize('code', [
    RotatedPlanarCode(3, 3),
    RotatedPlanarCode(3, 4),
    RotatedPlanarCode(4, 3),
    RotatedPlanarCode(4, 4),
])
def test_rotated_planar_smwpm_decoder_plaquette_indices(code):
    plaquette_indices = _rpsd._plaquette_indices(code)
    # check size
    n_rows, n_cols = code.size
    assert plaquette_indices.shape == (n_rows + 1, n_cols + 1), 'plaquette_indices wrong shape'
    # unpack into set to check uniqueness
    plaquette_indices_set = set((x, y) for x, y in plaquette_indices.flatten())
    assert len(plaquette_indices_set) == (n_rows + 1) * (n_cols + 1), 'plaquette_indices not unique'
    # check virtual or in-bounds
    for index in plaquette_indices.flatten():
        assert code.is_virtual_plaquette(index) or code.is_in_plaquette_bounds(index), (
            'plaquette_indices are not virtual or in-bounds.')


@pytest.mark.parametrize('code, a, b, expected', [
    # code, ((t, x, y), is_row), ((t, x, y), is_row), expected_distance

    # bulk cases
    (RotatedPlanarCode(5, 5), ((0, 0, 0), True), ((0, 2, 0), True), 2),  # row: in bulk
    (RotatedPlanarCode(5, 5), ((0, 0, 0), False), ((0, 0, 2), False), 2),  # col: in bulk

    # bulk / boundary cases
    (RotatedPlanarCode(5, 5), ((0, -1, 2), True), ((0, 2, 2), True), 3),  # row: real boundary to bulk
    (RotatedPlanarCode(5, 5), ((0, 1, -1), False), ((0, 1, 2), False), 3),  # col: real boundary to bulk
    (RotatedPlanarCode(5, 5), ((0, 0, 1), True), ((0, 4, 1), True), 4),  # row: bulk to real boundary
    (RotatedPlanarCode(5, 5), ((0, 3, 3), False), ((0, 3, -1), False), 4),  # col: bulk to real boundary

    (RotatedPlanarCode(5, 5), ((0, -1, 1), True), ((0, 0, 1), True), 1),  # row: virt boundary to bulk
    (RotatedPlanarCode(5, 5), ((0, 1, 4), False), ((0, 1, 3), False), 1),  # col: virt boundary to bulk
    (RotatedPlanarCode(5, 5), ((0, 2, 2), True), ((0, 4, 2), True), 2),  # row: bulk to virt boundary
    (RotatedPlanarCode(5, 5), ((0, 2, 1), False), ((0, 2, -1), False), 2),  # col: bulk to virt boundary

    # boundary / boundary cases
    (RotatedPlanarCode(5, 5), ((0, 1, -1), True), ((0, 3, -1), True), 2),  # row: real to real boundary (bottom)
    (RotatedPlanarCode(5, 5), ((0, 0, 4), True), ((0, 2, 4), True), 2),  # row: real to real boundary (top)
    (RotatedPlanarCode(5, 5), ((0, -1, 0), False), ((0, -1, 2), False), 2),  # col: real to real boundary (left)
    (RotatedPlanarCode(5, 5), ((0, 4, 3), False), ((0, 4, 1), False), 2),  # col: real to real boundary (right)

    (RotatedPlanarCode(5, 5), ((0, 2, -1), True), ((0, 0, -1), True), 2),  # row: virt to virt boundary (bottom)
    (RotatedPlanarCode(5, 5), ((0, 1, 4), True), ((0, 3, 4), True), 2),  # row: virt to virt boundary (top)
    (RotatedPlanarCode(5, 5), ((0, -1, 3), False), ((0, -1, 1), False), 2),  # col: virt to virt boundary (left)
    (RotatedPlanarCode(5, 5), ((0, 4, 0), False), ((0, 4, 2), False), 2),  # col: virt to virt boundary (right)

    (RotatedPlanarCode(5, 5), ((0, 3, -1), True), ((0, 0, -1), True), 3),  # row: real to virt boundary (bottom)
    (RotatedPlanarCode(5, 5), ((0, 2, 4), True), ((0, 3, 4), True), 1),  # row: real to virt boundary (top)
    (RotatedPlanarCode(5, 5), ((0, -1, 0), False), ((0, -1, 3), False), 3),  # col: real to virt boundary (left)
    (RotatedPlanarCode(5, 5), ((0, 4, 3), False), ((0, 4, 2), False), 1),  # col: real to virt boundary (right)

    # corner / boundary cases (adjacent)
    (RotatedPlanarCode(5, 5), ((0, -1, -1), True), ((0, 0, -1), True), 1),  # row: virt to virt boundary (bottom)
    (RotatedPlanarCode(5, 5), ((0, 3, 4), True), ((0, 4, 4), True), 1),  # row: virt to virt boundary (top)
    (RotatedPlanarCode(5, 5), ((0, -1, 3), False), ((0, -1, 4), False), 1),  # col: virt to virt boundary (left)
    (RotatedPlanarCode(5, 5), ((0, 4, 0), False), ((0, 4, -1), False), 1),  # col: virt to virt boundary (right)

    # corner / boundary cases (non-adjacent)
    (RotatedPlanarCode(5, 5), ((0, -1, -1), True), ((0, 2, -1), True), 3),  # row: virt to virt boundary (bottom)
    (RotatedPlanarCode(5, 5), ((0, 1, 4), True), ((0, 4, 4), True), 3),  # row: virt to virt boundary (top)
    (RotatedPlanarCode(5, 5), ((0, -1, 4), False), ((0, -1, 1), False), 3),  # col: virt to virt boundary (left)
    (RotatedPlanarCode(5, 5), ((0, 4, -1), False), ((0, 4, 2), False), 3),  # col: virt to virtual boundary (right)

    (RotatedPlanarCode(5, 5), ((0, 3, -1), True), ((0, -1, -1), True), 4),  # row: real to virt boundary (bottom)
    (RotatedPlanarCode(5, 5), ((0, 2, 4), True), ((0, 4, 4), True), 2),  # row: real to virt boundary (top)
    (RotatedPlanarCode(5, 5), ((0, -1, 0), False), ((0, -1, 4), False), 4),  # col: real to virt boundary (left)
    (RotatedPlanarCode(5, 5), ((0, 4, 1), False), ((0, 4, -1), False), 2),  # col: real to virt boundary (right)
])
def test_rotated_planar_smwpm_decoder_distance(code, a, b, expected):
    assert _rpsd._distance(code, 1, a, b) == expected, 'Distance not as expected'


# @pytest.mark.parametrize('code, a, b', [
#     # code, ((t, x, y), is_row), ((t, x, y), is_row)
#     (RotatedPlanarCode(5, 5), ((0, 0, 0), True), ((0, 2, 0), False)),  # orthogonals
#     (RotatedPlanarCode(5, 5), ((0, 0, 0), True), ((0, 0, 2), True)),  # distinct rows with infinite bias
#     (RotatedPlanarCode(5, 5), ((0, 0, 0), False), ((0, 2, 0), False)),  # distinct columns with infinite bias
# ])
# def test_rotated_planar_smwpm_decoder_distance_invalid(code, a, b):
#     with pytest.raises(ValueError):
#         RotatedPlanarSMWPMDecoder._distance(code, 1, a, b)


@pytest.mark.parametrize('code, time_steps, a, b, error_probability, measurement_error_probability, eta', [
    # code, ((t, x, y), is_row), ((t, x, y), is_row)
    (RotatedPlanarCode(5, 5), 1, ((0, 0, 0), True), ((0, 2, 0), False), None, None, None),  # orthogonals
    (RotatedPlanarCode(5, 5), 1, ((0, 0, 0), True), ((0, 0, 2), True), None, None, None),  # distinct rows inf bias
    (RotatedPlanarCode(5, 5), 1, ((0, 0, 0), False), ((0, 2, 0), False), None, None, None),  # distinct cols inf bias
    (RotatedPlanarCode(5, 5), 5, ((0, 0, 0), True), ((2, 0, 0), True), None, None, 10),  # invalid q for time steps
    (RotatedPlanarCode(5, 5), 5, ((0, 0, 0), True), ((2, 0, 0), True), None, 0, 10),  # invalid q for time steps
    (RotatedPlanarCode(5, 5), 5, ((0, 0, 0), True), ((2, 0, 0), True), None, 1, 10),  # invalid q for time steps
    (RotatedPlanarCode(5, 5), 5, ((0, 0, 0), True), ((0, 2, 0), True), None, None, 10),  # invalid p for parallel steps
    (RotatedPlanarCode(5, 5), 5, ((0, 0, 0), True), ((0, 2, 0), True), 0, None, 10),  # invalid p for parallel steps
    (RotatedPlanarCode(5, 5), 5, ((0, 0, 0), False), ((0, 2, 0), False), None, None, 10),  # invalid p for diagonal stps
    (RotatedPlanarCode(5, 5), 5, ((0, 0, 0), False), ((0, 2, 0), False), 0, None, 10),  # invalid p for diagonal steps
    (RotatedPlanarCode(5, 5), 5, ((0, 0, 0), False), ((0, 2, 0), False), 0.2, 0.1, None),  # inf eta for diagonal steps

])
def test_rotated_planar_smwpm_decoder_distance_invalid(code, time_steps, a, b, error_probability,
                                                       measurement_error_probability, eta):
    with pytest.raises(ValueError):
        _rpsd._distance(code, time_steps, a, b, error_probability, measurement_error_probability, eta)


def test_rotated_planar_smwpm_decoder_space_step_weights():
    p = 0.1
    eta = 0.5
    parallel_step_wt_half = _rpsd._step_weight_parallel(eta, p)
    diagonal_step_wt_half = _rpsd._step_weight_diagonal(eta, p)
    print('eta=', eta, 'p=', p, 'parallel_wt=', parallel_step_wt_half, 'diagonal_wt=', diagonal_step_wt_half)
    assert 0 < parallel_step_wt_half == diagonal_step_wt_half
    eta = 10
    parallel_step_wt_10 = _rpsd._step_weight_parallel(eta, p)
    diagonal_step_wt_10 = _rpsd._step_weight_diagonal(eta, p)
    print('eta=', eta, 'p=', p, 'parallel_wt=', parallel_step_wt_10, 'diagonal_wt=', diagonal_step_wt_10)
    assert 0 < parallel_step_wt_10 < diagonal_step_wt_10
    eta = 100
    parallel_step_wt_100 = _rpsd._step_weight_parallel(eta, p)
    diagonal_step_wt_100 = _rpsd._step_weight_diagonal(eta, p)
    print('eta=', eta, 'p=', p, 'parallel_wt=', parallel_step_wt_100, 'diagonal_wt=', diagonal_step_wt_100)
    assert 0 < parallel_step_wt_100 < diagonal_step_wt_100
    assert 0 < parallel_step_wt_100 < parallel_step_wt_10
    assert 0 < diagonal_step_wt_10 < diagonal_step_wt_100
    eta = 1000
    parallel_step_wt_1000 = _rpsd._step_weight_parallel(eta, p)
    diagonal_step_wt_1000 = _rpsd._step_weight_diagonal(eta, p)
    print('eta=', eta, 'p=', p, 'parallel_wt=', parallel_step_wt_1000, 'diagonal_wt=', diagonal_step_wt_1000)
    assert 0 < parallel_step_wt_1000 < diagonal_step_wt_1000
    assert 0 < parallel_step_wt_1000 < parallel_step_wt_100
    assert 0 < diagonal_step_wt_100 < diagonal_step_wt_1000


def test_rotated_planar_smwpm_decoder_time_step_weights_ftp():
    # infinite bias
    eta = None
    time_step_wt_10pc = _rpsd._step_weight_time(0.1)
    parallel_step_wt_10pc = _rpsd._step_weight_parallel(eta, 0.1)
    time_step_wt_20pc = _rpsd._step_weight_time(0.2)
    parallel_step_wt_20pc = _rpsd._step_weight_parallel(eta, 0.2)
    assert 0 < time_step_wt_20pc < time_step_wt_10pc
    assert 0 < parallel_step_wt_20pc < parallel_step_wt_10pc
    assert time_step_wt_10pc == parallel_step_wt_10pc
    assert time_step_wt_20pc == parallel_step_wt_20pc
    # finite bias
    eta = 100
    parallel_step_wt_10pc = _rpsd._step_weight_parallel(eta, 0.1)
    parallel_step_wt_20pc = _rpsd._step_weight_parallel(eta, 0.2)
    assert 0 < time_step_wt_10pc < parallel_step_wt_10pc
    assert 0 < time_step_wt_20pc < parallel_step_wt_20pc


def test_rotated_planar_smwpm_decoder_step_weights_invalid_ftp():
    with pytest.raises(ValueError):
        _rpsd._step_weight_time(q=None)
    with pytest.raises(ValueError):
        _rpsd._step_weight_time(q=0)
    with pytest.raises(ValueError):
        _rpsd._step_weight_time(q=1)
    with pytest.raises(ValueError):
        _rpsd._step_weight_parallel(eta=100, p=None)
    with pytest.raises(ValueError):
        _rpsd._step_weight_parallel(eta=100, p=0)
    with pytest.raises(ValueError):
        _rpsd._step_weight_diagonal(eta=100, p=None)
    with pytest.raises(ValueError):
        _rpsd._step_weight_diagonal(eta=100, p=0)
    with pytest.raises(ValueError):
        _rpsd._step_weight_diagonal(eta=None, p=0.1)


@pytest.mark.parametrize('code, time_steps, a, b, eta, exp_delta_time, exp_delta_parallel, exp_delta_diagonal', [

    # code, time_steps, ((t, x, y), is_row), ((t, x, y), is_row), eta, expected_deltas

    # eta = 10
    # between rows
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 0, 0), True), 10, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 0, 1), True), 10, 0, 1, 1),  # row: 0x1 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 1, 0), True), 10, 0, 1, 0),  # row: 1x0 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 0, 2), True), 10, 0, 0, 2),  # row: 0x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 2, 0), True), 10, 0, 2, 0),  # row: 2x0 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 1, 1), True), 10, 0, 0, 1),  # row: 1x1 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 2, 2), True), 10, 0, 0, 2),  # row: 2x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 3, 2), True), 10, 0, 1, 2),  # row: 3x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 2, 3), True), 10, 0, 1, 3),  # row: 2x3 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 4, 2), True), 10, 0, 2, 2),  # row: 4x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 2, 4), True), 10, 0, 0, 4),  # row: 2x4 box
    # between columns
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 0, 0), False), 10, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 0, 1), False), 10, 0, 1, 0),  # row: 0x1 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 1, 0), False), 10, 0, 1, 1),  # row: 1x0 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 0, 2), False), 10, 0, 2, 0),  # row: 0x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 2, 0), False), 10, 0, 0, 2),  # row: 2x0 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 1, 1), False), 10, 0, 0, 1),  # row: 1x1 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 2, 2), False), 10, 0, 0, 2),  # row: 2x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 3, 2), False), 10, 0, 1, 3),  # row: 3x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 2, 3), False), 10, 0, 1, 2),  # row: 2x3 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 4, 2), False), 10, 0, 0, 4),  # row: 4x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 2, 4), False), 10, 0, 2, 2),  # row: 2x4 box
    # between columns and times
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((0, 0, 0), False), 10, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((1, 0, 1), False), 10, 1, 1, 0),  # row: 0x1 box
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((2, 1, 0), False), 10, 2, 1, 1),  # row: 1x0 box
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((3, 0, 2), False), 10, 3, 2, 0),  # row: 0x2 box
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((4, 2, 0), False), 10, 4, 0, 2),  # row: 2x0 box
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((5, 1, 1), False), 10, 5, 0, 1),  # row: 1x1 box
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((6, 2, 2), False), 10, 5, 0, 2),  # row: 2x2 box
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((7, 3, 2), False), 10, 4, 1, 3),  # row: 3x2 box
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((8, 2, 3), False), 10, 3, 1, 2),  # row: 2x3 box
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((9, 4, 2), False), 10, 2, 0, 4),  # row: 4x2 box
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((10, 2, 4), False), 10, 1, 2, 2),  # row: 2x4 box

    # eta = 100
    # between rows
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 0, 0), True), 100, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 0, 1), True), 100, 0, 1, 1),  # row: 0x1 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 1, 0), True), 100, 0, 1, 0),  # row: 1x0 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 0, 2), True), 100, 0, 0, 2),  # row: 0x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 2, 0), True), 100, 0, 2, 0),  # row: 2x0 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 1, 1), True), 100, 0, 0, 1),  # row: 1x1 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 2, 2), True), 100, 0, 0, 2),  # row: 2x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 3, 2), True), 100, 0, 1, 2),  # row: 3x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 2, 3), True), 100, 0, 1, 3),  # row: 2x3 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 4, 2), True), 100, 0, 2, 2),  # row: 4x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 2, 4), True), 100, 0, 0, 4),  # row: 2x4 box
    # between columns
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 0, 0), False), 100, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 0, 1), False), 100, 0, 1, 0),  # row: 0x1 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 1, 0), False), 100, 0, 1, 1),  # row: 1x0 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 0, 2), False), 100, 0, 2, 0),  # row: 0x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 2, 0), False), 100, 0, 0, 2),  # row: 2x0 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 1, 1), False), 100, 0, 0, 1),  # row: 1x1 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 2, 2), False), 100, 0, 0, 2),  # row: 2x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 3, 2), False), 100, 0, 1, 3),  # row: 3x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 2, 3), False), 100, 0, 1, 2),  # row: 2x3 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 4, 2), False), 100, 0, 0, 4),  # row: 4x2 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 2, 4), False), 100, 0, 2, 2),  # row: 2x4 box

    # eta = None
    # between rows
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 0, 0), True), None, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 1, 0), True), None, 0, 1, 0),  # row: 1x0 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), True), ((0, 2, 0), True), None, 0, 2, 0),  # row: 2x0 box
    # between columns
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 0, 0), False), None, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 0, 1), False), None, 0, 1, 0),  # row: 0x1 box
    (RotatedPlanarCode(11, 11), 1, ((0, 0, 0), False), ((0, 0, 2), False), None, 0, 2, 0),  # row: 0x2 box
    # between columns and time
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((0, 0, 0), False), None, 0, 0, 0),  # row: 0x0 box (w x h)
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((4, 0, 1), False), None, 4, 1, 0),  # row: 0x1 box
    (RotatedPlanarCode(11, 11), 11, ((0, 0, 0), False), ((8, 0, 2), False), None, 3, 2, 0),  # row: 0x2 box

])
def test_rotated_planar_smwpm_decoder_distance_ftp(code, time_steps, a, b, eta, exp_delta_time, exp_delta_parallel,
                                                   exp_delta_diagonal):
    p, q = 0.2, 0.1
    expected_distance = 0
    if exp_delta_time:
        expected_distance += exp_delta_time * _rpsd._step_weight_time(q)
    if exp_delta_parallel:
        expected_distance += exp_delta_parallel * _rpsd._step_weight_parallel(eta, p)
    if exp_delta_diagonal:
        expected_distance += exp_delta_diagonal * _rpsd._step_weight_diagonal(eta, p)
    assert _rpsd._distance(code, time_steps, a, b,
                           eta=eta, error_probability=p, measurement_error_probability=q) == expected_distance, (
        'Distance with bias not as expected')


@pytest.mark.parametrize('error_pauli, expected', [
    # nodes (within edges) are sorted for comparison (automatically in test).
    # edges are in an unsorted set.
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)),  # bulk
     {(((0, 1, 1), True), ((0, 2, 1), True), 1),  # bottom edge
      (((0, 1, 2), True), ((0, 2, 2), True), 1),  # top edge
      (((0, 1, 2), False), ((0, 1, 1), False), 1),  # left edge
      (((0, 2, 2), False), ((0, 2, 1), False), 1),  # right edge
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2), (3, 2)),  # bulk
     {(((0, 1, 1), True), ((0, 3, 1), True), 2),  # bottom edge
      (((0, 1, 2), True), ((0, 3, 2), True), 2),  # top edge
      (((0, 1, 2), False), ((0, 1, 1), False), 1),  # left edge
      (((0, 3, 2), False), ((0, 3, 1), False), 1),  # right edge
      }),
])
def test_rotated_planar_smwpm_decoder_graph(error_pauli, expected):
    # parameters
    code = error_pauli.code
    time_steps = 1
    error = error_pauli.to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    # ensure syndrome is 2d
    syndrome = np.expand_dims(syndrome, axis=0)
    # call
    graph = _rpsd._graph(code, time_steps, syndrome)
    # prepare actual
    # i.e. filter out edges to virtual plaquettes, sort nodes within edges, and extract a_node, b_node, weight
    actual = {(*sorted((a_node, b_node)), weight) for (a_node, b_node), weight in graph.items()
              if not (code.is_virtual_plaquette(a_node[0][1:]) or code.is_virtual_plaquette(b_node[0][1:]))}
    # prepare expected (sort nodes within edges)
    expected = set((*sorted((a_node, b_node)), weight) for a_node, b_node, weight in expected)
    # check
    assert actual == expected, 'Edges between real plaquettes not as expected.'


@pytest.mark.parametrize('error_pauli, eta, error_probability, expected', [
    # nodes (within edges) are sorted for comparison (manually).
    # edges are sorted by weight, a_node, b_node, where True > False (manually).
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)).site('X', (3, 3)),
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
def test_rotated_planar_smwpm_decoder_graph_with_bias(error_pauli, eta, error_probability, expected):
    # parameters
    code = error_pauli.code
    time_steps = 1
    error = error_pauli.to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    # ensure syndrome is 2d
    syndrome = np.expand_dims(syndrome, axis=0)
    # call
    graph = _rpsd._graph(code, time_steps, syndrome, error_probability, eta=eta)
    # prepare actual
    # i.e. filter out edges to virtual plaquettes,
    #      sort edges by weight, a_node, b_node where nodes are sorted within edges,
    #      and extract a_node, b_node
    actual = tuple((a_node, b_node) for weight, a_node, b_node in
                   sorted((weight, *sorted((a_node, b_node))) for (a_node, b_node), weight in graph.items()
                          if not (code.is_virtual_plaquette(a_node[0][1:])
                                  or code.is_virtual_plaquette(b_node[0][1:]))))
    # check
    assert actual == expected, 'Edges between real plaquettes not as expected.'


@pytest.mark.parametrize('code, error, syndrome, expected', [
    # nodes (within edges) are sorted for comparison (automatically in test).
    # edges are in an unsorted set.

    (*_code_error_syndrome(  # 2 time-steps, 1 Y in bulk, 1 measurement error
        RotatedPlanarCode(5, 5),  # code
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
      }
     ),
])
def test_rotated_planar_smwpm_decoder_graph_ftp(code, error, syndrome, expected):
    # call
    graph = _rpsd._graph(code, len(syndrome), syndrome)
    # prepare actual
    # i.e. filter out edges to virtual plaquettes, sort nodes within edges, and extract a_node, b_node, weight
    actual = {(*sorted((a_node, b_node)), weight) for (a_node, b_node), weight in graph.items()
              if not (code.is_virtual_plaquette(a_node[0][1:]) or code.is_virtual_plaquette(b_node[0][1:]))}
    # prepare expected (sort nodes within edges)
    expected = set((*sorted((a_node, b_node)), weight) for a_node, b_node, weight in expected)
    # check
    assert actual == expected, 'Edges between real plaquettes not as expected.'


@pytest.mark.parametrize('error_pauli, expected', [
    # edges starting up from bottom left and then clockwise

    # BULK
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)),  # . bulk
     {frozenset({((0, 1, 1), False), ((0, 1, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((0, 2, 2), True)}),  # right
      frozenset({((0, 2, 2), False), ((0, 2, 1), False)}),  # down
      frozenset({((0, 2, 1), True), ((0, 1, 1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2), (3, 2)),  # .. bulk
     {frozenset({((0, 1, 1), False), ((0, 1, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((0, 3, 2), True)}),  # right
      frozenset({((0, 3, 2), False), ((0, 3, 1), False)}),  # down
      frozenset({((0, 3, 1), True), ((0, 1, 1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2), (2, 3)),  # : bulk
     {frozenset({((0, 1, 1), False), ((0, 1, 3), False)}),  # up
      frozenset({((0, 1, 3), True), ((0, 2, 3), True)}),  # right
      frozenset({((0, 2, 3), False), ((0, 2, 1), False)}),  # down
      frozenset({((0, 2, 1), True), ((0, 1, 1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2), (2, 3), (3, 3), (3, 2)),  # :: bulk
     {frozenset({((0, 1, 1), False), ((0, 1, 3), False)}),  # up
      frozenset({((0, 1, 3), True), ((0, 3, 3), True)}),  # right
      frozenset({((0, 3, 3), False), ((0, 3, 1), False)}),  # down
      frozenset({((0, 3, 1), True), ((0, 1, 1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2), (3, 3)),  # / in bulk
     {frozenset({((0, 1, 1), False), ((0, 1, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((0, 3, 2), True)}),  # right
      frozenset({((0, 3, 2), False), ((0, 3, 3), False)}),  # up
      frozenset({((0, 3, 3), True), ((0, 2, 3), True)}),  # left
      frozenset({((0, 2, 3), False), ((0, 2, 1), False)}),  # down
      frozenset({((0, 2, 1), True), ((0, 1, 1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 3), (3, 2)),  # \ in bulk
     {frozenset({((0, 1, 2), False), ((0, 1, 3), False)}),  # up
      frozenset({((0, 1, 3), True), ((0, 2, 3), True)}),  # right
      frozenset({((0, 2, 3), False), ((0, 2, 1), False)}),  # down
      frozenset({((0, 2, 1), True), ((0, 3, 1), True)}),  # right
      frozenset({((0, 3, 1), False), ((0, 3, 2), False)}),  # up
      frozenset({((0, 3, 2), True), ((0, 1, 2), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2), (2, 3), (3, 2)),  # :. in bulk
     {frozenset({((0, 1, 1), False), ((0, 1, 3), False)}),  # up
      frozenset({((0, 1, 3), True), ((0, 2, 3), True)}),  # right
      frozenset({((0, 2, 3), False), ((0, 2, 2), False)}),  # down
      frozenset({((0, 2, 2), True), ((0, 3, 2), True)}),  # right
      frozenset({((0, 3, 2), False), ((0, 3, 1), False)}),  # down
      frozenset({((0, 3, 1), True), ((0, 1, 1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 3), (3, 3), (3, 2)),  # ': in bulk
     {frozenset({((0, 1, 2), False), ((0, 1, 3), False)}),  # up
      frozenset({((0, 1, 3), True), ((0, 3, 3), True)}),  # right
      frozenset({((0, 3, 3), False), ((0, 3, 1), False)}),  # down
      frozenset({((0, 3, 1), True), ((0, 2, 1), True)}),  # left
      frozenset({((0, 2, 1), False), ((0, 2, 2), False)}),  # up
      frozenset({((0, 2, 2), True), ((0, 1, 2), True)}),  # left
      }),

    # CORNER SW
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0)),  # . in sw corner
     {frozenset({((0, -1, -1), False), ((0, -1, 0), False)}),  # up
      frozenset({((0, -1, 0), True), ((0, 0, 0), True)}),  # right
      frozenset({((0, 0, 0), False), ((0, 0, -1), False)}),  # down
      frozenset({((0, 0, -1), True), ((0, -1, -1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (1, 0)),  # .. in sw corner
     {frozenset({((0, -1, -1), False), ((0, -1, 0), False)}),  # up
      frozenset({((0, -1, 0), True), ((0, 1, 0), True)}),  # right
      frozenset({((0, 1, 0), False), ((0, 1, -1), False)}),  # down
      frozenset({((0, 1, -1), True), ((0, -1, -1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (0, 1)),  # : in sw corner
     {frozenset({((0, -1, -1), False), ((0, -1, 1), False)}),  # up
      frozenset({((0, -1, 1), True), ((0, 0, 1), True)}),  # right
      frozenset({((0, 0, 1), False), ((0, 0, -1), False)}),  # down
      frozenset({((0, 0, -1), True), ((0, -1, -1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (0, 1), (1, 1), (1, 0)),  # :: in sw corner
     {frozenset({((0, -1, -1), False), ((0, -1, 1), False)}),  # up
      frozenset({((0, -1, 1), True), ((0, 1, 1), True)}),  # right
      frozenset({((0, 1, 1), False), ((0, 1, -1), False)}),  # down
      frozenset({((0, 1, -1), True), ((0, -1, -1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (1, 0), (2, 0)),  # ... in sw corner
     {frozenset({((0, -1, -1), False), ((0, -1, 0), False)}),  # up
      frozenset({((0, -1, 0), True), ((0, 2, 0), True)}),  # right
      frozenset({((0, 2, 0), False), ((0, 2, -1), False)}),  # down
      frozenset({((0, 2, -1), True), ((0, -1, -1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (0, 1), (0, 2)),  # ! in sw corner
     {frozenset({((0, -1, -1), False), ((0, -1, 2), False)}),  # up
      frozenset({((0, -1, 2), True), ((0, 0, 2), True)}),  # right
      frozenset({((0, 0, 2), False), ((0, 0, -1), False)}),  # down
      frozenset({((0, 0, -1), True), ((0, -1, -1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (1, 1)),  # / in sw corner
     {frozenset({((0, -1, -1), False), ((0, -1, 0), False)}),  # up
      frozenset({((0, -1, 0), True), ((0, 1, 0), True)}),  # right
      frozenset({((0, 1, 0), False), ((0, 1, 1), False)}),  # up
      frozenset({((0, 1, 1), True), ((0, 0, 1), True)}),  # left
      frozenset({((0, 0, 1), False), ((0, 0, -1), False)}),  # down
      frozenset({((0, 0, -1), True), ((0, -1, -1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 1), (1, 0)),  # \ in sw corner
     {frozenset({((0, -1, 0), False), ((0, -1, 1), False)}),  # up
      frozenset({((0, -1, 1), True), ((0, 0, 1), True)}),  # right
      frozenset({((0, 0, 1), False), ((0, 0, -1), False)}),  # down
      frozenset({((0, 0, -1), True), ((0, 1, -1), True)}),  # right
      frozenset({((0, 1, -1), False), ((0, 1, 0), False)}),  # up
      frozenset({((0, 1, 0), True), ((0, -1, 0), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (0, 1), (1, 0)),  # :. in sw corner
     {frozenset({((0, -1, -1), False), ((0, -1, 1), False)}),  # up
      frozenset({((0, -1, 1), True), ((0, 0, 1), True)}),  # right
      frozenset({((0, 0, 1), False), ((0, 0, 0), False)}),  # down
      frozenset({((0, 0, 0), True), ((0, 1, 0), True)}),  # right
      frozenset({((0, 1, 0), False), ((0, 1, -1), False)}),  # down
      frozenset({((0, 1, -1), True), ((0, -1, -1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 1), (1, 1), (1, 0)),  # ': in sw corner
     {frozenset({((0, -1, 0), False), ((0, -1, 1), False)}),  # up
      frozenset({((0, -1, 1), True), ((0, 1, 1), True)}),  # right
      frozenset({((0, 1, 1), False), ((0, 1, -1), False)}),  # down
      frozenset({((0, 1, -1), True), ((0, 0, -1), True)}),  # left
      frozenset({((0, 0, -1), False), ((0, 0, 0), False)}),  # up
      frozenset({((0, 0, 0), True), ((0, -1, 0), True)}),  # left
      }),

    # CORNERS (NW, NE SE)
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 4)),  # . in nw corner
     {frozenset({((0, -1, 3), False), ((0, -1, 4), False)}),  # up
      frozenset({((0, -1, 4), True), ((0, 0, 4), True)}),  # right
      frozenset({((0, 0, 4), False), ((0, 0, 3), False)}),  # down
      frozenset({((0, 0, 3), True), ((0, -1, 3), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (4, 4)),  # . in ne corner
     {frozenset({((0, 3, 3), False), ((0, 3, 4), False)}),  # up
      frozenset({((0, 3, 4), True), ((0, 4, 4), True)}),  # right
      frozenset({((0, 4, 4), False), ((0, 4, 3), False)}),  # down
      frozenset({((0, 4, 3), True), ((0, 3, 3), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (4, 0)),  # . in se corner
     {frozenset({((0, 3, -1), False), ((0, 3, 0), False)}),  # up
      frozenset({((0, 3, 0), True), ((0, 4, 0), True)}),  # right
      frozenset({((0, 4, 0), False), ((0, 4, -1), False)}),  # down
      frozenset({((0, 4, -1), True), ((0, 3, -1), True)}),  # left
      }),

    # BOUNDARIES (N, E, S, W)
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (1, 4)),  # . on n boundary
     {frozenset({((0, 0, 3), False), ((0, 0, 4), False)}),  # up
      frozenset({((0, 0, 4), True), ((0, 1, 4), True)}),  # right
      frozenset({((0, 1, 4), False), ((0, 1, 3), False)}),  # down
      frozenset({((0, 1, 3), True), ((0, 0, 3), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (1, 4), (2, 4)),  # .. on n boundary
     {frozenset({((0, 0, 3), False), ((0, 0, 4), False)}),  # up
      frozenset({((0, 0, 4), True), ((0, 2, 4), True)}),  # right
      frozenset({((0, 2, 4), False), ((0, 2, 3), False)}),  # down
      frozenset({((0, 2, 3), True), ((0, 0, 3), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (4, 2)),  # . on e boundary
     {frozenset({((0, 3, 1), False), ((0, 3, 2), False)}),  # up
      frozenset({((0, 3, 2), True), ((0, 4, 2), True)}),  # right
      frozenset({((0, 4, 2), False), ((0, 4, 1), False)}),  # down
      frozenset({((0, 4, 1), True), ((0, 3, 1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (4, 2), (4, 3)),  # : on e boundary
     {frozenset({((0, 3, 1), False), ((0, 3, 3), False)}),  # up
      frozenset({((0, 3, 3), True), ((0, 4, 3), True)}),  # right
      frozenset({((0, 4, 3), False), ((0, 4, 1), False)}),  # down
      frozenset({((0, 4, 1), True), ((0, 3, 1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (1, 0)),  # . on s boundary
     {frozenset({((0, 0, -1), False), ((0, 0, 0), False)}),  # up
      frozenset({((0, 0, 0), True), ((0, 1, 0), True)}),  # right
      frozenset({((0, 1, 0), False), ((0, 1, -1), False)}),  # down
      frozenset({((0, 1, -1), True), ((0, 0, -1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (1, 0), (2, 0)),  # .. on s boundary
     {frozenset({((0, 0, -1), False), ((0, 0, 0), False)}),  # up
      frozenset({((0, 0, 0), True), ((0, 2, 0), True)}),  # right
      frozenset({((0, 2, 0), False), ((0, 2, -1), False)}),  # down
      frozenset({((0, 2, -1), True), ((0, 0, -1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 2)),  # . on w boundary
     {frozenset({((0, -1, 1), False), ((0, -1, 2), False)}),  # up
      frozenset({((0, -1, 2), True), ((0, 0, 2), True)}),  # right
      frozenset({((0, 0, 2), False), ((0, 0, 1), False)}),  # down
      frozenset({((0, 0, 1), True), ((0, -1, 1), True)}),  # left
      }),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 2), (0, 3)),  # : on w boundary
     {frozenset({((0, -1, 1), False), ((0, -1, 3), False)}),  # up
      frozenset({((0, -1, 3), True), ((0, 0, 3), True)}),  # right
      frozenset({((0, 0, 3), False), ((0, 0, 1), False)}),  # down
      frozenset({((0, 0, 1), True), ((0, -1, 1), True)}),  # left
      }),
])
def test_rotated_planar_smwpm_decoder_matching(error_pauli, expected):
    # parameters
    code = error_pauli.code
    time_steps = 1
    error = error_pauli.to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    # ensure syndrome is 2d
    syndrome = np.expand_dims(syndrome, axis=0)
    # calls
    graph = _rpsd._graph(code, time_steps, syndrome)
    matches = _rpsd._matching(graph)
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
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)),
     None, 0.1,
     {frozenset({((0, 1, 1), False), ((0, 1, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((0, 2, 2), True)}),  # right
      frozenset({((0, 2, 2), False), ((0, 2, 1), False)}),  # down
      frozenset({((0, 2, 1), True), ((0, 1, 1), True)}),  # left
      }),
    # / bulk, eta=3
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)).site('X', (3, 3)),  # / in center of bulk, eta=10
     10, 0.1,
     {frozenset({((0, 1, 1), False), ((0, 1, 2), False)}),  # up
      frozenset({((0, 1, 2), True), ((0, 3, 3), True)}),  # right (and up)
      frozenset({((0, 3, 3), False), ((0, 2, 1), False)}),  # down (and left)
      frozenset({((0, 2, 1), True), ((0, 1, 1), True)}),  # left
      }),
])
def test_rotated_planar_smwpm_decoder_matching_with_bias(error_pauli, eta, error_probability, expected):
    # parameters
    code = error_pauli.code
    time_steps = 1
    error = error_pauli.to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    # ensure syndrome is 2d
    syndrome = np.expand_dims(syndrome, axis=0)
    # calls
    graph = _rpsd._graph(code, time_steps, syndrome, error_probability, eta=eta)
    matches = _rpsd._matching(graph)
    # prepare actual (convert mates to frozenset)
    actual = set()
    for ((a_index, a_is_row), (b_index, b_is_row)) in matches:
        if a_index != b_index:
            actual.add(frozenset(((a_index, a_is_row), (b_index, b_is_row))))
    # check
    assert actual == expected, 'Matches not as expected.'


@pytest.mark.parametrize('code, error, syndrome, p, q, eta, expected', [
    # edges starting up from bottom left and then clockwise

    (*_code_error_syndrome(  # 2 time-steps, Y in bulk, 1 measurement error
        RotatedPlanarCode(5, 5),  # code
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
        RotatedPlanarCode(5, 5),  # code
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
        RotatedPlanarCode(5, 5),  # code
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
def test_rotated_planar_smwpm_decoder_matching_ftp(code, error, syndrome, p, q, eta, expected):
    # parameters
    # calls
    graph = _rpsd._graph(code, len(syndrome), syndrome, p, q, eta)
    matches = _rpsd._matching(graph)
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
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)),  # . bulk
     [[(0, 1, 1), (0, 1, 2), (0, 2, 2), (0, 2, 1)],
      ]),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2), (3, 3)),  # / in bulk
     [[(0, 1, 1), (0, 1, 2), (0, 3, 2), (0, 3, 3), (0, 2, 3), (0, 2, 1)],
      ]),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 3), (3, 2)),  # \ in bulk
     [[(0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 2, 1), (0, 3, 1), (0, 3, 2)],
      ]),

    # CORNER SW
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0)),  # . in sw corner
     [[(0, -1, -1), (0, -1, 0), (0, 0, 0), (0, 0, -1)],
      ]),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (1, 1)),  # / in sw corner
     [[(0, -1, -1), (0, -1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, -1)],
      ]),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 1), (1, 0)),  # \ in sw corner
     [[(0, -1, 0), (0, -1, 1), (0, 0, 1), (0, 0, -1), (0, 1, -1), (0, 1, 0)],
      ]),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (1, 1), (2, 2)),  # .*' in sw corner
     [[(0, -1, -1), (0, -1, 0), (0, 1, 0), (0, 1, 2), (0, 2, 2), (0, 2, 1), (0, 0, 1), (0, 0, -1)],
      ]),

    # CORNERS (NW, NE SE)
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 4)),  # . in nw corner
     [[(0, -1, 3), (0, -1, 4), (0, 0, 4), (0, 0, 3)],
      ]),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (4, 4)),  # . in ne corner
     [[(0, 3, 3), (0, 3, 4), (0, 4, 4), (0, 4, 3)],
      ]),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (4, 0)),  # . in se corner
     [[(0, 3, -1), (0, 3, 0), (0, 4, 0), (0, 4, -1)],
      ]),

    # TWO CLUSTERS
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (2, 2)),  # . in sw corner and . in bulk
     [[(0, -1, -1), (0, -1, 0), (0, 0, 0), (0, 0, -1)],
      [(0, 1, 1), (0, 1, 2), (0, 2, 2), (0, 2, 1)],
      ]),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (1, 1), (4, 4)),  # / in sw corner and . in ne corner
     [[(0, -1, -1), (0, -1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, -1)],
      [(0, 3, 3), (0, 3, 4), (0, 4, 4), (0, 4, 3)],
      ]),
])
def test_rotated_planar_smwpm_decoder_clusters(error_pauli, expected):
    # parameters
    code = error_pauli.code
    time_steps = 1
    error = error_pauli.to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    # ensure syndrome is 2d
    syndrome = np.expand_dims(syndrome, axis=0)
    # calls
    graph = _rpsd._graph(code, time_steps, syndrome)
    matches = _rpsd._matching(graph)
    clusters = _rpsd._clusters(matches)
    # check
    assert clusters == expected, 'Clusters not as expected.'


@pytest.mark.parametrize('error_pauli, eta, error_probability, expected', [
    # edges starting up from bottom left and then clockwise

    # BULK
    # . in bulk, eta=None (infinite)
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)),
     None, 0.1,
     [[(0, 1, 1), (0, 1, 2), (0, 2, 2), (0, 2, 1)],
      ]),
    # / in bulk, eta=10
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)).site('X', (3, 3)),
     10, 0.1,
     [[(0, 1, 1), (0, 1, 2), (0, 3, 3), (0, 2, 1)],
      ]),

    # TWO CLUSTERS
    # . in sw corner and . in bulk, eta=10
    (RotatedPlanarCode(7, 7).new_pauli().site('Y', (0, 0), (3, 3)).site('X', (4, 4)),
     10, 0.1,
     [[(0, -1, -1), (0, -1, 0), (0, 0, 0), (0, 0, -1)],
      [(0, 2, 2), (0, 2, 3), (0, 4, 4), (0, 3, 2)],
      ]),

    # ISOLATED Y DEFECTS
    # Y defect in SW corner, eta=10
    (RotatedPlanarCode(7, 7).new_pauli().site('X', (0, 1), (1, 1)).site('Z', (2, 0), (2, 1)),
     10, 0.1,
     [[(0, 1, 1), (0, 2, 1)],
      ]),
    # Y defects in SW and NE corners, eta=10
    (RotatedPlanarCode(7, 7).new_pauli()
     .site('X', (0, 1), (1, 1), (6, 5), (5, 5))
     .site('Z', (2, 0), (2, 1), (4, 5), (4, 6)),
     10, 0.1,
     [[(0, 1, 1), (0, 2, 1)],
      [(0, 3, 4), (0, 4, 4)],
      ]),
    # Y defects in SW and NE corners, eta=10
    (RotatedPlanarCode(9, 9).new_pauli()
     .site('X', (0, 1), (1, 1), (8, 7), (7, 7))
     .site('Z', (2, 0), (2, 1), (6, 7), (6, 8)),
     10, 0.1,
     [[(0, 1, 1), (0, 2, 1)],
      [(0, 5, 6), (0, 6, 6)],
      ]),
    # Y defects in SW and NE corners and neutral cluster in center, eta=10
    (RotatedPlanarCode(9, 9).new_pauli()
     .site('X', (0, 1), (1, 1), (8, 7), (7, 7))
     .site('Z', (2, 0), (2, 1), (6, 7), (6, 8))
     .site('Y', (3, 4), (4, 4), (5, 4)),
     10, 0.1,
     [[(0, 1, 1), (0, 2, 1)],
      [(0, 2, 3), (0, 2, 4), (0, 5, 4), (0, 5, 3)],
      [(0, 5, 6), (0, 6, 6)],
      ]),
    # Y defects in SW and NE corners and neutral cluster in center, eta=10
    (RotatedPlanarCode(11, 11).new_pauli()
     .site('X', (0, 1), (1, 1), (10, 9), (9, 9))
     .site('Z', (2, 0), (2, 1), (8, 9), (8, 10))
     .site('Y', (4, 5), (5, 5), (6, 5)),
     10, 0.1,
     [[(0, 1, 1), (0, 2, 1)],
      [(0, 3, 4), (0, 3, 5), (0, 6, 5), (0, 6, 4)],
      [(0, 7, 8), (0, 8, 8)],
      ]),
    # Y defects in SW and NE corners and 2 neutral clusters in center, eta=10
    (RotatedPlanarCode(11, 11).new_pauli()
     .site('X', (0, 1), (1, 1), (10, 8), (9, 8), (8, 8))
     .site('Z', (2, 0), (2, 1), (7, 8), (7, 9), (7, 10))
     .site('Y', (3, 4), (6, 5)),
     10, 0.1,
     [[(0, 1, 1), (0, 2, 1)],
      [(0, 2, 3), (0, 2, 4), (0, 3, 4), (0, 3, 3)],
      [(0, 5, 4), (0, 5, 5), (0, 6, 5), (0, 6, 4)],
      [(0, 6, 7), (0, 7, 7)],
      ]),
])
def test_rotated_planar_smwpm_decoder_clusters_with_bias(error_pauli, eta, error_probability, expected):
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
    graph = _rpsd._graph(code, time_steps, syndrome, error_probability, eta=eta)
    matches = _rpsd._matching(graph)
    clusters = _rpsd._clusters(matches)
    print('### clusters=', clusters)
    print('### expected=', expected)
    _print_clusters(code, clusters)
    # check
    assert clusters == expected, 'Clusters not as expected.'


@pytest.mark.parametrize('code, error, syndrome, expected', [
    # edges starting up from bottom left and then clockwise

    (*_code_error_syndrome(  # 2 time-steps, Y in bulk, 1 measurement error
        RotatedPlanarCode(5, 5),  # code
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
def test_rotated_planar_smwpm_decoder_clusters_ftp(code, error, syndrome, expected):
    # calls
    graph = _rpsd._graph(code, len(syndrome), syndrome)
    matches = _rpsd._matching(graph)
    clusters = _rpsd._clusters(matches)
    # check
    assert clusters == expected, 'Clusters not as expected.'


@pytest.mark.parametrize('code, cluster, expected', [
    # On-lattice (no Y-defect)
    (RotatedPlanarCode(5, 5),
     [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],  # ZXZX
     ([(0, 0, 1), (0, 1, 0)], [(0, 0, 0), (0, 1, 1)], None)
     ),
    # On-lattice (no Y-defect, 1 measurement error)
    (RotatedPlanarCode(5, 5),
     [(0, 0, 0), (0, 0, 1), (1, 1, 1), (0, 1, 0)],  # ZXZX
     ([(0, 0, 1), (0, 1, 0)], [(0, 0, 0), (1, 1, 1)], None)
     ),
    # On-lattice (Y-defect)
    (RotatedPlanarCode(5, 5),
     [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 2, 0)],  # ZXZZ
     ([], [(0, 0, 0), (0, 1, 1)], ((0, 0, 1), (0, 2, 0)))
     ),
    # On-lattice (Y-defect, 1 measurement error)
    (RotatedPlanarCode(5, 5),
     [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 2, 0)],  # ZXZZ
     ([], [(0, 0, 0), (0, 1, 1)], ((0, 0, 1), (1, 2, 0)))
     ),
    # Off-lattice (no Y-defect)
    (RotatedPlanarCode(5, 5),
     [(0, -1, -1), (0, -1, 0), (0, 0, 0), (0, 0, -1)],  # ZXZX
     ([(0, -1, 0), (0, 0, -1)], [(0, -1, -1), (0, 0, 0)], None)
     ),
    # Off-lattice (no Y-defect, 1 measurement error)
    (RotatedPlanarCode(5, 5),
     [(0, -1, -1), (1, -1, 0), (0, 0, 0), (0, 0, -1)],  # ZXZX
     ([(1, -1, 0), (0, 0, -1)], [(0, -1, -1), (0, 0, 0)], None)
     ),
    # Off-lattice (Y-defect)
    (RotatedPlanarCode(5, 5),
     [(0, -1, -1), (0, -1, 0), (0, 1, 0), (0, 2, -1)],  # ZXXX
     ([(0, -1, 0), (0, 1, 0)], [], ((0, 2, -1), (0, -1, -1)))
     ),
    # Off-lattice (Y-defect, 1 measurment error)
    (RotatedPlanarCode(5, 5),
     [(1, -1, -1), (0, -1, 0), (0, 1, 0), (0, 2, -1)],  # ZXXX
     ([(0, -1, 0), (0, 1, 0)], [], ((0, 2, -1), (1, -1, -1)))
     ),
    # Partially off-lattice (Y-defect)
    (RotatedPlanarCode(5, 5),
     [(0, 0, -1), (0, -1, 0), (0, 1, 0), (0, 1, -1)],  # XXXZ
     ([(0, 0, -1), (0, -1, 0)], [], ((0, 1, 0), (0, 1, -1)))
     ),
    # Partially off-lattice (Y-defect, 1 measurement error)
    (RotatedPlanarCode(5, 5),
     [(0, 0, -1), (0, -1, 0), (0, 1, 0), (1, 1, -1)],  # XXXZ
     ([(0, 0, -1), (0, -1, 0)], [], ((0, 1, 0), (1, 1, -1)))
     ),
])
def test_rotated_planar_smwpm_decoder_cluster_to_paths_and_defect_ftp(code, cluster, expected):
    x_path, z_path, y_defect = _rpsd._cluster_to_paths_and_defect(code, cluster)
    print()
    print('actual:')
    print(x_path, z_path, y_defect)
    print()
    print('expected:')
    print(expected)
    assert (x_path, z_path, y_defect) == expected


@pytest.mark.parametrize('code, a_index, b_index, expected', [
    # between Z plaquettes
    (RotatedPlanarCode(5, 5), (2, 2), (2, 2),
     RotatedPlanarCode(5, 5).new_pauli()),  # same site
    (RotatedPlanarCode(5, 5), (0, 2), (2, 2),
     RotatedPlanarCode(5, 5).new_pauli().site('X', (1, 2), (2, 2))),  # along row
    (RotatedPlanarCode(5, 5), (2, 0), (2, 2),
     RotatedPlanarCode(5, 5).new_pauli().site('X', (2, 1), (2, 2))),  # along column
    (RotatedPlanarCode(5, 5), (0, 2), (3, 1),
     RotatedPlanarCode(5, 5).new_pauli().site('X', (1, 2), (2, 2), (3, 2))),  # dog-leg row
    (RotatedPlanarCode(5, 5), (0, 2), (3, 3),
     RotatedPlanarCode(5, 5).new_pauli().site('X', (1, 3), (2, 3), (3, 3))),  # dog-leg row
    (RotatedPlanarCode(5, 5), (2, 0), (1, 3),
     RotatedPlanarCode(5, 5).new_pauli().site('X', (2, 1), (2, 2), (2, 3))),  # dog-leg column
    (RotatedPlanarCode(5, 5), (2, 0), (3, 3),
     RotatedPlanarCode(5, 5).new_pauli().site('X', (3, 1), (3, 2), (3, 3))),  # dog-leg column
    (RotatedPlanarCode(5, 5), (0, 4), (3, -1),
     RotatedPlanarCode(5, 5).new_pauli().site('X', (1, 4), (2, 3), (3, 2), (3, 1), (3, 0))),  # extreme dog-leg column
    # between X plaquettes
    (RotatedPlanarCode(5, 5), (2, 1), (2, 1),
     RotatedPlanarCode(5, 5).new_pauli()),  # same site
    (RotatedPlanarCode(5, 5), (0, 1), (2, 1),
     RotatedPlanarCode(5, 5).new_pauli().site('Z', (1, 1), (2, 1))),  # along row
    (RotatedPlanarCode(5, 5), (2, 3), (2, 1),
     RotatedPlanarCode(5, 5).new_pauli().site('Z', (2, 2), (2, 3))),  # along column
    (RotatedPlanarCode(5, 5), (0, 1), (3, 2),
     RotatedPlanarCode(5, 5).new_pauli().site('Z', (1, 2), (2, 2), (3, 2))),  # dog-leg row
    (RotatedPlanarCode(5, 5), (0, 1), (3, 0),
     RotatedPlanarCode(5, 5).new_pauli().site('Z', (1, 1), (2, 1), (3, 1))),  # dog-leg row
    (RotatedPlanarCode(5, 5), (-1, 0), (4, 3),
     RotatedPlanarCode(5, 5).new_pauli().site('Z', (0, 1), (1, 2), (2, 3), (3, 3), (4, 3))),  # extreme dog-leg row
    (RotatedPlanarCode(5, 5), (2, 3), (1, 0),
     RotatedPlanarCode(5, 5).new_pauli().site('Z', (2, 1), (2, 2), (2, 3))),  # dog-leg column
    (RotatedPlanarCode(5, 5), (2, 3), (3, 0),
     RotatedPlanarCode(5, 5).new_pauli().site('Z', (3, 1), (3, 2), (3, 3))),  # dog-leg column
])
def test_rotated_planar_smwpm_decoder_path_operator(code, a_index, b_index, expected):
    path_operator = _rpsd._path_operator(code, a_index, b_index)
    path_pauli = code.new_pauli(path_operator)
    print()
    print('actual:')
    print(path_pauli)
    print()
    print('expected:')
    print(expected)
    assert path_pauli == expected


@pytest.mark.parametrize('code, a_index, b_index', [
    (RotatedPlanarCode(5, 5), (0, 0), (0, 1)),  # Z to X plaquettes
    (RotatedPlanarCode(5, 5), (-1, 0), (0, 0)),  # X to Z plaquettes
])
def test_rotated_planar_smwpm_decoder_path_operator_invalid(code, a_index, b_index):
    with pytest.raises(ValueError):
        _rpsd._path_operator(code, a_index, b_index)


@pytest.mark.parametrize('error_pauli', [
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (1, 1))),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 0), (4, 4))),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (0, 2), (1, 2))),
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 0), (2, 1))),
])
def test_rotated_planar_smwpm_decoder_decode(error_pauli):
    print()
    print('error:')
    print(error_pauli)
    error = error_pauli.to_bsf()
    code = error_pauli.code
    decoder = RotatedPlanarSMWPMDecoder()
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


@pytest.mark.parametrize('code, error, syndrome', [

    _code_error_syndrome(  # 2 time-steps, Y in bulk, 1 measurement error
        RotatedPlanarCode(5, 5),  # code
        [  # step_errors
            {'Y': [(2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [],
        ]),
    _code_error_syndrome(  # 2 time-steps, Y in bulk, 2 measurement errors
        RotatedPlanarCode(5, 5),  # code
        [  # step_errors
            {'Y': [(2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [(0, 0)],
        ]),
    _code_error_syndrome(  # 2 time-steps, 2 Y in bulk, 1 measurement error
        RotatedPlanarCode(5, 5),  # code
        [  # step_errors
            {'Y': [(1, 1), (2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [],
        ]),
    _code_error_syndrome(  # 2 time-steps, 2 Y in bulk, 2 measurement errors
        RotatedPlanarCode(5, 5),  # code
        [  # step_errors
            {'Y': [(1, 1), (2, 2)]},
            {},
        ],
        [  # step_measurement_errors
            [(2, 2)],
            [(2, 3)],
        ]),
])
def test_rotated_planar_smwpm_decoder_decode_ftp(code, error, syndrome):
    print()
    print('error:')
    print(error)
    decoder = RotatedPlanarSMWPMDecoder()
    print()
    print('syndrome:')
    print(syndrome)
    recovery = decoder.decode_ftp(code, len(syndrome), syndrome)
    print()
    print('recovery:')
    print(code.new_pauli(recovery))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


@pytest.mark.parametrize('time_steps, a_node, b_node, expected', [
    (1,
     _rpsd._ClusterNode([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], (0, 0, 1), (0, 0, 0)),
     _rpsd._ClusterNode([(0, 2, 3), (0, 2, 4), (0, 3, 4), (0, 3, 3)], (0, 2, 3), (0, 2, 4)),
     3  # manhattan distance between (0, 1, 1) and (0, 2, 3)
     ),
    (5,
     _rpsd._ClusterNode([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], (0, 0, 1), (0, 0, 0)),
     _rpsd._ClusterNode([(2, 2, 3), (2, 2, 4), (2, 3, 4), (2, 3, 3)], (2, 2, 3), (2, 2, 4)),
     5  # manhattan distance between (0, 1, 1) and (2, 2, 3)
     ),
    (5,
     _rpsd._ClusterNode([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], (0, 0, 1), (0, 0, 0)),
     _rpsd._ClusterNode([(4, 2, 3), (4, 2, 4), (4, 3, 4), (4, 3, 3)], (4, 2, 3), (4, 2, 4)),
     4  # periodic manhattan distance between (0, 1, 1) and (4, 2, 3)
     ),
])
def test_rotated_planar_smwpm_decoder_cluster_distance_ftp(time_steps, a_node, b_node, expected):
    distance = _rpsd._cluster_distance(time_steps, a_node, b_node)
    assert distance == expected, 'Cluster distance not as expected'


def test_rotated_planar_smwpm_decoder_cluster_distance_twins_ftp():
    time_steps = 5
    a_node = _rpsd._ClusterNode([(0, 0, 0), (1, 0, 1), (2, 1, 1), (3, 1, 0)], (1, 0, 1), (0, 0, 0))
    b_node = _rpsd._ClusterNode([(0, 0, 0), (1, 0, 1), (2, 1, 1), (3, 1, 0)], (1, 0, 1), (0, 0, 0))
    distance = _rpsd._cluster_distance(time_steps, a_node, b_node)
    expected = 0  # manhattan distance between twin clusters
    assert distance == expected, 'Cluster distance not as expected'


@pytest.mark.parametrize('code', [
    RotatedPlanarCode(3, 3),
    RotatedPlanarCode(3, 4),
    RotatedPlanarCode(4, 3),
    RotatedPlanarCode(3, 5),
    RotatedPlanarCode(5, 3),
    RotatedPlanarCode(4, 4),
    RotatedPlanarCode(4, 6),
    RotatedPlanarCode(6, 4),
])
def test_rotated_planar_smwpm_decoder_cluster_corner_indices(code):
    for x_plaquette_index, z_plaquette_index in _rpsd._cluster_corner_indices(code):
        assert code.is_x_plaquette(x_plaquette_index)
        assert code.is_z_plaquette(z_plaquette_index)
        assert code.is_virtual_plaquette(x_plaquette_index)
        assert code.is_virtual_plaquette(z_plaquette_index)


def test_rotated_planar_smwpm_decoder_cluster_graph_and_matching():
    code = RotatedPlanarCode(7, 7)
    error = code.new_pauli().site('X', (0, 1), (1, 1)).site('Z', (2, 0), (2, 1)).to_bsf()
    syndrome = pt.bsp(error, code.stabilizers.T)
    # ensure syndrome is 2d (it would be 1d if time_steps == 1)
    syndrome = np.expand_dims(syndrome, axis=0) if syndrome.ndim == 1 else syndrome
    time_steps = len(syndrome)
    eta = 10
    error_probability = 0.1
    measurement_error_probability = 0.0
    # identity recovery
    recovery = code.new_pauli().to_bsf()
    # SYMMETRY DECODING
    # prepare graph
    graph = _rpsd._graph(code, time_steps, syndrome, error_probability, measurement_error_probability, eta)
    # minimum weight matching
    matches = _rpsd._matching(graph)
    # cluster matches
    clusters = _rpsd._clusters(matches)
    # add recovery from fusing within clusters
    recovery ^= _rpsd._recovery(code, clusters)
    # residual cluster syndrome
    cluster_syndrome = np.bitwise_xor.reduce(syndrome) ^ pt.bsp(recovery, code.stabilizers.T)
    assert np.any(cluster_syndrome), 'There should be one isolated Y defect'
    # CLUSTER GRAPH
    # prepare cluster graph
    cluster_graph = _rpsd._cluster_graph(code, time_steps, clusters)
    # i.e. filter out edges virtual nodes, sort by node indices,
    # and extract as {((x_index, z_index), (x_index, z_index), weight), ...}
    actual = {(*sorted(((a.x_index, a.z_index), (b.x_index, b.z_index))), weight)
              for (a, b), weight in cluster_graph.items() if not (True is a.is_virtual is b.is_virtual)}
    # expected
    expected = {
        (((0, 0, -1), (0, -1, -1)), ((0, 2, 1), (0, 1, 1)), 3),
        (((0, -1, 6), (0, -1, 5)), ((0, 2, 1), (0, 1, 1)), 6),
        (((0, 2, 1), (0, 1, 1)), ((0, 5, 6), (0, 6, 6)), 8),
        (((0, 2, 1), (0, 1, 1)), ((0, 6, -1), (0, 6, 0)), 5),
    }
    assert actual == expected, 'Cluster graph not as expected'
    # CLUSTER MATCHING
    cluster_matches = _rpsd._matching(cluster_graph)
    # prepare actual as {((x_index, z_index), (x_index, z_index)), ...} excluding matches between virtual nodes
    actual = set((*sorted(((a.x_index, a.z_index), (b.x_index, b.z_index))),)
                 for a, b in cluster_matches if not (True is a.is_virtual is b.is_virtual))
    # expected
    expected = {(((0, 0, -1), (0, -1, -1)), ((0, 2, 1), (0, 1, 1)))}
    assert actual == expected, 'Cluster matches not as expected'


@pytest.mark.parametrize('error_pauli, bias, error_probability', [
    # BULK
    # . in bulk, bias=None (infinite)
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)), None, 0.1),
    # / in bulk, bias=10
    (RotatedPlanarCode(5, 5).new_pauli().site('Y', (2, 2)).site('X', (3, 3)), 10, 0.1),
    # TWO CLUSTERS
    # . in sw corner and . in bulk, bias=10
    (RotatedPlanarCode(7, 7).new_pauli().site('Y', (0, 0), (3, 3)).site('X', (4, 4)), 10, 0.1),
    # ISOLATED Y DEFECTS
    # Y defect in SW corner, bias=10
    (RotatedPlanarCode(7, 7).new_pauli().site('X', (0, 1), (1, 1)).site('Z', (2, 0), (2, 1)), 10, 0.1),
    # Y defects in SW and NE corners, bias=3 (fuses in bulk)
    (RotatedPlanarCode(7, 7).new_pauli()
     .site('X', (0, 1), (1, 1), (6, 5), (5, 5))
     .site('Z', (2, 0), (2, 1), (4, 5), (4, 6)), 10, 0.1),
    # Y defects in SW and NE corners, bias=3 (fuses to corners)
    (RotatedPlanarCode(9, 9).new_pauli()
     .site('X', (0, 1), (1, 1), (8, 7), (7, 7))
     .site('Z', (2, 0), (2, 1), (6, 7), (6, 8)), 10, 0.1),
    # Y defects in SW and NE corners and neutral cluster in center, bias=10 (fuses through neutral cluster in bulk)
    (RotatedPlanarCode(9, 9).new_pauli()
     .site('X', (0, 1), (1, 1), (8, 7), (7, 7))
     .site('Z', (2, 0), (2, 1), (6, 7), (6, 8))
     .site('Y', (3, 4), (4, 4), (5, 4)), 10, 0.1),
    # Y defects in SW and NE corners and neutral cluster in center, bias=10 (fuses to corners, ignores neutral cluster)
    (RotatedPlanarCode(11, 11).new_pauli()
     .site('X', (0, 1), (1, 1), (10, 9), (9, 9))
     .site('Z', (2, 0), (2, 1), (8, 9), (8, 10))
     .site('Y', (4, 5), (5, 5), (6, 5)), 10, 0.1),
    # Y defects in SW and NE corners and 2 neutral clusters in center, bias=10 (fuses through neutral clusters in bulk)
    (RotatedPlanarCode(11, 11).new_pauli()
     .site('X', (0, 1), (1, 1), (10, 8), (9, 8), (8, 8))
     .site('Z', (2, 0), (2, 1), (7, 8), (7, 9), (7, 10))
     .site('Y', (3, 4), (6, 5)), 10, 0.1),
])
def test_rotated_planar_smwpm_decoder_decode_with_bias(error_pauli, bias, error_probability):
    print()
    print('error:')
    print(error_pauli)
    error = error_pauli.to_bsf()
    code = error_pauli.code
    decoder = RotatedPlanarSMWPMDecoder()
    error_model = BitPhaseFlipErrorModel() if bias is None else BiasedDepolarizingErrorModel(bias)
    syndrome = pt.bsp(error, code.stabilizers.T)
    print()
    print('syndrome:')
    print(code.ascii_art(syndrome=syndrome))
    recovery = decoder.decode(code, syndrome, error_model=error_model, error_probability=error_probability,
                              measurement_error_probability=0.0)
    print()
    print('recovery:')
    print(code.new_pauli(recovery))
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome), (
        'recovery {} does not give the same syndrome as the error {}'.format(recovery, error))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))


@pytest.mark.parametrize('code, error, syndrome, p, q, eta', [

    # 2 time-steps, . in bulk, bias=None (infinite), 1 measurement error
    (*_code_error_syndrome(
        RotatedPlanarCode(5, 5),  # code
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
        RotatedPlanarCode(5, 5),  # code
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
def test_rotated_planar_smwpm_decoder_decode_with_bias_ftp(code, error, syndrome, p, q, eta):
    print()
    print('error:')
    print(error)
    decoder = RotatedPlanarSMWPMDecoder()
    error_model = BitPhaseFlipErrorModel() if eta is None else BiasedDepolarizingErrorModel(eta)
    print()
    print('syndrome:')
    print(syndrome)
    recovery = decoder.decode_ftp(code, len(syndrome), syndrome, error_model=error_model, error_probability=p,
                                  measurement_error_probability=q)
    print()
    print('recovery:')
    print(code.new_pauli(recovery))
    assert np.all(pt.bsp(recovery ^ error, code.stabilizers.T) == 0), (
        'recovery ^ error ({} ^ {}) does not commute with stabilizers.'.format(recovery, error))
