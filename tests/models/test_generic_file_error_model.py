import math
import os

import numpy as np
import pytest
from click.testing import CliRunner  # use for isolated_filesystem feature

from qecsim import app
from qecsim import paulitools as pt
from qecsim.models.generic import FileErrorModel
from qecsim.models.basic import FiveQubitCode
from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarMPSDecoder

FILES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_generic_file_error_model_files')


@pytest.mark.parametrize('filename, start', [
    (os.path.join(FILES_DIR, 'fivequbitcode-errors-p0.4-bias10.jsonl'), 0),
    (os.path.join(FILES_DIR, 'fivequbitcode-errors-p0.4-bias10.jsonl'), 3),
])
def test_file_error_model_init(filename, start):
    FileErrorModel(filename, start)  # no error raised


@pytest.mark.parametrize('filename, start', [
    (os.path.join(FILES_DIR, 'fivequbitcode-errors-p0.4-bias10.jsonl'), 'blah'),
    (os.path.join(FILES_DIR, 'fivequbitcode-errors-p0.4-bias10.jsonl'), -1),
    (os.path.join(FILES_DIR, 'fivequbitcode-errors-p0.4-bias10.jsonl'), 0.4),
    (None, 0),
])
def test_file_error_model_init_invalid_parameters(filename, start):
    with pytest.raises((ValueError, TypeError), match=r"^FileErrorModel") as exc_info:
        FileErrorModel(filename, start)  # expected error raised
    print(exc_info)


@pytest.mark.parametrize('filename', [
    "this_file_does_not_exist.txt",
    "this_one_also.json",
    "and_this_one",
])
def test_file_error_model_file_not_found(filename):
    with pytest.raises(FileNotFoundError):
        FileErrorModel(filename)


def test_file_error_model_init_default_parameters():
    FileErrorModel(os.path.join(FILES_DIR, 'fivequbitcode-errors-p0.4-bias10.jsonl'))  # no error raised


def test_file_error_model_init_extra_header():
    fem = FileErrorModel(os.path.join(FILES_DIR, 'fivequbitcode-errors-p0.4-bias10.jsonl'))
    assert fem.bias == 10


@pytest.mark.parametrize('filename', [
    os.path.join(FILES_DIR, 'invalid-extra-header-1.jsonl'),
    os.path.join(FILES_DIR, 'invalid-extra-header-2.jsonl'),
    os.path.join(FILES_DIR, 'invalid-extra-header-3.jsonl'),
    os.path.join(FILES_DIR, 'invalid-extra-header-4.jsonl'),
])
def test_file_error_model_init_invalid_extra_headers(filename):
    with pytest.raises(ValueError):
        FileErrorModel(filename)


def test_file_error_model_probability_distribution():
    fem = FileErrorModel(os.path.join(FILES_DIR, 'fivequbitcode-errors-p0.4-bias10.jsonl'))
    pd = fem.probability_distribution(fem._probability)
    assert isinstance(pd, tuple)
    assert len(pd) == 4
    assert pd[0] + pd[1] + pd[2] + pd[3] == 1


def test_file_error_model_generate():
    fem = FileErrorModel(os.path.join(FILES_DIR, 'fivequbitcode-errors-p0.4-bias10.jsonl'))
    fqc = FiveQubitCode()
    for _ in range(10):
        error = fem.generate(fqc, 0.4)
        assert isinstance(error, np.ndarray)
        assert len(error) == 10


def test_file_error_model_generate_skip_to_start():
    print()
    fem = FileErrorModel(os.path.join(FILES_DIR, 'fivequbitcode-errors-p0.4-bias10.jsonl'), 4)
    fqc = FiveQubitCode()
    packed_errors = (
        pt.pack(fem.generate(fqc, 0.4)),
        pt.pack(fem.generate(fqc, 0.4)),
        pt.pack(fem.generate(fqc, 0.4)),
    )
    expected_packed_errors = (("8400", 10), ("5280", 10), ("1080", 10))
    assert packed_errors == expected_packed_errors


@pytest.mark.parametrize('filename', [
    os.path.join(FILES_DIR, 'invalid_line1.json'),
    os.path.join(FILES_DIR, 'invalid_line2.json'),
    os.path.join(FILES_DIR, 'invalid_line3.json'),
    os.path.join(FILES_DIR, 'repeated_dic_key.json'),
    os.path.join(FILES_DIR, 'invalid_structure1.json'),
    os.path.join(FILES_DIR, 'invalid_structure2.json'),
    os.path.join(FILES_DIR, 'no_prob_in_header.json'),
    os.path.join(FILES_DIR, 'no_label_in_header.json'),
])
def test_file_error_model_invalid_file_header(filename):
    with pytest.raises(ValueError):
        FileErrorModel(filename)


def test_file_error_model_probability_distribution_invalid_probability_parameter():
    fem = FileErrorModel(os.path.join(FILES_DIR, 'fivequbitcode-errors-p0.4-bias10.jsonl'))
    with pytest.raises(ValueError):
        fem.probability_distribution(0.3)


def test_file_error_model__probability_distribution_no_probability_distribution_in_header():
    fem = FileErrorModel(os.path.join(FILES_DIR, 'no_probability_distribution.json'))
    with pytest.raises(ValueError):
        fem.probability_distribution(0.4)


def test_file_error_model_generate_invalid_probability_parameter():
    fem = FileErrorModel(os.path.join(FILES_DIR, 'fivequbitcode-errors-p0.4-bias10.jsonl'))
    fqc = FiveQubitCode()
    with pytest.raises(ValueError):
        fem.generate(fqc, 0.3)


def test_file_error_model_generate_invalid_error_lines():
    fem = FileErrorModel(os.path.join(FILES_DIR, 'header_lines_after_body_lines.json'))
    fqc = FiveQubitCode()
    with pytest.raises(ValueError):
        fem.generate(fqc, 0.4)
        fem.generate(fqc, 0.4)


def test_file_error_model_generate_no_more_errors_available():
    fem = FileErrorModel(os.path.join(FILES_DIR, 'no_more_errors_available.json'))
    fqc = FiveQubitCode()
    with pytest.raises(EOFError):
        fem.generate(fqc, 0.4)
        fem.generate(fqc, 0.4)
        fem.generate(fqc, 0.4)


def test_file_error_model_generate_no_more_errors_available_skip_to_start():
    fem = FileErrorModel(os.path.join(FILES_DIR, 'no_more_errors_available.json'), 1)
    fqc = FiveQubitCode()
    with pytest.raises(EOFError):
        fem.generate(fqc, 0.4)
        fem.generate(fqc, 0.4)


@pytest.mark.parametrize('filename', [
    os.path.join(FILES_DIR, 'incorrect_length_in_packed_error.json'),
    os.path.join(FILES_DIR, 'incorrect_length_in_packed_error2.json')
])
def test_file_error_model_generate_incorrect_length_in_packed_error(filename):
    fem = FileErrorModel(filename)
    fqc = FiveQubitCode()
    with pytest.raises(ValueError):
        fem.generate(fqc, 0.4)


# TESTS FOR GENERATED SAMPLES BELOW

@pytest.mark.parametrize('code, filename, decoder', [
    (RotatedPlanarCode(5, 5),
     os.path.join(FILES_DIR, 'rotated_planar_code_size_5_J_0.1_p_2.json'),
     RotatedPlanarMPSDecoder(chi=8)),
    (RotatedPlanarCode(5, 5),
     os.path.join(FILES_DIR, 'rotated_planar_code_size_5_J_0.1_p_4.json'),
     RotatedPlanarMPSDecoder(chi=8)),
    (RotatedPlanarCode(5, 5),
     os.path.join(FILES_DIR, 'rotated_planar_code_size_5_J_0.1_p_6.json'),
     RotatedPlanarMPSDecoder(chi=8)),
])
def test_file_error_model_generated_sample_error_probability(code, filename, decoder):
    with CliRunner().isolated_filesystem():  # isolate from logging_qecsim.ini
        # error model and probability from sample
        error_model = FileErrorModel(filename, start=1000)
        e_prob = error_model._probability
        # runs (repeat many times to ensure physical_error_rate is close to error_probability)
        max_runs = 100
        data = app.run(code, error_model, decoder, e_prob, max_runs)  # no error raised
        p_rate = data['physical_error_rate']
        p_var = data['error_weight_pvar'] / (data['n_k_d'][0] ** 2)  # physical_error_rate_pvar (power of 2 is correct)
        p_std = math.sqrt(p_var)  # physical_error_rate_std
        assert p_rate - p_std < e_prob < p_rate + p_std, (
            'physical_error_rate={} is not within 1 std={} of error_probability={}'.format(p_rate, p_std, e_prob))
