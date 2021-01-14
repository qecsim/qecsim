"""
This module contains functions to generate and merge stabilizer code simulation data.
"""
import collections
import itertools
import json
import logging
import statistics
import time

import numpy as np

from qecsim import paulitools as pt
from qecsim.model import DecodeResult

logger = logging.getLogger(__name__)


def _run_once(mode, code, time_steps, error_model, decoder, error_probability, measurement_error_probability, rng):
    """Implements run_once and run_once_ftp functions"""

    # assumptions
    assert (mode == 'ideal' and time_steps == 1) or mode == 'ftp'

    # generate step_error, step_syndrome and step_measurement_error for each time step
    step_errors, step_syndromes, step_measurement_errors = [], [], []
    for _ in range(time_steps):
        # step_error: random error based on error probability
        step_error = error_model.generate(code, error_probability, rng)
        step_errors.append(step_error)
        # step_syndrome: stabilizers that do not commute with the error
        step_syndrome = pt.bsp(step_error, code.stabilizers.T)
        step_syndromes.append(step_syndrome)
        # step_measurement_error: random syndrome bit flips based on measurement_error_probability
        if measurement_error_probability:
            step_measurement_error = rng.choice(
                (0, 1),
                size=step_syndrome.shape,
                p=(1 - measurement_error_probability, measurement_error_probability)
            )
        else:
            step_measurement_error = np.zeros(step_syndrome.shape, dtype=int)
        step_measurement_errors.append(step_measurement_error)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: step_errors={}'.format(step_errors))
        logger.debug('run: step_syndromes={}'.format(step_syndromes))
        logger.debug('run: step_measurement_errors={}'.format(step_measurement_errors))

    # error: sum of errors at each time step
    error = np.bitwise_xor.reduce(step_errors)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: error={}'.format(error))

    # syndrome: apply measurement_error at times t-1 and t to syndrome at time t
    syndrome = []
    for t in range(time_steps):
        syndrome.append(step_measurement_errors[t - 1] ^ step_syndromes[t] ^ step_measurement_errors[t])
    # convert syndrome to 2d numpy array
    syndrome = np.array(syndrome)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: syndrome={}'.format(syndrome))

    # decoding: boolean or best match recovery operation based on decoder
    ctx = {'error_model': error_model, 'error_probability': error_probability, 'error': error,
           'step_errors': step_errors, 'measurement_error_probability': measurement_error_probability,
           'step_measurement_errors': step_measurement_errors}
    # convert syndrome to 1d if mode is 'ideal'
    if mode == 'ideal':  # convert syndrome to 1d and call decode
        decoding = decoder.decode(code, syndrome[0], **ctx)
    if mode == 'ftp':  # call decode_ftp
        decoding = decoder.decode_ftp(code, time_steps, syndrome, **ctx)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: decoding={}'.format(decoding))

    # check for DecodeResult or recovery operation from decoder
    if isinstance(decoding, DecodeResult):
        # if decoder returns DecodeResult, success flag defines success
        success = decoding.success
    else:
        # otherwise, treat decoder return value as recovery operation
        recovery = decoding
        # recovered code
        recovered = recovery ^ error
        # success checks
        commutes_with_stabilizers = np.all(pt.bsp(recovered, code.stabilizers.T) == 0)
        if not commutes_with_stabilizers:
            log_data = {
                # models
                'code': repr(code),
                'error_model': repr(error_model),
                'decoder': repr(decoder),
                # variables
                'error': pt.pack(error),
                'recovery': pt.pack(recovery),
                # step variables
                'step_errors': [pt.pack(v) for v in step_errors],
                'step_measurement_errors': [pt.pack(v) for v in step_measurement_errors],
            }
            logger.warning('RECOVERY DOES NOT RETURN TO CODESPACE: {}'.format(json.dumps(log_data, sort_keys=True)))
        commutes_with_logicals = np.all(pt.bsp(recovered, code.logicals.T) == 0)
        success = commutes_with_stabilizers and commutes_with_logicals
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('run: commutes_with_stabilizers={}'.format(commutes_with_stabilizers))
            logger.debug('run: commutes_with_logicals={}'.format(commutes_with_logicals))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: success={}'.format(success))

    data = {
        'error_weight': pt.bsf_wt(np.array(step_errors)),
        'success': success
    }

    return data


def run_once(code, error_model, decoder, error_probability, rng=None):
    r"""
    Run a stabilizer code error-decode-recovery (ideal) simulation and return run data.

    Assumptions:

    * Error probability is float in [0, 1].

    Notes:

    * The simulation is as follows:

        * generate Pauli ``error`` by passing ``code`` and ``error_probability`` to
          :meth:`qecsim.model.ErrorModel.generate`.
        * evaluate ``syndrome`` as ``error`` :math:`\odot` ``code.stabilizers``:math:`^T`.
        * resolve ``decoding`` by passing ``code`` and ``syndrome`` to :meth:`qecsim.model.Decoder.decode`.
        * if ``decoding`` is boolean:

            * define ``success`` as ``decoding``.

        * else:

            * define ``recovery`` as ``decoding``.
            * assert that (``recovery`` :math:`\oplus` ``error``) :math:`\odot` ``code.stabilizers``:math:`^T = 0`, by
              construction.
            * define ``success`` as (``recovery`` :math:`\oplus` ``error``) :math:`\odot`
              ``code.logicals``:math:`^T = 0`.

    * :math:`\oplus` denotes binary addition defined as addition modulo 2, or equivalently exclusive-or.
    * See :func:`qecsim.paulitools.bsp` for definition of :math:`\odot`.
    * In addition to ``code`` and ``syndrome``, the following keyword parameters are passed as context to
      :meth:`qecsim.model.Decoder.decode`: ``error_model``, ``error_probability``, ``error``. Furthermore, in order to
      enable decoders to handle ideal and fault-tolerant decoding consistently, the following keyword parameters and
      default values are passed as context: ``step_errors=[error]``, ``measurement_error_probability=0.0`` and
      ``step_measurement_errors=[np.zeros(syndrome.shape)]``. Most decoders will ignore these parameters.
    * The returned data is in the following format:

    ::

        {
            'error_weight': 2,  # number qubits acted on non-trivially by error
            'success': True,    # recovery successful, i.e. (recovery ^ error) commutes with all logical operators
        }

    :param code: Stabilizer code.
    :type code: StabilizerCode
    :param error_model: Error model.
    :type error_model: ErrorModel
    :param decoder: Decoder.
    :type decoder: Decoder
    :param error_probability: Error probability.
    :type error_probability: float
    :param rng: Random number generator for error generation. (default=None resolves to numpy.random.default_rng())
    :type rng: numpy.random.Generator
    :return: error_weight and success flag.
    :rtype: dict
    :raises ValueError: if error_probability is not in [0, 1].
    """

    # validate parameters
    if not (0 <= error_probability <= 1):
        raise ValueError('Error probability must be in [0, 1].')

    # defaults
    rng = np.random.default_rng() if rng is None else rng

    return _run_once('ideal', code, 1, error_model, decoder, error_probability, 0.0, rng)


def run_once_ftp(code, time_steps, error_model, decoder, error_probability, measurement_error_probability=None,
                 rng=None):
    r"""
    Run a stabilizer code error-decode-recovery (fault-tolerant time-periodic) simulation and return run data.

    Assumptions:

    * Time steps is integer >= 1.
    * Probabilities, where defined, are float in [0, 1].

    Notes:

    * The simulation is as follows:

        * for each time step :math`t`:

            * generate Pauli ``step_errors[t]`` by passing ``code`` and ``error_probability`` to
              :meth:`qecsim.model.ErrorModel.generate`.
            * evaluate ``step_syndromes[t]`` as ``step_errors[t]`` :math:`\odot` ``code.stabilizers``:math:`^T`.
            * generate ``step_measurement_errors[t]`` as syndrome bit-flips.
            * generate ``syndrome[t]`` as ``step_measurement_errors[t-1]`` :math:`\oplus` ``step_syndromes[t]``
              :math:`\oplus` ``step_measurement_errors[t]``.

        * evaluate ``error`` as :math:`\bigoplus` ``step_errors``.
        * resolve ``decoding`` by passing ``code``, ``time_steps`` and ``syndrome`` to
          :meth:`qecsim.model.DecoderFTP.decode_ftp`.
        * if ``decoding`` is boolean:

            * define ``success`` as ``decoding``.

        * else:

            * define ``recovery`` as ``decoding``.
            * assert that (``recovery`` :math:`\oplus` ``error``) :math:`\odot` ``code.stabilizers``:math:`^T = 0`, by
              construction.
            * define ``success`` as (``recovery`` :math:`\oplus` ``error``) :math:`\odot`
              ``code.logicals``:math:`^T = 0`.

    * :math:`\oplus` denotes binary addition defined as addition modulo 2, or equivalently exclusive-or.
    * See :func:`qecsim.paulitools.bsp` for definition of :math:`\odot`.
    * In addition to ``code``, ``time_steps`` and ``syndrome``, the following keyword parameters are passed as context
      to :meth:`qecsim.model.DecoderFTP.decode_ftp`: ``error_model``, ``error_probability``, ``error``, ``step_errors``,
      ``measurement_error_probability`` and ``step_measurement_errors``. Most decoders will ignore these parameters.
    * The returned data is in the following format:

    ::

        {
            'error_weight': 2,  # number qubits acted on non-trivially by all step errors
            'success': True,    # recovery successful, i.e. (recovery ^ error) commutes with all logical operators
        }

    :param code: Stabilizer code.
    :type code: StabilizerCode
    :param time_steps: Number of time steps.
    :type time_steps: int
    :param error_model: Error model.
    :type error_model: ErrorModel
    :param decoder: Fault-tolerant time-periodic decoder.
    :type decoder: DecoderFTP
    :param error_probability: Error probability.
    :type error_probability: float
    :param measurement_error_probability: Measurement error probability.
           (default=None, None=error_probability or 0.0 if single time step)
    :type measurement_error_probability: float
    :param rng: Random number generator for error generation. (default=None resolves to numpy.random.default_rng())
    :type rng: numpy.random.Generator
    :return: error_weight and success flag.
    :rtype: dict
    :raises ValueError: if time_steps is not >= 1.
    :raises ValueError: if error_probability is not in [0, 1].
    :raises ValueError: if measurement_error_probability is not None or in [0, 1].
    """

    # validate parameters
    if not time_steps >= 1:
        raise ValueError('Time steps must be integer >= 1.')
    if not (0 <= error_probability <= 1):
        raise ValueError('Error probability must be in [0, 1].')
    if not (measurement_error_probability is None or (0 <= measurement_error_probability <= 1)):
        raise ValueError('Measurement error probability must be None or in [0, 1].')

    # defaults
    if measurement_error_probability is None:
        measurement_error_probability = 0.0 if time_steps == 1 else error_probability
    rng = np.random.default_rng() if rng is None else rng

    return _run_once('ftp', code, time_steps, error_model, decoder, error_probability, measurement_error_probability,
                     rng)


def _run(mode, code, time_steps, error_model, decoder, error_probability, measurement_error_probability,
         max_runs=None, max_failures=None, random_seed=None):
    """Implements run and run_ftp functions"""

    # assumptions
    assert (mode == 'ideal' and time_steps == 1) or mode == 'ftp'

    # derived defaults
    if max_runs is None and max_failures is None:
        max_runs = 1

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: code={}, time_steps={}, error_model={}, decoder={}, error_probability={},'
                     'measurement_error_probability={} max_runs={}, max_failures={}, random_seed={}.'
                     .format(code, time_steps, error_model, decoder, error_probability,
                             measurement_error_probability, max_runs, max_failures, random_seed))

    wall_time_start = time.perf_counter()

    runs_data = {
        'code': code.label,
        'n_k_d': code.n_k_d,
        'time_steps': time_steps,
        'error_model': error_model.label,
        'decoder': decoder.label,
        'error_probability': error_probability,
        'measurement_error_probability': measurement_error_probability,
        'n_run': 0,
        'n_success': 0,
        'n_fail': 0,
        'error_weight_total': 0,
        'error_weight_pvar': 0.0,
        'logical_failure_rate': 0.0,
        'physical_error_rate': 0.0,
        'wall_time': 0.0,
    }

    # if random_seed is None, unpredictable entropy is pulled from the OS, which we log for reproducibility
    seed_sequence = np.random.SeedSequence(random_seed)
    logger.info('run: np.random.SeedSequence.entropy={}'.format(seed_sequence.entropy))
    rng = np.random.default_rng(seed_sequence)

    error_weights = []  # list of error_weight from current run

    while ((max_runs is None or runs_data['n_run'] < max_runs)
           and (max_failures is None or runs_data['n_fail'] < max_failures)):
        # run simulation
        data = _run_once(mode, code, time_steps, error_model, decoder, error_probability, measurement_error_probability,
                         rng)
        # increment run counts
        runs_data['n_run'] += 1
        if data['success']:
            runs_data['n_success'] += 1
        else:
            runs_data['n_fail'] += 1
        # append error weight
        error_weights.append(data['error_weight'])

    # error weight statistics
    runs_data['error_weight_total'] = sum(error_weights)
    runs_data['error_weight_pvar'] = statistics.pvariance(error_weights)

    # rate statistics
    _add_rate_statistics(runs_data)

    # record wall_time
    runs_data['wall_time'] = time.perf_counter() - wall_time_start

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('run: aggregated_data={}'.format(runs_data))

    return runs_data


def run(code, error_model, decoder, error_probability, max_runs=None, max_failures=None, random_seed=None):
    """
    Execute stabilizer code error-decode-recovery (ideal) simulation many times and return aggregated runs data.

    See :func:`run_once` for details of a single run.

    Notes:

    * The simulation is run one or more times as determined by ``max_runs`` and ``max_failures``:

        * If ``max_runs`` specified, stop after ``max_runs`` runs.
        * If ``max_failures`` specified, stop after ``max_failures`` failures.
        * If ``max_runs`` and ``max_failures`` unspecified, run once.

    * The returned data is in the following format:

    ::

        {
            'code': '5-qubit',                      # given code.label
            'n_k_d': (5, 1, 3),                     # given code.n_k_d
            'time_steps': 1,                        # always 1 for ideal simulation
            'error_model': 'Depolarizing',          # given error_model.label
            'decoder': 'Naive',                     # given decoder.label
            'error_probability': 0.0,               # given error_probability
            'measurement_error_probability': 0.0    # alwyas 0.0 for ideal simulation
            'n_run': 0,                             # count of runs
            'n_success': 0,                         # count of successful recovery
            'n_fail': 0,                            # count of failed recovery
            'error_weight_total': 0,                # sum of error_weight over n_run runs
            'error_weight_pvar': 0.0,               # pvariance of error_weight over n_run runs
            'logical_failure_rate': 0.0,            # n_fail / n_run
            'physical_error_rate': 0.0,             # error_weight_total / n_k_d[0] / time_steps / n_run
            'wall_time': 0.0,                       # wall-time for run in fractional seconds
        }

    :param code: Stabilizer code.
    :type code: StabilizerCode
    :param error_model: Error model.
    :type error_model: ErrorModel
    :param decoder: Decoder.
    :type decoder: Decoder
    :param error_probability: Error probability.
    :type error_probability: float
    :param max_runs: Maximum number of runs. (default=None or 1 if max_failures unspecified, unrestricted=None)
    :type max_runs: int
    :param max_failures: Maximum number of failures. (default=None, unrestricted=None)
    :type max_failures: int
    :param random_seed: Error generation random seed. (default=None, unseeded=None)
    :type random_seed: int
    :return: Aggregated runs data.
    :rtype: dict
    :raises ValueError: if error_probability is not in [0, 1].
    """

    # validate parameters
    if not (0 <= error_probability <= 1):
        raise ValueError('Error probability must be in [0, 1].')
    return _run('ideal', code, 1, error_model, decoder, error_probability, 0.0, max_runs, max_failures, random_seed)


def run_ftp(code, time_steps, error_model, decoder, error_probability,
            measurement_error_probability=None, max_runs=None, max_failures=None, random_seed=None):
    """
    Execute stabilizer code error-decode-recovery (fault-tolerant time-periodic) simulation many times and return
    aggregated runs data.

    See :func:`run_once_ftp` for details of a single run.

    Notes:

    * The simulation is run one or more times as determined by ``max_runs`` and ``max_failures``:

        * If ``max_runs`` specified, stop after ``max_runs`` runs.
        * If ``max_failures`` specified, stop after ``max_failures`` failures.
        * If ``max_runs`` and ``max_failures`` unspecified, run once.

    * The returned data is in the following format:

    ::

        {
            'code': '5-qubit',                      # given code.label
            'n_k_d': (5, 1, 3),                     # given code.n_k_d
            'time_steps': 1,                        # given number of time steps
            'error_model': 'Depolarizing',          # given error_model.label
            'decoder': 'NaiveFTP',                  # given decoder.label
            'error_probability': 0.0,               # given error_probability
            'measurement_error_probability': 0.0    # given measurement_error_probability
            'n_run': 0,                             # count of runs
            'n_success': 0,                         # count of successful recovery
            'n_fail': 0,                            # count of failed recovery
            'error_weight_total': 0,                # sum of error_weight over n_run runs
            'error_weight_pvar': 0.0,               # pvariance of error_weight over n_run runs
            'logical_failure_rate': 0.0,            # n_fail / n_run
            'physical_error_rate': 0.0,             # error_weight_total / n_k_d[0] / time_steps / n_run
            'wall_time': 0.0,                       # wall-time for run in fractional seconds
        }

    :param code: Stabilizer code.
    :type code: StabilizerCode
    :param time_steps: Number of time steps.
    :type time_steps: int
    :param error_model: Error model.
    :type error_model: ErrorModel
    :param decoder: Fault-tolerant time-periodic decoder.
    :type decoder: DecoderFTP
    :param error_probability: Error probability.
    :type error_probability: float
    :param measurement_error_probability: Measurement error probability.
           (default=None, None=error_probability or 0.0 if single time step)
    :type measurement_error_probability: float
    :param max_runs: Maximum number of runs. (default=None or 1 if max_failures unspecified, unrestricted=None)
    :type max_runs: int
    :param max_failures: Maximum number of failures. (default=None, unrestricted=None)
    :type max_failures: int
    :param random_seed: Error generation random seed. (default=None, unseeded=None)
    :type random_seed: int
    :return: Aggregated runs data.
    :rtype: dict
    :raises ValueError: if time_steps is not >= 1.
    :raises ValueError: if error_probability is not in [0, 1].
    :raises ValueError: if measurement_error_probability is not None or in [0, 1].
    """

    # validate parameters
    if not (0 <= error_probability <= 1):
        raise ValueError('Error probability must be in [0, 1].')
    if not time_steps >= 1:
        raise ValueError('Time steps must be integer >= 1.')
    if not (measurement_error_probability is None or (0 <= measurement_error_probability <= 1)):
        raise ValueError('Measurement error probability must be None or in [0, 1].')

    # derived defaults
    if measurement_error_probability is None:
        measurement_error_probability = 0.0 if time_steps == 1 else error_probability

    return _run('ftp', code, time_steps, error_model, decoder, error_probability, measurement_error_probability,
                max_runs, max_failures, random_seed)


def _add_rate_statistics(runs_data):
    """Add `logical_failure_rate` and `physical_error_rate` as defined in :func:`_run`."""
    # extract data
    time_steps = runs_data['time_steps']
    n_run = runs_data['n_run']
    n_fail = runs_data['n_fail']
    error_weight_total = runs_data['error_weight_total']
    code_n_qubits = runs_data['n_k_d'][0]
    # add rate statistics
    runs_data['logical_failure_rate'] = n_fail / n_run
    runs_data['physical_error_rate'] = error_weight_total / code_n_qubits / time_steps / n_run


def merge(*data_list):
    """
    Merge any number of lists of aggregated runs data.

    Notes:

    * The runs data is in the format specified in :func:`run` and :func:`fun_ftp`.
    * Merged data is grouped by: `(code, n_k_d, error_model, decoder, error_probability, time_steps,
      measurement_error_probability)`.
    * The following values are summed: `n_run`, `n_success`, `n_fail`, `error_weight_total`, `wall_time`.
    * The following values are recalculated: `logical_failure_rate`, `physical_error_rate`.
    * The following values are *not* currently recalculated: `error_weight_pvar`.

    :param data_list: List of aggregated runs data.
    :type data_list: list of dict
    :return: Merged list of aggregated runs data.
    :rtype: list of dict
    """
    # define group keys, value keys and zero values
    grp_keys = ('code', 'n_k_d', 'error_model', 'decoder', 'error_probability', 'time_steps',
                'measurement_error_probability')
    val_keys = ('n_run', 'n_fail', 'n_success', 'error_weight_total', 'wall_time')
    zero_vals = (0, 0, 0, 0, 0.0)
    # map of groups to sums (use ordered dict to preserve order as much as possible).
    grps_to_sums = collections.OrderedDict()
    # iterate through single list from given data lists
    for runs_data in itertools.chain(*data_list):
        # support for 0.10 and 0.15 files: define defaults, create new data with defaults overwritten by data
        defaults_0_16 = {'time_steps': 1, 'measurement_error_probability': 0.0}
        runs_data = dict(itertools.chain(defaults_0_16.items(), runs_data.items()))
        # extract group and values from data (note: force lists to tuples so group_id is hashable)
        group_id = tuple(tuple(v) if isinstance(v, list) else v for v in (runs_data[k] for k in grp_keys))
        values = tuple(runs_data[k] for k in val_keys)
        # get sums (or zeros if not found)
        sums = grps_to_sums.get(group_id, zero_vals)
        # add values to sums
        sums = tuple(sum(x) for x in zip(values, sums))
        # put summed_values
        grps_to_sums[group_id] = sums
    # flatten grps_to_sums
    merged_data_list = [dict(zip(grp_keys + val_keys, group_id + sums)) for group_id, sums in grps_to_sums.items()]
    # add rate statistics
    for runs_data in merged_data_list:
        _add_rate_statistics(runs_data)
    return merged_data_list
