"""
This module contains the qecsim command line interface (CLI).

New codes, error models and decoders can be offered in the CLI by adding them to the _CODE_PARAMETER,
_ERROR_MODEL_PARAMETER and _DECODER_PARAMETER variables respectively, and updating the docstring of :func:`run`.

New FTP codes, error models and decoders can be offered in the CLI by adding them to the _FTP_CODE_PARAMETER,
_FTP_ERROR_MODEL_PARAMETER and _FTP_DECODER_PARAMETER variables respectively, and updating the docstring of
:func:`run_ftp`.
"""

import ast
import json
import logging
import re
from collections import namedtuple

import click

import qecsim
from qecsim import app
from qecsim import util
# from qecsim.models.color import Color666Code
# from qecsim.models.color import Color666MPSDecoder
# from qecsim.models.generic import BiasedDepolarizingErrorModel
# from qecsim.models.generic import BiasedYXErrorModel
from qecsim.models.generic import BitFlipErrorModel
from qecsim.models.generic import BitPhaseFlipErrorModel
# from qecsim.models.generic import CenterSliceErrorModel
from qecsim.models.generic import DepolarizingErrorModel
# from qecsim.models.generic import FileErrorModel
from qecsim.models.generic import NaiveDecoder
from qecsim.models.generic import PhaseFlipErrorModel
from qecsim.models.linear import FiveQubitCode, SteaneCode

# from qecsim.models.planar import PlanarAFCXErrorModel, PlanarAFCXDecoder
# from qecsim.models.planar import PlanarAVCXErrorModel, PlanarAVCXDecoder
# from qecsim.models.planar import PlanarCMWPMDecoder, PlanarMWPMDecoder
# from qecsim.models.planar import PlanarCode
# from qecsim.models.planar import PlanarMPSDecoder, PlanarRMPSDecoder
# from qecsim.models.planar import PlanarYDecoder
# from qecsim.models.rotatedplanar import RotatedPlanarCode
# from qecsim.models.rotatedplanar import RotatedPlanarMPSDecoder, RotatedPlanarRMPSDecoder
# from qecsim.models.rotatedplanar import RotatedPlanarSMWPMDecoder
# from qecsim.models.rotatedplanarxz import RotatedPlanarXZCode
# from qecsim.models.rotatedplanarxz import RotatedPlanarXZRMPSDecoder
# from qecsim.models.rotatedtoric import RotatedToricCode
# from qecsim.models.rotatedtoric import RotatedToricSMWPMDecoder
# from qecsim.models.toric import ToricCode
# from qecsim.models.toric import ToricMWPMDecoder

logger = logging.getLogger(__name__)


class _ConstructorParamType(click.ParamType):
    """
    Constructor param type that accepts parameters in the format ``name(<args>)``.
    """
    name = 'constructor'
    ConstructorValue = namedtuple('ConstructorValue', 'param value constructor arguments')

    def __init__(self, constructors):
        """
        Initialise new constructor parameter type.

        :param constructors: Map of constructor names to constructor functions.
        :type constructors: dict of str to function
        """
        self._constructors = constructors

    def get_metavar(self, param):
        """See ``click.ParamType.get_metavar``"""
        return '[{}]'.format('|'.join(sorted(self._constructors.keys())))

    def get_missing_message(self, param):
        """See ``click.ParamType.get_missing_message``"""
        return '(choose from {})'.format(', '.join(sorted(self._constructors.keys())))

    def convert(self, value, param, ctx):
        """
        Convert value to named tuple (param, value, constructor, arguments).

        If the value is correctly formatted as ``name`` or ``name(<args>)`` then:

        * param and value are passed through.
        * constructor is resolved using the constructors map.
        * arguments is resolved to a tuple using a literal evaluation of args.

        The returned tuple ``my_cstr`` can be used to construct a new object using:
        ``my_cstr.constructor(*my_cstr.arguments)``

        See ``click.ParamType.convert`` for more details.

        :param value: Parameter value.
        :type value: str
        :param param: Parameter.
        :type param: click.Parameter
        :param ctx: Context.
        :type ctx: click.Context
        :return: Named tuple (param, value, constructor, arguments)
        :rtype: ConstructorValue
        :raises BadParameter: if the value cannot be parsed or does not correspond to valid constructor or arguments.
        """
        # pass through None or already converted
        if value is None or isinstance(value, _ConstructorParamType.ConstructorValue):
            return value

        # constructor regex match
        constructor_match = re.fullmatch(r'''
            # match 'toric(3,3)' as {'constructor_name': 'toric', 'constructor_args': '3,3'}
            (?P<constructor_name>[\w.]+)  # capture constructor_name, e.g. 'toric'
            (?:\(\s*                      # skip opening parenthesis and leading whitespace
                (?P<constructor_args>.*?) # capture constructor_args, e.g. '3,3'
            ,?\s*\))?                     # skip trailing comma, trailing whitespace and closing parenthesis
        ''', value, re.VERBOSE)

        # check format
        if constructor_match is None:
            self.fail('{} (format as name(<args>))'.format(value), param, ctx)

        # convert constructor_name to constructor
        constructor_name = constructor_match.group('constructor_name')
        if constructor_name in self._constructors.keys():
            # select constructor from map
            constructor = self._constructors[constructor_name]
        else:
            self.fail('{} (choose from {})'.format(value, ', '.join(sorted(self._constructors.keys()))), param, ctx)

        # convert constructor_args to arguments tuple
        constructor_args = constructor_match.group('constructor_args')
        if constructor_args:
            try:
                # eval args as literal (add comma to force tuple)
                arguments = ast.literal_eval(constructor_args + ',')
            except Exception as ex:
                self.fail('{} (failed to parse arguments "{}")'.format(value, ex), param, ctx)
        else:
            # no args -> empty tuple
            arguments = tuple()

        return _ConstructorParamType.ConstructorValue(param, value, constructor, arguments)

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self._constructors)


@click.group()
@click.version_option(version=qecsim.__version__, prog_name='qecsim')
def cli():
    """
    qecsim - quantum error correction simulator using stabilizer codes.

    See python qecsim.pyz COMMAND --help for command-specific help.
    """
    util.init_logging()


# custom param types
_CODE_PARAMETER = _ConstructorParamType({
    # add new codes here mapping name -> constructor
    # 'color666': Color666Code,
    'linear.five_qubit': FiveQubitCode,
    'linear.steane': SteaneCode,
    # 'planar': PlanarCode,
    # 'rotated_planar': RotatedPlanarCode,
    # 'rotated_planar_xz': RotatedPlanarXZCode,
    # 'rotated_toric': RotatedToricCode,
    # 'toric': ToricCode,
})
_ERROR_MODEL_PARAMETER = _ConstructorParamType({
    # add new error_models here mapping name -> constructor
    'generic.depolarizing': DepolarizingErrorModel,
    'generic.bit_flip': BitFlipErrorModel,
    'generic.phase_flip': PhaseFlipErrorModel,
    'generic.bit_phase_flip': BitPhaseFlipErrorModel,
    # 'generic.biased_depolarizing': BiasedDepolarizingErrorModel,
    # 'generic.biased_y_x': BiasedYXErrorModel,
    # 'generic.file': FileErrorModel,
    # 'generic.center_slice': CenterSliceErrorModel,
    # 'planar.afcx': PlanarAFCXErrorModel,
    # 'planar.avcx': PlanarAVCXErrorModel,
})
_DECODER_PARAMETER = _ConstructorParamType({
    # add new decoders here mapping name -> constructor
    # 'color666.mps': Color666MPSDecoder,
    'generic.naive': NaiveDecoder,
    # 'planar.afcx': PlanarAFCXDecoder,
    # 'planar.avcx': PlanarAVCXDecoder,
    # 'planar.cmwpm': PlanarCMWPMDecoder,
    # 'planar.mps': PlanarMPSDecoder,
    # 'planar.mwpm': PlanarMWPMDecoder,
    # 'planar.rmps': PlanarRMPSDecoder,
    # 'planar.y': PlanarYDecoder,
    # 'rotated_planar.mps': RotatedPlanarMPSDecoder,
    # 'rotated_planar.rmps': RotatedPlanarRMPSDecoder,
    # 'rotated_planar.smwpm': RotatedPlanarSMWPMDecoder,
    # 'rotated_planar_xz.rmps': RotatedPlanarXZRMPSDecoder,
    # 'rotated_toric.smwpm': RotatedToricSMWPMDecoder,
    # 'toric.mwpm': ToricMWPMDecoder,
})


# custom validators
def _validate_error_probability(ctx, param, value):
    if not (0 <= value <= 1):
        raise click.BadParameter('{} is not in [0.0, 1.0]'.format(value), ctx, param)
    return value


def _validate_error_probabilities(ctx, param, value):
    for v in value:
        _validate_error_probability(ctx, param, v)
    return value


@cli.command()
@click.argument('t_code', type=_CODE_PARAMETER, metavar='CODE')
@click.argument('t_error_model', type=_ERROR_MODEL_PARAMETER, metavar='ERROR_MODEL')
@click.argument('t_decoder', type=_DECODER_PARAMETER, metavar='DECODER')
@click.argument('error_probabilities', required=True, nargs=-1, type=float, metavar='ERROR_PROBABILITY...',
                callback=_validate_error_probabilities)
@click.option('--max-failures', '-f', type=click.IntRange(min=1), metavar='INT',
              help='Maximum number of failures for each probability.')
@click.option('--max-runs', '-r', type=click.IntRange(min=1), metavar='INT',
              help='Maximum number of runs for each probability.  [default: 1 if max-failures unspecified]')
@click.option('--output', '-o', default='-', type=click.Path(allow_dash=True), metavar='FILENAME',
              help='Output file. (Writes to log if file exists).')
@click.option('--random-seed', '-s', type=click.IntRange(min=0, max=2 ** 32 - 1), metavar='INT',
              help='Random seed for qubit error generation. (Re-applied for each probability).')
def run(t_code, t_error_model, t_decoder, error_probabilities, max_failures, max_runs, output, random_seed):
    """
    Simulate quantum error correction.

    Arguments:

    \b
     CODE                  Stabilizer code in format name(<args>)
      color666(size)                     Color 6.6.6 (size INT odd >=3)
      linear.five_qubit                  Linear 5-qubit
      linear.steane                      Linear Steane
      planar(rows, cols)                 Planar (rows INT >= 2, cols INT >= 2)
      rotated_planar(rows, cols)         Rotated planar (rows INT >= 3,
                                                         cols INT >= 3)
      rotated_planar_xz(size)            Rotated planar XZ (size INT odd >= 3)
      toric(rows, cols)                  Toric (rows INT >= 2, cols INT >= 2)

    \b
     ERROR_MODEL           Error model in format name(<args>)
      generic.biased_depolarizing(bias, ...) Biased (bias FLOAT >= 0, [axis] CHAR)
      generic.biased_y_x(bias)           Biased Y v. X (bias FLOAT >= 0)
      generic.bit_flip                   Pr I,X,Y,Z is 1-p,p,0,0
      generic.bit_phase_flip             Pr I,X,Y,Z is 1-p,0,p,0
      generic.center_slice(lim, pos)     Slice (lim 3-tuple of FLOAT, pos FLOAT)
      generic.depolarizing               Pr I,X,Y,Z is 1-p,p/3,p/3,p/3
      generic.file(filename, start)      File (filename STR, [start] INT >= 0)
      generic.phase_flip                 Pr I,X,Y,Z is 1-p,0,0,p
      planar.afcx(correlation)           AFCX (correlation FLOAT >= 0)
      planar.avcx(correlation)           AVCX (correlation FLOAT >= 0)

    \b
     DECODER               Decoder in format name(<args>)
      color666.mps(chi, ...)             MPS ([chi] INT, ...)
      generic.naive(max_qubits)          Naive ([max_qubits] INT)
      planar.afcx(chi, ...)              AFCX ([chi] INT >=0, [mode] CHAR, ...)
      planar.avcx(chi, ...)              AVCX ([chi] INT >=0, [mode] CHAR, ...)
      planar.cmwpm(factor, ...)          Converging MWPM ([factor] FLOAT >=0, ...)
      planar.mps(chi, ...)               MPS ([chi] INT >=0, [mode] CHAR, ...)
      planar.mwpm                        MWPM
      planar.rmps(chi, ...)              RMPS ([chi] INT >=0, [mode] CHAR, ...)
      planar.y                           Y-noise
      rotated_planar.mps(chi, ...)       MPS ([chi] INT >=0, [mode] CHAR, ...)
      rotated_planar.rmps(chi, ...)      RMPS ([chi] INT >=0, [mode] CHAR, ...)
      rotated_planar.smwpm               Symmetry MWPM
      rotated_planar_xz.rmps(chi, ...)   RMPS ([chi] INT >=0, [mode] CHAR, ...)
      toric.mwpm                         MWPM

    \b
     ERROR_PROBABILITY...  One or more probabilities as FLOAT in [0.0, 1.0]

    Examples:

     python qecsim.pyz run -r 5 "color666(7)" "generic.bit_flip" "color666.mps(16)" 0.1

     python qecsim.pyz run -r 10 "linear.five_qubit" "generic.depolarizing" "generic.naive" 0.05 0.1

     python qecsim.pyz run -f 5 "linear.steane" "generic.phase_flip" "generic.naive" 0.05 0.1 0.15

     python qecsim.pyz run -r 20 "planar(7,7)" "generic.bit_flip" "planar.mps(6, 'a')" 0.101 0.102 0.103

     python qecsim.pyz run -o "data.json" -f 5 -r 50 -s 5 "toric(3,3)" "generic.bit_flip" "toric.mwpm" 0.1
    """
    # INPUT
    try:
        code = t_code.constructor(*t_code.arguments)  # FiveQubitCode()
    except Exception as ex:
        raise click.BadParameter('{} (failed to construct code "{}")'.format(t_code.value, ex), param=t_code.param)
    code.validate()
    try:
        error_model = t_error_model.constructor(*t_error_model.arguments)  # DepolarizingErrorModel()
    except Exception as ex:
        raise click.BadParameter('{} (failed to construct error model "{}")'.format(t_error_model.value, ex),
                                 param=t_error_model.param)
    try:
        decoder = t_decoder.constructor(*t_decoder.arguments)  # NaiveDecoder()
    except Exception as ex:
        raise click.BadParameter('{} (failed to construct decoder "{}")'.format(t_decoder.value, ex),
                                 param=t_decoder.param)

    logger.info('RUN STARTING: code={}, error_model={}, decoder={}, error_probabilities={}, max_failures={}, '
                'max_runs={}, random_seed={}.'
                .format(code, error_model, decoder, error_probabilities, max_failures, max_runs, random_seed))

    # RUN
    data = []
    for error_probability in error_probabilities:
        runs_data = app.run(code, error_model, decoder, error_probability,
                            max_runs=max_runs, max_failures=max_failures, random_seed=random_seed)
        data.append(runs_data)

    logger.info('RUN COMPLETE: data={}'.format(data))

    # OUTPUT
    if output == '-':
        # write to stdout
        click.echo(json.dumps(data, sort_keys=True))
    else:
        try:
            # attempt to save to output filename (mode='x' -> fail if file exists)
            with open(output, 'x') as f:
                json.dump(data, f, sort_keys=True)
        except IOError as ex:
            logger.error('recovered data:\n{}'.format(json.dumps(data, sort_keys=True)))
            raise click.ClickException('{} (failed to open output file "{}")'.format(output, ex))


# custom FT param types
_FTP_CODE_PARAMETER = _ConstructorParamType({
    # add new codes here mapping name -> constructor
    # 'rotated_planar': RotatedPlanarCode,
    # 'rotated_toric': RotatedToricCode,
})
_FTP_ERROR_MODEL_PARAMETER = _ConstructorParamType({
    # add new error_models here mapping name -> constructor
    'generic.depolarizing': DepolarizingErrorModel,
    'generic.bit_flip': BitFlipErrorModel,
    'generic.phase_flip': PhaseFlipErrorModel,
    'generic.bit_phase_flip': BitPhaseFlipErrorModel,
    # 'generic.biased_depolarizing': BiasedDepolarizingErrorModel,
    # 'generic.biased_y_x': BiasedYXErrorModel,
    # 'generic.center_slice': CenterSliceErrorModel,
})
_FTP_DECODER_PARAMETER = _ConstructorParamType({
    # add new decoders here mapping name -> constructor
    # 'rotated_planar.smwpm': RotatedPlanarSMWPMDecoder,
    # 'rotated_toric.smwpm': RotatedToricSMWPMDecoder,
})


# custom FT validators
def _validate_measurement_error_probability(ctx, param, value):
    if not (value is None or (0 <= value <= 1)):
        raise click.BadParameter('{} is not in [0.0, 1.0]'.format(value), ctx, param)
    return value


@cli.command()
@click.argument('t_code', type=_FTP_CODE_PARAMETER, metavar='CODE')
@click.argument('time_steps', type=click.IntRange(min=1), metavar='TIME_STEPS')
@click.argument('t_error_model', type=_FTP_ERROR_MODEL_PARAMETER, metavar='ERROR_MODEL')
@click.argument('t_decoder', type=_FTP_DECODER_PARAMETER, metavar='DECODER')
@click.argument('error_probabilities', required=True, nargs=-1, type=float, metavar='ERROR_PROBABILITY...',
                callback=_validate_error_probabilities)
@click.option('--max-failures', '-f', type=click.IntRange(min=1), metavar='INT',
              help='Maximum number of failures for each probability.')
@click.option('--max-runs', '-r', type=click.IntRange(min=1), metavar='INT',
              help='Maximum number of runs for each probability. [default: 1 if max_failures unspecified]')
@click.option('--measurement-error-probability', '-m', type=float, default=None,
              callback=_validate_measurement_error_probability,
              help='Measurement error probability [default: 0.0 if TIME_STEPS == 1 else ERROR_PROBABILITY].')
@click.option('--output', '-o', default='-', type=click.Path(allow_dash=True), metavar='FILENAME',
              help='Output file. (Writes to log if file exists).')
@click.option('--random-seed', '-s', type=click.IntRange(min=0, max=2 ** 32 - 1), metavar='INT',
              help='Random seed for qubit error generation. (Re-applied for each probability).')
def run_ftp(t_code, time_steps, t_error_model, t_decoder, error_probabilities, max_failures, max_runs,
            measurement_error_probability, output, random_seed):
    """
    Simulate fault-tolerant (time-periodic) quantum error correction.

    Arguments:

    \b
     CODE                  Stabilizer code in format name(<args>)
      rotated_planar(rows, cols)         Rotated planar (rows INT >= 3,
                                                         cols INT >= 3)
      rotated_toric(rows, cols)          Rotated toric (rows INT even >= 2,
                                                        cols INT even >= 2)

    \b
     TIME_STEPS            Number of time steps as INT >= 1

    \b
     ERROR_MODEL           Error model in format name(<args>)
      generic.biased_depolarizing(bias, ...) Biased (bias FLOAT >= 0, [axis] CHAR)
      generic.biased_y_x(bias)           Biased Y v. X (bias FLOAT >= 0)
      generic.bit_flip                   Pr I,X,Y,Z is 1-p,p,0,0
      generic.bit_phase_flip             Pr I,X,Y,Z is 1-p,0,p,0
      generic.center_slice(lim, pos)     Slice (lim 3-tuple of FLOAT, pos FLOAT)
      generic.depolarizing               Pr I,X,Y,Z is 1-p,p/3,p/3,p/3
      generic.phase_flip                 Pr I,X,Y,Z is 1-p,0,0,p

    \b
     DECODER               Decoder in format name(<args>)
      rotated_planar.smwpm               Symmetry MWPM
      rotated_toric.smwpm                Symmetry MWPM

    \b
     ERROR_PROBABILITY...  One or more probabilities as FLOAT in [0.0, 1.0]

    Examples:

     python qecsim.pyz run-ftp -r 5 "rotated_planar(13,13)" 13 "generic.bit_phase_flip" "rotated_planar.smwpm" 0.1 0.2

     python qecsim.pyz run-ftp -r 5 -m 0.05 "rotated_toric(6,6)" 4 "generic.bit_phase_flip" "rotated_toric.smwpm" 0.1

     python qecsim.pyz run-ftp -r 5 -o "data.json" "rotated_planar(7,7)" 7 "generic.depolarizing" "rotated_planar.smwpm"
    0.1
    """
    # INPUT
    try:
        code = t_code.constructor(*t_code.arguments)  # RotatedPlanar(7, 7)
    except Exception as ex:
        raise click.BadParameter('{} (failed to construct code "{}")'.format(t_code.value, ex), param=t_code.param)
    code.validate()
    try:
        error_model = t_error_model.constructor(*t_error_model.arguments)  # DepolarizingErrorModel()
    except Exception as ex:
        raise click.BadParameter('{} (failed to construct error model "{}")'.format(t_error_model.value, ex),
                                 param=t_error_model.param)
    try:
        decoder = t_decoder.constructor(*t_decoder.arguments)  # RotatedPlanarSMWPMDecoder()
    except Exception as ex:
        raise click.BadParameter('{} (failed to construct decoder "{}")'.format(t_decoder.value, ex),
                                 param=t_decoder.param)

    logger.info('RUN STARTING: code={}, time_steps={}, error_model={}, decoder={}, error_probabilities={}, '
                'max_failures={}, max_runs={}, measurement_error_probability={}, random_seed={}.'
                .format(code, time_steps, error_model, decoder, error_probabilities, max_failures, max_runs,
                        measurement_error_probability, random_seed))

    # RUN
    data = []
    for error_probability in error_probabilities:
        runs_data = app.run_ftp(code, time_steps, error_model, decoder, error_probability,
                                measurement_error_probability=measurement_error_probability,
                                max_runs=max_runs, max_failures=max_failures, random_seed=random_seed)
        data.append(runs_data)

    logger.info('RUN COMPLETE: data={}'.format(data))

    # OUTPUT
    if output == '-':
        # write to stdout
        click.echo(json.dumps(data, sort_keys=True))
    else:
        try:
            # attempt to save to output filename (mode='x' -> fail if file exists)
            with open(output, 'x') as f:
                json.dump(data, f, sort_keys=True)
        except IOError as ex:
            logger.error('recovered data:\n{}'.format(json.dumps(data, sort_keys=True)))
            raise click.ClickException('{} (failed to open output file "{}")'.format(output, ex))


@cli.command()
@click.argument('data_file', required=True, nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--output', '-o', default='-', type=click.Path(allow_dash=True), metavar='FILENAME',
              help='Output file. (Writes to log if file exists).')
def merge(data_file, output):
    """
    Merge simulation data files.

    Arguments:

    \b
     DATA_FILE...          One or more data files

    Examples:

     python qecsim.pyz merge "data1.json" "data2.json" "data3.json"

     python qecsim.pyz merge -o "merged.json" data*.json

    """
    # INPUT
    input_data = []

    # extract data from input files
    for input_file in data_file:
        try:
            with open(input_file, 'r') as f:
                input_data.append(json.load(f))
        except ValueError as ex:
            raise click.ClickException('{} (failed to parse JSON data "{}")'.format(input_file, ex))

    # MERGE
    data = app.merge(*input_data)

    # OUTPUT
    if output == '-':
        # write to stdout
        click.echo(json.dumps(data, sort_keys=True))
    else:
        try:
            # attempt to save to output filename (mode='x' -> fail if file exists)
            with open(output, 'x') as f:
                json.dump(data, f, sort_keys=True)
        except IOError as ex:
            logger.error('recovered data:\n{}'.format(json.dumps(data, sort_keys=True)))
            raise click.ClickException('{} (failed to open output file "{}")'.format(output, ex))
