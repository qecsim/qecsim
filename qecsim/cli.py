"""
This module contains the qecsim command line interface (CLI).

New codes, error models and decoders can be offered in the CLI by adding them to the _CODE_PARAMETER,
_ERROR_MODEL_PARAMETER and _DECODER_PARAMETER variables respectively, and updating the docstring of :func:`run`.

New FTP codes, error models and decoders can be offered in the CLI by adding them to the _FTP_CODE_PARAMETER,
_FTP_ERROR_MODEL_PARAMETER and _FTP_DECODER_PARAMETER variables respectively, and updating the docstring of
:func:`run_ftp`.
"""

import logging

import click

import qecsim

logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version=qecsim.__version__, prog_name='qecsim')
def cli():
    """
    qecsim - quantum error correction simulator using stabilizer codes.

    See python qecsim.pyz COMMAND --help for command-specific help.
    """


@cli.command()
@click.argument('t_code', metavar='CODE')
@click.argument('t_error_model', metavar='ERROR_MODEL')
@click.argument('t_decoder', metavar='DECODER')
@click.argument('error_probabilities', required=True, nargs=-1, type=float, metavar='ERROR_PROBABILITY...')
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
    print(t_code, t_error_model, t_decoder, error_probabilities, max_failures, max_runs, output, random_seed)

@cli.command()
@click.argument('t_code', metavar='CODE')
@click.argument('time_steps', type=click.IntRange(min=1), metavar='TIME_STEPS')
@click.argument('t_error_model', metavar='ERROR_MODEL')
@click.argument('t_decoder', metavar='DECODER')
@click.argument('error_probabilities', required=True, nargs=-1, type=float, metavar='ERROR_PROBABILITY...')
@click.option('--max-failures', '-f', type=click.IntRange(min=1), metavar='INT',
              help='Maximum number of failures for each probability.')
@click.option('--max-runs', '-r', type=click.IntRange(min=1), metavar='INT',
              help='Maximum number of runs for each probability. [default: 1 if max_failures unspecified]')
@click.option('--measurement-error-probability', '-m', type=float, default=None,
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
    print(t_code, time_steps, t_error_model, t_decoder, error_probabilities, max_failures, max_runs,
          measurement_error_probability, output, random_seed)


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
    print(data_file, output)
