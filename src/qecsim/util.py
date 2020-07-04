"""
This module contains utility functions.
"""
import ctypes
import itertools
import logging
import logging.config
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# default qecsim home configuration directory, e.g. resolves to ~/.qecsim.
QECSIM_CFG_HOME = '.qecsim'
# environment variable to override qecsim home configuration directory, e.g. /my_qecsim_cfg_dir
QECSIM_CFG_ENV_KEY = 'QECSIM_CFG'


def _resolve_path(relative_path):
    """
    Resolve path relative to $QECSIM_CFG/, ./, and ~/.qecsim/.

    Notes:

    * Path resolves to ``$QECSIM_CFG/<relative_path>`` (if ``QECSIM_CFG`` env variable is set).
    * Path resolves to  ``./<relative_path>`` or ``~/.qecsim/<relative_path>`` (otherwise).

    :param relative_path: Relative path, e.g. Path('clib/libpypm.so')
    :type relative_path: Path or str
    :return: Path to file.
    :rtype: Path
    :raises FileNotFoundError: if none of the resolved paths exists.
    """
    # get config path from env
    qecsim_cfg_path = os.getenv(QECSIM_CFG_ENV_KEY)
    # list of paths tried for exception reporting
    paths = []
    # if env variable set the only try relative to that
    if qecsim_cfg_path is not None:  # i.e. is set but could be empty string
        # $QECSIM_CFG/<relative_path>
        path = Path(qecsim_cfg_path) / relative_path
        paths.append(path)
    else:  # otherwise try relative to ./ and ~/.qecsim/
        # ./<relative_path>
        path = Path(relative_path)
        paths.append(path)
        if not path.exists():
            # ~/.qecsim/<relative_path>
            path = Path.home() / QECSIM_CFG_HOME / relative_path
            paths.append(path)
    # raise exception is no resolved path exists.
    if not path.exists():
        raise FileNotFoundError('Resolved paths do not exist: {}.'.format(', '.join(str(p) for p in paths)))
    return path


def init_logging():
    """
    Initialise logging.

    Notes:

    * Configuration loaded from ``$QECSIM_CFG/logging_qecsim.ini`` (if ``QECSIM_CFG`` env variable is set).
    * Configuration loaded from ``./logging_qecsim.ini`` or ``~/.qecsim/logging_qecsim.ini`` (otherwise).
    * If configuration not found then a basic configuration is used.

    """
    try:
        path = _resolve_path('logging_qecsim.ini')
        logging.config.fileConfig(path, disable_existing_loggers=False)
    except FileNotFoundError:
        # default silently to basic logging config
        logging.basicConfig()
    except Exception:
        # default noisily to basic logging config
        logging.basicConfig()
        logger.exception('Failed to load logging configuration. Defaulting to basic configuration.')


def load_clib(filename):
    """
    Load clib shared library.

    Notes:

    * Library loaded from ``$QECSIM_CFG/clib/<filename>`` (if ``QECSIM_CFG`` env variable is set).
    * Library loaded from ``./clib/<filename>`` or ``~/.qecsim/clib/<filename>`` (otherwise).

    :param filename: Library file name, e.g. 'libpypm.so'.
    :type filename: str
    :return: Library
    :rtype: ctypes.CDLL
    :raises FileNotFoundError: if filename does not resolve to an existing file.
    :raises OSError: if filename cannot be loaded as a shared library.
    """
    try:
        path = _resolve_path(Path('clib', filename))
        # load clib shared library
        return ctypes.CDLL(path)
    except OSError:
        logger.exception('Failed to load clib: {}.'.format(filename))
        raise


def chunker(iterable, chunk_len):
    """
    Returns an iterator of iterables of length chunk or less (for the final chunk).

    :param iterable: Iterable.
    :type iterable: iterable
    :param chunk_len: Chunk length.
    :type chunk_len: int
    :return: An iterator of iterables of size chunk or less
    :rtype: iterator
    """
    it = iter(iterable)
    while True:
        try:
            yield itertools.chain([next(it)], itertools.islice(it, chunk_len - 1))
        except StopIteration:  # fix for Python 3.6 deprecation warning, see PEP 479
            return
