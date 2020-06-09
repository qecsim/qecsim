"""
This module contains utility functions.
"""
import ctypes
import fcntl
import functools
import gzip
import itertools
import json
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


def file_cache(filename, compress=False):
    """
    File cache decorator.

    Attributes:

    * enabled (bool):   If the cache is enabled (default=True).
    * _filename (str):  Filename of cache file (read-only).
    * _compress (bool): If cache file is compressed with gzip (read-only).

    Notes:

    * Cache file is JSON formatted (and gzipped if compress is True).
    * Cache key is a string built from repr of arguments to cached function.
    * Return value of the function should be convertible to a JSON type.
    * Cache file is read for each hit so typically this decorator should be decorated by a memory cache decorator such
      as ``functools.lru_cache``.

    :param filename: The filename of cache file.
    :type filename: str
    :param compress: Compress cache file with gzip. (default=False)
    :type compress: bool
    :return: File cache decorator.
    :rtype: function
    """

    def file_cache_decorator(func):

        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            if not func_wrapper.enabled:
                return func(*args, **kwargs)
            # define key: "[arg1, arg2][('key1', val1), ('key2', val2)]"
            key = repr(list(args)) + (repr(sorted(kwargs.items())) if kwargs else '')
            # load cache from file
            path = Path(filename)
            try:
                with path.open('rb' if compress else 'r') as f:
                    try:
                        fcntl.flock(f, fcntl.LOCK_SH)  # get read lock
                        data = f.read()  # file might be empty
                        if compress:
                            cache = json.loads(gzip.decompress(data).decode('utf-8')) if data else {}
                        else:
                            cache = json.loads(data) if data else {}
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)  # release read lock
            except FileNotFoundError:
                cache = {}
            # get value from cache
            try:
                val = cache[key]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('file_cache: HIT. func={}, filename={}, key={}, val={!r}'.format(
                        func.__qualname__, filename, key, val))
            except KeyError:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('file_cache: MISS. func={}, filename={}, key={}'.format(
                        func.__qualname__, filename, key))
                # get value from function
                val = func(*args, **kwargs)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('file_cache: ADD_ITEM. func={}, filename={}, item={!r}: {!r},'.format(
                        func.__qualname__, filename, key, val))
                # update file cache
                path.touch(exist_ok=True)  # make sure file exists

                with path.open('rb+' if compress else 'r+') as f:
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX)  # get write lock
                        data = f.read()  # file might be empty
                        if compress:
                            cache = json.loads(gzip.decompress(data).decode('utf-8')) if data else {}
                        else:
                            cache = json.loads(data) if data else {}
                        cache[key] = val
                        f.seek(0)  # write to beginning of file
                        if compress:
                            f.write(gzip.compress(json.dumps(cache).encode('utf-8')))
                        else:
                            json.dump(cache, f, sort_keys=True, indent=2)
                        f.truncate()  # strip extra data
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)  # release write lock
            return val

        func_wrapper.enabled = True
        func_wrapper._filename = filename
        func_wrapper._compress = compress

        return func_wrapper

    return file_cache_decorator


# def touch_open(filename, *args, **kwargs):
#     # Open the file in R/W and create if it doesn't exist. *Don't* pass O_TRUNC
#     fd = os.open(filename, os.O_RDWR | os.O_CREAT)
#     # Encapsulate the low-level file descriptor in a python file object
#     return os.fdopen(fd, *args, **kwargs)


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
