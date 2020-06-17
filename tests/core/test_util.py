import gzip
import json
import logging
from pathlib import Path

from click.testing import CliRunner  # use for isolated_filesystem feature
from qecsim import util


def test_init_logging_fall_through(monkeypatch):
    # logging fall-through
    monkeypatch.setenv('QECSIM_LOGGING_CFG', 'blah.ini')
    util.init_logging()  # no error raised


def test_file_cache(caplog):
    filename = '.qecsim.test.test_config._hello.file_cache.json'

    @util.file_cache(filename)
    def _hello(name):
        return 'Hello {}'.format(name)

    with CliRunner().isolated_filesystem(), caplog.at_level(logging.DEBUG):
        assert _hello('Bob') == 'Hello Bob'
        assert 'MISS' in caplog.text, 'Bob in file_cache.'
        assert _hello('Bob') == 'Hello Bob'
        assert 'HIT' in caplog.text, 'Bob not in file_cache.'
        assert _hello('Alice') == 'Hello Alice'

        with open(filename) as f:
            cache = json.load(f)  # no error raised

        assert cache == {
            "['Alice']": 'Hello Alice',
            "['Bob']": 'Hello Bob',
        }, 'File cache not updated.'

        print(caplog.text)


def test_file_cache_compressed(caplog):
    filename = '.qecsim.test.test_config._hello.file_cache.json.gz'

    @util.file_cache(filename, compress=True)
    def _hello(name):
        return 'Hello {}'.format(name)

    with CliRunner().isolated_filesystem(), caplog.at_level(logging.DEBUG):
        assert _hello('Bob') == 'Hello Bob'
        assert 'MISS' in caplog.text, 'Bob in file_cache.'
        assert _hello('Bob') == 'Hello Bob'
        assert 'HIT' in caplog.text, 'Bob not in file_cache.'
        assert _hello('Alice') == 'Hello Alice'

        with gzip.open(filename, mode='rt') as f:
            cache = json.load(f)  # no error raised

        assert cache == {
            "['Alice']": 'Hello Alice',
            "['Bob']": 'Hello Bob',
        }, 'File cache not updated.'

        print(caplog.text)


def test_file_cache_empty_file(caplog):
    filename = '.qecsim.test.test_config._hello.file_cache.json'

    @util.file_cache(filename)
    def _hello(name):
        return 'Hello {}'.format(name)

    with CliRunner().isolated_filesystem(), caplog.at_level(logging.DEBUG):
        Path(filename).touch()

        assert _hello('Bob') == 'Hello Bob'
        assert 'MISS' in caplog.text, 'Bob in file_cache when not expected.'
        assert _hello('Bob') == 'Hello Bob'
        assert 'HIT' in caplog.text, 'Bob not in file_cache when expected.'
        assert _hello('Alice') == 'Hello Alice'

        with open(filename) as f:
            cache = json.load(f)  # no error raised

        assert cache == {
            "['Alice']": 'Hello Alice',
            "['Bob']": 'Hello Bob',
        }, 'File cache not updated.'

        print(caplog.text)


def test_file_cache_existing_file(caplog):
    filename = '.qecsim.test.test_config._hello.file_cache.json'

    @util.file_cache(filename)
    def _hello(name):
        return 'Hello {}'.format(name)

    with CliRunner().isolated_filesystem(), caplog.at_level(logging.DEBUG):
        with open(filename, 'x') as f:
            json.dump({
                "['Alice']": 'Hola Alice',
            }, f)

        assert _hello('Bob') == 'Hello Bob'
        assert 'MISS' in caplog.text, 'Bob in file_cache when not expected.'
        assert _hello('Alice') == 'Hola Alice'
        assert 'HIT' in caplog.text, 'Alice not in file_cache when expected.'

        with open(filename) as f:
            cache = json.load(f)  # no error raised

        assert cache == {
            "['Alice']": 'Hola Alice',
            "['Bob']": 'Hello Bob',
        }, 'File cache not updated.'

        print(caplog.text)


def test_file_cache_attributes(caplog):
    hello_filename, hello_compress = '._hello.file_cache.json.gz', True
    bye_filename, bye_compress = '._bye.file_cache.json', False

    @util.file_cache(hello_filename, compress=hello_compress)
    def _hello(name):
        return 'Hello {}'.format(name)

    @util.file_cache(bye_filename, compress=bye_compress)
    def _bye(name):
        return 'Bye {}'.format(name)

    with CliRunner().isolated_filesystem(), caplog.at_level(logging.DEBUG):
        assert _hello._filename == hello_filename
        assert _bye._filename == bye_filename

        assert _hello._compress == hello_compress
        assert _bye._compress == bye_compress

        assert _hello.enabled
        assert _bye.enabled

        _hello.enabled = False
        assert not _hello.enabled
        assert _bye.enabled

        assert _hello('Bob') == 'Hello Bob'
        assert not caplog.text

        assert _bye('Bob') == 'Bye Bob'
        assert 'MISS' in caplog.text, 'Bob in bye file_cache.'

        assert _bye('Bob') == 'Bye Bob'
        assert 'HIT' in caplog.text, 'Bob not in bye file_cache.'

        assert not Path(hello_filename).exists(), 'hello file_cache exists.'

        with open(bye_filename) as f:
            bye_cache = json.load(f)  # no error raised
        assert bye_cache == {
            "['Bob']": 'Bye Bob',
        }, 'File cache not updated.'

        print(caplog.text)


def test_chunker_iterator():
    # test with iterator of iterables
    my_length = 10
    my_iterator = ((i, i) for i in range(my_length))
    my_chunk_len = 3
    count = 0
    for chunk in util.chunker(my_iterator, my_chunk_len):
        chunk = list(chunk)
        print(chunk)
        assert len(chunk) <= my_chunk_len, 'Invalid chunk length'
        expected_chunk = [(i, i) for i in range(count * my_chunk_len, min((count + 1) * my_chunk_len, my_length))]
        assert chunk == expected_chunk
        count += 1


def test_chunker_iterable():
    # test with iterable of iterables
    my_length = 10
    my_iterable = [(i, i) for i in range(my_length)]
    my_chunk_len = 3
    count = 0
    for chunk in util.chunker(my_iterable, my_chunk_len):
        chunk = list(chunk)
        print(chunk)
        assert len(chunk) <= my_chunk_len, 'Invalid chunk length'
        expected_chunk = [(i, i) for i in range(count * my_chunk_len, min((count + 1) * my_chunk_len, my_length))]
        assert chunk == expected_chunk
        count += 1
