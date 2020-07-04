from qecsim import util


def test_init_logging_fall_through(monkeypatch):
    # logging fall-through
    monkeypatch.setenv('QECSIM_LOGGING_CFG', 'blah.ini')
    util.init_logging()  # no error raised


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
