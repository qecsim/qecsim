import functools
import json
import operator
import re

from qecsim import paulitools as pt
from qecsim.model import ErrorModel, cli_description

_RE_ATTR_NAME = re.compile(r"^[a-zA-Z]\w*$")
_RE_COMMENT_BLANK_LINE = re.compile(r"^\s*(//.*)?$")


class _JSONLines:
    """Utility class to read objects from JSON lines file with push/pull buffer."""

    def __init__(self, filename):
        """Initialise JSON lines
        :raises OSError: if filename cannot be opened for reading.
        """
        self._buffer = []  # buffer for pushed objects
        self._file_stream = open(filename)

    def __del__(self):
        if hasattr(self, '_file_stream'):
            self._file_stream.close()

    def pull(self):
        """Pull object from buffer or JSON lines file, skipping blank and comment lines.
        :raises ValueError: if JSON cannot be decoded.
        :raises EOFError: if no object found before end of file.
        """
        try:  # pop from buffer
            return self._buffer.pop()
        except IndexError:  # read from file
            for line in self._file_stream:
                if _RE_COMMENT_BLANK_LINE.match(line):  # skip comment or blank line
                    continue
                return json.loads(line)
            raise EOFError('End of JSON lines file encountered unexpectedly.')

    def push(self, obj):
        """Push object to buffer."""
        self._buffer.append(obj)


@cli_description('File (filename STR, [start] INT >= 0)')
class FileErrorModel(ErrorModel):
    """
    Implements a file error model.

    In addition to the members defined in :class:`qecsim.model.ErrorModel`, it provides properties given by additional
    headers in the file.

    The expected file format is new-line delimited JSON + JS comments:
    ::

        // Comment lines start with //; comment lines are ignored.
        // Blank lines are also ignored.
        // Non-comment lines are JSON-encoded per line.
        //
        // Header lines are JSON objects containing one or more keys.
        // A dictionary is built from header lines; if keys are repeated across header lines then an exception is raised.
        // The following header key is required; it gives the marginal error probability per qubit.
        {"probability": 0.4}
        // The following header key is required; it identifies the algorithm/parameters used to generate the errors.
        {"label": "Biased (bias=10)"}
        // The following header key is desirable; it gives the (I, X, Y, Z) marginal probability distribution per qubit.
        {"probability_distribution": [0.6, 0.018181818181818184, 0.36363636363636365, 0.018181818181818184]}
        // Additional header keys are optional; they are added as properties to the error model.
        {"bias": 10}
        // All header lines must appear before body lines.
        //
        // Body lines are JSON lists containing a single error per line.
        // Each error unpacked using qecsim.paulitools.unpack into a NumPy array in binary symplectic form.
        // (The following errors are suitable for the qecsim.models.linear.FiveQubitCode)
        ["f380", 10]
        ["2940", 10]
        ["ce00", 10]
        ["7bc0", 10]
        ["8400", 10]
        ["5280", 10]
        ["1080", 10]
        ["d680", 10]
        ["4a40", 10]
        ["4a40", 10]

    Notes:

    * Header key "probability" is used to validate the error probability assumed in simulations matches that of the
      generated errors.
    * Header key "label" is used to identify simulations that can be merged for statistical purposes.
    * Header key "probability_distribution" is used to make that information available to decoders that make use of it.
    * Extra header keys are made available as object attributes to decoders that make use of them. The key format must
      be a valid Python attribute name, not start with an underscore, and not shadow an existing attribute.
    """

    def __init__(self, filename, start=0):
        """
        Initialise new file error model.

        :param filename: Name of file containing generated errors.
        :type filename: str or path-like
        :param start: Index of first error to serve. (default=0)
        :type start: int
        :raises ValueError: if start is not >=0.
        :raises OSError: if filename cannot be opened for reading.
        :raises ValueError: if file parsing fails.
        :raises EOFError: if start error unavailable.
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI. (operator.index ensures the parameter can be treated as an int)
            # open file for reading as JSON lines object
            self._json_lines = _JSONLines(filename)  # can raise TypeError / OSError / FileNotFoundError
            if not operator.index(start) >= 0:
                raise ValueError("FileErrorModel valid start values are integer >= 0")
        except TypeError as ex:
            raise TypeError('FileErrorModel invalid parameter type') from ex
        # init parameters
        self._filename = filename
        self._start = start
        # load header
        header = {}
        while True:  # pull objects until first error (i.e. non-dict object)
            obj = self._json_lines.pull()
            if not isinstance(obj, dict):
                self._json_lines.push(obj)
                break
            if obj.keys() & header.keys():  # test new keys not already in header
                raise ValueError('Error file has repeated header keys.')
            header.update(obj)  # update header keys
        # extract attributes
        try:
            self._probability = float(header.pop('probability'))
            self._label = str(header.pop('label'))
            self._probability_distribution = header.pop('probability_distribution', None)  # default None
        except KeyError as e:
            raise ValueError("Error file is missing required header key: {}.".format(e))
        # TODO: maybe validate probability_distribution?
        # skip to start error index
        for _ in range(self._start):
            self._json_lines.pull()
        # make additional header values available as object attributes
        for k, v in header.items():
            if _RE_ATTR_NAME.match(k) and not hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError("Error file extra header key has invalid format: {}.".format(k))

    @functools.lru_cache()
    def probability_distribution(self, probability):
        """
        Return the single-qubit probability distribution amongst Pauli I, X, Y and Z.

        :param probability: Overall probability of an error on a single qubit.
        :type probability: float
        :return: Tuple of probability distribution in the format (Pr(I), Pr(X), Pr(Y), Pr(Z)).
        :rtype: 4-tuple of float
        :raises ValueError: if probability does not equal probability given in file header.
        :raises ValueError: if probability_distribution not given in file header.
        """
        if probability != self._probability:
            raise ValueError("Probability does not match probability given in file header")
        if self._probability_distribution:
            return tuple(self._probability_distribution)
        raise ValueError("Probability distribution not given in file header")

    def generate(self, code, probability, rng=None):
        """
        Return next error from file.

        :param code: Stabilizer code.
        :type code: StabilizerCode
        :param probability: Overall probability of an error on a single qubit.
        :type probability: float
        :param rng: Random number generator. (default=None, ignored)
        :type rng: numpy.random.Generator
        :return: Next error as binary symplectic vector.
        :rtype: numpy.array (1d)
        :raises ValueError: if probability does not equal probability given in file header.
        :raises EOFError: if next error unavailable.
        :raises ValueError: if file parsing fails.
        :raises ValueError: if length of error is inconsistent with number of qubits in code.
        """
        if probability != self._probability:
            raise ValueError("Probability does not match probability given in file header")
        # pull error, unpack and validate
        error = pt.unpack(self._json_lines.pull())
        if len(error) != 2 * code.n_k_d[0]:
            raise ValueError("Length of error inconsistent with number of qubits in code")
        return error

    @property
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""
        return self._label

    def __repr__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, self._filename, self._start)
