"""
This module contains base classes relevant to simulating stabilizer codes and a CLI description class decorator.
"""

import abc
import functools

import numpy as np

from qecsim import paulitools as pt
from qecsim.error import QecsimError

ATTR_CLI_DESCRIPTION = '__qecsim_cli_desc'


def cli_description(description):
    """
    CLI description class decorator.

    Notes:

    * Adds the attribute ``__qecsim_cli_desc`` to the class with the value of the given description.
    * The description is used by :mod:`qecsim.cli` to generate CLI help messages.
    * Typically it describes the model and parameters in a human-readable form; the model type (i.e. code, error model,
      decoder) is not included.
    * For examples, see :class:`qecsim.models.planar.PlanarCode`, :class:`qecsim.models.generic.BitPhaseFlipErrorModel`
      and :class:`qecsim.models.planar.PlanarMPSDecoder`.

    :param description: CLI description.
    :type description: str
    :return: CLI description class decorator.
    :rtype: function
    """

    def _decorator(cls):
        setattr(cls, ATTR_CLI_DESCRIPTION, description)
        return cls

    return _decorator


class StabilizerCode(metaclass=abc.ABCMeta):
    """
    Defines stabilizer code properties and methods.

    This class cannot be instantiated directly, see :class:`qecsim.models.basic.FiveQubitCode` for an example
    implementation.
    """

    @property
    @abc.abstractmethod
    def stabilizers(self):
        """
        Stabilizer generators as binary symplectic vector or matrix.

        Notes:

        * Each row is a stabilizer generator.
        * The set must include at least a full set of generators but it may include additional stabilizers to simplify
          the decoding of syndromes. (E.g. all plaquette / vertex stabilizers on a surface code).

        :rtype: numpy.array (1d or 2d)
        """

    @property
    @abc.abstractmethod
    def logical_xs(self):
        """
        Logical X operators as binary symplectic vector or matrix.

        Notes:

        * Each row is a logical X operator.
        * The order of logical X operators matches that of logical Z operators given by :meth:`logical_zs`, with one for
          each logical qubit.

        :rtype: numpy.array (1d or 2d)
        """

    @property
    @abc.abstractmethod
    def logical_zs(self):
        """
        Logical Z operators as binary symplectic vector or matrix.

        Notes:

        * Each row is a logical Z operator.
        * The order of logical Z operators matches that of logical X operators given by :meth:`logical_xs`, with one for
          each logical qubit.

        :rtype: numpy.array (1d or 2d)
        """

    @property
    @functools.lru_cache()
    def logicals(self):
        """
        Logical operators as binary symplectic matrix.

        Notes:

        * Each row is a logical operator.
        * All logical X operators are stacked above all logical Z operators, in the order given by :meth:`logical_xs`
          and :meth:`logical_zs`.

        :rtype: numpy.array (2d)
        """
        return np.vstack((self.logical_xs, self.logical_zs))

    @property
    @abc.abstractmethod
    def n_k_d(self):
        """
        Descriptor of the code in the format (n, k, d).

        Notes:

        * n == number of physical qubits.
        * k == number of logical qubits.
        * d == distance of the code. (Optional. None if not known).

        :rtype: 3-tuple of int
        """

    @property
    @abc.abstractmethod
    def label(self):
        """
        Label suitable for use in plots.

        :rtype: str
        """

    def validate(self):
        r"""
        Perform various sanity checks.

        Sanity checks:

        * :math:`stabilizers \odot stabilisers^T = 0`
        * :math:`stabilizers \odot logicals^T = 0`
        * :math:`logicals \odot logicals^T = \Lambda`

        See :func:`qecsim.paulitools.bsp` for definition of :math:`\odot` and :math:`\Lambda`.

        :raises QecsimError: if the stabilizers or logicals fail the sanity checks.
        """
        if not np.all(pt.bsp(self.stabilizers, self.stabilizers.T) == 0):
            raise QecsimError('Stabilizers do not mutually commute.')
        if not np.all(pt.bsp(self.stabilizers, self.logicals.T) == 0):
            raise QecsimError('Stabilizers do not commute with logicals.')
        # twisted identity with shape (len(logicals), len(logicals))
        i1, i2 = np.hsplit(np.identity(len(self.logicals), dtype=int), 2)
        expected = np.hstack((i2, i1))
        if not np.array_equal(pt.bsp(self.logicals, self.logicals.T), expected):
            raise QecsimError('Logicals do not commute as expected.')


class ErrorModel(metaclass=abc.ABCMeta):
    """
    Defines error model properties and methods.

    This class cannot be instantiated directly, see :class:`qecsim.models.generic.DepolarizingErrorModel` for an example
    implementation.
    """

    def probability_distribution(self, probability):
        """
        Return the single-qubit probability distribution amongst Pauli I, X, Y and Z.

        Notes:

        * Implementing this method is **optional**. It is **not** invoked by any core modules. By default, it raises
          :class:`NotImplementedError`.
        * Since this method is often useful for decoders, it is provided as a template and subclasses are encouraged to
          implement it when appropriate, particularly for IID error models.

        :param probability: Overall probability of an error on a single qubit.
        :type probability: float
        :return: Tuple of probability distribution in the format (Pr(I), Pr(X), Pr(Y), Pr(Z)).
        :rtype: 4-tuple of float
        :raises NotImplementedError: Unless implemented in a subclass.
        """
        raise NotImplementedError("Attempt to invoke non-implemented optional method: {}.probability_distribution"
                                  .format(type(self).__qualname__))

    @abc.abstractmethod
    def generate(self, code, probability, rng=None):
        """
        Generate new error.

        :param code: Stabilizer code.
        :type code: StabilizerCode
        :param probability: Overall probability of an error on a single qubit.
        :type probability: float
        :param rng: Random number generator. (default=None resolves to numpy.random.default_rng())
        :type rng: numpy.random.Generator
        :return: New error as binary symplectic vector.
        :rtype: numpy.array (1d)
        """

    @property
    @abc.abstractmethod
    def label(self):
        """
        Label suitable for use in plots.

        :rtype: str
        """


class Decoder(metaclass=abc.ABCMeta):
    """
    Defines decoder properties and methods.

    This class cannot be instantiated directly, see :class:`qecsim.models.generic.NaiveDecoder` for an example
    implementation.
    """

    @abc.abstractmethod
    def decode(self, code, syndrome, **kwargs):
        """
        Resolve recovery operation for given syndrome.

        Assumptions:

        * The syndrome has length equal to the number of stabilizers.
        * A syndrome element value of 0 or 1 indicates that the corresponding stabilizer commutes or does not commute
          with the error, respectively.

        Notes:

        * The keyword parameters ``kwargs`` may be provided by the client with context values such as error_model,
          error_probability and error, see :func:`qecsim.app.run_once`. Most implementations will ignore such
          parameters; however, if they are used, implementations should declare them explicitly and treat them as
          optional.
        * This method typically returns a recovery operation but it may, alternatively, return :class:`DecodeResult`
          to indicate success/failure more explicitly.

        :param code: Stabilizer code.
        :type code: StabilizerCode
        :param syndrome: Syndrome as binary vector.
        :type syndrome: numpy.array (1d)
        :param kwargs: Optional context parameters passed by a client.
        :type kwargs: dict
        :return: Recovery operation as binary symplectic vector, or decode result indicating recovery success.
        :rtype: numpy.array (1d) or DecodeResult
        """

    @property
    @abc.abstractmethod
    def label(self):
        """
        Label suitable for use in plots.

        :rtype: str
        """


class DecoderFTP(metaclass=abc.ABCMeta):
    """
    Defines (fault-tolerant time-periodic) decoder properties and methods.

    This class cannot be instantiated directly, see :class:`qecsim.models.rotatedtoric.RotatedToricSMWPMDecoder` for an
    example implementation.
    """

    @abc.abstractmethod
    def decode_ftp(self, code, time_steps, syndrome, **kwargs):
        """
        Resolve recovery operation for given (fault-tolerant time-periodic) syndrome.

        Assumptions:

        * The syndrome has shape (number of time steps, number of stabilizers).
        * In the absence of a measurement error, a syndrome element value of 0 or 1 indicates that the corresponding
          stabilizer commutes or does not commute with the error, respectively.
        * The presence of a measurement error inverts the value of the corresponding syndrome element.

        Notes:

        * The keyword parameters ``kwargs`` may be provided by the client with context values such as error_model,
          error_probability, error, step_errors, measurement_error_probability and step_measurement_errors, see
          :func:`qecsim.app.run_once_ftp`. Most implementations will ignore such parameters; however, if they are used,
          implementations should declare them explicitly and treat them as optional.
        * This method typically returns a recovery operation but it may, alternatively, return :class:`DecodeResult`
          to indicate success/failure more explicitly.

        :param code: Stabilizer code.
        :type code: StabilizerCode
        :param time_steps: Number of time steps.
        :type time_steps: int
        :param syndrome: Syndrome as binary array.
        :type syndrome: numpy.array (2d)
        :param kwargs: Optional context parameters passed by a client.
        :type kwargs: dict
        :return: Recovery operation as binary symplectic vector, or decode result indicating recovery success.
        :rtype: numpy.array (1d) or DecodeResult
        """

    @property
    @abc.abstractmethod
    def label(self):
        """
        Label suitable for use in plots.

        :rtype: str
        """


class DecodeResult:
    """Represents the result of decoding.

    Typically decoders return a recovery operation and delegate the evaluation
    of success and logical commutations to :mod:`qecsim.app`. Optionally,
    decoders may return an instance of this class to partially or completely
    override the evaluation of success and logical commutations. Additionally,
    decoders can provide custom values to be summed across runs.

    Notes:

    * ``success`` and/or ``logical_commutations``, if not None, are used by
      :mod:`qecsim.app` to override the usual evaluation of success and logical
      commutations.
    * ``recovery``, if not None, is used by :mod:`qecsim.app` to evaluate any
      values unspecified by ``success`` and/or ``logical_commutations``.
    * ``success`` and ``recovery`` must not both be None; this ensures that
      :mod:`qecsim.app` can resolve a success value.
    * Logical commutations, as resolved by :mod:`qecsim.app`, and custom values
      must be consistent across identically parameterized simulation runs, i.e.
      always None or always equal length arrays; this ensures that
      :mod:`qecsim.app` can sum results across multiple runs.

    See also :class:`Decoder` and :class:`DecoderFTP`.

    """

    def __init__(self, success=None, logical_commutations=None, recovery=None, custom_values=None):
        """
        Initialise new decode result.

        :param success: If the decoding was successful (default=None).
        :type success: bool
        :param logical_commutations: Logical commutations as binary vector or None (default=None).
        :type logical_commutations: numpy.array (1d)
        :param recovery: Recovery operation as binary symplectic vector (default=None).
        :type recovery: numpy.array (1d)
        :param custom_values: Custom values as numeric vector or None (default=None).
        :type custom_values: numpy.array (1d)
        :raises QecsimError: If both success and recovery are unspecified (i.e. None).
        """
        if success is None and recovery is None:
            raise QecsimError('At least one of success or recovery must be specified.')
        self.success = success
        self.logical_commutations = logical_commutations
        self.recovery = recovery
        self.custom_values = custom_values

    def __repr__(self):
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            type(self).__name__, self.success, self.logical_commutations, self.recovery, self.custom_values)
