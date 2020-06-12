import functools
import math

from qecsim.model import cli_description
from qecsim.models.generic import SimpleErrorModel


@cli_description('Biased (bias FLOAT >= 0, [axis] CHAR)')
class BiasedDepolarizingErrorModel(SimpleErrorModel):
    """
    Implements a biased-depolarizing error model.

    In addition to the members defined in :class:`qecsim.model.ErrorModel`, it provides the following method:

    * Get bias: :meth:`bias`.
    * Get axis: :meth:`axis`.


    The probability distribution for a given error probability p given axis = 'Y' is:

    * 1 - p: I (i.e. no error)
    * 1 / (2 * (bias + 1)) * p: X
    * bias / (bias + 1) * p: Y
    * 1 / (2 * (bias + 1)) * p: Z


    Notes:

    * Probability distributions are defined analogously for axis = 'X' and axis = 'Z'.
    * bias = 0.5 corresponds to the standard depolarizing error model.
    """

    def __init__(self, bias, axis='Y'):
        """
        Initialise new biased-depolarizing error model.

        :param bias: Bias in favour of axis errors relative to  off-axis errors.
        :type bias: float
        :param axis: Axis towards which the noise is biased (default='Y', values='X', 'Y', 'Z')
        :type axis: str
        :raises ValueError: if bias is not > 0.
        :raises ValueError: if axis is not in ('X', 'Y', 'Z') (lowercase accepted).
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI
            if not (bias > 0 and math.isfinite(bias)):
                raise ValueError("BiasedDepolarizingErrorModel valid bias values are number > 0")
            if axis not in ('x', 'y', 'z', 'X', 'Y', 'Z'):
                raise ValueError("BiasedDepolarizingErrorModel valid axis values are ('X', 'Y', 'Z')")
        except TypeError as ex:
            raise TypeError('BiasedDepolarizingErrorModel invalid parameter type') from ex
        self._bias = bias
        self._axis = axis.upper()

    @property
    def bias(self):
        """
        Bias.

        :rtype: float
        """
        return self._bias

    @property
    def axis(self):
        """
        Axis.

        :rtype: str
        """
        return self._axis

    @functools.lru_cache()
    def probability_distribution(self, probability):
        """See :meth:`qecsim.model.ErrorModel.probability_distribution`"""
        # low and high-rate error probabilities
        p_lr = 1 / (2 * (self._bias + 1)) * probability
        p_hr = self._bias / (self._bias + 1) * probability
        # along given axis
        if self.axis == 'X':
            p_x, p_y, p_z = p_hr, p_lr, p_lr
        elif self.axis == 'Y':
            p_x, p_y, p_z = p_lr, p_hr, p_lr
        elif self.axis == 'Z':
            p_x, p_y, p_z = p_lr, p_lr, p_hr
        # with no-error sum to 1
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z

    @property
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""
        return 'Biased-depolarizing (bias={!r}, axis={!r})'.format(self._bias, self._axis)

    def __repr__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, self._bias, self._axis)


@cli_description('Biased Y v. X (bias FLOAT >= 0)')
class BiasedYXErrorModel(SimpleErrorModel):
    """
    Implements a biased-Y-X error model.

    In addition to the members defined in :class:`qecsim.model.ErrorModel`, it provides the following method:

    * Get bias: :meth:`bias`.

    The probability distribution for a given error probability p is:

    * 1 - p: I (i.e. no error)
    * rx * (1 - ry): X
    * ry * (1 - rx): Y
    * rx * ry: Z

    where rx, ry, rz are rates found by solving:

    * p = px + py + pz
    * bias = py / px

    Note: bias = 0 corresponds to the standard bit-flip error model (i.e. pure X-noise).
    """

    def __init__(self, bias):
        """
        Initialise new biased-Y-X error model.

        :param bias: Bias in favour of Y errors relative to X errors.
        :type bias: float
        :raises ValueError: if bias is not >=0.
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI
            if not (bias >= 0 and math.isfinite(bias)):
                raise ValueError("BiasedYXErrorModel valid bias values are number >= 0")
        except TypeError as ex:
            raise TypeError('BiasedYXErrorModel invalid parameter type') from ex
        self._bias = bias

    @property
    def bias(self):
        """
        Bias.

        :rtype: float
        """
        return self._bias

    @staticmethod
    def _rate_x(bias, probability):
        if bias == 0:  # zero-bias => pure X-noise
            return probability
        h, p = bias, probability
        return 1 / 2 * (1 + h + p - h * p - math.sqrt(-4 * p + (1 + h + p - h * p) ** 2))

    @staticmethod
    def _rate_y(bias, probability):
        if bias == 0:  # zero-bias => pure X-noise
            return 0
        h, p = bias, probability
        return 1 / (2 * h) * (1 + h - p + h * p - math.sqrt(-4 * p + (1 + h + p - h * p) ** 2))

    @functools.lru_cache()
    def probability_distribution(self, probability):
        """See :meth:`qecsim.model.ErrorModel.probability_distribution`"""
        # rates
        r_x = self._rate_x(self._bias, probability)
        r_y = self._rate_y(self._bias, probability)
        # probabilities
        p_x = r_x * (1 - r_y)
        p_y = r_y * (1 - r_x)
        p_z = r_x * r_y
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z

    @property
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""
        return 'Biased-Y-X (bias={!r})'.format(self._bias)

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self._bias)
