import abc
import functools

import numpy as np

from qecsim import paulitools as pt
from qecsim.model import ErrorModel, cli_description


class SimpleErrorModel(ErrorModel):
    """
    Implements a simple error model that generates an error based on the number of qubits and probability distribution.

    This class cannot be instantiated directly, see :class:`qecsim.models.generic.DepolarizingErrorModel` for an example
    implementation.
    """

    @abc.abstractmethod
    def probability_distribution(self, probability):
        """See :meth:`qecsim.model.ErrorModel.probability_distribution`"""

    def generate(self, code, probability, rng=None):
        """
        See :meth:`qecsim.model.ErrorModel.generate`

        Notes:

        * This method delegates to :meth:`~qecsim.model.ErrorModel.probability_distribution` to find the probability of
          I, X, Y, Z operators on each qubit.
        """
        rng = np.random.default_rng() if rng is None else rng
        n_qubits = code.n_k_d[0]
        error_pauli = ''.join(rng.choice(
            ('I', 'X', 'Y', 'Z'),
            size=n_qubits,
            p=self.probability_distribution(probability)
        ))
        return pt.pauli_to_bsf(error_pauli)

    @property
    @abc.abstractmethod
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""

    def __repr__(self):
        return '{}()'.format(type(self).__name__)


@cli_description('Pr I,X,Y,Z is 1-p,p/3,p/3,p/3')
class DepolarizingErrorModel(SimpleErrorModel):
    """
    Implements a depolarizing error model.

    The probability distribution for a given error probability p is:

    * (1 - p): I (i.e. no error)
    * p/3: X
    * p/3: Y
    * p/3: Z
    """

    @functools.lru_cache()
    def probability_distribution(self, probability):
        """See :meth:`qecsim.model.ErrorModel.probability_distribution`"""
        p_x = p_y = p_z = probability / 3
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z

    @property
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""
        return 'Depolarizing'


@cli_description('Pr I,X,Y,Z is 1-p,p,0,0')
class BitFlipErrorModel(SimpleErrorModel):
    """
    Implements a bit-flip error model.

    The probability distribution for a given error probability p is:

    * (1 - p): I (i.e. no error)
    * p: X
    * 0: Y
    * 0: Z
    """

    @functools.lru_cache()
    def probability_distribution(self, probability):
        """See :meth:`qecsim.model.ErrorModel.probability_distribution`"""
        p_x = probability
        p_y = p_z = 0
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z

    @property
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""
        return 'Bit-flip'


@cli_description('Pr I,X,Y,Z is 1-p,0,0,p')
class PhaseFlipErrorModel(SimpleErrorModel):
    """
    Implements a phase-flip error model.

    The probability distribution for a given error probability p is:

    * (1 - p): I (i.e. no error)
    * 0: X
    * 0: Y
    * p: Z
    """

    @functools.lru_cache()
    def probability_distribution(self, probability):
        """See :meth:`qecsim.model.ErrorModel.probability_distribution`"""
        p_x = p_y = 0
        p_z = probability
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z

    @property
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""
        return 'Phase-flip'


@cli_description('Pr I,X,Y,Z is 1-p,0,p,0')
class BitPhaseFlipErrorModel(SimpleErrorModel):
    """
    Implements a bit-phase-flip error model.

    The probability distribution for a given error probability p is:

    * (1 - p): I (i.e. no error)
    * 0: X
    * p: Y
    * 0: Z
    """

    @functools.lru_cache()
    def probability_distribution(self, probability):
        """See :meth:`qecsim.model.ErrorModel.probability_distribution`"""
        p_x = 0
        p_y = probability
        p_z = 0
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z

    @property
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""
        return 'Bit-phase-flip'
