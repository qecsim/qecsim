import functools

import numpy as np
from qecsim.error import QecsimException
from qecsim.model import cli_description
from qecsim.models.generic import SimpleErrorModel


@cli_description('Slice (lim 3-tuple of FLOAT, pos FLOAT)')
class CenterSliceErrorModel(SimpleErrorModel):
    """
    Implements a center-slice error model.

    In addition to the members defined in :class:`qecsim.model.ErrorModel`, it provides the following properties:

    * Get limit (normalized): :meth:`lim`.
    * Get position: :meth:`pos`.
    * Get ratio: :meth:`ratio`.
    * Get negative limit: :meth:`neg_lim`.

    The (normalized) limit defines a point on the boundary of the triangle with vertices (1, 0, 0),
    (0, 1, 0), and (0, 0, 1) in Euclidean x, y, z coordinates. The center point is defined to be at (1/3, 1/3, 1/3).
    The negative limit defines the point at the other intersection of the boundary of the triangle and the line through
    the limit and the center. Positive position defines a point on a linear scale along this line from the center, at
    position 0, to the limit, at position 1. Similarly, negative position defines a point on the line from the center
    to the negative limit, at position -1. The ratio, r_x:r_y:r_z, corresponds to the coordinates (r_x, r_y, r_z) of the
    point defined by position. With such a definition, ratio is always normalized to sum to 1.

    The probability distribution for a given error probability p is:

    * 1 - p: I (i.e. no error)
    * r_x * p: X
    * r_y * p: Y
    * r_z * p: Z

    Notes:

    * pos = 0 corresponds to the standard depolarizing error model.
    * lim = (1, 0, 0) and pos = 1 corresponds to pure bit-flip noise.
    * lim = (0, 1, 0) and pos = 1 corresponds to pure bit-phase-flip noise.
    * lim = (0, 0, 1) and pos = 1 corresponds to pure phase-flip noise.
    """

    def __init__(self, lim, pos):
        """
        Initialise new center-slice error model.

        :param lim: Limit (possibly unnormalized).
        :type lim: 3-tuple of float
        :param pos: Position.
        :type pos: float
        :raises ValueError: if lim is not of length 3 with 1 or 2 zeros.
        :raises ValueError: if stp is not -1.0 <= pos <= 1.0.
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI
            if not (len(lim) == 3 and np.count_nonzero(lim) in (1, 2)):
                raise ValueError("CenterSliceErrorModel valid lim values are 3-tuples of number with 1 or 2 zeros.")
            if not -1.0 <= pos <= 1.0:
                raise ValueError("CenterSliceErrorModel valid pos values -1.0 <= number <= 1.0.")
        except TypeError as ex:
            raise TypeError('CenterSliceErrorModel invalid parameter type') from ex
        self._lim = self._normalize(np.array(lim))
        self._pos = pos

    @staticmethod
    def _normalize(r):
        """params: lim:np.array(3d). return:lim:np.array(3d)."""
        return r / np.linalg.norm(r, ord=1)

    @staticmethod
    def _ratio(lim, pos):
        """params: lim:np.array(3d), pos:float. return:lim:np.array(3d)."""
        center = np.array([1 / 3, 1 / 3, 1 / 3])
        # if negative position, use negative limit with absolute position
        lim = lim if pos >= 0 else CenterSliceErrorModel._neg_lim(lim)
        pos = np.abs(pos)
        return pos * lim + (1 - pos) * center

    @staticmethod
    def _neg_lim(lim):
        """params: lim:np.array(3d). return:lim:np.array(3d)."""
        pX, pY, pZ, pO, pC = map(np.array, ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0), (1 / 3, 1 / 3, 1 / 3)))
        pL = lim
        # find opposing limit through center point
        for p1, p2, p3 in ((pX, pY, pZ), (pY, pZ, pX), (pZ, pX, pY)):
            if not np.dot(pL, p1):  # lim lies in 23-plane
                # if lim closer to p2 than p3
                if np.linalg.norm(p2 - pL) <= np.linalg.norm(p3 - pL):
                    pN = p2  # opposing limit at intersect with 31-plane
                else:
                    pN = p3  # opposing limit at intersect with 12-plane
                # return slice and plane interest
                csem = CenterSliceErrorModel
                return csem._normalize(csem._line_plane_intersect(pN, pO, pC - pL, pL))
        raise QecsimException('CenterSliceErrorModel: failed to find negative-limit.')

    @staticmethod
    def _line_plane_intersect(plane_normal, plane_point, line_direction, line_point):
        """params: *:np.array(3d). return:np.array(3d)."""
        # see https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python
        w = line_point - plane_point
        si = -plane_normal.dot(w) / plane_normal.dot(line_direction)
        return w + si * line_direction + plane_point

    @property
    def lim(self):
        """
        Limit (normalized).

        :rtype: 3-tuple of float
        """
        return tuple(self._lim)

    @property
    def pos(self):
        """
        Position.

        :rtype: float
        """
        return self._pos

    @property
    @functools.lru_cache()
    def ratio(self):
        """
        Ratio.

        :rtype: 3-tuple of float
        """
        return tuple(self._ratio(self._lim, self._pos))

    @property
    @functools.lru_cache()
    def neg_lim(self):
        """
        Negative limit.

        :rtype: 3-tuple of float
        """
        return tuple(self._neg_lim(self._lim))

    @functools.lru_cache()
    def probability_distribution(self, probability):
        """See :meth:`qecsim.model.ErrorModel.probability_distribution`"""
        p_x, p_y, p_z = np.array(self.ratio) * probability
        p_i = 1 - sum((p_x, p_y, p_z))
        return p_i, p_x, p_y, p_z

    @property
    def label(self):
        """See :meth:`qecsim.model.ErrorModel.label`"""
        return 'Center-slice (lim={!r}, pos={!r})'.format(self.lim, self.pos)

    def __repr__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, self.lim, self.pos)
