import functools
import itertools
import operator

import numpy as np

from qecsim.model import StabilizerCode, cli_description
from qecsim.models.rotatedplanar import RotatedPlanarPauli


@cli_description('Rotated planar (rows INT >= 3, cols INT >= 3)')
class RotatedPlanarCode(StabilizerCode):
    r"""
    Implements a rotated planar mixed boundary code defined by its lattice size.

    In addition to the members defined in :class:`qecsim.model.StabilizerCode`, it provides several lattice methods as
    described below.

    Lattice methods:

    * Get size: :meth:`size`.
    * Get plaquette type: :meth:`is_x_plaquette`, :meth:`is_z_plaquette`, :meth:`is_virtual_plaquette`.
    * Get and test bounds: :meth:`site_bounds`, :meth:`is_in_site_bounds`, :meth:`is_in_plaquette_bounds`.
    * Resolve a syndrome to plaquettes: :meth:`syndrome_to_plaquette_indices`.
    * Construct a Pauli operator on the lattice: :meth:`new_pauli`.

    Indices:

    * Indices are in the format (x, y).
    * Qubit sites (vertices) are indexed by (x, y) coordinates with the origin at the lower left qubit.
    * Stabilizer plaquettes are indexed by (x, y) coordinates such that the lower left corner of the plaquette is on the
      qubit site at (x, y).
    * X-type stabilizer plaquette indices satisfy (x-y) % 2 == 1.
    * Z-type stabilizer plaquette indices satisfy (x-y) % 2 == 0.

    For example, qubit site indices on a 3 x 3 lattice:
    ::

             (0,2)-----(1,2)-----(2,2)
               |         |         |
               |         |         |
               |         |         |
             (0,1)-----(1,1)-----(2,1)
               |         |         |
               |         |         |
               |         |         |
             (0,0)-----(1,0)-----(2,0)

    For example, stabilizer plaquette types and indices on a 3 x 3 lattice:
    ::

                 -------
                /   Z   \
               |  (0,2)  |
               +---------+---------+-----
               |    X    |    Z    |  X  \
               |  (0,1)  |  (1,1)  |(2,1) |
               |         |         |     /
          -----+---------+---------+-----
         /  X  |    Z    |    X    |
        |(-1,0)|  (0,0)  |  (1,0)  |
         \     |         |         |
          -----+---------+---------+
                         |    Z    |
                          \ (1,-1)/
                           -------
    """

    MIN_SIZE = (3, 3)

    def __init__(self, rows, columns):
        """
        Initialise new rotated planar code.

        :param rows: Number of rows in lattice.
        :type rows: int
        :param columns: Number of columns in lattice.
        :type columns: int
        :raises ValueError: if (rows, columns) smaller than (3, 3) in either dimension.
        :raises TypeError: if any parameter is of an invalid type.
        """
        min_rows, min_cols = self.MIN_SIZE
        try:  # paranoid checking for CLI. (operator.index ensures the parameter can be treated as an int)
            if operator.index(rows) < min_rows or operator.index(columns) < min_cols:
                raise ValueError('{} minimum size is {}.'.format(type(self).__name__, self.MIN_SIZE))
        except TypeError as ex:
            raise TypeError('{} invalid parameter type'.format(type(self).__name__)) from ex
        self._size = rows, columns

    # < StabilizerCode interface methods >

    @property
    @functools.lru_cache()
    def n_k_d(self):
        """See :meth:`qecsim.model.StabilizerCode.n_k_d`"""
        # n = r*c, k = 1, d = min(r, c)
        rows, cols = self.size
        return rows * cols, 1, min(rows, cols)

    @property
    def label(self):
        """See :meth:`qecsim.model.StabilizerCode.label`"""
        return 'Rotated planar {}x{}'.format(*self.size)

    @property
    @functools.lru_cache()
    def stabilizers(self):
        """See :meth:`qecsim.model.StabilizerCode.stabilizers`"""
        return np.array([self.new_pauli().plaquette(i).to_bsf() for i in self._plaquette_indices])

    @property
    @functools.lru_cache()
    def logical_xs(self):
        """See :meth:`qecsim.model.StabilizerCode.logical_xs`"""
        return np.array([self.new_pauli().logical_x().to_bsf()])

    @property
    @functools.lru_cache()
    def logical_zs(self):
        """See :meth:`qecsim.model.StabilizerCode.logical_zs`"""
        return np.array([self.new_pauli().logical_z().to_bsf()])

    # </ StabilizerCode interface methods >

    @property
    def size(self):
        """
        Size of the lattice in format (rows, columns), e.g. (5, 5).

        :rtype: 2-tuple of int
        """
        return self._size

    @classmethod
    def is_x_plaquette(cls, index):
        """
        Return True if the plaquette index specifies an X-type plaquette, irrespective of lattice bounds.

        :param index: Index in the format (x, y).
        :type index: 2-tuple of int
        :return: If the index specifies an X-type plaquette.
        :rtype: bool
        """
        x, y = index
        return (x - y) % 2 == 1

    @classmethod
    def is_z_plaquette(cls, index):
        """
        Return True if the plaquette index specifies an Z-type plaquette, irrespective of lattice bounds.

        :param index: Index in the format (x, y).
        :type index: 2-tuple of int
        :return: If the index specifies an Z-type plaquette.
        :rtype: bool
        """
        return not cls.is_x_plaquette(index)

    @property
    def site_bounds(self):
        """
        Maximum x and y value that an index coordinate can take.

        :rtype: 2-tuple of int
        """
        # max_row, max_col
        rows, cols = self.size
        return cols - 1, rows - 1  # max_x, max_y

    def is_in_site_bounds(self, index):
        """
        Return True if the site index is within lattice bounds inclusive.

        :param index: Index in the format (x, y).
        :type index: 2-tuple of int
        :return: If the index is within lattice bounds inclusive.
        :rtype: bool
        """
        x, y = index
        max_site_x, max_site_y = self.site_bounds
        return 0 <= x <= max_site_x and 0 <= y <= max_site_y

    @functools.lru_cache(maxsize=2 ** 14)  # O(n) per code, so for 101x101 code
    def is_in_plaquette_bounds(self, index):
        """
        Return True if the plaquette index is within lattice bounds inclusive.

        :param index: Index in the format (x, y).
        :type index: 2-tuple of int
        :return: If the index is within lattice bounds inclusive.
        :rtype: bool
        """
        x, y = index
        max_site_x, max_site_y = self.site_bounds
        # derive min and max x bounds allowing for boundary plaquettes
        min_x = -1 if y % 2 == 0 else 0
        if max_site_x % 2 == 0:  # even max_site_x (i.e. odd number of columns)
            max_x = max_site_x - 1 if y % 2 == 0 else max_site_x
        else:
            max_x = max_site_x if y % 2 == 0 else max_site_x - 1
        # derive min and max y bounds allowing for boundary plaquettes
        min_y = 0 if x % 2 == 0 else -1
        if max_site_y % 2 == 0:  # even max_site_y (i.e. odd number of rows)
            max_y = max_site_y if x % 2 == 0 else max_site_y - 1
        else:  # odd max_site_y (i.e. even number of rows)
            max_y = max_site_y - 1 if x % 2 == 0 else max_site_y
        # evaluate in bounds
        return min_x <= x <= max_x and min_y <= y <= max_y

    def is_virtual_plaquette(self, index):
        """
        Return True if the plaquette index specifies a virtual plaquette
        (i.e. index is on the boundary but not within lattice bounds).

        :param index: Index in the format (x, y).
        :type index: 2-tuple of int
        :return: If the index specifies a virtual plaquette.
        :rtype: bool
        """
        x, y = index
        max_site_x, max_site_y = self.site_bounds
        # index is on boundary but not within lattice bounds.
        return (x == -1 or x == max_site_x or y == -1 or y == max_site_y) and not self.is_in_plaquette_bounds(index)

    @property
    @functools.lru_cache()
    def _plaquette_indices(self):
        """
        Return a list of the plaquette indices of the lattice.

        Notes:

        * Each index is in the format (x, y).
        * Indices are in order of increasing type, y, x. (Z-type first)

        :return: List of indices in the format (x, y).
        :rtype: list of 2-tuple of int
        """
        max_site_x, max_site_y = self.site_bounds
        z_plaquette_indices, x_plaquette_indices = [], []
        for y in range(-1, max_site_y + 2):
            for x in range(-1, max_site_x + 2):
                index = x, y
                if self.is_in_plaquette_bounds(index):
                    if self.is_z_plaquette(index):
                        z_plaquette_indices.append(index)
                    else:
                        x_plaquette_indices.append(index)
        return list(itertools.chain(z_plaquette_indices, x_plaquette_indices))

    def syndrome_to_plaquette_indices(self, syndrome):
        """
        Returns the indices of the plaquettes associated with the non-commuting stabilizers identified by the syndrome.

        :param syndrome: Binary vector identifying commuting and non-commuting stabilizers by 0 and 1 respectively.
        :type syndrome: numpy.array (1d)
        :return: List of plaquette indices.
        :rtype: list of 2-tuple of int
        """
        return set(tuple(index) for index in np.array(self._plaquette_indices)[syndrome.nonzero()])

    def __eq__(self, other):
        if type(other) is type(self):
            return self._size == other._size
        return NotImplemented

    def __hash__(self):
        return hash(self._size)

    def __repr__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, *self.size)

    def ascii_art(self, syndrome=None, pauli=None, plaquette_labels=None, site_labels=None):
        """
        Return ASCII art style lattice showing primal lattice lines with syndrome bits and Pauli operators as given.

        Notes:

        * Optional plaquette_labels override syndrome.
        * Optional site_labels override pauli.

        :param syndrome: Syndrome (optional) as binary vector.
        :type syndrome: numpy.array (1d)
        :param pauli: Rotated planar Pauli (optional)
        :type pauli: RotatedPlanarPauli
        :param plaquette_labels: Dictionary of plaquette indices as (x, y) to single-character labels (optional).
        :type plaquette_labels: dict of (int, int) to char
        :param site_labels: Dictionary of site indices as (x, y) to single-character labels (optional).
        :type site_labels: dict of (int, int) to char
        :return: ASCII art style lattice.
        :rtype: str
        """
        # See https://unicode-table.com/en/blocks/box-drawing/ for box-drawing unicode characters
        max_site_x, max_site_y = self.site_bounds
        syndrome_indices = set() if syndrome is None else self.syndrome_to_plaquette_indices(syndrome)
        pauli = self.new_pauli() if pauli is None else pauli
        plaquette_labels = {} if plaquette_labels is None else plaquette_labels
        site_labels = {} if site_labels is None else site_labels

        # Build row templates
        # e.g. (where @=plaquette, o=site, .=virtual_plaquette):
        #
        # . /-@-\ . /-@-\ .   .     :plaquette_row_top_even
        #   o---o---o---o---o-\     :site_row_top_even
        # . |#@#| @ |#@#| @ |#@     :plaquette_row_odd
        # /-o---o---o---o---o-/     :site_row_odd
        # @#| @ |#@#| @ |#@#| .     :plaquette_row_even
        # \-o---o---o---o---o-\     :t_site_row_even
        # . |#@#| @ |#@#| @ |#@     :plaquette_row_odd
        # /-o---o---o---o---o-/     :site_row_odd
        # @#| @ |#@#| @ |#@#| .     :plaquette_row_even
        # \-o---o---o---o---o       :site_row_bottom
        # .   . \-@-/ . \-@-/ .     :plaquette_row_bottom
        #
        # e.g (if top row odd):
        #
        # .   . /-@-\ . /-@-\ .     :plaquette_row_top_odd
        # /-o---o---o---o---o       :site_row_top_odd
        #
        # Common chars
        c_dot = chr(0x00B7)
        c_dash = chr(0x2500)
        c_bar = chr(0x2502)
        c_angle_nw = chr(0x250C)
        c_angle_ne = chr(0x2510)
        c_angle_sw = chr(0x2514)
        c_angle_se = chr(0x2518)
        c_shade = chr(0x2591)
        # Common char sequences
        cs_pn = c_angle_nw + c_dash + '{}' + c_dash + c_angle_ne  # '/-{}-\'
        cs_pnw = c_angle_nw + c_dash  # '/-'
        cs_pw = '{}' + c_shade  # ' #'
        cs_psw = c_angle_sw + c_dash  # '\-'
        cs_pne = c_dash + c_angle_ne  # '-\'
        cs_pe = c_shade + '{}'  # '# '
        cs_pse = c_dash + c_angle_se  # '-/'
        cs_ps = c_angle_sw + c_dash + '{}' + c_dash + c_angle_se  # '\-{}-/'
        cs_pbulkx = c_bar + c_shade + '{}' + c_shade  # '|#{}#'
        cs_pbulkz = c_bar + ' {} '  # '| {} '
        cs_sbulk = '{}' + c_dash * 3  # '{}---'
        # booleans to control placement of boundary plaquettes
        odd_rows = max_site_y % 2 == 0
        odd_cols = max_site_x % 2 == 0
        if odd_rows:
            # . /-@-\ . /-@-\ .   .
            t_plaquette_row_top = ('{} ' + cs_pn + ' ') * ((max_site_x + 1) // 2) + ('{}   {}' if odd_cols else '{}')
            #   o---o---o---o---o-\
            t_site_row_top = '  ' + cs_sbulk * max_site_x + '{}' + (cs_pne if odd_cols else '  ')
        else:
            # .   . /-@-\ . /-@-\ .
            t_plaquette_row_top = '{}   {}' + (' ' + cs_pn + ' {}') * (max_site_x // 2) + ('' if odd_cols else '   {}')
            # /-o---o---o---o---o
            t_site_row_top = cs_pnw + cs_sbulk * max_site_x + '{}' + (cs_pne if not odd_cols else '  ')
        #   |#@#| @ |#@#| @ |#@
        t_plaquette_row_odd = ('{} ' + ''.join(([cs_pbulkx, cs_pbulkz] * max_site_x)[:max_site_x])
                               + c_bar + (cs_pe if odd_cols else ' {}'))
        # /-o---o---o---o---o-/
        t_site_row_odd = cs_pnw + cs_sbulk * max_site_x + '{}' + (cs_pse if odd_cols else cs_pne)
        # @#| @ |#@#| @ |#@#| .
        t_plaquette_row_even = (cs_pw + ''.join(([cs_pbulkz, cs_pbulkx] * max_site_x)[:max_site_x])
                                + c_bar + (cs_pe if not odd_cols else ' {}'))
        # \-o---o---o---o---o-\
        t_site_row_even = cs_psw + cs_sbulk * max_site_x + '{}' + (cs_pne if odd_cols else cs_pse)
        # \-o---o---o---o---o
        t_site_row_bottom = cs_psw + cs_sbulk * max_site_x + '{}' + (cs_pse if not odd_cols else '  ')
        # .   . \-@-/ . \-@-/ .
        t_plaquette_row_bottom = '{}   {}' + (' ' + cs_ps + ' {}') * (max_site_x // 2) + ('' if odd_cols else '   {}')

        # Parameter extraction functions
        def _site_parameters(y):
            indices = [i for i in ((x, y) for x in range(max_site_x + 1))]
            parameters = []
            for i in indices:
                if i in site_labels:
                    parameters.append(site_labels[i])
                else:
                    op = pauli.operator(i)
                    parameters.append(c_dot if op == 'I' else op)
            return parameters

        def _plaquette_parameters(y):
            indices = [i for i in ((x, y) for x in range(-1, max_site_x + 1))]
            parameters = []
            for i in indices:
                is_z_plaquette = self.is_z_plaquette(i)
                is_virtual_plaquette = self.is_virtual_plaquette(i)
                if is_virtual_plaquette:
                    parameters.append(plaquette_labels.get(i, ' '))
                elif i in plaquette_labels:
                    parameters.append(plaquette_labels[i])
                elif i in syndrome_indices:
                    parameters.append('Z' if is_z_plaquette else 'X')
                elif i[0] == -1 or i[0] == max_site_x:
                    parameters.append(c_bar)
                elif i[1] == -1 or i[1] == max_site_y:
                    parameters.append(c_dash)
                else:
                    parameters.append(' ' if is_z_plaquette else c_shade)
            return parameters

        # Append templates to text with parameters
        text = []
        # top rows
        text.append(t_plaquette_row_top.format(*_plaquette_parameters(max_site_y)))
        text.append(t_site_row_top.format(*_site_parameters(max_site_y)))
        # middle rows
        for y in range(max_site_y - 1, 0, -1):
            if y % 2 == 0:
                text.append(t_plaquette_row_even.format(*_plaquette_parameters(y)))
                text.append(t_site_row_even.format(*_site_parameters(y)))
            else:
                text.append(t_plaquette_row_odd.format(*_plaquette_parameters(y)))
                text.append(t_site_row_odd.format(*_site_parameters(y)))
        # bottom rows
        text.append(t_plaquette_row_even.format(*_plaquette_parameters(0)))
        text.append(t_site_row_bottom.format(*_site_parameters(0)))
        text.append(t_plaquette_row_bottom.format(*_plaquette_parameters(-1)))

        return '\n'.join(text)

    def new_pauli(self, bsf=None):
        """
        Convenience constructor of planar Pauli for this code.

        Notes:

        * For performance reasons, the new Pauli is a view of the given bsf. Modifying one will modify the other.

        :param bsf: Binary symplectic representation of Pauli. (Optional. Defaults to identity.)
        :type bsf: numpy.array (1d)
        :return: Rotated planar Pauli
        :rtype: RotatedPlanarPauli
        """
        return RotatedPlanarPauli(self, bsf)
