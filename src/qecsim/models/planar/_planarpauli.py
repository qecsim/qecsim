import numpy as np


class PlanarPauli:
    """
    Defines a Pauli operator on a planar lattice.

    Notes:

    * This is a utility class used by planar implementations of the core models.
    * It is typically instantiated using :meth:`qecsim.models.planar.PlanarCode.new_pauli`

    Use cases:

    * Construct a planar Pauli operator by applying site, plaquette, path and logical operators: :meth:`site`,
      :meth:`plaquette`, :meth:`path`, :meth:`logical_x`, :meth:`logical_z`.
    * Get the single Pauli operator applied to a given site: :meth:`operator`
    * Convert to binary symplectic form: :meth:`to_bsf`.
    * Copy a planar Pauli operator: :meth:`copy`.
    """

    def __init__(self, code, bsf=None):
        """
        Initialise new planar Pauli.

        Notes:

        * For performance reasons, the new Pauli is a view of the given bsf. Modifying one will modify the other.

        :param code: The planar code.
        :type code: PlanarCode
        :param bsf: Binary symplectic representation of Pauli. (Optional. Defaults to identity.)
        :type bsf: numpy.array (1d)
        """
        self._code = code
        self._from_bsf(bsf)

    def _from_bsf(self, bsf):
        # initialise lattices for X and Z operators from bsf
        n_qubits = self.code.n_k_d[0]
        if bsf is None:
            # initialise identity lattices for X and Z operators
            self._xs = np.zeros(n_qubits, dtype=int)
            self._zs = np.zeros(n_qubits, dtype=int)
        else:
            assert len(bsf) == 2 * n_qubits, 'BSF {} has incompatible length'.format(bsf)
            assert np.array_equal(bsf % 2, bsf), 'BSF {} is not in binary form'.format(bsf)
            # initialise lattices for X and Z operators from bsf
            self._xs, self._zs = np.hsplit(bsf, 2)  # split out Xs and Zs

    def _flatten_site_index(self, index):
        """Return 1-d index from 2-d index for internal storage."""
        r, c = index
        assert self.code.is_site(index), 'Invalid site index: {}.'.format(index)
        assert self.code.is_in_bounds(index), 'Out of bounds index: {}.'.format(index)
        rows, cols = self.code.size
        # row_in_lattice * lattice_cols + col_in_lattice + lattice_offset
        return (r // 2) * (cols - c % 2) + (c // 2) + (r % 2 * rows * cols)

    @property
    def code(self):
        """
        The planar code.

        :rtype: PlanarCode
        """
        return self._code

    def copy(self):
        """
        Returns a copy of this Pauli that references the same code but is backed by a copy of the bsf.

        :return: A copy of this Pauli.
        :rtype: PlanarPauli
        """
        return self.code.new_pauli(bsf=np.copy(self.to_bsf()))

    def operator(self, index):
        """
        Returns the operator on the site identified by the index.

        :param index: Index identifying a site in the format (row, column).
        :type index: 2-tuple of int
        :return: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :rtype: str
        :raises IndexError: If index is not an in-bounds site index.
        """
        # check valid in-bounds index
        if not (self.code.is_site(index) and self.code.is_in_bounds(index)):
            raise IndexError('{} is not an in-bounds site index for code of size {}.'.format(index, self.code.size))
        # extract binary x and z
        flat_index = self._flatten_site_index(index)
        x = self._xs[flat_index]
        z = self._zs[flat_index]
        # return Pauli
        if x == 1 and z == 1:
            return 'Y'
        if x == 1:
            return 'X'
        if z == 1:
            return 'Z'
        else:
            return 'I'

    def site(self, operator, *indices):
        """
        Apply the operator to site identified by the index.

        Notes:

        * Index is in the format (row, column).
        * Operations on sites that lie outside the lattice have no effect on the lattice.

        :param operator: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :type operator: str
        :param indices: Any number of indices identifying a site in the format (row, column).
        :type indices: Any number of 2-tuple of int
        :return: self (to allow chaining)
        :rtype: PlanarPauli
        :raises IndexError: If index is not a site index.
        """
        for index in indices:
            # check valid index
            if not self.code.is_site(index):
                raise IndexError('{} is not a site index.'.format(index))
            # apply if index within lattice
            if self.code.is_in_bounds(index):
                # flip sites
                flat_index = self._flatten_site_index(index)
                if operator in ('X', 'Y'):
                    self._xs[flat_index] ^= 1
                if operator in ('Z', 'Y'):
                    self._zs[flat_index] ^= 1
        return self

    def plaquette(self, index):
        """
        Apply a plaquette operator at the given index.

        Notes:

        * Index is in the format (row, column).
        * If the primal lattice is indexed (i.e. row % 2 = 1), then Z operators are applied around the plaquette.
          (This is equivalent to a vertex operator on the dual lattice.)
        * If the dual lattice is indexed (i.e. row % 2 = 0), then X operators are applied around the plaquette.
          (This is equivalent to a vertex operator on the primal lattice.)
        * Parts of plaquettes that lie outside the lattice have no effect on the lattice.

        :param index: Index identifying the plaquette in the format (row, column).
        :type index: 2-tuple of int
        :return: self (to allow chaining)
        :rtype: PlanarPauli
        :raises IndexError: If index is not a plaquette index.
        """
        r, c = index
        # check valid index
        if not self.code.is_plaquette(index):
            raise IndexError('{} is not a plaquette index.'.format(index))
        # apply Zs if primal lattice, or Xs otherwise
        operator = 'Z' if self.code.is_primal(index) else 'X'
        # flip plaquette sites
        self.site(operator, (r - 1, c))  # North
        self.site(operator, (r + 1, c))  # South
        self.site(operator, (r, c - 1))  # West
        self.site(operator, (r, c + 1))  # East
        return self

    def path(self, a_index, b_index):
        """
        Apply the shortest taxi-cab path of operators between the plaquettes indexed by A and B.

        Notes:

        * Indices are in the format (row, column).
        * Both indices must index the same lattice, see :meth:`qecsim.models.planar.PlanarCode.is_primal` /
          :meth:`qecsim.models.planar.PlanarCode.is_dual`.
        * Plaquettes not indexed within the lattice are said to be virtual, see
          :meth:`qecsim.models.planar.PlanarCode.bounds`.
        * Paths proceed in the following directions in order: North/South, West/East. Therefore if one plaquette lies
          beyond both boundaries the path will meet the boundary as dictated by the directions defined here.
        * If both plaquettes are virtual then they are considered connected by a zero length path.
        * Parts of paths that lie outside the lattice have no effect on the lattice.

        :param a_index: Index identifying a plaquette in the format (row, column).
        :type a_index: 2-tuple of int
        :param b_index: Index identifying a plaquette in the format (row, column).
        :type b_index: 2-tuple of int
        :return: self (to allow chaining)
        :rtype: PlanarPauli
        :raises IndexError: If indices are not plaquette indices on the same lattice.
        """
        # steps from A to B
        row_steps, col_steps = self.code.translation(a_index, b_index)
        # apply Xs if plaquette on primal lattice, or Zs otherwise
        operator = 'X' if self.code.is_primal(a_index) else 'Z'
        # current index
        c_r, c_c = a_index
        while row_steps < 0:  # heading north
            # flip current then decrement row
            self.site(operator, (c_r - 1, c_c))
            c_r -= 2
            row_steps += 1
        while row_steps > 0:  # heading south
            # flip current then increment row
            self.site(operator, (c_r + 1, c_c))
            c_r += 2
            row_steps -= 1
        while col_steps < 0:  # heading west
            # flip current then decrement col
            self.site(operator, (c_r, c_c - 1))
            c_c -= 2
            col_steps += 1
        while col_steps > 0:  # heading east
            # flip current then increment col
            self.site(operator, (c_r, c_c + 1))
            c_c += 2
            col_steps -= 1
        return self

    def logical_x(self):
        """
        Apply a logical X operator, i.e. column of X on horizontal-edge sites of primal lattice.

        Notes:

        * The column of X is applied to the rightmost column to allow optimisation of the MPS decoder.

        :return: self (to allow chaining)
        :rtype: PlanarPauli
        """
        max_row, max_col = self.code.bounds
        self.site('X', *((row, max_col) for row in range(0, max_row + 1, 2)))
        return self

    def logical_z(self):
        """
        Apply a logical Z operator, i.e. row of Z on horizontal-edge sites of primal lattice.

        Notes:

        * The row of Z is applied to the bottom row to allow optimisation of the MPS decoder.

        :return: self (to allow chaining)
        :rtype: PlanarPauli
        """
        max_row, max_col = self.code.bounds
        self.site('Z', *((max_row, col) for col in range(0, max_col + 1, 2)))
        return self

    def __eq__(self, other):
        if isinstance(other, PlanarPauli):
            return np.array_equal(self._xs, other._xs) and np.array_equal(self._zs, other._zs)
        return NotImplemented

    def __repr__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, self.code, self.to_bsf())

    def __str__(self):
        """
        ASCII art style lattice showing primal lattice lines and Pauli operators.

        :return: Informal string representation.
        :rtype: str
        """
        return self.code.ascii_art(pauli=self)

    def to_bsf(self):
        """
        Binary symplectic representation of Pauli.

        Notes:

        * For performance reasons, the returned bsf is a view of this Pauli. Modifying one will modify the other.

        :return: Binary symplectic representation of Pauli.
        :rtype: numpy.array (1d)
        """
        return np.concatenate((self._xs, self._zs))
