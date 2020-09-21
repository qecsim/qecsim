import numpy as np


class Color666Pauli:
    """
    Defines a Pauli operator on a color 6.6.6 lattice.

    Notes:

    * This is a utility class used by color 6.6.6 implementations of the core models.
    * It is typically instantiated using :meth:`qecsim.models.color.Color666Code.new_pauli`

    Use cases:

    * Construct a planar Pauli operator by applying site, plaquette, path and logical operators: :meth:`site`,
      :meth:`plaquette`, :meth:`logical_x`, :meth:`logical_z`.
    * Get the single Pauli operator applied to a given site: :meth:`operator`
    * Convert to binary symplectic form: :meth:`to_bsf`.
    * Copy a planar Pauli operator: :meth:`copy`.
    """

    def __init__(self, code, bsf=None):
        """
        Initialise new color 6.6.6 Pauli.

        Notes:

        * For performance reasons, the new Pauli is a view of the given bsf. Modifying one will modify the other.

        :param code: The color 6.6.6 code.
        :type code: Color666Code
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
        return int(
            # number of sites in rows 0 to r - 1 for all columns
            ((((2 * r + 1) ** 2) / 3 + 1) // 4)
            # number of sites in row r for columns 0 to c - 1
            + ((2 * c + (2 - r % 3)) // 3)
        )

    @property
    def code(self):
        """
        The color 6.6.6 code.

        :rtype: Color666Code
        """
        return self._code

    def copy(self):
        """
        Returns a copy of this Pauli that references the same code but is backed by a copy of the bsf.

        :return: A copy of this Pauli.
        :rtype: Color666Pauli
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
        :rtype: Color666Pauli
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

    def plaquette(self, operator, index):
        """
        Apply a plaquette operator at the given index.

        Notes:

        * Index is in the format (row, column).
        * Parts of plaquettes that lie outside the lattice have no effect on the lattice.

        :param operator: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :type operator: str
        :param index: Index identifying the plaquette in the format (row, column).
        :type index: 2-tuple of int
        :return: self (to allow chaining)
        :rtype: Color666Pauli
        :raises IndexError: If index is not a plaquette index.
        """
        r, c = index
        # check valid index
        if not self.code.is_plaquette(index):
            raise IndexError('{} is not a plaquette index.'.format(index))
        # flip plaquette sites
        self.site(operator, (r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c), (r + 1, c + 1))
        return self

    def logical_x(self):
        """
        Apply a logical X operator, i.e. column of X along leftmost sites.

        Notes:

        * The column of X is applied to the leftmost column to allow optimisation of the MPS decoder.

        :return: self (to allow chaining)
        :rtype: Color666Pauli
        """
        for row in range(self.code.bound + 1):
            index = row, 0
            if self.code.is_site(index):
                self.site('X', index)
        return self

    def logical_z(self):
        """
        Apply a logical Z operator, i.e. column of Z along leftmost sites.

        Notes:

        * The column of Z is applied to the leftmost column to allow optimisation of the MPS decoder.

        :return: self (to allow chaining)
        :rtype: Color666Pauli
        """
        for row in range(self.code.bound + 1):
            index = row, 0
            if self.code.is_site(index):
                self.site('Z', index)
        return self

    def __eq__(self, other):
        if type(other) is type(self):
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
