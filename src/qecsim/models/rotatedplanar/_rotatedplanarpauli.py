import numpy as np


class RotatedPlanarPauli:
    """
    Defines a Pauli operator on a rotated planar lattice.

    Notes:

    * This is a utility class used by rotated planar implementations of the core models.
    * It is typically instantiated using :meth:`qecsim.models.rotatedplanar.RotatedPlanarCode.new_pauli`

    Use cases:

    * Construct a rotated planar Pauli operator by applying site, plaquette, and logical operators: :meth:`site`,
      :meth:`plaquette`, :meth:`logical_x`, :meth:`logical_z`.
    * Get the single Pauli operator applied to a given site: :meth:`operator`
    * Convert to binary symplectic form: :meth:`to_bsf`.
    * Copy a rotated planar Pauli operator: :meth:`copy`.
    """

    def __init__(self, code, bsf=None):
        """
        Initialise new rotated planar Pauli.

        Notes:

        * For performance reasons, the new Pauli is a view of the given bsf. Modifying one will modify the other.

        :param code: The rotated planar code.
        :type code: RotatedPlanarCode
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
        x, y = index
        assert self.code.is_in_site_bounds(index), 'Out of bounds index: {}.'.format(index)
        rows, cols = self.code.size
        # x + y * lattice_cols
        return x + y * cols

    @property
    def code(self):
        """
        The rotated planar code.
        
        :rtype: RotatedPlanarCode
        """
        return self._code

    def copy(self):
        """
        Returns a copy of this Pauli that references the same code but is backed by a copy of the bsf.

        :return: A copy of this Pauli.
        :rtype: RotatedPlanarCode
        """
        return self.code.new_pauli(bsf=np.copy(self.to_bsf()))

    def operator(self, index):
        """
        Returns the operator on the site identified by the index.

        :param index: Index identifying a site in the format (x, y).
        :type index: 2-tuple of int
        :return: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :rtype: str
        :raises IndexError: If index is not in-bounds.
        """
        # check valid in-bounds index
        if not self.code.is_in_site_bounds(index):
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

        * Index is in the format (x, y).
        * Applying operators on sites that lie outside the lattice have no effect on the lattice.

        :param operator: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :type operator: str
        :param indices: Any number of indices identifying sites in the format (x, y).
        :type indices: Any number of 2-tuple of int
        :return: self (to allow chaining)
        :rtype: RotatedPlanarPauli
        """
        for index in indices:
            # apply if index within lattice
            if self.code.is_in_site_bounds(index):
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

        * Index is in the format (x, y).
        * If an Z-type plaquette is indexed (i.e. (x - y) % 2 == 0), then Z operators are applied around the plaquette.
        * If an X-type plaquette is indexed (i.e. (x - y) % 2 == 1), then X operators are applied around the plaquette.
        * Applying plaquette operators on plaquettes that lie outside the lattice have no effect on the lattice.

        :param index: Index identifying the plaquette in the format (x, y).
        :type index: 2-tuple of int
        :return: self (to allow chaining)
        :rtype: RotatedPlanarPauli
        """
        x, y = index
        # apply if index within lattice
        if self.code.is_in_plaquette_bounds(index):
            # apply Zs if Z-type plaquette, or Xs otherwise
            operator = 'Z' if self.code.is_z_plaquette(index) else 'X'
            # flip plaquette sites
            self.site(operator, (x, y))  # SW
            self.site(operator, (x, y + 1))  # NW
            self.site(operator, (x + 1, y + 1))  # NE
            self.site(operator, (x + 1, y))  # SE
        return self

    def logical_x(self):
        """
        Apply a logical X operator, i.e. row of X between X-type boundaries (i.e. left to right).

        Notes:

        * Operators are applied to the bottom row to allow optimisation of the MPS decoder.

        :return: self (to allow chaining)
        :rtype: RotatedPlanarPauli
        """
        max_site_x, max_site_y = self.code.site_bounds
        self.site('X', *((x, 0) for x in range(0, max_site_x + 1)))
        return self

    def logical_z(self):
        """
        Apply a logical Z operator, i.e. column of Z between Z-type boundaries (i.e. bottom to top).

        Notes:

        * Operators are applied to the rightmost column to allow optimisation of the MPS decoder.

        :return: self (to allow chaining)
        :rtype: RotatedPlanarPauli
        """
        max_site_x, max_site_y = self.code.site_bounds
        self.site('Z', *((max_site_x, y) for y in range(0, max_site_y + 1)))
        return self

    def __eq__(self, other):
        if isinstance(other, RotatedPlanarPauli):
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
