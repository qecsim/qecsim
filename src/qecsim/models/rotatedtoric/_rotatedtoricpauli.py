import numpy as np


class RotatedToricPauli:
    """
    Defines a Pauli operator on a rotated toric lattice.

    Notes:

    * This is a utility class used by rotated toric implementations of the core models.
    * It is typically instantiated using :meth:`qecsim.models.rotatedtoric.RotatedToricCode.new_pauli`

    Use cases:

    * Construct a rotated toric Pauli operator by applying site, plaquette, path and logical operators: :meth:`site`,
      :meth:`plaquette`, :meth:`path`, :meth:`logical_x1`, :meth:`logical_x2`, :meth:`logical_z1`, :meth:`logical_z2`.
    * Get the single Pauli operator applied to a given site: :meth:`operator`
    * Convert to binary symplectic form: :meth:`to_bsf`.
    * Copy a rotated toric Pauli operator: :meth:`copy`.
    """

    def __init__(self, code, bsf=None):
        """
        Initialise new rotated toric Pauli.

        Notes:

        * For performance reasons, the new Pauli is a view of the given bsf. Modifying one will modify the other.

        :param code: The rotated toric code.
        :type code: RotatedToricCode
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
        """Return 1-d index from 2-d index for internal storage. (index is expected to be in bounds.)"""
        assert self.code.is_in_bounds(index), 'Out of bounds index: {}.'.format(index)
        x, y = index
        max_x, max_y = self.code.bounds
        return x + y * (max_x + 1)

    def _mod_index(self, index):
        """Return 2-d index modulo lattice dimensions."""
        x, y = index
        max_x, max_y = self.code.bounds
        return x % (max_x + 1), y % (max_y + 1)

    @property
    def code(self):
        """
        The rotated toric code.

        :rtype: RotatedToricCode
        """
        return self._code

    def copy(self):
        """
        Returns a copy of this Pauli that references the same code but is backed by a copy of the bsf.

        :return: A copy of this Pauli.
        :rtype: RotatedToricPauli
        """
        return self.code.new_pauli(bsf=np.copy(self.to_bsf()))

    def operator(self, index):
        """
        Returns the operator on the site identified by the index.

        Notes:

        * Index is in the format (x, y).
        * Index is modulo lattice dimensions, i.e. on a (2, 2) lattice, (2, -1) indexes the same site as (0, 1).

        :param index: Index identifying a site in the format (x, y).
        :type index: 2-tuple of int
        :return: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :rtype: str
        """
        index = self._mod_index(index)
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
        * Index is modulo lattice dimensions, i.e. on a (2, 2) lattice, (2, -1) indexes the same site as (0, 1).

        :param operator: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :type operator: str
        :param indices: Any number of indices identifying sites in the format (x, y).
        :type indices: Any number of 2-tuple of int
        :return: self (to allow chaining)
        :rtype: RotatedToricPauli
        """
        for index in indices:
            index = self._mod_index(index)
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
        * Index is modulo lattice dimensions, i.e. on a (2, 2) lattice, (2, -1) indexes the same plaquette as (0, 1).
        * If an Z-type plaquette is indexed (i.e. (x - y) % 2 == 0), then Z operators are applied around the plaquette.
        * If an X-type plaquette is indexed (i.e. (x - y) % 2 == 1), then X operators are applied around the plaquette.

        :param index: Index identifying the plaquette in the format (x, y).
        :type index: 2-tuple of int
        :return: self (to allow chaining)
        :rtype: RotatedToricPauli
        """
        x, y = index
        # apply Zs if Z-type plaquette, or Xs otherwise
        operator = 'Z' if self.code.is_z_plaquette(index) else 'X'
        # flip plaquette sites
        self.site(operator, (x, y))  # SW
        self.site(operator, (x, y + 1))  # NW
        self.site(operator, (x + 1, y + 1))  # NE
        self.site(operator, (x + 1, y))  # SE
        return self

    def logical_x1(self):
        """
        Apply a logical X1 operator, i.e. column of X at sizes with x = 0.

        :return: self (to allow chaining)
        :rtype: RotatedToricPauli
        """
        max_x, max_y = self.code.bounds
        self.site('X', *((0, y) for y in range(max_y + 1)))
        return self

    def logical_x2(self):
        """
        Apply a logical X2 operator, i.e. row of X at sites with y = 0.

        :return: self (to allow chaining)
        :rtype: RotatedToricPauli
        """
        max_x, max_y = self.code.bounds
        self.site('X', *((x, 0) for x in range(max_x + 1)))
        return self

    def logical_z1(self):
        """
        Apply a logical Z1 operator, i.e. row of Z at sites with y = 0.

        :return: self (to allow chaining)
        :rtype: RotatedToricPauli
        """
        max_x, max_y = self.code.bounds
        self.site('Z', *((x, 0) for x in range(max_x + 1)))
        return self

    def logical_z2(self):
        """
        Apply a logical Z2 operator, i.e. column of Z at sizes with x = 0.

        :return: self (to allow chaining)
        :rtype: RotatedToricPauli
        """
        max_x, max_y = self.code.bounds
        self.site('Z', *((0, y) for y in range(max_y + 1)))
        return self

    def path(self, a_index, b_index):
        """
        Apply the shortest taxi-cab path of operators between the plaquettes indexed by A and B.

        Notes:

        * Indices are in the format (x, y).
        * Indices are modulo lattice dimensions, i.e. on a (2, 2) lattice, (2, -1) indexes the same site as (0, 1).
        * Both plaquettes must be of the same type, i.e. X or Z.
        * If X plaquettes are indexed then Z operators are applied.
        * If Z plaquettes are indexed then X operators are applied.

        :param a_index: Plaquette index as (x, y).
        :type a_index: (int, int)
        :param b_index: Plaquette index as (x, y).
        :type b_index: (int, int)
        :return: self (to allow chaining)
        :rtype: ToricPauli
        :raises IndexError: If plaquettes are not of the same type (i.e. X or Z).
        """

        # if start and end plaquette indices apply identity operator
        if a_index == b_index:
            return self

        # steps from A to B
        x_steps, y_steps = self.code.translation(a_index, b_index)
        # start index
        current_x, current_y = a_index
        # build path (diagonal until inline then straight up/down or left/right)
        path_indices = []
        while x_steps or y_steps:
            if x_steps > 0:
                current_x += 1
                x_steps -= 1
            elif x_steps < 0:
                if path_indices:  # only move left after first step (because plaquettes indexed by lower-left corner)
                    current_x -= 1
                x_steps += 1
            if y_steps > 0:
                current_y += 1
                y_steps -= 1
            elif y_steps < 0:
                if path_indices:  # only move down after first step (because plaquettes indexed by lower-left corner)
                    current_y -= 1
                y_steps += 1
            # add next_index to path
            path_indices.append((current_x, current_y))

        # apply Xs if Z plaquettes, or Zs otherwise
        operator = 'X' if self.code.is_z_plaquette(a_index) else 'Z'
        self.site(operator, *path_indices)
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
