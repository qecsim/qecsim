import numpy as np


class ToricPauli:
    """
    Defines a Pauli operator on a toric lattice.

    Notes:

    * This is a utility class used by toric implementations of the core models.
    * It is typically instantiated using :meth:`qecsim.models.toric.ToricCode.new_pauli`

    Use cases:

    * Construct a toric Pauli operator by applying site, plaquette, path and logical operators: :meth:`site`,
      :meth:`plaquette`, :meth:`path`, :meth:`logical_x1`, :meth:`logical_x2`, :meth:`logical_z1`, :meth:`logical_z2`.
    * Get the single Pauli operator applied to a given site: :meth:`operator`
    * Convert to binary symplectic form: :meth:`to_bsf`.
    * Copy a toric Pauli operator: :meth:`copy`.
    """

    def __init__(self, code, bsf=None):
        """
        Initialise new toric Pauli.

        Notes:

        * For performance reasons, the new Pauli is a view of the given bsf. Modifying one will modify the other.

        :param code: The toric code.
        :type code: ToricCode
        :param bsf: Binary symplectic representation of Pauli. (Optional. Defaults to identity.)
        :type bsf: numpy.array (1d)
        """
        self._code = code
        if bsf is None:
            # initialise identity lattices for X and Z operators
            self._xs = np.zeros(self.code.shape, dtype=int)
            self._zs = np.zeros(self.code.shape, dtype=int)
        else:
            assert np.array_equal(bsf % 2, bsf), 'BSF {} is not in binary form'.format(bsf)
            # initialise lattices for X and Z operators from bsf
            xs_flat, zs_flat = np.hsplit(bsf, 2)
            self._xs = xs_flat.reshape(self.code.shape)
            self._zs = zs_flat.reshape(self.code.shape)

    @property
    def code(self):
        """
        The toric code.

        :rtype: ToricCode
        """
        return self._code

    def copy(self):
        """
        Returns a copy of this Pauli that references the same code but is backed by a copy of the bsf.

        :return: A copy of this Pauli.
        :rtype: ToricPauli
        """
        return self.code.new_pauli(bsf=np.copy(self.to_bsf()))

    def operator(self, index):
        """
        Returns the operator on the site identified by the index.

        Notes:

        * Index is in the format (lattice, row, column).
        * Index is modulo the lattice shape, i.e. on a (5, 5) lattice, (2, 6, -1) indexes the same site as (0, 1, 4).

        :param index: Index identifying a site in the format (lattice, row, column).
        :type index: 3-tuple of int
        :return: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :rtype: str
        """
        # index modulo shape
        index = tuple(np.mod(index, self.code.shape))
        x = self._xs[index]
        z = self._zs[index]
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

        * Index is in the format (lattice, row, column).
        * Index is modulo the lattice shape, i.e. on a (5, 5) lattice, (2, 6, -1) indexes the same site as (0, 1, 4).

        :param operator: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :type operator: str
        :param indices: Any number of indices identifying a site in the format (lattice, row, column).
        :type indices: Any number of 3-tuple of int
        :return: self (to allow chaining)
        :rtype: ToricPauli
        """
        for index in indices:
            # index modulo shape
            index = tuple(np.mod(index, self.code.shape))
            # flip sites
            if operator in ('X', 'Y'):
                self._xs[index] ^= 1
            if operator in ('Z', 'Y'):
                self._zs[index] ^= 1
        return self

    def plaquette(self, index):
        """
        Apply a plaquette operator at the given index.

        Notes:

        * Index is in the format (lattice, row, column).
        * If the primal lattice is indexed (i.e. lattice=0), then Z operators are applied around the indexed plaquette.
          (This is equivalent to a vertex operator on the dual lattice.)
        * If the dual lattice is indexed (i.e. lattice=1), then X operators are applied around the indexed plaquette.
          (This is equivalent to a vertex operator on the primal lattice.)
        * Index is modulo the lattice shape, i.e. on a (5, 5) lattice, (2, 6, -1) indexes the same plaquette as
          (0, 1, 4).

        :param index: Index identifying the plaquette in the format (lattice, row, column).
        :type index: 3-tuple of int
        :return: self (to allow chaining)
        :rtype: ToricPauli
        """
        # index modulo shape
        la, r, c = np.mod(index, self.code.shape)
        # apply Zs if primal lattice, or Xs otherwise
        operator = 'Z' if la == self.code.PRIMAL_INDEX else 'X'
        # flip plaquette sites
        self.site(operator, (la, r, c))  # North
        self.site(operator, (la, r + 1, c))  # South
        self.site(operator, (la + 1, r + la, c - la))  # West
        self.site(operator, (la + 1, r + la, c - la + 1))  # East
        return self

    def path(self, a_index, b_index):
        """
        Apply the shortest taxi-cab path of operators between the plaquettes indexed by A and B.

        Notes:

        * Indices are in the format (lattice, row, column).
        * Both indices must index the same lattice.
        * If the primal lattice is indexed (i.e. lattice=0) then X operators are applied along the shortest path on
          dual lattice edges.
        * If the dual lattice is indexed (i.e. lattice=1) then Z operators are applied along the shortest path on
          primal lattice edges.
        * Indices are modulo the lattice shape, i.e. on a (5, 5) lattice, (2, 6, -1) indexes the same plaquette as
          (0, 1, 4).

        :param a_index: Index identifying a plaquette in the format (lattice, row, column).
        :type a_index: 3-tuple of int
        :param b_index: Index identifying a plaquette in the format (lattice, row, column).
        :type b_index: 3-tuple of int
        :return: self (to allow chaining)
        :rtype: ToricPauli
        :raises IndexError: If indices do not index the same valid lattice.
        """
        # index of plaquette A
        a_l, a_r, a_c = np.mod(a_index, self.code.shape)
        # apply Xs if plaquettes on primal lattice, or Zs otherwise
        operator = 'X' if a_l == self.code.PRIMAL_INDEX else 'Z'
        # steps from A to B
        row_steps, col_steps = self.code.translation(a_index, b_index)
        # current index
        c_l, c_r, c_c = a_l, a_r, a_c
        # start on north site
        while row_steps < 0:  # heading north
            # flip current then decrement row
            self.site(operator, (c_l, c_r, c_c))
            c_r -= 1
            row_steps += 1
        while row_steps > 0:  # heading south
            # increment row and then flip current
            c_r += 1
            self.site(operator, (c_l, c_r, c_c))
            row_steps -= 1
        # switch to west site
        c_l, c_r, c_c = c_l + 1, c_r + c_l, c_c - c_l
        while col_steps < 0:  # heading west
            # flip current then decrement col
            self.site(operator, (c_l, c_r, c_c))
            c_c -= 1
            col_steps += 1
        while col_steps > 0:  # heading east
            # increment col then flip current
            c_c += 1
            self.site(operator, (c_l, c_r, c_c))
            col_steps -= 1
        return self

    def logical_x1(self):
        """
        Apply a logical X1 operator, i.e. column of X on horizontal-edge sites of plaquettes on primal.

        :return: self (to allow chaining)
        :rtype: ToricPauli
        """
        # flip x_operators[primal_lattice, all_rows, middle column]
        self._xs[self.code.PRIMAL_INDEX, :, self._xs.shape[2] // 2] ^= 1
        return self

    def logical_x2(self):
        """
        Apply a logical X2 operator, i.e. row of X on horizontal-edge sites of plaquettes on dual.

        :return: self (to allow chaining)
        :rtype: ToricPauli
        """
        # flip x_operators[dual_lattice, middle_row, all_columns]
        self._xs[self.code.DUAL_INDEX, self._xs.shape[1] // 2, :] ^= 1
        return self

    def logical_z1(self):
        """
        Apply a logical Z1 operator, i.e. row of Z on horizontal-edge sites of plaquettes on primal.

        :return: self (to allow chaining)
        :rtype: ToricPauli
        """
        # flip z_operators[primal_lattice, middle_row, all_columns]
        self._zs[self.code.PRIMAL_INDEX, self._zs.shape[1] // 2, :] ^= 1
        return self

    def logical_z2(self):
        """
        Apply a logical Z2 operator, i.e. column of Z on horizontal-edge sites of plaquettes on dual.

        :return: self (to allow chaining)
        :rtype: ToricPauli
        """
        # flip z_operators[dual_lattice, all_rows, middle_column]
        self._zs[self.code.DUAL_INDEX, :, self._zs.shape[2] // 2] ^= 1
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
        return np.concatenate((self._xs.flatten(), self._zs.flatten()))
