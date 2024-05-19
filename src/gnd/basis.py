# basis.py
""" This module contains the Basis class along with helper functions to construct specific bases. """
import itertools as it
import numpy as np
import jax.numpy as jnp
import jax
from gnd.utils import multikron


@jax.jit
def trace_dot_jit(x, y):
    """ Return tr(xy). """
    return jnp.trace(x @ y)


def traces(b_1, b_2):
    """ Return tr(xy) for all permutations of x in b_1 and y in b_2. """
    len_1 = b_1.shape[0]
    len_2 = b_2.shape[0]
    indices = np.stack([[i, j] for i in range(len_1) for j in range(len_2)])

    jself = jnp.array(b_1)
    jother = jnp.array(b_2)
    carry = jnp.empty((len_1, len_2), dtype=np.complex128)

    def scan_body(c, idx):
        idx, jdx = idx
        c = c.at[idx, jdx].set(trace_dot_jit(jself[idx], jother[jdx]))
        return c, None

    carry, _ = jax.lax.scan(scan_body, init=carry, xs=indices)
    return carry


class Basis:
    """
    This class represents a basis for a Hilbert space.

    Parameters
    ----------
    basis : np.ndarray
        the set of arrays the basis is to consists of, must be shape (3, dqudits**n, dqudits**n).
    dqudits : int
        the dimension of the qudits that the basis is constructed for.

    Attributes
    ----------
    basis : np.ndarray
        the set of arrays the basis is to consists of, must be shape (3, dqudits**n, dqudits**n).
    dqudits : int
        the dimension of the qudits that the basis is constructed for.
    nqudits : int
        the number of qudits the basis covers.
    dim : int
        the dimension of the basis operators.

    Methods
    -------
    linear_span(parameters)
        Represents the linear span of this basis given a parameterisation.
    overlap(other)
        Represents the overlap between this basis and another.
    verify()
        Verifies that the basis is orthogonal.
    """

    def __init__(self, basis: np.ndarray, dqudits=2):
        if basis.ndim != 3:
            raise ValueError('basis must be a rank 3 tensor')

        nqudits = np.emath.logn(dqudits, basis.shape[1])
        if (basis.shape[1] != basis.shape[2]) or (nqudits != int(nqudits)):
            raise ValueError('basis must be a tensor of shape (x, dqudits**nqudits, dqudits**nqudits),' +
                             f'received {basis.shape} and dqudits={dqudits} giving non integer nqudits={nqudits}')
        self.basis = basis
        self.dim = basis.shape[1]
        self.nqudits = int(nqudits)

    def linear_span(self, parameters):
        """ Return the linear span of this basis given a parameterisation. TODO - expand docstring """
        parameters = np.reshape(parameters, (-1, 1, 1))
        return np.einsum('nij,nij->ij', parameters, self.basis)

    def overlap(self, other):
        """ Return the overlap between this basis and another. TODO - expand docstring """
        out = traces(self.basis, other.basis)
        return ~np.isclose(np.sum(out, axis=0), 0)

    def verify(self):
        """ Return True if basis is orthogonal. """
        out = traces(self.basis, self.basis)
        return np.allclose(np.diag(np.diag(out)), out)

    def __len__(self):
        return self.basis.shape[0]


def construct_two_body_pauli_basis(n: int):
    """ Return Basis instance consisting of two-body Pauli terms. """
    eye = np.eye(2).astype(np.complex128)
    paulis = [
        eye,
        np.array([[0, 1], [1, 0]], np.complex128),
        np.array([[0, -1j], [1j, 0]], np.complex128),
        np.array([[1, 0], [0, -1]], np.complex128)
    ]

    basis = [multikron(comb) for comb in list(it.product(paulis, repeat=n))[1:]
             if np.sum(np.any(comb != eye, (1, 2))) <= 2]

    return Basis(np.stack(basis))


def construct_full_pauli_basis(n: int):
    """ Return Basis instance consisting of all Pauli terms. """
    eye = np.eye(2).astype(np.complex128)
    paulis = [
        eye,
        np.array([[0, 1], [1, 0]], np.complex128),
        np.array([[0, -1j], [1j, 0]], np.complex128),
        np.array([[1, 0], [0, -1]], np.complex128)
    ]

    basis = [multikron(comb) for comb in list(it.product(paulis, repeat=n))[1:]]

    return Basis(np.stack(basis))


def construct_two_body_gellmann_basis(n: int):
    """ Return Basis instance consisting of two-body Gell-Mann terms. """
    eye = np.eye(3).astype(np.complex128)
    gellmans = [
        eye,
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], np.complex128),
        np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], np.complex128),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.complex128),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], np.complex128),
        np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], np.complex128),
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], np.complex128),
        np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], np.complex128),
        1/np.sqrt(3) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], np.complex128)
    ]

    basis = [multikron(comb) for comb in list(it.product(gellmans, repeat=n))[1:]
             if np.sum(np.any(comb != eye, (1, 2))) <= 2]

    return Basis(np.stack(basis), dqudits=3)


def construct_full_gellmann_basis(n: int):
    """ Return Basis instance consisting of all Gell-Mann terms. """
    eye = np.eye(3).astype(np.complex128)
    gellmans = [
        eye,
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], np.complex128),
        np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], np.complex128),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.complex128),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], np.complex128),
        np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], np.complex128),
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], np.complex128),
        np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], np.complex128),
        1/np.sqrt(3) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], np.complex128)
    ]

    basis = [multikron(comb) for comb in list(it.product(gellmans, repeat=n))[1:]]

    return Basis(np.stack(basis), dqudits=3)
