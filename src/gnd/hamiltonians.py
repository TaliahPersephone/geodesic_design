import numpy as np
import scipy.linalg as spla
import sympy as sp
from gnd.utils import multikron


def each_body_term(Us, neyes=2):
    """ Return U tensored with I to apply to each qutrit. """
    eye = np.eye(3)
    eyes = [eye] * (neyes)

    return np.array([multikron(np.roll(Us + eyes, j, axis=0)) for j in range(neyes+1)])


def rydberg_hamiltonian(nqudits=3, extended=True):
    """ Return the Hamiltonian of a qutrit Rydberg system. """
    # Single body terms
    eye = np.eye(3)
    ket_0 = np.matrix([[1], [0], [0]], dtype=np.complex128)
    ket_1 = np.matrix([[0], [1], [0]], dtype=np.complex128)
    ket_r = np.matrix([[0], [0], [1]], dtype=np.complex128)

    m_r1s = each_body_term([ket_r @ ket_1.H], 2)
    m_10s = each_body_term([ket_1 @ ket_0.H], 2)
    m_r0s = each_body_term([ket_r @ ket_0.H], 2)

    m_rr = ket_r @ ket_r.H
    m_rrs = each_body_term([m_rr], 2)
    m_2rrs = each_body_term([m_rr, m_rr], 1)

    # Omegas
    omega_r1 = sp.symarray('omega_r1', nqudits).reshape((-1, 1, 1))
    omega_10 = sp.symarray('omega_10', nqudits).reshape((-1, 1, 1))
    omega_r0 = sp.symarray('omega_r0', nqudits).reshape((-1, 1, 1))

    # phi
    phi_r1 = sp.symarray('phi_r1', nqudits)
    e_phi_r1 = np.array([sp.exp(phi * sp.I) for phi in phi_r1]).reshape((-1, 1, 1))
    e_phi_r1 = e_phi_r1 * m_r1s
    h_r1 = np.array([sp.Matrix(j) + sp.Matrix(j).H for j in e_phi_r1]).reshape((-1, 3**nqudits, 3**nqudits))
    h_r1 = sp.Matrix(np.einsum('njk,nab->ab', omega_r1, h_r1))

    phi_10 = sp.symarray('phi_10', nqudits)
    e_phi_10 = np.array([sp.exp(phi * sp.I) for phi in phi_10]).reshape((-1, 1, 1))
    e_phi_10 = e_phi_10 * m_10s
    h_10 = np.array([sp.Matrix(j) + sp.Matrix(j).H for j in e_phi_10]).reshape((-1, 3**nqudits, 3**nqudits))
    h_10 = sp.Matrix(np.einsum('njk,nab->ab', omega_10, h_10))

    phi_r0 = sp.symarray('phi_r0', nqudits)
    e_phi_r0 = np.array([sp.exp(phi * sp.I) for phi in phi_r0]).reshape((-1, 1, 1))
    e_phi_r0 = e_phi_r0 * m_r0s
    h_r0 = np.array([sp.Matrix(j) + sp.Matrix(j).H for j in e_phi_r0]).reshape((-1, 3**nqudits, 3**nqudits))
    h_r0 = sp.Matrix(np.einsum('njk,nab->ab', omega_r0, h_r0))

    # delta
    delta = sp.symarray('delta', nqudits).reshape((-1, 1, 1))
    h_rr = sp.Matrix(np.einsum('njk,nab->ab', delta, m_rrs))

    # B
    B = sp.symarray('B', nqudits-1).reshape((-1, 1, 1))
    h_2rr = sp.Matrix(np.einsum('njk,nab->ab', B, m_2rrs))

    # Put it together
    return h_r1 + h_10 + h_r0 + h_rr + h_2rr, np.concatenate([omega_r1.flatten(),phi_r1.flatten(), omega_10.flatten(), phi_10.flatten(), omega_r0.flatten(), phi_r0.flatten(), delta.flatten(), B.flatten()])
