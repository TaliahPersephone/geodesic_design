# unitaries.py
""" This module contains definitions of unitaries as numpy matrices, for use as targets or otherwise. """
import numpy as np
import sympy as sp
from gnd.utils import multikron


def _pauli_x(dqudits=2):
    """
    Return a single qudit pauli X operator:
        sum over j in {0,..,d-1} : |j><j+1|
    """
    return np.roll(np.eye(dqudits), dqudits)


def _pauli_z(dqudits=2):
    """
    Return a single qudit pauli Z operator:
        sum over j in {0,..,d-1} : ω**j |j><j|
        ω = exp(2πi/d)
    """
    # We use sympy here to limit float errors
    phases = [np.complex128(sp.exp(2 * sp.pi * sp.I * j / dqudits)) for j in range(dqudits)]
    return np.diag(phases)


def phase_flip(dqudits=2):
    """ Return an operator that flips the phase of |d-1>. """
    phase_flip = np.eye(dqudits)
    phase_flip[-1, -1] = -1
    return phase_flip


def swap(dqudits=2):
    """ Return a two qudit swap operator. """
    # Use a 3rd dimension to easily reorder the columns by a transpose, then return to the original shape
    return np.eye(dqudits**2).reshape(dqudits, dqudits, -1).T.reshape(dqudits**2, -1)


def cx_u(U, ncontrol=1, dqudits=2):
    """
    Return a controlled U operator, applied when each control has a value of d-1.
    U need not operate on qudits of the same dimension as the controls.

    Parameters
    ----------
    U : np.array like
        The operator to be controlled, which will act on the last qudits.
    ncontrol : int, default = 1
        How many qudits to control U.
    dqudits : int, default = 2
        The dimension of the qudits, the default of 2 results in qubits.

    Returns
    -------
    c_u : np.array
    """
    # U is an M by M matrix
    u_m = U.shape[0]

    # Create I of the correct size
    c_u = np.eye(dqudits ** ncontrol * u_m)

    # Assign U to |(d-1)><(d-1)|⊗ ncontrol
    c_u[-u_m:, -u_m:] = U

    return c_u


def cnot(nqudits=3, dqudits=2):
    """ Return a controled X operator. """
    ncontrol = nqudits-1
    return cx_u(_pauli_x(dqudits), ncontrol, dqudits), f'C{ncontrol}Not'


def cz(nqudits=3, dqudits=2):
    """ Return a controlled Z operator. """
    ncontrol = nqudits-1
    return cx_u(_pauli_z(dqudits), ncontrol, dqudits), f'C{ncontrol}Z'


def cphase(nqudits=3, dqudits=2):
    """ Return a controlled PHASE operator. """
    ncontrol = nqudits-1
    return cx_u(phase_flip(dqudits), ncontrol, dqudits), f'C{ncontrol}PHASE'


def cswap(nqudits=3, dqudits=2):
    """ Return a controlled SWAP operator. """
    ncontrol = nqudits-2
    return cx_u(swap(dqudits), ncontrol, dqudits), f'C{ncontrol}SWAP'


def qft(nqudits=3, dqudits=2):
    """ Return a qft operator. """
    D = dqudits ** nqudits
    w = sp.exp(2 * sp.pi * sp.I / D)
    qft = [[np.complex128(w ** (i * j)) for i in range(D)] for j in range(D)]
    return np.array(qft) / np.sqrt(D), f'QFT_{nqudits}'


def _wk_u_parity(U, k=4, dqudits=2):
    """
    Return a k-weighted U parity check operator.
    This is used to construct the k-weighted X and Z parity checks.

    Parameters
    ----------
    U : np.array like
        The U operator with which to perform the parity check, typically Pauli X or Z.
    k : int
        The weight of the parity check.
    dqudits : int, default = 2
        The dimension of the qudits, the default of 2 results in qubits.

    Returns
    -------
    wkp_u : np.array
    """
    eye = np.eye(dqudits)
    X = _pauli_x(dqudits)

    wkp_u = 1/2 * (
        multikron([*[eye]*(k+1)])
        + multikron([*[U]*(k), eye])
        + multikron([*[eye]*(k), X])
        - multikron([*[U]*(k), X])
    )

    return wkp_u


def wkpx(nqudits=5, dqudits=2):
    """ Return k-weighted X parity check, see _wk_u_parity for details. """
    k = nqudits-1
    return _wk_u_parity(_pauli_x(dqudits), k, dqudits), f'W{k}pX'


def wkpz(nqudits=5, dqudits=2):
    """ Return k-weighted Z parity check, see _wk_u_parity for details. """
    k = nqudits-1
    return _wk_u_parity(_pauli_z(dqudits), k, dqudits), f'W{k}pZ'
