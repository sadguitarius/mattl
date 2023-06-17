import numpy as np


def iir_ap(N=None, f=None, phi=None, W=None):
    w = np.pi * f
    phi = phi
    W = np.sqrt(W)
    A = np.exp(
        -1j * np.kron(np.atleast_2d(w).T, (np.arange(N - 1, -1, -1)))
    ) - np.exp(
        1j
        * (
            np.tile(np.atleast_2d(phi).T, N)
            - np.kron(np.atleast_2d(w).T, (np.arange(1, N + 1)))
        )
    )
    b = np.exp(1j * phi) - np.exp(-1j * N * w)
    A = np.tile(np.atleast_2d(W).T, N) * A
    b = np.array([W * b]).T
    a = np.linalg.lstsq(
        np.vstack([np.real(A), np.imag(A)]),
        np.vstack([np.real(b), np.imag(b)]),
        rcond=None,
    )
    a = np.vstack([1, a[0]])
    return a
