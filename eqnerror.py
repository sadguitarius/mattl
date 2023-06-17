# [a,b] = eqnerror(M,N,w,D,W,iter);
#
# IIR filter design using equation error method
#
# if the input argument 'iter' is specified and if iter > 1, then
# the equation error method is applied iteratively trying to
# determine the true L2 solution of the nonlinear IIR design problem
#
# M     numerator order
# N     denominator order
# a     denominator coefficients (length N+1), a(1) = 1
# b     numerator coefficients (length M+1)
# w     frequency vector in [0,pi], where pi is Nyquist
# D     desired complex frequency response at frequencies w
# W     weight vector defined at frequencies w
# iter  optional; number of iterations for non-linear solver
#
# author: Mathias C. Lang, 2016-02-28
# mattsdspblog@gmail.com

import numpy as np
from scipy.signal import freqz


def eqnerror(M, N, w, D, W, iter=1):
    if (np.max(w) > np.pi or np.min(w) < 0):
        raise ValueError('w must be in [0,pi]')
    L = len(w)
    if (len(D) != L):
        raise ValueError('D and w must have the same lengths.')
    if (len(W) != L):
        raise ValueError('W and w must have the same lengths.')

    D0 = D
    W0 = W
    A0 = np.r_[-D0[:, None] * np.exp(-1j*w*np.arange(N+1)), np.exp(-1j*w*np.arange(M+1))]
    den = np.ones(L, 1)

    for k in range(iter):
        W = W0/np.abs(den)
        A = A0[:, None] * W
        D = D0 * W
        x = np.linalg.lstsq(np.r_[np.real(A), np.imag(A)], np.r_[np.real(D), np.imag(D)])
        a = np.r_[1, x[:N]]
        b = x[N:M+N]
        den = freqz(a, 1, w)

    return a, b
