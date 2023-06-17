import numpy as np
from .levin import levin


def lslevin(N=None, om=None, D=None, W=None):
    # h = lslevin(N,om,D,W)
    # Complex Least Squares FIR filter design using Levinson's algorithm

    # h      filter impulse response
    # N      filter length
    # om     frequency grid (0 <= om <= pi)
    # D      complex desired frequency response on the grid om
    # W      positive weighting function on the grid om

    # example: length 61 bandpass, band edges [.23,.3,.5,.57]*pi,
    # weighting 1 in passband and 10 in stopbands, desired passband
    # group delay 20 samples

    # om=pi*[linspace(0,.23,230),linspace(.3,.5,200),linspace(.57,1,430)];
    # D=[zeros(1,230),exp(-j*om(231:430)*20),zeros(1,430)];
    # W=[10*ones(1,230),ones(1,200),10*ones(1,430)];
    # h = lslevin(61,om,D,W);

    # Author: Mathias C. Lang, Vienna University of Technology
    # 1998-07
    # mattsdspblog@gmail.com

    L = len(om)
    # DR = real(D)
    # DI = imag(D)
    a = np.zeros(N)
    b = np.copy(a)
    # Set up vectors for quadratic objective function
    # (avoid building matrices)
    dvec = D
    evec = np.ones(L)
    e1 = np.exp(1j * om)
    for i in range(N):
        a[i] = W @ np.real(evec)
        b[i] = W @ np.real(dvec)
        evec = evec * e1
        dvec = dvec * e1

    a = a / L
    b = b / L
    # Compute weighted l2 solution
    h = levin(a, b)
    return h
