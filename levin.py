import numpy as np


def levin(a=None, b=None):
    # function x = levin(a,b)
    # solves system of complex linear equations toeplitz(a)*x=b
    # using Levinson's algorithm
    # a ... first row of positive definite Hermitian Toeplitz matrix
    # b ... right hand side vector

    # Author: Mathias C. Lang, Vienna University of Technology, AUSTRIA
    # 1997-09
    # mattsdspblog@gmail.com

    n = len(a)
    t = np.ones(1)
    alpha = a[0]
    x = np.atleast_1d(b[0] / a[0])
    for i in np.arange(1, n):
        k = -(a[i:0:-1]) @ t / alpha
        t = np.r_[t, 0] + k * np.r_[np.conj(t), 0][::-1]
        alpha = alpha * (1 - np.abs(k) ** 2)
        k = (b[i] - np.transpose(a[i:0:-1]) @ x) / alpha
        x = np.r_[x, 0] + k * np.conj(t)[::-1]

    return x
