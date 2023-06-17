# function h = cfirls(N,f,d,w,hreal)
# Least-squares FIR filter design with prescribed magnitude and phase
# responses and with possibly complex coefficients
#
# h     real or complex-valued impulse response
# N     desired filter length
# f     frequency grid
#       	for complex-valued filters ('hreal=0'): -1 <= f < 1
# 		( '1' corresponds to Nyquist) or 0 <= f < 2
# 		for real-valued filters ('hreal=1'): 0 <= f <= 1
# d     desired complex-valued frequency response on the grid f:
#       d = ( desired_magnitude ) .* exp( 1i * desired_phase )
# w     (positive) weighting function on the grid f
# hreal	flag for constraining the filter coefficients h to be real-valued
# 		hreal=1: h is real-valued
# 		hreal=0: h can be complex-valued
# 			 h will be real-valued (up to numerical accuracy) if for
# 			 every grid point f[i] there is a point -f[i] and the
# 			 desired response at -f[i] is the complex conjugate of d
# 			 at f[i], and the weights at f[i] and -f[i] must be equal

# differences with firls.m:
# 1. the desired phase response can be non-linear
# 2. the filter coefficients can be complex (i.e., the specified
# 	 frequency response does not need to be symmetrical)
# 3. the desired magnitude doesn't need to be piecewise linear,
# 	 but it is specified as a vector defined on a frequency grid
# 4. the weighting does not need to be piecewise constant

# EXAMPLE 1: real-valued linear phase bandpass filter
# N = 61;
# tau = (N-1)/2;
# f = [linspace(0,.23,230),linspace(.3,.5,200),linspace(.57,1,430)];
# d = [zeros(1,230),ones(1,200),zeros(1,430)] .* exp(-j*tau*pi*f);
# w = [10*ones(1,230),ones(1,200),10*ones(1,430)];
# h = cfirls(N,f,d,w,1);

# EXAMPLE 2: same as Ex. 1 but with reduced delay in the passband
# N = 61;
# tau = 25;
# f = [linspace(0,.23,230),linspace(.3,.5,200),linspace(.57,1,430)];
# d = [zeros(1,230),ones(1,200),zeros(1,430)] .* exp(-j*tau*pi*f);
# w = [10*ones(1,230),ones(1,200),10*ones(1,430)];
# h = cfirls(N,f,d,w,1);

# EXAMPLE 3: linear phase complex low pass filter
# N = 51;
# tau = (N-1)/2;
# f = [linspace(-1,-.18,328),linspace(-.1,.3,160),linspace(.38,1,248)];
# d = [zeros(1,328),ones(1,160),zeros(1,248)].*exp(-1i*pi*f*tau);
# w = [10*ones(1,328),ones(1,160),10*ones(1,248)];
# h = cfirls(N,f,d,w);

# EXAMPLE 4: same as Ex. 3 but with reduced delay in the passband
# N = 51;
# tau = 20;
# f = [linspace(-1,-.18,328),linspace(-.1,.3,160),linspace(.38,1,248)];
# d = [zeros(1,328),ones(1,160),zeros(1,248)].*exp(-1i*pi*f*tau);
# w = [10*ones(1,328),ones(1,160),10*ones(1,248)];
# h = cfirls(N,f,d,w);
#
# author: Mathias Lang, 2018-01-01
# revision 2022-10-17: added option 'hreal' for designing exactly
# real-valued filters
# mattsdspblog@gmail.com

import numpy as np


def cfirls(N, f, d, w, hreal=0):
    L = len(f)
    if len(d) != L:
        raise ValueError('d and f must have the same lengths.')
    if len(w) != L:
        raise ValueError('w and f must have the same lengths.')
    if N < 1:
        raise ValueError('N must be greater than 0.')
    if not np.all(w):
        raise ValueError('All elements of the weight vector w must be positive.')
    minf = np.min(f)
    maxf = np.max(f)
    if hreal:
        if (minf < 0 or maxf > 1):
            raise ValueError('f must be in the range [0,1] for real-valued filters.')
    else:
        if not ((minf >= -1 and maxf <= 1) or (minf >= 0 and maxf <= 2)):
            raise ValueError('f must be in the interval [-1,1) or [0,2).')

    w = np.sqrt(w)

    # solve overdetermined system in a least squares sense
    A = w[:, None] * (np.exp(-1j * np.pi * f[:, None] * np.arange(N)))
    b = w * d
    if hreal:
        h = np.linalg.lstsq(np.r_[np.real(A), np.imag(A)], np.r_[np.real(b), np.imag(b)], rcond=None)[0]
    else:
        h = np.linalg.lstsq(A, b, rcond=None)[0]
    return h
