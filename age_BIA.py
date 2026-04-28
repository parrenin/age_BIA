# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import sys
import numpy as np
from scipy.linalg import toeplitz
import yaml
import matplotlib.pyplot as plt
import time
import math
import os
import resource


# Registration of start time
START_TIME = time.perf_counter()


# Setting experiment directory
datadir = sys.argv[1]
if datadir[-1] != '/':
    datadir = datadir+'/'
print('Parameters directory is: ', datadir)


# ---------------------------------------------------------
# Reading parameters.yml file
# ---------------------------------------------------------

# Default values for parameters, to prevent spyder errors
x_left = 0.
x_right = 100.
x_step = 1.
Q_left = 0.
Qm_left = 0.

yamls = open(datadir+'parameters.yml').read()
para = yaml.load(yamls, Loader=yaml.FullLoader)
globals().update(para)


# -----------------------------------------------------
# Loading files for Geographic data, arrays creations
# -----------------------------------------------------

# Steady accumulation
x_a, a_measure = np.loadtxt(datadir+'accumulation.txt', unpack=True)

# Melting
x_m, m_measure = np.loadtxt(datadir+'melting.txt', unpack=True)

# Sliding rate
x_s, s_measure = np.loadtxt(datadir+'sliding.txt', unpack=True)

# Lliboutry parameter
x_p, p_measure = np.loadtxt(datadir+'p_Lliboutry.txt', unpack=True)

# Surface
x_S, S_measure = np.loadtxt(datadir+'surface.txt', unpack=True)

# Thickness
x_H, H_measure = np.loadtxt(datadir+'thickness.txt', unpack=True)

# Tube width
x_Y, Y_measure = np.loadtxt(datadir+'tube_width.txt', unpack=True)


# --------------------
# Interpolation
# --------------------

x = np.arange(x_left, x_right+x_step, x_step)

a = np.interp(x, x_a, a_measure)
m = np.interp(x, x_m, m_measure)
sliding = np.interp(x, x_s, s_measure)
p = np.interp(x, x_p, p_measure)
S = np.interp(x, x_S, S_measure)
H = np.interp(x, x_H, H_measure)
Y = np.interp(x, x_Y, Y_measure)


# Computation of total flux Q
dQdx = (x[1:]-x[:-1]) * 1000 * \
    (a[:-1] * Y[:-1] +
     0.5 * ((a[1:]-a[:-1]) * Y[:-1] + (Y[1:]-Y[:-1])
            * a[:-1]) +
     1./3 * (a[1:]-a[:-1]) * (Y[1:]-Y[:-1]))

dQdx = np.insert(dQdx, 0, Q_left)
Q_fld = np.cumsum(dQdx)

# Computation of basal melting flux Qm
dQmdx = (x[1:]-x[:-1]) * 1000 * \
    (m[:-1] * Y[:-1] +
     0.5 * ((m[1:]-m[:-1]) * Y[:-1] + (Y[1:]-Y[:-1])
            * m[:-1]) +
     1./3 * (m[1:]-m[:-1]) * (Y[1:]-Y[:-1]))
dQmdx = np.insert(dQmdx, 0, Qm_left)
Qm_fld = np.cumsum(dQmdx)


# ------------------------------------
# Determination of accumulation areas
# ------------------------------------

accu_starts = [0]
accu_ends = []
for i in range(len(x)-1):
    if x[i] > 0. and x[i+1] <=0.:
        accu_ends.append(i-1)
    if x[i] <=0 and x[i+1] >0:
        accu_starts.append(i+1)
        

    