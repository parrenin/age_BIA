# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import sys
import numpy as np
from scipy.optimize import fsolve
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
# Fonctions definition
# ---------------------------------------------------------

def fct_omega(zeta, p):
    return 1 - (p + 2) / (p + 1) * (1 - zeta) + 1 / (p + 1) * (1 - zeta) ** (p + 2)

def fct_zeta(omega, p):
    # Définir la fonction dont on cherche la racine
    def equation(zeta):
        return fct_omega(zeta, p) - omega

    # Résoudre numériquement
    zeta_solution = fsolve(equation, omega)  # omega is an initial value
    return zeta_solution[0]

# ---------------------------------------------------------
# Reading parameters.yml file
# ---------------------------------------------------------

# Default values for parameters, to prevent spyder errors
x_left = 0.
x_right = 100.
x_step = 1.
z_step = 5.
Q_left = 1000.
Qm_left = 100.
fig_format = 'pdf'

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

B = S - H

# Computation of total flux Q
dQdx = (x[1:]-x[:-1]) * 1000 * \
    (a[:-1] * Y[:-1] +
     0.5 * ((a[1:]-a[:-1]) * Y[:-1] + (Y[1:]-Y[:-1])
            * a[:-1]) +
     1./3 * (a[1:]-a[:-1]) * (Y[1:]-Y[:-1]))

dQdx = np.insert(dQdx, 0, Q_left)
Q = np.cumsum(dQdx)
if np.any(Q<=0):
    print('Warning: total flux becomes negatif at some point! Exiting...')
    sys.exit()

# Computation of basal melting flux Qm
dQmdx = (x[1:]-x[:-1]) * 1000 * \
    (m[:-1] * Y[:-1] +
     0.5 * ((m[1:]-m[:-1]) * Y[:-1] + (Y[1:]-Y[:-1])
            * m[:-1]) +
     1./3 * (m[1:]-m[:-1]) * (Y[1:]-Y[:-1]))
dQmdx = np.insert(dQmdx, 0, Qm_left)
Qm = np.cumsum(dQmdx)


# ------------------------------------
# Determination of accumulation areas
# ------------------------------------

accu_starts = [0]
accu_ends = []
for i in range(1, len(x)-1):
    if Q[i] > Q[i-1] and Q[i] >= Q[i+1] : #To be checked
        accu_ends.append(i)
    elif Q[i] <= Q[i-1] and Q[i] < Q[i+1] : #To be checked
        accu_starts.append(i+1)
  
        
# --------------------------------------
# Determination of trajectories
# --------------------------------------

trajs = []
for k in np.arange(H[0], 0., -z_step):
    xx = np.array([x[0]])
    zz = np.array([k])
    zeta = zz/H[0]
    omega = fct_omega(zeta, p[0]) 
    qq = Qm[0] + (Q[0]-Qm[0]) * omega
    traj = {'x':xx, 'z':zz}
    j = 1
    while j < len(x) and Q[j] >= qq and Qm[j] <= qq :
        traj['x'] = np.append(traj['x'], x[j])
        omega = (qq-Qm[j])/(Q[j]-Qm[j])
        zeta = fct_zeta(omega, p[j])
        traj['z'] = np.append(traj['z'], B[j] + zeta * H[j])
        j += 1
    trajs.insert(0, traj)
for k in range(len(accu_starts)):
    istart = accu_starts[k]
    iend = accu_ends[k]
    for i in range(istart, iend+1):
        xx = np.array([x[i]])
        zz = np.array([S[i]])
        traj = {'x':xx, 'z':zz}
        j = i+1
        while j < len(x) and Q[j] >= Q[i] and Qm[j] <= Q[i] :
            traj['x'] = np.append(traj['x'], x[j])
            omega = (Q[i]-Qm[j])/(Q[j]-Qm[j])
            zeta = fct_zeta(omega, p[j])
            traj['z'] = np.append(traj['z'], B[j] + zeta * H[j])
            j += 1
        trajs.append(traj)
        


# ----------------------------------------
# Plotting of trajectories
# ----------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x, a, color='k', linewidth=2)
ax1.plot(x, np.zeros_like(x), color='k', linewidth=2, linestyle='--')
ax1.set_ylabel('a (m/yr)', fontsize=19)

ax2.plot(x, S, label='Surface', color='k', linewidth=2.)
ax2.plot(x, B, label='Bedrock', color='k', linewidth=2.)
for traj in trajs:
    ax2.plot(traj['x'], traj['z'], color='k', linestyle='--')
ax2.set_xlabel(r'$x$ (km)', fontsize=19)
ax2.set_ylabel(r'$z$ (m)', fontsize=19)
plt.savefig(datadir+'stream_lines_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')


# Program execution time
MESSAGE = 'Program execution time: '+str(time.perf_counter()-START_TIME)+' s.'
print(MESSAGE)
if os.name != 'nt':
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max memory usage: '+str(mem)+' kbytes')