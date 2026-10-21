# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# TODO: plot Y
# TODO: exact formula for the age increase
# TDO: use panda for inputs
# TODO: Parametrize input files names?
# TODO: comparison files for unconformities
# TODO: can we interpolate age/origin/etc. on the grid?
# TODO: Use a gridded interpolation for inversion of omega, instead of fsolve?
# TODO: Output for isochrones
# TODO: Clean parameter file (regroup parameters, etc.)
# TODO: Remove 'a0' and 'x0' from trajs and profs since we have 'i0' now
# TODO: Can we take ux with an average zeta and an average p? Instead of the geometric mean of ux1 and ux2. This would prevent a singularity at bedrock.
# TODO: Can we avoid this repetition of code for the upstream and surface trajectories? Maybe by concatenating lists of i0, z0, etc.
# TODO: trajs and profs: store zeta, not z_ie.
# TODO: Implement varying z grid as well

import sys
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d
import yaml
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm, Normalize
import time
import os
import resource
import warnings
import pandas as pd
import math
from abia_math import grid

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
z_step = 5.
Q_left = 1000.
Qm_left = 100.
fig_format = 'pdf'
fig_age_iso = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
step_traj = 5
scaling_age = 10.
ic = {}
color_core = 'r'
lw_core = 2
ls_core = 'dashed'
temp_factor = None
obs_bedrock = None
comp_isochrones = None
comp_iso_ages = None
comp_surf_vel = None
comp_surf_age = None
x_start = None
solid_fraction = None
output_fl = False
fig_ic_slope = False
x_grid = {}
fig_slope_maxrange = None

yamls = open(datadir+'parameters.yml').read()
para = yaml.load(yamls, Loader=yaml.FullLoader)
globals().update(para)


# ---------------------------------------------------------
# Fonctions definition
# ---------------------------------------------------------

def derivative(arr, i):
    if i==0:
        return (arr[1]-arr[0])/(x[1]-x[0])
    elif i==len(arr)-1:
        return (arr[-1]-arr[-2])/(x[-1]-x[-2])
    elif i>0 and i<len(arr)-1:
        return (arr[i+1]-arr[i-1])/(x[i+1]-x[i-1])
    else:
        print('i index out of array length in derivative fct.')

def fct_omega(zeta, p):
    return 1 - (p + 2) / (p + 1) * (1 - zeta) + 1 / (p + 1) * (1 - zeta) ** (p + 2)

def fct_domega_dzeta(zeta, p):
    return (p + 2) / (p + 1) * (1 - (1 - zeta) ** (p + 1))

def fct_d2omega_dzeta2(zeta, p):
    return (p + 2) * (1 - zeta)**p

def fct_domega_dp(zeta, p):
    term1 = (1-zeta)/(p+1)**2
    term2 = -(1-zeta)**(p+2)/(p+1)**2
    if zeta < 1:
        term3 = (1-zeta)**(p+2)*math.log(1-zeta)/(p+1)
    else:
        term3 = 0.
    return term1 + term2 + term3

def fct_zeta(omega, p):
    # Définir la fonction dont on cherche la racine
    def equation(zeta):
        return fct_omega(zeta, p) - omega
    def derivative(zeta):
        return fct_domega_dzeta(zeta, p)

    # Résoudre numériquement
    zeta_solution = fsolve(equation, omega, fprime=derivative)  # omega is an initial value
    return zeta_solution[0]

def integrand(zeta, p, H, a, mu):
    """Intégrande : H / (a * (omega(zeta) + mu) / (1 + mu)) = H*(1+mu) / (a*(omega(zeta) + mu))"""
    return (H * (1 + mu)) / (a * (fct_omega(zeta, p) + mu))

def integrate_age(zeta_min, zeta_max, p, H, a, mu):
    """Intègre l'intégrande entre zeta_min et zeta_max."""
    result, _ = quad(
        lambda z: integrand(z, p, H, a, mu),
        zeta_min,
        zeta_max
    )
    return result

def fct_gamma_int(j, qq, zeta):
    return Y[j]/QH[j]**2*qq*H_ie[j] / fct_domega_dzeta(zeta, p[j])**3 \
        *fct_d2omega_dzeta2(zeta, p[j])
    


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


# Reading of observed bedrock when different from mechanical bedrock
if obs_bedrock is not None:
    readarray = np.loadtxt(datadir+obs_bedrock, unpack=True)
    obs_bed_x = readarray[0, :]
    obs_bed_depth = readarray[1, :]

# ---------------------------------------------------------------
# Reading of comparison isochrones
# ---------------------------------------------------------------

if comp_isochrones is not None:
    readarray = np.loadtxt(datadir+comp_isochrones, unpack=True)
    cp_iso_x = readarray[0, :]
    cp_iso_depth = readarray[1:, :]
    cp_iso_nb = cp_iso_depth.shape[0]
    cp_iso_z = np.interp(cp_iso_x, x_S, S_measure) - cp_iso_depth
if comp_iso_ages is not None:
    readarray = np.loadtxt(datadir+comp_iso_ages, unpack = True)
    cp_iso_ages = readarray[0, :]
    cp_iso_ages_sigma = readarray[1, :]

# ---------------------------------------------------------------
# Loading data from temporal_factor.txt
# ---------------------------------------------------------------

if temp_factor != None:
    age_R, R = np.loadtxt(datadir+temp_factor, unpack=True)
    age_R = np.append(age_R, age_R[-1]+1)
    age_R = np.append(age_R, 1e9)
    R = np.append(R, 1)
    R = np.append(R, 1)
    steady_age_R = np.concatenate((np.array([age_R[0]]),
                               (age_R[1:] - age_R[:-1]) * R[:-1]))
    steady_age_R = np.cumsum(steady_age_R)
else:
    age_R = np.array([0, 1e9])
    steady_age_R = age_R
    R = np.array([1, 1])

# ----------------------------------------------------------------
# Loading data from solid_fraction.txt
# ----------------------------------------------------------------

if solid_fraction != None:
    D_depth, D_D = np.loadtxt(datadir+solid_fraction, unpack=True)
    D_depth = np.append(D_depth, D_depth[-1]+0.01)
    D_depth = np.append(D_depth, 10000)
    D_D = np.append(D_D, 1)
    D_D = np.append(D_D, 1)

    D_depth_ie = np.cumsum(np.concatenate((np.array([0]),
                                       D_D[:-1] * (D_depth[1:]-D_depth[:-1]))))
else:
    D_depth_ie = np.array([0, 10000])
    D_depth = D_depth_ie

# -----------------------------------------------------------
# Reading of comparison files for ice cores
# -----------------------------------------------------------
for name in ic:
    if ic[name]['comp'] is not None:
        df = pd.read_csv(datadir+ic[name]['comp'], sep=None, comment='#', engine='python')
        ic[name]['cp_depth'] = df['depth'].to_numpy(dtype=float)
        ic[name]['cp_age'] = df['age'].to_numpy(dtype=float)
        ic[name]['cp_x'] = df['x_origin'].to_numpy(dtype=float)
        ic[name]['cp_tau'] = df['thinning'].to_numpy(dtype=float)
        ic[name]['cp_slope'] = df['slope'].to_numpy(dtype=float)

# -----------------------------------------------------------
# Reading of comparison files for surface velocity and age
# -----------------------------------------------------------
if comp_surf_vel != None:
    cp_surf_vel_x, cp_surf_vel_vel = np.loadtxt(datadir+comp_surf_vel, unpack=True)
if comp_surf_age != None:
    cp_surf_age_x, cp_surf_age_age = np.loadtxt(datadir+comp_surf_age, unpack=True)

# --------------------------
# Interpolation and fluxes
# --------------------------

#x = np.arange(x_left, x_right+x_step, x_step)
x = grid(x_grid)

a = np.interp(x, x_a, a_measure)
m = np.interp(x, x_m, m_measure)
sliding = np.interp(x, x_s, s_measure)
p = np.interp(x, x_p, p_measure)
S = np.interp(x, x_S, S_measure)
H = np.interp(x, x_H, H_measure)
Y = np.interp(x, x_Y, Y_measure)

B = S - H

if obs_bedrock != None:
    obs_bed_depth = np.interp(x, obs_bed_x, obs_bed_depth)
    obs_bed_z = S - obs_bed_depth

x *= 1000

# Computation of total flux Q
dQdx = (x[1:]-x[:-1]) * \
    (a[:-1] * Y[:-1] +
     0.5 * ((a[1:]-a[:-1]) * Y[:-1] + (Y[1:]-Y[:-1])
            * a[:-1]) +
     1./3 * (a[1:]-a[:-1]) * (Y[1:]-Y[:-1]))

dQdx = np.insert(dQdx, 0, Q_left)
Q = np.cumsum(dQdx)
if np.any(Q[1:]<=0):
    print('Warning: total flux becomes negatif at some point! Exiting...')
    sys.exit()
elif Q[0] < 0:
    print('Total flux at start cannot be <0. Exiting...')
    sys.exit()

# Computation of basal melting flux Qm
dQmdx = (x[1:]-x[:-1]) * \
    (m[:-1] * Y[:-1] +
     0.5 * ((m[1:]-m[:-1]) * Y[:-1] + (Y[1:]-Y[:-1])
            * m[:-1]) +
     1./3 * (m[1:]-m[:-1]) * (Y[1:]-Y[:-1]))
dQmdx = np.insert(dQmdx, 0, Qm_left)
Qm = np.cumsum(dQmdx)

mu = Qm / (Q - Qm)
QH = Q - Qm

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    surf_vel = Q / Y * fct_domega_dzeta(1., p) / H


# ----------------------------------------------------------
# DELTA H, H_ie and S_ie
# ----------------------------------------------------------

DELTA_H = D_depth[-1] - D_depth_ie[-1]
print('DELTA_H:', DELTA_H)
S_ie = S - DELTA_H
H_ie = H - DELTA_H

# ------------------------------------
# Determination of accumulation areas
# ------------------------------------

if x_start == None and Q[0]>0: 
    accu_starts = [0]
elif x_start == None and Q[0] == 0:
    accu_starts = [1]
else:
    accu_starts = [np.argmin(np.abs(x_start-x))]
accu_ends = []
for i in range(1, len(x)-1):
    if Q[i] > Q[i-1] and Q[i] >= Q[i+1] : #To be checked
        accu_ends.append(i)
    elif Q[i] <= Q[i-1] and Q[i] < Q[i+1] : #To be checked
        accu_starts.append(i+1)


# --------------------------------------
# Initialisation of trajectories
# --------------------------------------

# Initialisation of trajectories and profiles
trajs = []
profs = []
for i in range(len(x)):
    prof = {'z_ie': np.array([]), 'z': np.array([]), 'steady_age': np.array([]),
            'x0': np.array([]), 'unconf': [], 'a0': np.array([]), 'i0': np.array([]),
            'tau': np.array([]), 'zieb_iso': np.array([]), 'ziea_iso': np.array([]),
            'slope': np.array([])}
    profs.append(prof)

# Initialisation of trajectories from the upstream border
if Q[0]>0:
    zz_start = B[0] + ((S_ie[0] - B[0]) % z_step)
    for z in np.arange(zz_start, S_ie[0], z_step):
        xx = np.array([x[0]])
        zz = np.array([z])
        with warnings.catch_warnings(): #FIXME: why do we need this 'with'?
            warnings.simplefilter("ignore")
            age = np.array([integrate_age((zz-B[0])/H_ie[0], 1, p[0], H_ie[0], a[0], mu[0])])
        zeta = (zz-B[0])/H_ie[0]
        omega = fct_omega(zeta, p[0]) 
        qq = Qm[0] + (Q[0]-Qm[0]) * omega
        gammaint = np.array([ H_ie[0]/a[0]*Q[0]/QH[0]* (1/fct_domega_dzeta(zeta, p[0]) - 1/fct_domega_dzeta(1, p[0])) ])
        tau = np.array([qq/Q[0]])
        traj = {'q': qq, 'x': xx, 'z_ie': zz, 'steady_age': age, 'in': 'upstream',
                'out': None, 'x0': x[0], 'a0': a[0], 'i0': 0, 'gammaint': gammaint,
                'tau': tau, 'zieb_iso': np.array([np.nan]), 'ziea_iso': np.array([]),
                'slope': np.array([0.])}
        trajs.append(traj)

# Initialisation of trajectories from the surface
index_unconf = []
for k in range(len(accu_starts)):
    if k>0:
        index_unconf.append(len(trajs))
    istart = accu_starts[k]
    try:
        iend = accu_ends[k]
    except:
        iend = len(x)-1
    print('Accu area n°', k, 'starts at x=',x[istart],'km and ends at x=', x[iend], 'km')
    for i in range(istart, iend+1):
        xx = np.concatenate([np.ones(i)*np.nan, np.array([x[i]])])
        zz = np.concatenate([np.ones(i)*np.nan, np.array([S_ie[i]])])
        age = np.concatenate([np.ones(i)*np.nan, np.array([0.])])
        gammaint = np.concatenate([np.ones(i)*np.nan, np.array([0.])])
        tau = np.concatenate([np.ones(i)*np.nan, np.array([1.])])
        sie_slope = derivative(S_ie, i)
        slope = np.concatenate([np.ones(i)*np.nan, np.array([sie_slope])])
        zieb_iso = np.concatenate([np.ones(i)*np.nan, np.array([S_ie[i-1]])])
        ziea_iso = np.ones(i)*np.nan
        traj = {'q': Q[i], 'x':xx, 'z_ie':zz, 'steady_age': age, 'in': 'surf', 'out': None,
                'x0': x[i], 'a0': a[i], 'i0': i, 'gammaint': gammaint, 'tau': tau,
                'zieb_iso': zieb_iso, 'ziea_iso': ziea_iso, 'slope': slope}
        trajs.append(traj)

# ------------------------------
# Following of trajectories
# ------------------------------

for traj in trajs:
    i0 = traj['i0']
    qq = traj['q']

    profs[i0]['z_ie'] = np.append(profs[i0]['z_ie'], traj['z_ie'][-1])
    profs[i0]['steady_age'] = np.append(profs[i0]['steady_age'], traj['steady_age'][-1])
    profs[i0]['tau'] = np.append(profs[i0]['tau'], traj['tau'][-1])
    profs[i0]['slope'] = np.append(profs[i0]['slope'], traj['slope'][-1])
    profs[i0]['x0'] = np.append(profs[i0]['x0'], traj['x0'])
    profs[i0]['a0'] = np.append(profs[i0]['a0'], traj['a0'])
    profs[i0]['i0'] = np.append(profs[i0]['i0'], traj['i0'])
    profs[i0]['zieb_iso'] = np.append(profs[i0]['zieb_iso'], traj['zieb_iso'][-1])

    j = i0 + 1
    while j < len(x) :
        traj['x'] = np.append(traj['x'], x[j])
        if Q[j] >= qq and Qm[j] <= qq and traj['out'] == None: 
            omega = (qq-Qm[j])/(Q[j]-Qm[j])
            zeta = fct_zeta(omega, p[j])
            traj['z_ie'] = np.append(traj['z_ie'], B[j] + zeta * H_ie[j])
            zetab = (traj['z_ie'][-2]-B[j-1])/H_ie[j-1]
            kappab = Y[j-1]*H_ie[j-1]/QH[j-1]/fct_domega_dzeta(zetab, p[j-1])
            kappa = Y[j]*H_ie[j]/QH[j]/fct_domega_dzeta(zeta, p[j])
            dage = (traj['x'][-1]-traj['x'][-2]) * (kappa+kappab)/2
            traj['steady_age'] = np.append(traj['steady_age'], traj['steady_age'][-1]+dage)
            dgammaint = (traj['x'][-1]-traj['x'][-2]) * \
                np.sqrt(fct_gamma_int(j, qq, zeta) * fct_gamma_int(j-1, qq, zetab))
            gammaint = traj['gammaint'][-1]+dgammaint
            traj['gammaint'] = np.append(traj['gammaint'], gammaint)
            i0 = traj['i0']
            gamma = a[i0]*fct_domega_dzeta(1, p[i0])*QH[i0]/Q[i0]/H_ie[i0]*gammaint
            #FIXME: define kappa and kappab and reuse for slope
            tau = qq/Q[j] / (fct_domega_dzeta(zeta, p[j])*QH[j]/Q[j]/H_ie[j]) \
                * (fct_domega_dzeta(1, p[i0])*QH[i0]/Q[i0]/H_ie[i0]) / (1+gamma)
            traj['tau'] = np.append(traj['tau'], tau)
            traj_term = (Y[j]*H_ie[j]/QH[j]/fct_domega_dzeta(zeta, p[j])*traj['a0']*qq\
                /Q[j]*QH[i0]/Q[i0]/H_ie[i0]*fct_domega_dzeta(1., p[i0])/(1+gamma) - qq*a[j]*Y[j]/Q[j]**2) #FIXME: factorize Y and 1/Q
            traj_term *= H_ie[j]/fct_domega_dzeta(zeta, p[j])*Q[j]/QH[j]   #FIXME: divide by D
            melt_term = -(m[j]*Q[j]-a[j]*Qm[j])*Y[j]/QH[j]**2*(1-qq/Q[j])*H_ie[j]/fct_domega_dzeta(zeta, p[j]) #FIXME: Divide by D
            flow_term = -fct_domega_dp(zeta, p[j])*derivative(p, j)/fct_domega_dzeta(zeta, p[j])*H_ie[j] #FIXME: Divide by D
            geom_term = derivative(B, j)+zeta*derivative(H_ie, j) #FIXME: divide by D
            slope = traj_term+melt_term+flow_term+geom_term
            traj['slope'] = np.append(traj['slope'], slope)
            zieb_iso = traj['z_ie'][-2]-dage*traj['tau'][-2]*traj['a0']
            traj['zieb_iso'] = np.append(traj['zieb_iso'], zieb_iso)
            ziea_iso = traj['z_ie'][-1]+dage*tau*traj['a0']
            if j == i0+1 and traj['in'] == 'surface':
                ziea_iso = S_ie[i+1]
            traj['ziea_iso'] = np.append(traj['ziea_iso'], ziea_iso)
            if j == len(x)-1:
                traj['out'] = 'downstream'
            profs[j]['z_ie'] = np.append(profs[j]['z_ie'], traj['z_ie'][-1])
            profs[j]['steady_age'] = np.append(profs[j]['steady_age'], traj['steady_age'][-1])
            profs[j]['x0'] = np.append(profs[j]['x0'], traj['x0'])
            profs[j]['a0'] = np.append(profs[j]['a0'], traj['a0'])
            profs[j]['i0'] = np.append(profs[j]['i0'], traj['i0'])
            profs[j]['tau'] = np.append(profs[j]['tau'], traj['tau'][-1])
            profs[j]['slope'] = np.append(profs[j]['slope'], traj['slope'][-1])
            profs[j]['zieb_iso'] = np.append(profs[j]['zieb_iso'], traj['zieb_iso'][-1])
            profs[j-1]['ziea_iso'] = np.append(profs[j-1]['ziea_iso'], traj['ziea_iso'][-1])
            if i0 in accu_starts[1:]:
                profs[j]['unconf'].append(len(profs[j]['z_ie'])-1)
        elif Q[j] < qq or traj['out'] == 'surf':
            if traj['out'] != 'surf':
                profs[j-1]['ziea_iso'] = np.append(profs[j-1]['ziea_iso'], np.nan)
            traj['z_ie'] = np.append(traj['z_ie'], np.nan)
            traj['steady_age'] = np.append(traj['steady_age'], np.nan)
            traj['tau'] = np.append(traj['tau'], np.nan)
            traj['slope'] = np.append(traj['slope'], np.nan)
            traj['out'] = 'surf'
            traj['zieb_iso'] = np.append(traj['zieb_iso'], np.nan)
            traj['ziea_iso'] = np.append(traj['ziea_iso'], np.nan)
        elif Qm[j] > qq or traj['out'] == 'bed':
            if traj['out'] != 'bed':
                profs[j-1]['ziea_iso'] = np.append(profs[j-1]['ziea_iso'], np.nan)
            traj['z_ie'] = np.append(traj['z_ie'], np.nan)
            traj['steady_age'] = np.append(traj['steady_age'], np.nan)
            traj['tau'] = np.append(traj['tau'], np.nan)
            traj['slope'] = np.append(traj['slope'], np.nan)
            traj['out'] = 'bed'
            traj['zieb_iso'] = np.append(traj['zieb_iso'], np.nan)
            traj['ziea_iso'] = np.append(traj['ziea_iso'], np.nan)
        j += 1
    traj['ziea_iso'] = np.append(traj['ziea_iso'], np.nan)
    traj['z'] = S-np.interp(S_ie-traj['z_ie'], D_depth_ie, D_depth)
    traj['zb_iso'] = np.nan*np.ones_like(traj['z'])
    traj['zb_iso'][1:] = S[:-1]-np.interp(S_ie[:-1]-traj['zieb_iso'][1:], D_depth_ie, D_depth)
    traj['za_iso'] = np.nan*np.ones_like(traj['z'])
    traj['za_iso'][:-1] = S[1:]-np.interp(S_ie[1:]-traj['ziea_iso'][:-1], D_depth_ie, D_depth)
    traj['slope_diff'] = np.nan*np.ones_like(traj['z'])
    traj['slope_diff'][1:-1] = (traj['za_iso'][1:-1]-traj['zb_iso'][1:-1])/(x[2:]-x[:-2])
    traj['age'] = np.interp(traj['steady_age'], steady_age_R, age_R)


for i in range(len(x)-1): #FIXME: why -1?
    profs[i]['z'] = S[i]-np.interp(S_ie[i]-profs[i]['z_ie'], D_depth_ie, D_depth)
    if i>=1:
        profs[i]['zb_iso'] = S[i-1]-np.interp(S_ie[i-1]-profs[i]['zieb_iso'], D_depth_ie, D_depth)
        profs[i]['za_iso'] = S[i+1]-np.interp(S_ie[i+1]-profs[i]['ziea_iso'], D_depth_ie, D_depth)
        profs[i]['slope_diff'] = (profs[i]['za_iso']-profs[i]['zb_iso'])/(x[i+1]-x[i-1])
    else:
        profs[i]['zb_iso'] = np.nan*np.ones_like(profs[i]['z'])
        profs[i]['za_iso'] = np.nan*np.ones_like(profs[i]['z'])
        profs[i]['slope_diff'] = np.nan*np.ones_like(profs[i]['z'])
    profs[i]['age'] = np.interp(profs[i]['steady_age'], steady_age_R, age_R)
#    profs[i]['tau'] = (profs[i]['age'][1:]-profs[i]['age'][:-1]) / (profs[i]['a0'][1:]+profs[i]['a0'][:-1]) *2
    if comp_isochrones:
        profs[i]['cp_iso_depth'] = np.zeros(cp_iso_nb)
        profs[i]['cp_iso_z'] = np.zeros(cp_iso_nb)
        for j in range(cp_iso_nb):
            profs[i]['cp_iso_depth'][j] = np.interp(x[i], cp_iso_x, cp_iso_depth[j,:])
            profs[i]['cp_iso_z'][j] = np.interp(x[i]/1000., cp_iso_x, cp_iso_z[j, :])

    
# ----------------------------------------
# Computation of surface quantities
# ----------------------------------------

surf_steady_age = np.zeros_like(x)
surf_tau = np.zeros_like(x)
surf_slope = np.zeros_like(x)
for i in range(len(x)):
    try: #FIXME: can we avoid this try by being smarter?
        surf_steady_age[i] = interp1d(profs[i]['z'], profs[i]['steady_age'], kind='linear', fill_value="extrapolate")(S[i])
        surf_tau[i] = interp1d(profs[i]['z'], profs[i]['tau'], kind='linear', fill_value="extrapolate")(S[i])
        surf_slope[i] = interp1d(profs[i]['z'], profs[i]['slope'], kind='linear', fill_value="extrapolate")(S[i])
    except:
        surf_steady_age[i] = np.nan
        surf_tau[i] = np.nan
        surf_slope[i] = np.nan
surf_age = np.interp(surf_steady_age, steady_age_R, age_R)        

print('Before creating figures:',
      round(time.perf_counter()-START_TIME, 4), 's.')        

# ----------------------------------------
# Preparing for figures
# ----------------------------------------
# FIXME: can we make a function here instead of repeating for every variable?


# Concatenation in lists for trajectories plots
x_list = []
z_list = []
age_list = []
tau_list = []
slope_list = []
in_list = []
for i in range(0, len(trajs), step_traj):
    traj = trajs[i]
    x_list.append(traj['x'])
    z_list.append(traj['z'])
    age_list.append(traj['age'])
    tau_list.append(traj['tau'])
    slope_list.append(traj['slope'])
    in_list.append(traj['in'])

# Creation of domains and unformities

mat_x = np.array([trajs[0]['x']])
mat_z = np.array([trajs[0]['z']])
mat_age = np.array([trajs[0]['age']])
mat_tau = np.array([trajs[0]['tau']])
mat_slope = np.array([trajs[0]['slope']])
if len(index_unconf) ==0:
    end = len(trajs)
else:
    end = index_unconf[0]
for i in range(1, end):
#    print(np.shape(mat_x), np.shape(np.array([trajs[i]['x']])), trajs[i]['z'][0])
    mat_x = np.concatenate([mat_x, np.array([trajs[i]['x']])])
    mat_z = np.concatenate([mat_z, np.array([trajs[i]['z']])])
    mat_age = np.concatenate([mat_age, np.array([trajs[i]['age']])])
    mat_tau = np.concatenate([mat_tau, np.array([trajs[i]['tau']])])
    mat_slope = np.concatenate([mat_slope, np.array([trajs[i]['slope']])])
domains_list = [{'mat_x': mat_x, 'mat_z': mat_z, 'mat_age': mat_age,
                 'mat_tau': mat_tau, 'mat_slope': mat_slope}]
unconf_list = []


for k in range(len(index_unconf)):
    traj_z = trajs[index_unconf[k]]['z']  # FIXME: average with traj below, but not so easy...
    unconf_list.append(traj_z)
    mat_x = np.array([trajs[index_unconf[k]]['x']])
    mat_z = np.array([trajs[index_unconf[k]]['z']])
    mat_age = np.array([trajs[index_unconf[k]]['age']])
    mat_tau = np.array([trajs[index_unconf[k]]['tau']])
    mat_slope = np.array([trajs[index_unconf[k]]['slope']])
    if k == len(index_unconf)-1:
        end = len(trajs)
    else:
        end = index_unconf[k+1]
    for i in range(index_unconf[k]+1, end):
    #    print(np.shape(mat_x), np.shape(np.array([trajs[i]['x']])), trajs[i]['z'][0])
        mat_x = np.concatenate([mat_x, np.array([trajs[i]['x']])])
        mat_z = np.concatenate([mat_z, np.array([trajs[i]['z']])])
        mat_age = np.concatenate([mat_age, np.array([trajs[i]['age']])])
        mat_tau = np.concatenate([mat_tau, np.array([trajs[i]['tau']])])
        mat_slope = np.concatenate([mat_slope, np.array([trajs[i]['slope']])])
    domains_list.append({'mat_x': mat_x, 'mat_z': mat_z, 'mat_age': mat_age, 'mat_tau': mat_tau, 'mat_slope': mat_slope})


# Quantities for ice cores
     
for name in ic:
    ic[name]['XX'] = ic[name]['x'] * np.ones(2)
    ic[name]['S'] = np.interp(ic[name]['x'], x, S)
    ic[name]['ZZ'] = np.array([ic[name]['S'], ic[name]['S']-ic[name]['max_depth']])
    ic[name]['DD'] = np.array([0., ic[name]['max_depth']])

# ---------------------
# Trajectories figure
# ---------------------

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, 
                               figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x/1000., a*R[0], color='k', linewidth=2)
ax1.set_ylabel('PD accu. (m/yr)', fontsize=14)
ax1.plot(x/1000., np.zeros_like(x), color='k', linewidth=2, linestyle='--')

ax1b = ax1.twinx()
ax1b.plot(x/1000., surf_vel*R[0], color='r', linewidth=2)
ax1b.set_ylabel('PD surf. vel. (m/yr)', color='r', fontsize=14)
if comp_surf_vel != None:
    ax1b.plot(cp_surf_vel_x, cp_surf_vel_vel, color='r', linewidth=2, linestyle='--')  
ax1b.tick_params(axis='y', colors='r')

ax2.plot(x/1000., S, label='Surface', color='k', linewidth=2.)
if obs_bedrock is not None:
    ax2.fill_between(x/1000., obs_bed_z, B, color='grey', label='Stagnant ice')
    ax2.plot(x/1000., obs_bed_z, label='Bedrock', color='k', linewidth=2.)
else:
    ax2.plot(x/1000., B, label='Bedrock', color='k', linewidth=2.)

ax2top = ax2.twiny()
ax2top.set_xlabel(r"direction of the flow $\longrightarrow$", fontsize=20)
ax2top.xaxis.set_ticks([])

for xx, zz, iin in zip(x_list, z_list, in_list):
    if iin == 'upstream': 
        ax2.plot(xx/1000., zz, color='k')
    elif iin == 'surf':
        ax2.plot(xx/1000., zz, color='b')

for k in range(len(index_unconf)):
    traj = trajs[index_unconf[k]]
    ax2.plot(traj['x']/1000., traj['z'], label='Discontinuities', color='k', linewidth=4., linestyle='--')

ax2.set_xlabel('horizontal distance (km)', fontsize=19)
ax2.set_ylabel('elevation (m)', fontsize=19)

for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], ic[name]['S']), ha='center',
                 va='bottom', color=color_core)

plt.savefig(datadir+'trajectories_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()



# -----------------------------
# Trajectories figure vs depth
# -----------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, 
                               figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x/1000., a*R[0], color='k', linewidth=2)
ax1.set_ylabel('PD accu. (m/yr)', fontsize=14)
ax1.plot(x/1000., np.zeros_like(x), color='k', linewidth=2, linestyle='--')

ax1b = ax1.twinx()
ax1b.plot(x/1000., surf_vel*R[0], color='r', linewidth=2)
ax1b.set_ylabel('PD surf. vel. (m/yr)', color='r', fontsize=14)
if comp_surf_vel != None:
    ax1b.plot(cp_surf_vel_x, cp_surf_vel_vel, color='r', linewidth=2, linestyle='--')
ax1b.tick_params(axis='y', colors='r')

ax2.plot(x/1000., np.zeros_like(S), label='Surface', color='k', linewidth=2.)
if obs_bedrock is not None:
    ax2.fill_between(x/1000., obs_bed_depth, S-B, color='grey', label='Stagnant ice')
    ax2.plot(x/1000., obs_bed_depth, label='Bedrock', color='k', linewidth=2.)
else:
    ax2.plot(x/1000., S-B, label='Bedrock', color='k', linewidth=2.)

ax2top = ax2.twiny()
ax2top.set_xlabel(r"direction of the flow $\longrightarrow$", fontsize=20)
ax2top.xaxis.set_ticks([])

for xx, zz, iin in zip(x_list, z_list, in_list):
    if iin == 'upstream': 
        ax2.plot(xx/1000., S-zz, color='k')
    elif iin == 'surf':
        ax2.plot(xx/1000., S-zz, color='b')

for k in range(len(index_unconf)):
    traj = trajs[index_unconf[k]]
    ax2.plot(traj['x']/1000., S-traj['z'], label='Discontinuities', color='k', linewidth=4., linestyle='--')

ax2.set_xlabel('horizontal distance (km)', fontsize=19)
ax2.set_ylabel('depth (m)', fontsize=19)
ax2.set_ylim(ax2.get_ylim()[::-1])

for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['DD'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], 0.), ha='center',
                 va='bottom', color=color_core)

plt.savefig(datadir+'trajectories_x_depth.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()

print('After simple trajectories figures:',
      round(time.perf_counter()-START_TIME, 4), 's.')        


# ---------------------------
# Color trajectories figure
# ---------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, 
                               figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x/1000., a*R[0], color='k', linewidth=2)
ax1.set_ylabel('PD accu. (m/yr)', fontsize=14)
ax1.plot(x/1000., np.zeros_like(x), color='k', linewidth=2, linestyle='--')

ax1b = ax1.twinx()
ax1b.plot(x/1000., surf_vel*R[0], color='r', linewidth=2)
ax1b.set_ylabel('PD surf. vel. (m/yr)', color='r', fontsize=14)
if comp_surf_vel != None:
    ax1b.plot(cp_surf_vel_x, cp_surf_vel_vel, color='r', linewidth=2, linestyle='--')
ax1b.tick_params(axis='y', colors='r')

age_min = 0.
age_max = max(np.nanmax(age/1000.) for age in age_list)
# print(age_max)
linthresh = age_max/scaling_age  # Seuil en dessous duquel l'échelle devient linéaire (évite log(0))
norm = SymLogNorm(linthresh=linthresh, linscale=1, vmin=age_min, vmax=age_max)

ax2.plot(x/1000., S, label='Surface', color='k', linewidth=2.)
if obs_bedrock is not None:
    ax2.fill_between(x/1000., obs_bed_z, B, color='grey', label='Stagnant ice')
    ax2.plot(x/1000., obs_bed_z, label='Bedrock', color='k', linewidth=2.)
else:
    ax2.plot(x/1000., B, label='Bedrock', color='k', linewidth=2.)

ax2top = ax2.twiny()
ax2top.set_xlabel(r"direction of the flow $\longrightarrow$", fontsize=20)
ax2top.xaxis.set_ticks([])

for xx, zz, aa in zip(x_list, z_list, age_list):
    # Créer des segments entre les points consécutifs
    points = np.array([xx/1000., zz]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Créer une LineCollection pour cette ligne
    lc = LineCollection(segments, cmap='plasma_r', norm=norm)
    lc.set_array(aa/1000.)  # Associer les valeurs z aux segments
    lc.set_linewidth(2)
    ax2.add_collection(lc)

for k in range(len(index_unconf)):
    traj = trajs[index_unconf[k]]
    ax2.plot(traj['x']/1000., traj['z'], label='Discontinuities', color='k', linewidth=4., linestyle='--')

ax2.set_xlabel('horizontal distance (km)', fontsize=19)
ax2.set_ylabel('elevation (m)', fontsize=19)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='plasma_r', norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
cb.set_label(label='age (kyr)', size=18)

for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], ic[name]['S']), ha='center',
                 va='bottom', color=color_core)


plt.savefig(datadir+'trajectories_color_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()


# -----------------------------------
# Color trajectories figure vs depth 
# -----------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, 
                               figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x/1000., a*R[0], color='k', linewidth=2)
ax1.set_ylabel('PD accu. (m/yr)', fontsize=14)
ax1.plot(x/1000., np.zeros_like(x), color='k', linewidth=2, linestyle='--')

ax1b = ax1.twinx()
ax1b.plot(x/1000., surf_vel*R[0], color='r', linewidth=2)
ax1b.set_ylabel('PD surf. vel. (m/yr)', color='r', fontsize=14)
if comp_surf_vel != None:
    ax1b.plot(cp_surf_vel_x, cp_surf_vel_vel, color='r', linewidth=2, linestyle='--')
ax1b.tick_params(axis='y', colors='r')

age_min = 0.
age_max = max(np.nanmax(age/1000.) for age in age_list)
# print(age_max)
linthresh = age_max/scaling_age  # Seuil en dessous duquel l'échelle devient linéaire (évite log(0))
norm = SymLogNorm(linthresh=linthresh, linscale=1, vmin=age_min, vmax=age_max)

ax2.plot(x/1000., np.zeros_like(S), label='Surface', color='k', linewidth=2.)
if obs_bedrock is not None:
    ax2.fill_between(x/1000., obs_bed_depth, S-B, color='grey', label='Stagnant ice')
    ax2.plot(x/1000., obs_bed_depth, label='Bedrock', color='k', linewidth=2.)
else:
    ax2.plot(x/1000., S-B, label='Bedrock', color='k', linewidth=2.)

ax2top = ax2.twiny()
ax2top.set_xlabel(r"direction of the flow $\longrightarrow$", fontsize=20)
ax2top.xaxis.set_ticks([])

for xx, zz, aa in zip(x_list, z_list, age_list):
    # Créer des segments entre les points consécutifs
    points = np.array([xx/1000., S-zz]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Créer une LineCollection pour cette ligne
    lc = LineCollection(segments, cmap='plasma_r', norm=norm)
    lc.set_array(aa/1000.)  # Associer les valeurs z aux segments
    lc.set_linewidth(2)
    ax2.add_collection(lc)

for k in range(len(index_unconf)):
    traj = trajs[index_unconf[k]]
    ax2.plot(traj['x']/1000., S-traj['z'], label='Discontinuities', color='k', linewidth=4., linestyle='--')

ax2.set_xlabel('horizontal distance (km)', fontsize=19)
ax2.set_ylabel('depth (m)', fontsize=19)
ax2.set_ylim(ax2.get_ylim()[::-1])

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='plasma_r', norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
cb.set_label(label='age (kyr)', size=18)

for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['DD'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], 0.), ha='center',
                 va='bottom', color=color_core)


plt.savefig(datadir+'trajectories_color_x_depth.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()


print('After color trajectories figures:',
      round(time.perf_counter()-START_TIME, 4), 's.')        


# -----------
# Age figure
# -----------

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, 
                               figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x/1000., a*R[0], color='k', linewidth=2)
ax1.set_ylabel('PD accu. (m/yr)', fontsize=14)
ax1.plot(x/1000., np.zeros_like(x), color='k', linewidth=2, linestyle='--')

ax1b = ax1.twinx()
ax1b.plot(x/1000., surf_age/1000., color='r', linewidth=2)
if comp_surf_age != None:
    ax1b.plot(cp_surf_age_x, cp_surf_age_age/1000., color='r', linestyle='None', marker='o')
ax1b.set_ylabel('surface age (kyr)', color='r', fontsize=14)
ax1b.tick_params(axis='y', colors='r')
bottom, top = ax1b.get_ylim()
if top < 1.:
    top = 1.
ax1b.set_ylim(bottom, top)

age_min = 0.
age_max = max(np.nanmax(age/1000.) for age in age_list)
linthresh = age_max/scaling_age  # Seuil en dessous duquel l'échelle devient linéaire (évite log(0))
norm = SymLogNorm(linthresh=linthresh, linscale=1, vmin=age_min, vmax=age_max)
levels_iso = np.array(fig_age_iso)

ax2.plot(x/1000., S, label='Surface', color='k', linewidth=2.)
if obs_bedrock is not None:
    ax2.fill_between(x/1000., obs_bed_z, B, color='grey', label='Stagnant ice')
    ax2.plot(x/1000., obs_bed_z, label='Bedrock', color='k', linewidth=2.)
else:
    ax2.plot(x/1000., B, label='Bedrock', color='k', linewidth=2.)

ax2top = ax2.twiny()
ax2top.set_xlabel(r"direction of the flow $\longrightarrow$", fontsize=20)
ax2top.xaxis.set_ticks([])

if comp_isochrones is not None:
    for i in range(cp_iso_nb):
        ax2.plot(cp_iso_x, cp_iso_z[i, :],
                 color='w', linewidth=1)

normalized_levels = np.linspace(0, 1, 100)
levels = norm.inverse(normalized_levels)
for dom in domains_list:
    ax2.contourf(dom['mat_x']/1000., dom['mat_z'], dom['mat_age']/1000., levels=levels,
                 cmap='plasma_r', norm=norm)
    cp = ax2.contour(dom['mat_x']/1000., dom['mat_z'], dom['mat_age']/1000., levels=levels_iso, colors='k',
                     norm=norm)

for unconf in unconf_list:
    ax2.plot(x/1000., unconf, label='Discontinuities', color='k', linewidth=4.,
             linestyle='--')




ax2.set_xlabel('horizontal distance (km)', fontsize=19)
ax2.set_ylabel('elevation (m)', fontsize=19)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='plasma_r', norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
cb.set_label(label='age (kyr)', size=18)
cb.add_lines(cp)


for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], ic[name]['S']), ha='center',
                 va='bottom', color=color_core)

plt.savefig(datadir+'age_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()


# --------------------
# Age figure vs depth 
# --------------------

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, 
                               figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x/1000., a*R[0], color='k', linewidth=2)
ax1.set_ylabel('PD accu. (m/yr)', fontsize=14)
ax1.plot(x/1000., np.zeros_like(x), color='k', linewidth=2, linestyle='--')

ax1b = ax1.twinx()
ax1b.plot(x/1000., surf_age/1000., color='r', linewidth=2)
if comp_surf_age != None:
    ax1b.plot(cp_surf_age_x, cp_surf_age_age/1000., color='r', linestyle='None', marker='o')
ax1b.set_ylabel('surface age (kyr)', color='r', fontsize=14)
ax1b.tick_params(axis='y', colors='r')
bottom, top = ax1b.get_ylim()
if top < 1.:
    top = 1.
ax1b.set_ylim(bottom, top)

age_min = 0.
age_max = max(np.nanmax(age/1000.) for age in age_list)
linthresh = age_max/scaling_age  # Seuil en dessous duquel l'échelle devient linéaire (évite log(0))
norm = SymLogNorm(linthresh=linthresh, linscale=1, vmin=age_min, vmax=age_max)
levels_iso = np.array(fig_age_iso)

ax2.plot(x/1000., np.zeros_like(S), label='Surface', color='k', linewidth=2.)
if obs_bedrock is not None:
    ax2.fill_between(x/1000., obs_bed_depth, S-B, color='grey', label='Stagnant ice')
    ax2.plot(x/1000., obs_bed_depth, label='Bedrock', color='k', linewidth=2.)
else:
    ax2.plot(x/1000., S-B, label='Bedrock', color='k', linewidth=2.)

ax2top = ax2.twiny()
ax2top.set_xlabel(r"direction of the flow $\longrightarrow$", fontsize=20)
ax2top.xaxis.set_ticks([])

if comp_isochrones is not None:
    for i in range(cp_iso_nb):
        ax2.plot(cp_iso_x, cp_iso_depth[i, :],
                 color='w', linewidth=1)
        
normalized_levels = np.linspace(0, 1, 100)
levels = norm.inverse(normalized_levels)
for dom in domains_list:
    ax2.contourf(dom['mat_x']/1000., S-dom['mat_z'], dom['mat_age']/1000., levels=levels,
                 cmap='plasma_r', norm=norm)
    cp = ax2.contour(dom['mat_x']/1000., S-dom['mat_z'], dom['mat_age']/1000., levels=levels_iso, colors='k',
                     norm=norm)

for unconf in unconf_list:
    ax2.plot(x, S-unconf, label='Discontinuities', color='k', linewidth=4.,
             linestyle='--')


ax2.set_xlabel('horizontal distance (km)', fontsize=18)
ax2.set_ylabel('depth (m)', fontsize=18)
ax2.set_ylim(ax2.get_ylim()[::-1])

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='plasma_r', norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
cb.set_label(label='age (kyr)', size=18)
cb.add_lines(cp)


for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['DD'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], 0.), ha='center',
                 va='bottom', color=color_core)

plt.savefig(datadir+'age_x_depth.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()


print('After age figures:',
      round(time.perf_counter()-START_TIME, 4), 's.')        


# -----------------
# thinning figure
# -----------------

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, 
                               figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x/1000., a*R[0], color='k', linewidth=2)
ax1.set_ylabel('PD accu. (m/yr)', fontsize=14)
ax1.plot(x/1000., np.zeros_like(x), color='k', linewidth=2, linestyle='--')

ax1b = ax1.twinx()
ax1b.plot(x/1000., surf_tau, color='r', linewidth=2)
ax1b.set_ylabel('surface thinning', color='r', fontsize=14)
ax1b.tick_params(axis='y', colors='r')
bottom, top = ax1b.get_ylim()
if bottom > 0.9 and top < 1.1:
    bottom = 0.9
    top = 1.1
ax1b.set_ylim(bottom, top)

tau_min = 0.
tau_max = max(np.nanmax(tau) for tau in tau_list)
norm = Normalize(vmin=tau_min, vmax=tau_max)

ax2.plot(x/1000., S, label='Surface', color='k', linewidth=2.)
if obs_bedrock is not None:
    ax2.fill_between(x/1000., obs_bed_z, B, color='grey', label='Stagnant ice')
    ax2.plot(x/1000., obs_bed_z, label='Bedrock', color='k', linewidth=2.)
else:
    ax2.plot(x/1000., B, label='Bedrock', color='k', linewidth=2.)

ax2top = ax2.twiny()
ax2top.set_xlabel(r"direction of the flow $\longrightarrow$", fontsize=20)
ax2top.xaxis.set_ticks([])

levels = np.arange(tau_min, tau_max, 0.1)
for dom in domains_list:
    ax2.contourf(dom['mat_x']/1000., dom['mat_z'], dom['mat_tau'], levels=100,
                 cmap='plasma', norm=norm)
    cp = ax2.contour(dom['mat_x']/1000., dom['mat_z'], dom['mat_tau'], levels=levels,
                     colors='k', norm=norm)

for unconf in unconf_list:
    ax2.plot(x/1000., unconf, label='Discontinuities', color='k', linewidth=4.,
             linestyle='--')

ax2.set_xlabel('horizontal distance (km)', fontsize=19)
ax2.set_ylabel('elevation (m)', fontsize=19)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
cb.set_label(label='vertical thinning ratio (no unit)', size=18)
cb.add_lines(cp)


for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], ic[name]['S']), ha='center',
                 va='bottom', color=color_core)

plt.savefig(datadir+'thinning_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()


# ----------------------------
# thinning figure vs depth
# ----------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, 
                               figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x/1000., a*R[0], color='k', linewidth=2)
ax1.set_ylabel('PD accu. (m/yr)', fontsize=14)
ax1.plot(x/1000., np.zeros_like(x), color='k', linewidth=2, linestyle='--')

ax1b = ax1.twinx()
ax1b.plot(x/1000., surf_tau, color='r', linewidth=2)
ax1b.set_ylabel('surface thinning', color='r', fontsize=14)
ax1b.tick_params(axis='y', colors='r')
bottom, top = ax1b.get_ylim()
if bottom > 0.9:
    bottom = 0.9
    top = 1.1
ax1b.set_ylim(bottom, top)

tau_min = 0.
tau_max = max(np.nanmax(tau) for tau in tau_list)
norm = Normalize(vmin=tau_min, vmax=tau_max)

ax2.plot(x/1000., np.zeros_like(S), label='Surface', color='k', linewidth=2.)
if obs_bedrock is not None:
    ax2.fill_between(x/1000., obs_bed_depth, S-B, color='grey', label='Stagnant ice')
    ax2.plot(x/1000., obs_bed_depth, label='Bedrock', color='k', linewidth=2.)
else:
    ax2.plot(x/1000., S-B, label='Bedrock', color='k', linewidth=2.)

ax2top = ax2.twiny()
ax2top.set_xlabel(r"direction of the flow $\longrightarrow$", fontsize=20)
ax2top.xaxis.set_ticks([])

levels = np.arange(tau_min, tau_max, 0.1)
for dom in domains_list:
    ax2.contourf(dom['mat_x']/1000., S-dom['mat_z'], dom['mat_tau'], levels=100,
                 cmap='plasma', norm=norm)
    ax2.contour(dom['mat_x']/1000., S-dom['mat_z'], dom['mat_tau'], levels=levels,
                 colors='k', norm=norm)

for unconf in unconf_list:
    ax2.plot(x/1000., S-unconf, label='Discontinuities', color='k', linewidth=4.,
             linestyle='--')

ax2.set_xlabel('horizontal distance (km)', fontsize=19)
ax2.set_ylabel('depth (m)', fontsize=19)
ax2.set_ylim(ax2.get_ylim()[::-1])

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
cb.set_label(label='vertical thinning ratio (no unit)', size=18)
#cb.add_lines(cp)


for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['DD'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], 0.), ha='center',
                 va='bottom', color=color_core)

plt.savefig(datadir+'thinning_x_depth.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()


print('After thinning figures:',
      round(time.perf_counter()-START_TIME, 4), 's.')        




# -----------------
# slope figure
# -----------------

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, 
                               figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x/1000., a*R[0], color='k', linewidth=2)
ax1.set_ylabel('PD accu. (m/yr)', fontsize=14)
ax1.plot(x/1000., np.zeros_like(x), color='k', linewidth=2, linestyle='--')

ax1b = ax1.twinx()
ax1b.plot(x/1000., surf_slope, color='r', linewidth=2)
ax1b.set_ylabel('surf. layer slope', color='r', fontsize=14)
ax1b.tick_params(axis='y', colors='r')
bottom, top = ax1b.get_ylim()
if bottom > 0.9 and top < 1.1:
    bottom = 0.9
    top = 1.1
ax1b.set_ylim(bottom, top)

if fig_slope_maxrange == None:
    slope_min = min(np.nanmin(slope) for slope in slope_list)
    slope_max = max(np.nanmax(slope) for slope in slope_list)
    slope_max = max(-slope_min, slope_max)
else:
    slope_max = fig_slope_maxrange
norm = Normalize(vmin=-slope_max, vmax=slope_max)

ax2.plot(x/1000., S, label='Surface', color='k', linewidth=2.)
if obs_bedrock is not None:
    ax2.fill_between(x/1000., obs_bed_z, B, color='grey', label='Stagnant ice')
    ax2.plot(x/1000., obs_bed_z, label='Bedrock', color='k', linewidth=2.)
else:
    ax2.plot(x/1000., B, label='Bedrock', color='k', linewidth=2.)

ax2top = ax2.twiny()
ax2top.set_xlabel(r"direction of the flow $\longrightarrow$", fontsize=20)
ax2top.xaxis.set_ticks([])

#levels = np.arange(tau_min, tau_max, 0.1)
for dom in domains_list:
    ax2.contourf(dom['mat_x']/1000., dom['mat_z'], dom['mat_slope'], levels=100,
                 cmap='PuOr', norm=norm)
    # cp = ax2.contour(dom['mat_x'], dom['mat_z'], dom['mat_slope'], levels=levels,
    #                  colors='k', norm=norm)

for unconf in unconf_list:
    ax2.plot(x/1000., unconf, label='Discontinuities', color='k', linewidth=4.,
             linestyle='--')

ax2.set_xlabel('horizontal distance (km)', fontsize=19)
ax2.set_ylabel('elevation (m)', fontsize=19)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='PuOr', norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
cb.set_label(label='layer slope (no unit)', size=18)
#cb.add_lines(cp)


for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], ic[name]['S']), ha='center',
                 va='bottom', color=color_core)

plt.savefig(datadir+'slope_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()


# ----------------------------
# slope figure vs depth
# ----------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, 
                               figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x/1000., a*R[0], color='k', linewidth=2)
ax1.set_ylabel('PD accu. (m/yr)', fontsize=14)
ax1.plot(x/1000., np.zeros_like(x), color='k', linewidth=2, linestyle='--')

ax1b = ax1.twinx()
ax1b.plot(x/1000., surf_tau, color='r', linewidth=2)
ax1b.set_ylabel('surf. layer slope', color='r', fontsize=14)
ax1b.tick_params(axis='y', colors='r')
bottom, top = ax1b.get_ylim()
if bottom > 0.9:
    bottom = 0.9
    top = 1.1
ax1b.set_ylim(bottom, top)

if fig_slope_maxrange == None:
    slope_min = min(np.nanmin(slope) for slope in slope_list)
    slope_max = max(np.nanmax(slope) for slope in slope_list)
    slope_max = max(-slope_min, slope_max)
else:
    slope_max = fig_slope_maxrange
norm = Normalize(vmin=-slope_max, vmax=slope_max)

ax2.plot(x/1000., np.zeros_like(S), label='Surface', color='k', linewidth=2.)
if obs_bedrock is not None:
    ax2.fill_between(x/1000., obs_bed_depth, S-B, color='grey', label='Stagnant ice')
    ax2.plot(x/1000., obs_bed_depth, label='Bedrock', color='k', linewidth=2.)
else:
    ax2.plot(x/1000., S-B, label='Bedrock', color='k', linewidth=2.)

ax2top = ax2.twiny()
ax2top.set_xlabel(r"direction of the flow $\longrightarrow$", fontsize=20)
ax2top.xaxis.set_ticks([])

#levels = np.arange(tau_min, tau_max, 0.1)
for dom in domains_list:
    ax2.contourf(dom['mat_x']/1000., S-dom['mat_z'], dom['mat_slope'], levels=100,
                 cmap='PuOr', norm=norm)
    # ax2.contour(dom['mat_x'], S-dom['mat_z'], dom['mat_slope'], levels=levels,
    #              colors='k', norm=norm)

for unconf in unconf_list:
    ax2.plot(x/1000., S-unconf, label='Discontinuities', color='k', linewidth=4.,
             linestyle='--')

ax2.set_xlabel('horizontal distance (km)', fontsize=19)
ax2.set_ylabel('depth (m)', fontsize=19)
ax2.set_ylim(ax2.get_ylim()[::-1])

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='PuOr', norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
cb.set_label(label='layer slope (no unit)', size=18)
cb.add_lines(cp)


for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['DD'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], 0.), ha='center',
                 va='bottom', color=color_core)

plt.savefig(datadir+'slope_x_depth.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()


print('After slope figures:',
      round(time.perf_counter()-START_TIME, 4), 's.')        



# ----------------------------
# ice cores figures
# ----------------------------

for name in ic:
    ii = np.argmin(np.abs(x/1000. - ic[name]['x'])) # FIXME: interpolation is complicated...

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title(name + ' ice core', fontsize=20)
    prof_domains = profs[ii]['unconf'].copy()
    prof_domains.insert(0, 0)
    prof_domains.append(len(profs[ii]['z']))

    for i in range(len(prof_domains)-1):
        ax.plot(profs[ii]['age'][prof_domains[i]: prof_domains[i+1]]/1000., profs[ii]['z'][prof_domains[i]: prof_domains[i+1]], color='b')
    if ic[name]['comp'] != None:
        ax.plot(ic[name]['cp_age']/1000., S[ii]-ic[name]['cp_depth'], color='b', linestyle='--')
    if comp_isochrones is not None and comp_iso_ages is not None:
        ax.plot(cp_iso_ages/1000., profs[ii]['cp_iso_z'], color='b', linestyle='None', marker='o', markersize=5)
    ax.set_xlabel('age (kyr)', color='b', fontsize=15)
    ax.set_ylabel('elevation (m)', fontsize=15)
    ax.set_ylim(top=S[ii], bottom=S[ii]-ic[name]['max_depth'])
    jj = np.argmin(np.abs(profs[ii]['z']-(S[ii]-ic[name]['max_depth'])))
    max_age = profs[ii]['age'][jj]/1000.
    ax.set_xlim(left=0, right=max_age)
    ax.spines['bottom'].set_color('b')
    ax.tick_params(axis='x', colors='b')
    ax.hlines(profs[ii]['z'][prof_domains[1:-1]], 0., max_age, linestyles='--', linewidth=3., color = 'k')
    axb = ax.twinx()
    axb.set_ylim(top=0., bottom=S[ii]-B[ii])
    axb.set_ylabel('depth (m)', fontsize=15)
    
    ax2 = ax.twiny()
    for i in range(len(prof_domains)-1):
        ax2.plot(profs[ii]['x0'][prof_domains[i]: prof_domains[i+1]], profs[ii]['z'][prof_domains[i]: prof_domains[i+1]], color='r')
    if ic[name]['comp'] != None:
        ax2.plot(ic[name]['cp_x'], S[ii]-ic[name]['cp_depth'], color='r', linestyle='--')
    ax2.set_xlim(left=x_left)
    ax2.set_xlim(right=ic[name]['x'])
    ax2.set_xlabel('ice origin (km)', color='r', fontsize=15)
    ax2.spines['top'].set_color('r')
    ax2.tick_params(axis='x', colors='r')
    ax2.spines['bottom'].set_visible(False)

    ax3=ax.twiny()
    for i in range(len(prof_domains)-1):
        # tau = (profs[ii]['z_ie'][prof_domains[i]+1:prof_domains[i+1]]-profs[ii]['z_ie'][prof_domains[i]:prof_domains[i+1]-1]) / \
        #       (-profs[ii]['steady_age'][prof_domains[i]+1:prof_domains[i+1]]+profs[ii]['steady_age'][prof_domains[i]:prof_domains[i+1]-1]) / \
        #       (profs[ii]['a0'][prof_domains[i]+1:prof_domains[i+1]]+profs[ii]['a0'][prof_domains[i]:prof_domains[i+1]-1]) *2
        # zz = (profs[ii]['z'][prof_domains[i]+1:prof_domains[i+1]]+profs[ii]['z'][prof_domains[i]:prof_domains[i+1]-1] ) /2
        # ax3.plot(tau, zz, color='orange', linestyle='dotted')
        ax3.plot(profs[ii]['tau'][prof_domains[i]: prof_domains[i+1]], profs[ii]['z'][prof_domains[i]: prof_domains[i+1]], color='orange')
    if ic[name]['comp'] != None:
        ax3.plot(ic[name]['cp_tau'], S[ii]-ic[name]['cp_depth'], color='orange', linestyle='--')
    ax3.set_xlabel('vertical thinning ratio (no unit)', color='orange', fontsize=15)
    ax3.spines['top'].set_color('orange')
    ax3.spines['top'].set_position(('axes', 1.1))
    ax3.tick_params(axis='x', colors='orange')
    ax3.set_xlim(left=0)

    if fig_ic_slope:
        ax4=ax.twiny()
        for i in range(len(prof_domains)-1):
            # ax4.plot(profs[ii]['slope_diff'][prof_domains[i]: prof_domains[i+1]],
            #          profs[ii]['z'][prof_domains[i]: prof_domains[i+1]], color='violet', linestyle='dotted')
            ax4.plot(profs[ii]['slope'][prof_domains[i]: prof_domains[i+1]],
                     profs[ii]['z'][prof_domains[i]: prof_domains[i+1]],
                     color='violet')
        if ic[name]['comp'] != None:
            ax4.plot(ic[name]['cp_slope'], S[ii]-ic[name]['cp_depth'], color='violet', linestyle='--')
        ax4.set_xlabel('layer slope (no unit)', color='violet', fontsize=15)
        ax4.spines['top'].set_color('violet')
        ax4.spines['top'].set_position(('axes', 1.2))
        ax4.tick_params(axis='x', colors='violet')

    plt.savefig(datadir+name+'_ice_core.'+fig_format,
                    format=fig_format, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------
# R(t) - Age
# ---------------------------------------------------------------------

if temp_factor != None:
    fig, ax = plt.subplots()
    plt.stairs(R[:-2], age_R[:-1]/1000, baseline=None)
    plt.xlabel('time (kyr)', fontsize=15)
    plt.ylabel(r'$R(t) (relative unit)$', fontsize=15)
    plt.savefig(datadir+'temporal_factor.'+fig_format,
                format=fig_format, bbox_inches='tight')
    plt.close()


# ----------------------------------------------------------------------
# OUTPUTS
# ----------------------------------------------------------------------

# Flow line outputs
if output_fl:
    output = np.vstack((x/1000., Y, a*R[0], m*R[0], Q, Qm, surf_vel))
    np.savetxt(datadir+'flow_line_output.txt', np.transpose(output),
               header='x(km) tube_width PD_accu(m/yr) PD_melting(m/yr) total_flux melting_flux PD_surf_velocity(m/yr)')

# Ice cores outputs
for name in ic:
    ii = np.argmin(np.abs(x/1000. - ic[name]['x'])) # FIXME: interpolation is complicated...
    output = np.vstack((S[ii]-profs[ii]['z'], profs[ii]['age'], profs[ii]['x0'], profs[ii]['tau'], profs[ii]['a0']))
    prof_domains = profs[ii]['unconf']
    for i in range(len(prof_domains)):
        output = np.insert(output, prof_domains[i]+i, np.nan, axis=1)
    np.savetxt(datadir+name+'_ice_core_output.txt', np.transpose(output)[::-1],
               header='depth(m) age(yr) origin(km) thinning accu(m/yr')
    
    
# ----------------------------------------------------------------------
# Finishing
# ----------------------------------------------------------------------

# Program execution time
MESSAGE = 'Program execution time: '+str(time.perf_counter()-START_TIME)+' s.'
print(MESSAGE)
if os.name != 'nt':
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max memory usage: '+str(mem)+' kbytes')