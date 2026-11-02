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
# TODO: Output for isochrones
# TODO: Remove 'a0' and 'x0' from trajs and profs since we have 'i0' now
# TODO: trajs and profs: store zeta, not z_ie.
# TODO: Implement varying z grid as well
# TODO: Check why age is not correct at beginning of DC-BELDC flow line

import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm, Normalize
import time
import os
import resource
import pandas as pd
import math

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
x_left = 0. #FIXME: the 3 following lines do not seem necessary anymore
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
slope_diff = False
x_grid = {}
fig_slope_maxrange = None
nb_zeta = 1001
solid_fraction_profiles = None
dzie_D = 1.0

yamls = open(datadir+'parameters.yml').read()
para = yaml.load(yamls, Loader=yaml.FullLoader)
globals().update(para)


# ---------------------------------------------------------
# Fonctions definition
# ---------------------------------------------------------

def grid(para):
    start = para['start']
    end = para['end']
    try:
        nb_steps = para['nb_steps']
    except KeyError:
        resolution = para['resolution']
        nb_steps = math.floor((end-start)/resolution)
        end = start + resolution * nb_steps
    if para['type'] == 'regular':
        eps = (end-start)/nb_steps/2
        grid = np.arange(start, end+eps, (end-start)/nb_steps)
    elif para['type'] == 'linear_step':
        ratio = para['ratio']
        if ratio == None:
            ratio = 2/(nb_steps+1)
        eps = (1.-ratio)/nb_steps
        grid = np.arange(ratio, 2.-ratio+eps, (2.-2*ratio)/(nb_steps-1))
        grid = grid * (end-start)/nb_steps
        grid = np.cumsum(np.concatenate((np.array([start]), grid)))
    else:
        print('Type of grid not recognized.')
    try:
        inverted = para['inverted']
    except KeyError:
        inverted = False
    if inverted:
        grid = grid[::-1]
        grid = grid[:-1]-grid[1:]
        grid = np.cumsum(np.concatenate((np.array([start]), grid)))
    return grid

def d_dx(arr):
    der = np.ones_like(arr)*np.nan
    der[1:-1] = (arr[2:]-arr[:-2]) / (x[2:]-x[:-2])
    der[0] = (arr[1]-arr[0]) / (x[1]-x[0])
    der[-1] = (arr[-1]-arr[-2]) / (x[-1]-x[-2])
    return der

def fct_omegaD(zeta, i):
    return 1 - (p[i] + 2) / (p[i] + 1) * (1 - zeta) + 1 / (p[i] + 1) * (1 - zeta) ** (p[i] + 2)

def fct_omega(zeta, i):
    return s[i]*zeta + (1-s[i])*fct_omegaD(zeta, i)

def fct_domegaD_dzeta(zeta, i):
    return (p[i] + 2) / (p[i] + 1) * (1 - (1 - zeta) ** (p[i] + 1))

def fct_domega_dzeta(zeta, i):
    return s[i] + (1-s[i])*fct_domegaD_dzeta(zeta, i)

def fct_d2omegaD_dzeta2(zeta, i):
    return (p[i] + 2) * (1 - zeta)**p[i]

def fct_d2omega_dzeta2(zeta, i):
    return (1-s[i]) * fct_d2omegaD_dzeta2(zeta, i)

def fct_domegaD_dp(zeta, i):
    term1 = (1-zeta)/(p[i]+1)**2
    term2 = -(1-zeta)**(p[i]+2)/(p[i]+1)**2
    term3 = np.where(zeta < 1, (1-zeta)**(p[i]+2)*np.log(np.maximum(1-zeta, 1e-300))/(p[i]+1), 0.)
    return term1 + term2 + term3

def fct_domega_dp(zeta, i):
    return (1-s[i]) * fct_domegaD_dp(zeta, i)

def fct_domega_ds(zeta, i):
    return zeta - fct_omegaD(zeta, i)

def fct_domega_dx(zeta, i):
    return fct_domega_dp(zeta, i)*dp_dx[i] + fct_domega_ds(zeta, i)*ds_dx[i]

def fct_zeta(omega, i):
    return np.interp(omega, tab_omega[:, i], zeta_grid)

def fct_gamma_int(j, qq, zeta):
    return Y[j]/QH[j]**2*qq*H_ie[j] / fct_domega_dzeta(zeta, j)**3 \
        *fct_d2omega_dzeta2(zeta, j)
    


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


# -----------------------------------------------------------------
# Loading solid fraction profiles
# -----------------------------------------------------------------

def read_sf_profile(fname):
    Dd, DD = np.loadtxt(datadir+fname, unpack=True)
    Dd = np.append(Dd, Dd[-1]+0.01)
    Dd = np.append(Dd, 10000.)
    DD = np.append(DD, 1.)
    DD = np.append(DD, 1.)
    Die = np.cumsum(np.concatenate((np.array([0.]), DD[:-1]*(Dd[1:]-Dd[:-1]))))
    return Dd, DD, Die

sf_anchors = []   # (x_km, D_depth, D_D, D_depth_ie)
if solid_fraction_profiles is not None:
    xs_check = np.array([float(e['x']) for e in solid_fraction_profiles])
    if np.any(np.diff(xs_check) <= 0):
        raise ValueError('solid_fraction_profiles must be sorted by increasing x: '
                         + str(list(xs_check)))
    for e in solid_fraction_profiles:
        Dd, DD, Die = read_sf_profile(e['file'])
        sf_anchors.append((float(e['x']), Dd, DD, Die))
elif solid_fraction is not None:
    Dd, DD, Die = read_sf_profile(solid_fraction)
    sf_anchors.append((-np.inf, Dd, DD, Die))
else:
    sf_anchors.append((-np.inf, np.array([0., 10000.]), np.ones(2), np.array([0., 10000.])))

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
s = np.interp(x, x_s, s_measure)
S = np.interp(x, x_S, S_measure)
H = np.interp(x, x_H, H_measure)
Y = np.interp(x, x_Y, Y_measure)

B = S - H

if obs_bedrock != None:
    obs_bed_depth = np.interp(x, obs_bed_x, obs_bed_depth)
    obs_bed_z = S - obs_bed_depth

# Convert x in meters now
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

if Q[0] > 0:
    mu = Qm / (Q - Qm)
else:
    mu = np.zeros_like(Q)
    mu[1:] = Qm[1:] / (Q[1:] - Qm[1:])
    mu[0] = mu[1]
QH = Q - Qm


# ----------------------------------------------------------
# DELTA H, H_ie and S_ie
# ----------------------------------------------------------

xs_a = np.array([a[0] for a in sf_anchors])
dH_a = np.array([a[1][-1]-a[3][-1] for a in sf_anchors])
if len(sf_anchors) == 1:
    DELTA_H = np.full_like(x, dH_a[0])
else:
    DELTA_H = np.interp(x/1000., xs_a, dH_a)
print('DELTA_H: %.3f (x=0 km) .. %.3f (x=%.1f km)' % (DELTA_H[0], DELTA_H[-1], x[-1]/1000.))
S_ie = S - DELTA_H
H_ie = H - DELTA_H

# Grille commune de profondeur équivalent-glace (pas dzie_D)
v_max = 1.5*np.max(H_ie)
nv = max(int(np.ceil(v_max/dzie_D))+1, 2)
v_grid = np.linspace(0., v_max, nv)
nanc = len(sf_anchors)
Dprof = np.empty((nanc, nv)); Rprof = np.empty((nanc, nv))
for k in range(nanc):
    Dd, DD, Die = sf_anchors[k][1], sf_anchors[k][2], sf_anchors[k][3]
    Dprof[k] = np.interp(v_grid, Die, DD)
    Rprof[k] = np.interp(v_grid, Die, Dd)
if nanc == 1:
    D_col = np.tile(Dprof[0], (len(x), 1))
    R_col = np.tile(Rprof[0], (len(x), 1))
else:
    D_col = np.empty((len(x), nv)); R_col = np.empty((len(x), nv))
    for iv in range(nv):
        D_col[:, iv] = np.interp(x/1000., xs_a, Dprof[:, iv])
        R_col[:, iv] = np.interp(x/1000., xs_a, Rprof[:, iv])

# Integral term of d(1/D)/dx (varying density profile, for slope)
G = 1./D_col
Gx = np.zeros_like(G)
Gx[1:-1] = (G[2:]-G[:-2])/(x[2:]-x[:-2])[:, None]
Gx[0] = (G[1]-G[0])/(x[1]-x[0])
Gx[-1] = (G[-1]-G[-2])/(x[-1]-x[-2])
TIE = np.zeros_like(G)
TIE[:, 1:] = np.cumsum(0.5*(Gx[:, 1:]+Gx[:, :-1])*np.diff(v_grid)[None, :], axis=1)

# ----------------------------------------
# Calculation of surface velocity
# ----------------------------------------

surf_vel = np.ones_like(x)*np.nan
for i in range(len(x)): # FIXME: avoid this loop with a vector function
    surf_vel[i] = QH[i] / Y[i] * fct_domega_dzeta(1., i) / H_ie[i]


# ------------------------------------
# Determination of accumulation areas
# ------------------------------------

if x_start == None and Q[0]>0: 
    accu_starts = [0]
elif x_start == None and Q[0] == 0:
    accu_starts = [1]
else:
    accu_starts = [np.argmin(np.abs(x_start-x/1000.))]
accu_ends = []
for i in range(1, len(x)-1):
    if Q[i] > Q[i-1] and Q[i] >= Q[i+1] : #To be checked
        accu_ends.append(i)
    elif Q[i] <= Q[i-1] and Q[i] < Q[i+1] : #To be checked
        accu_starts.append(i+1)


# --------------------------------------
# Calculation of x derivatives
# --------------------------------------

dSie_dx = d_dx(S_ie)
dp_dx = d_dx(p)
ds_dx = d_dx(s)
dS_dx = d_dx(S)
dHie_dx = d_dx(H_ie)

# --------------------------------------
# Calculation of tab_omega
# --------------------------------------

zeta_grid = np.linspace(0, 1, nb_zeta)
tab_omega = np.ones((nb_zeta, len(x)))*np.nan
for i in range(len(x)):
    tab_omega[:, i] = fct_omega(zeta_grid, i)

print('Before initialisation of trajectories:', round(time.perf_counter()-START_TIME, 4), 's.')

# ----------------------------------------
# Calculating upstream age 
# ----------------------------------------

# Trapezoidal integration of the age on the upstream boundary
mu0 = Qm[0]/QH[0]
f0 = H_ie[0]*(1+mu0)/(a[0]*(tab_omega[:, 0]+mu0))
C0 = np.concatenate(([0.], np.cumsum(0.5*(f0[1:]+f0[:-1])*np.diff(zeta_grid))))

# --------------------------------------------
# Initialisation of trajectories (vectorized)
# --------------------------------------------

index_unconf = []
i0_l, q_l, a0_l, x0_l = [], [], [], []
in_l, z0_l, age0_l, gi0_l, tau0_l, slope0_l, zieb0_l = [], [], [], [], [], [], []

# From the upstream border
if Q[0] > 0:
    zz_start = B[0] + ((S_ie[0]-B[0]) % z_step)
    z_arr = np.arange(zz_start, S_ie[0], z_step)
    if z_arr.size:
        zeta = (z_arr-B[0])/H_ie[0]
        omega = fct_omega(zeta, 0)
        qq = Qm[0] + (Q[0]-Qm[0])*omega
        gi0 = H_ie[0]/a[0]*Q[0]/QH[0]*(1/fct_domega_dzeta(zeta, 0)-1/fct_domega_dzeta(1., 0))
        age0 = C0[-1] - np.interp(zeta, zeta_grid, C0)
        nup = z_arr.size
        i0_l += [0]*nup; q_l += list(qq);
        in_l += ['upstream']*nup; z0_l += list(z_arr)
        age0_l += list(age0); gi0_l += list(gi0)
        tau0_l += list(qq/Q[0]); slope0_l += [0.]*nup; zieb0_l += [np.nan]*nup

# From the surface
for k in range(len(accu_starts)):
    if k > 0:
        index_unconf.append(len(i0_l))
    istart = accu_starts[k]
    iend = accu_ends[k] if k < len(accu_ends) else len(x)-1
    print(f"Accu area n° {k} starts at x= {x[istart]/1000.:.3f} km and ends at x= {x[iend]/1000.:.3f} km")
    for i in range(istart, iend+1):
        i0_l.append(i); q_l.append(Q[i]);
        in_l.append('surf'); z0_l.append(S_ie[i]); age0_l.append(0.)
        gi0_l.append(0.); tau0_l.append(1.); slope0_l.append(dSie_dx[i])
        zieb0_l.append(S_ie[i-1])   # i=0 -> S_ie[-1], comme l'ancien code

ntraj = len(i0_l)
i0_arr = np.array(i0_l); q_arr = np.array(q_l)
a0_arr = a[i0_arr]
x0_arr = x[i0_arr]
in_arr = np.array(in_l)
z_ie = np.full((ntraj, len(x)), np.nan)
steady_age = np.full((ntraj, len(x)), np.nan)
gammaint = np.full((ntraj, len(x)), np.nan)
tau = np.full((ntraj, len(x)), np.nan)
slope = np.full((ntraj, len(x)), np.nan)
zieb_iso = np.full((ntraj, len(x)), np.nan)
ziea_iso = np.full((ntraj, len(x)), np.nan)

rows = np.arange(ntraj)
z_ie[rows, i0_arr] = z0_l
steady_age[rows, i0_arr] = age0_l
gammaint[rows, i0_arr] = gi0_l
tau[rows, i0_arr] = tau0_l
slope[rows, i0_arr] = slope0_l
zieb_iso[rows, i0_arr] = zieb0_l

# Constants by trajectory
QH0 = QH[i0_arr]; Q0 = Q[i0_arr]; H0 = H_ie[i0_arr] # FIXME: should be rather called Hie0, not H0
domega1_i0 = fct_domega_dzeta(1., i0_arr)
c_gamma = a[i0_arr] * domega1_i0 * QH0/Q0/H0
c_tr = QH0/Q0/H0 * domega1_i0

print('Before following trajectories:', round(time.perf_counter()-START_TIME, 4), 's.')

# ---------------------------------------------
# Follow-up of trajectories, batch by column
# ---------------------------------------------
with np.errstate(invalid='ignore', divide='ignore'): # FIXME: do we really need these ignores?
    for j in range(1, len(x)):
        entered = i0_arr <= j-1
        alive = entered & ~np.isnan(z_ie[:, j-1])
        idx = rows[alive]
        if idx.size == 0:
            continue
        qq = q_arr[idx]
        surf_exit = Q[j] < qq
        bed_exit = (Q[j] >= qq) & (Qm[j] > qq)
        exits = surf_exit | bed_exit
        if exits.any():
            z_ie[idx[exits], j:] = np.nan
            steady_age[idx[exits], j:] = np.nan
            gammaint[idx[exits], j:] = np.nan
            tau[idx[exits], j:] = np.nan
            slope[idx[exits], j:] = np.nan
            zieb_iso[idx[exits], j:] = np.nan
            ziea_iso[idx[exits], j-1] = np.nan
            idx = idx[~exits]
            if idx.size == 0:
                continue
            qq = q_arr[idx]
        omega = (qq-Qm[j])/(Q[j]-Qm[j])
        zeta = np.interp(omega, tab_omega[:, j], zeta_grid)
        z_ie[idx, j] = B[j] + zeta*H_ie[j]
        zetab = (z_ie[idx, j-1]-B[j-1])/H_ie[j-1]
        dom_dzeta = fct_domega_dzeta(zeta, j)
        dom_dzeta_b = fct_domega_dzeta(zetab, j-1)
        kappab = Y[j-1]*H_ie[j-1]/QH[j-1]/dom_dzeta_b
        kappa = Y[j]*H_ie[j]/QH[j]/dom_dzeta
        dage = (x[j]-x[j-1]) * np.sqrt(kappa*kappab)
        steady_age[idx, j] = steady_age[idx, j-1] + dage
        gi_j = Y[j]/QH[j]**2*qq*H_ie[j]/dom_dzeta**3*fct_d2omega_dzeta2(zeta, j)
        gi_jb = Y[j-1]/QH[j-1]**2*qq*H_ie[j-1]/dom_dzeta_b**3*fct_d2omega_dzeta2(zetab, j-1)
        gammaint[idx, j] = gammaint[idx, j-1] + (x[j]-x[j-1])*np.sqrt(gi_j*gi_jb)
        gamma = c_gamma[idx]*gammaint[idx, j]
        tau_j = qq/Q[j] / (dom_dzeta*QH[j]/Q[j]/H_ie[j]) * c_tr[idx] / (1+gamma)
        tau[idx, j] = tau_j
        traj_term = (H_ie[j]/QH[j]/dom_dzeta*a0_arr[idx]*c_tr[idx]/(1+gamma) - a[j]/Q[j])
        sl = traj_term*Y[j]*qq
        melt_term = -(m[j]*Q[j]-a[j]*Qm[j])*Y[j]/QH[j]*(1-qq/Q[j])
        sl = (sl + melt_term)/QH[j]
        sl = (sl - fct_domega_dx(zeta, j))*H_ie[j]/dom_dzeta
        geom_term = -(1-zeta)*dHie_dx[j]
        u_dep = S_ie[j] - z_ie[idx, j]        # depth ie
        D = np.interp(u_dep, v_grid, D_col[j])
        sl = (sl + geom_term)/D
        sl = sl - np.interp(u_dep, v_grid, TIE[j])   # term ∫ d(1/D)/dx
        sl = sl + dS_dx[j]  # FIXME: should be part of geom term
        slope[idx, j] = sl
        if slope_diff:
            zieb_iso[idx, j] = z_ie[idx, j-1]-dage*tau[idx, j-1]*a0_arr[idx]
            zia = z_ie[idx, j]+dage*tau_j*a0_arr[idx]
            first = (j == i0_arr[idx]+1) & (in_arr[idx] == 'surf')
            if first.any():
                zia = np.where(first, S_ie[i0_arr[idx[first]]+1], zia)
            ziea_iso[idx, j-1] = zia

print('After following trajectories:', round(time.perf_counter()-START_TIME, 4), 's.')

# ------------------------------
# Derived fields in 2D
# ------------------------------
Zmat = np.full_like(z_ie, np.nan)
for j in range(len(x)):
    Zmat[:, j] = S[j] - np.interp(S_ie[j]-z_ie[:, j], v_grid, R_col[j])
AgeMat = np.interp(steady_age.ravel(), steady_age_R, age_R).reshape(steady_age.shape)
Xmat = np.where(np.isnan(z_ie), np.nan, x)
if slope_diff:
    ziea_iso[:, -1] = np.nan
    ZBmat = np.full((ntraj, len(x)), np.nan)
    ZAmat = np.full((ntraj, len(x)), np.nan)
    for j in range(len(x)):
        if j >= 1:
            ZBmat[:, j] = S[j-1] - np.interp(S_ie[j-1] - zieb_iso[:, j], v_grid, R_col[j-1])
        if j <= len(x)-2:
            ZAmat[:, j] = S[j+1] - np.interp(S_ie[j+1] - ziea_iso[:, j], v_grid, R_col[j+1])
    SDmat = np.full((ntraj, len(x)), np.nan)
    SDmat[:, 1:-1] = (ZAmat[:, 1:-1]-ZBmat[:, 1:-1])/(x[2:]-x[:-2])[None, :]

# Advancement
advanced = np.zeros((ntraj, len(x)), dtype=bool)
for j in range(1, len(x)):
    advanced[:, j] = (i0_arr <= j-1) & ~np.isnan(z_ie[:, j-1]) & ~np.isnan(z_ie[:, j])
is_unconf = np.isin(i0_arr, accu_starts[1:]) if len(accu_starts) > 1 else np.zeros(ntraj, dtype=bool)

# ---------------------------------------------------------------------
# prof_col(ii) : vertical profile for column ii (view, no copy)
# ---------------------------------------------------------------------
def prof_col(ii):
    sel = (i0_arr == ii) | advanced[:, ii]
    pc = {}
    pc['z_ie'] = z_ie[sel, ii]
    pc['steady_age'] = steady_age[sel, ii]
    pc['tau'] = tau[sel, ii]
    pc['slope'] = slope[sel, ii]
    pc['x0'] = x0_arr[sel]
    pc['a0'] = a0_arr[sel]
    pc['i0'] = i0_arr[sel]
    if slope_diff:
        pc['zieb_iso'] = zieb_iso[sel, ii]
        if ii <= len(x)-2:
            adv_k = advanced[:, ii+1]
            alive_k = (i0_arr <= ii) & ~np.isnan(z_ie[:, ii])
            contrib = adv_k | (alive_k & ~adv_k)
            vals = ziea_iso[contrib, ii].copy()
            vals[alive_k[contrib] & ~adv_k[contrib]] = np.nan
            pc['ziea_iso'] = vals
        else:
            pc['ziea_iso'] = np.array([])
        if ii >= 1:
            pc['zb_iso'] = S[ii-1]-np.interp(S_ie[ii-1]-pc['zieb_iso'], v_grid, R_col[ii-1])
            pc['za_iso'] = S[ii+1]-np.interp(S_ie[ii+1]-pc['ziea_iso'], v_grid, R_col[ii+1])
            pc['slope_diff'] = (pc['za_iso']-pc['zb_iso'])/(x[ii+1]-x[ii-1])
        else:
            pc['zb_iso'] = np.nan*np.ones_like(pc['z'])
            pc['za_iso'] = np.nan*np.ones_like(pc['z'])
            pc['slope_diff'] = np.nan*np.ones_like(pc['z'])
    pc['z'] = S[ii]-np.interp(S_ie[ii]-pc['z_ie'], v_grid, R_col[ii])
    pc['age'] = np.interp(pc['steady_age'], steady_age_R, age_R)
    if len(accu_starts) > 1:
        adv_sel = advanced[:, ii] & is_unconf
        pc['unconf'] = (np.cumsum(sel)-1)[adv_sel].tolist() if adv_sel.any() else []
    else:
        pc['unconf'] = []
    if comp_isochrones is not None:
        pc['cp_iso_depth'] = np.zeros(cp_iso_nb)
        pc['cp_iso_z'] = np.zeros(cp_iso_nb)
        for j in range(cp_iso_nb):
            pc['cp_iso_depth'][j] = np.interp(x[ii], cp_iso_x, cp_iso_depth[j, :])
            pc['cp_iso_z'][j] = np.interp(x[ii]/1000., cp_iso_x, cp_iso_z[j, :])
    return pc

print('Before surface quantities:', round(time.perf_counter()-START_TIME, 4), 's.')

# ----------------------------------------
# Surface quantities (vectorized)
# ----------------------------------------

alive = ~np.isnan(z_ie)
ntraj_, nx_ = z_ie.shape
cols = np.arange(nx_)

# The two highest trajectories by column
top  = ntraj_-1 - np.argmax(alive[::-1, :], axis=0)
alive2 = alive.copy()
alive2[top, cols] = False
top2 = ntraj_-1 - np.argmax(alive2[::-1, :], axis=0)
ok = alive.any(axis=0) & alive2.any(axis=0)   # >= 2 trajs dans la colonne

z1, z2 = Zmat[top, cols], Zmat[top2, cols]
dz = z1 - z2
ok &= dz != 0

with np.errstate(divide='ignore', invalid='ignore'): #FIXME: is this ignore necessary?
    tfrac = np.where(ok, (S - z1)/np.where(dz == 0., 1., dz), 0.)
    def surf_extrap(Y):
        y1, y2 = Y[top, cols], Y[top2, cols]
        return np.where(ok, y1 + tfrac*(y1-y2), np.nan)

surf_steady_age = surf_extrap(steady_age)
surf_tau        = surf_extrap(tau)
surf_slope      = surf_extrap(slope)

surf_age = np.interp(surf_steady_age, steady_age_R, age_R)

print('Before creating figures:', round(time.perf_counter()-START_TIME, 4), 's.')

# ---------------------------------------------------------
# Preparing for figures (in 2D)
# ---------------------------------------------------------


cuts = [0] + list(index_unconf) + [ntraj]
domain_slices = [slice(cuts[k], cuts[k+1]) for k in range(len(cuts)-1)]
unconf_list = [Zmat[iu] for iu in index_unconf]

traj_sel = list(range(0, ntraj, step_traj))
age_max = max(np.nanmax(AgeMat[t]/1000.) for t in traj_sel)
linthresh = age_max/scaling_age
norm_age = SymLogNorm(linthresh=linthresh, linscale=1, vmin=0., vmax=age_max)

tau_max = max(np.nanmax(tau[t]) for t in traj_sel)
norm_tau = Normalize(vmin=0., vmax=tau_max)

if fig_slope_maxrange is None:
    slope_min = min(np.nanmin(slope[t]) for t in traj_sel)
    slope_max = max(np.nanmax(slope[t]) for t in traj_sel)
    slope_max = max(-slope_min, slope_max)
else:
    slope_max = fig_slope_maxrange
levels_slope = np.arange(-slope_max, slope_max+0.000001, slope_max/50)

for name in ic:
    ic[name]['XX'] = ic[name]['x'] * np.ones(2)
    ic[name]['S'] = np.interp(ic[name]['x'], x/1000., S)
    ic[name]['ZZ'] = np.array([ic[name]['S'], ic[name]['S']-ic[name]['max_depth']])
    ic[name]['DD'] = np.array([0., ic[name]['max_depth']])

# ---------------------------------------------------------
# Re-usable panels
# ---------------------------------------------------------

def top_accu(ax1, yred, ylabel, color='r', comp_x=None, comp_y=None,
             marker=None, ylim_fix=None):
    ax1.plot(x/1000., a*R[0], color='k', linewidth=2)
    ax1.set_ylabel('PD accu. (m/yr)', fontsize=14)
    ax1.plot(x/1000., np.zeros_like(x), color='k', linewidth=2, linestyle='--')
    ax1b = ax1.twinx()
    ax1b.plot(x/1000., yred, color=color, linewidth=2)
    if comp_x is not None:
        if marker is not None:
            ax1b.plot(comp_x, comp_y, color=color, linestyle='None', marker=marker)
        else:
            ax1b.plot(comp_x, comp_y, color=color, linewidth=2, linestyle='--')
    ax1b.set_ylabel(ylabel, color=color, fontsize=14)
    ax1b.tick_params(axis='y', colors=color)
    if ylim_fix is not None:
        bottom, top = ax1b.get_ylim()
        ax1b.set_ylim(ylim_fix(bottom, top))
    return ax1b

def bed_panel(ax2, depthmode):
    if obs_bedrock is not None:
        if not depthmode:
            ax2.plot(x/1000., S, label='Surface', color='k', linewidth=2.)
            ax2.fill_between(x/1000., obs_bed_z, B, color='grey', label='Stagnant ice')
            ax2.plot(x/1000., obs_bed_z, label='Bedrock', color='k', linewidth=2.)
        else:
            ax2.plot(x/1000., np.zeros_like(S), label='Surface', color='k', linewidth=2.)
            ax2.fill_between(x/1000., obs_bed_depth, S-B, color='grey', label='Stagnant ice')
            ax2.plot(x/1000., obs_bed_depth, label='Bedrock', color='k', linewidth=2.)
    else:
        if not depthmode:
            ax2.plot(x/1000., S, label='Surface', color='k', linewidth=2.)
            ax2.plot(x/1000., B, label='Bedrock', color='k', linewidth=2.)
        else:
            ax2.plot(x/1000., np.zeros_like(S), label='Surface', color='k', linewidth=2.)
            ax2.plot(x/1000., S-B, label='Bedrock', color='k', linewidth=2.)
    ax2top = ax2.twiny()
    ax2top.set_xlabel(r"direction of the flow $\longrightarrow$", fontsize=20)
    ax2top.xaxis.set_ticks([])

def plot_trajs(ax2, depthmode, tlist):
    for t in tlist:
        xx, zz = Xmat[t], Zmat[t]
        if in_arr[t] == 'upstream':
            ax2.plot(xx/1000., (S-zz) if depthmode else zz, color='k')
        elif in_arr[t] == 'surf':
            ax2.plot(xx/1000., (S-zz) if depthmode else zz, color='b')
    for iu in index_unconf:
        ax2.plot(Xmat[iu]/1000., (S-Zmat[iu]) if depthmode else Zmat[iu],
                 label='Discontinuities', color='k', linewidth=4., linestyle='--')

def plot_unconf(ax2, depthmode):
    for zu in unconf_list:
        ax2.plot(x/1000., (S-zu) if depthmode else zu,
                 label='Discontinuities', color='k', linewidth=4., linestyle='--')

def plot_color_trajs(ax2, depthmode, tlist, norm):
    for t in tlist:
        xx, zz, aa = Xmat[t], Zmat[t], AgeMat[t]
        points = np.array([xx/1000., (S-zz) if depthmode else zz]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='plasma_r', norm=norm)
        lc.set_array(aa/1000.)
        lc.set_linewidth(2)
        ax2.add_collection(lc)

def plot_ic(ax2, depthmode):
    for name in ic:
        ax2.plot(ic[name]['XX'], ic[name]['DD'] if depthmode else ic[name]['ZZ'],
                 linewidth=lw_core, color=color_core, linestyle=ls_core)
        ax2.annotate(name, (ic[name]['x'], 0. if depthmode else ic[name]['S']),
                     ha='center', va='bottom', color=color_core)

def savefig(fname):
    plt.savefig(datadir+fname+'.'+fig_format, format=fig_format, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# 1-2. Trajectories
# ---------------------------------------------------------

def traj_figure(depthmode, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,
                                   figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})
    top_accu(ax1, surf_vel*R[0], 'PD surf. vel. (m/yr)',
             comp_x=cp_surf_vel_x if comp_surf_vel is not None else None,
             comp_y=cp_surf_vel_vel if comp_surf_vel is not None else None)
    bed_panel(ax2, depthmode)
    plot_trajs(ax2, depthmode, traj_sel)
    ax2.set_xlabel('horizontal distance (km)', fontsize=19)
    ax2.set_ylabel('depth (m)' if depthmode else 'elevation (m)', fontsize=19)
    if depthmode:
        ax2.set_ylim(ax2.get_ylim()[::-1])
    plot_ic(ax2, depthmode)
    savefig(fname)

traj_figure(False, 'trajectories_x_z')
traj_figure(True, 'trajectories_x_depth')
print('After simple trajectories figures:', round(time.perf_counter()-START_TIME, 4), 's.')

# ---------------------------------------------------------
# 3-4. Colored trajectories
# ---------------------------------------------------------

def traj_color_figure(depthmode, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,
                                   figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})
    top_accu(ax1, surf_vel*R[0], 'PD surf. vel. (m/yr)',
             comp_x=cp_surf_vel_x if comp_surf_vel is not None else None,
             comp_y=cp_surf_vel_vel if comp_surf_vel is not None else None)
    bed_panel(ax2, depthmode)
    plot_color_trajs(ax2, depthmode, traj_sel, norm_age)
    for iu in index_unconf:
        ax2.plot(Xmat[iu]/1000., (S-Zmat[iu]) if depthmode else Zmat[iu],
                 label='Discontinuities', color='k', linewidth=4., linestyle='--')
    ax2.set_xlabel('horizontal distance (km)', fontsize=19)
    ax2.set_ylabel('depth (m)' if depthmode else 'elevation (m)', fontsize=19)
    if depthmode:
        ax2.set_ylim(ax2.get_ylim()[::-1])
    sm = plt.cm.ScalarMappable(cmap='plasma_r', norm=norm_age)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
    cb.set_label(label='age (kyr)', size=18)
    plot_ic(ax2, depthmode)
    savefig(fname)

traj_color_figure(False, 'trajectories_color_x_z')
traj_color_figure(True, 'trajectories_color_x_depth')
print('After color trajectories figures:', round(time.perf_counter()-START_TIME, 4), 's.')

# ---------------------------------------------------------
# 5-6. Age
# ---------------------------------------------------------

def age_figure(depthmode, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,
                                   figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})
    top_accu(ax1, surf_age/1000., 'surface age (kyr)',
             comp_x=cp_surf_age_x if comp_surf_age is not None else None,
             comp_y=cp_surf_age_age/1000. if comp_surf_age is not None else None,
             marker='o',
             ylim_fix=lambda b, t: (b, 1.) if t < 1. else (b, t))
    bed_panel(ax2, depthmode)
    if comp_isochrones is not None:
        for i in range(cp_iso_nb):
            ax2.plot(cp_iso_x, (cp_iso_depth if depthmode else cp_iso_z)[i, :],
                     color='w', linewidth=1)
    levels = norm_age.inverse(np.linspace(0, 1, 100))
    levels_iso = np.array(fig_age_iso)
    for sl in domain_slices:
        zt = (S[None, :]-Zmat[sl]) if depthmode else Zmat[sl]
        ax2.contourf(Xmat[sl]/1000., zt, AgeMat[sl]/1000., levels=levels,
                     cmap='plasma_r', norm=norm_age)
        cp = ax2.contour(Xmat[sl]/1000., zt, AgeMat[sl]/1000.,
                         levels=levels_iso, colors='k', norm=norm_age)
    plot_unconf(ax2, depthmode)
    ax2.set_xlabel('horizontal distance (km)', fontsize=19)
    ax2.set_ylabel('depth (m)' if depthmode else 'elevation (m)', fontsize=19)
    if depthmode:
        ax2.set_ylim(ax2.get_ylim()[::-1])
    sm = plt.cm.ScalarMappable(cmap='plasma_r', norm=norm_age)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
    cb.set_label(label='age (kyr)', size=18)
    cb.add_lines(cp)
    plot_ic(ax2, depthmode)
    savefig(fname)

age_figure(False, 'age_x_z')
age_figure(True, 'age_x_depth')
print('After age figures:', round(time.perf_counter()-START_TIME, 4), 's.')

# ---------------------------------------------------------
# 7-8. Thinning
# ---------------------------------------------------------

def thinning_figure(depthmode, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,
                                   figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})
    if depthmode:
        yfix = lambda b, t: (0.9, 1.1) if b > 0.9 else (b, t)
    else:
        yfix = lambda b, t: (0.9, 1.1) if (b > 0.9 and t < 1.1) else (b, t)
    top_accu(ax1, surf_tau, 'surface thinning', ylim_fix=yfix)
    bed_panel(ax2, depthmode)
    levels = np.arange(0., tau_max, 0.1)
    for sl in domain_slices:
        zt = (S[None, :]-Zmat[sl]) if depthmode else Zmat[sl]
        ax2.contourf(Xmat[sl]/1000., zt, tau[sl], levels=100,
                     cmap='plasma', norm=norm_tau)
        cp = ax2.contour(Xmat[sl]/1000., zt, tau[sl], levels=levels,
                         colors='k', norm=norm_tau)
    plot_unconf(ax2, depthmode)
    ax2.set_xlabel('horizontal distance (km)', fontsize=19)
    ax2.set_ylabel('depth (m)' if depthmode else 'elevation (m)', fontsize=19)
    if depthmode:
        ax2.set_ylim(ax2.get_ylim()[::-1])
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm_tau)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
    cb.set_label(label='vertical thinning ratio (no unit)', size=18)
    if not depthmode:
        cb.add_lines(cp)
    plot_ic(ax2, depthmode)
    savefig(fname)

thinning_figure(False, 'thinning_x_z')
thinning_figure(True, 'thinning_x_depth')
print('After thinning figures:', round(time.perf_counter()-START_TIME, 4), 's.')

# ---------------------------------------------------------
# 9-10. Slope
# ---------------------------------------------------------

def slope_figure(depthmode, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,
                                   figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})
    top_accu(ax1, surf_slope, 'surf. layer slope')
    bed_panel(ax2, depthmode)
    for sl in domain_slices:
        zt = (S[None, :]-Zmat[sl]) if depthmode else Zmat[sl]
        cp = ax2.contourf(Xmat[sl]/1000., zt, slope[sl],
                          levels=levels_slope, cmap='PuOr')
    plot_unconf(ax2, depthmode)
    ax2.set_xlabel('horizontal distance (km)', fontsize=19)
    ax2.set_ylabel('depth (m)' if depthmode else 'elevation (m)', fontsize=19)
    if depthmode:
        ax2.set_ylim(ax2.get_ylim()[::-1])
    cb = plt.colorbar(cp, pad=-0.06)
    cb.set_label(label='layer slope (no unit)', size=18)
    plot_ic(ax2, depthmode)
    savefig(fname)

slope_figure(False, 'slope_x_z')
slope_figure(True, 'slope_x_depth')
print('After slope figures:', round(time.perf_counter()-START_TIME, 4), 's.')

# ---------------------------------------------------------
# 11. Ice ores
# ---------------------------------------------------------

for name in ic:
    ii = np.argmin(np.abs(x/1000. - ic[name]['x']))
    pc = prof_col(ii)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title(name + ' ice core', fontsize=20)
    prof_domains = pc['unconf'].copy()
    prof_domains.insert(0, 0)
    prof_domains.append(len(pc['z']))

    for i in range(len(prof_domains)-1):
        ax.plot(pc['age'][prof_domains[i]:prof_domains[i+1]]/1000.,
                pc['z'][prof_domains[i]:prof_domains[i+1]], color='b')
    if ic[name]['comp'] is not None:
        ax.plot(ic[name]['cp_age']/1000., S[ii]-ic[name]['cp_depth'], color='b', linestyle='--')
    if comp_isochrones is not None and comp_iso_ages is not None:
        ax.plot(cp_iso_ages/1000., pc['cp_iso_z'], color='b', linestyle='None',
                marker='o', markersize=5)
    ax.set_xlabel('age (kyr)', color='b', fontsize=15)
    ax.set_ylabel('elevation (m)', fontsize=15)
    ax.set_ylim(top=S[ii], bottom=S[ii]-ic[name]['max_depth'])
    jj = np.argmin(np.abs(pc['z']-(S[ii]-ic[name]['max_depth'])))
    max_age = pc['age'][jj]/1000.
    ax.set_xlim(left=0, right=max_age)
    ax.spines['bottom'].set_color('b')
    ax.tick_params(axis='x', colors='b')
    ax.hlines(pc['z'][prof_domains[1:-1]], 0., max_age, linestyles='--',
              linewidth=3., color='k')
    axb = ax.twinx()
    axb.set_ylim(top=0., bottom=S[ii]-B[ii])
    axb.set_ylabel('depth (m)', fontsize=15)

    ax2 = ax.twiny()
    for i in range(len(prof_domains)-1):
        ax2.plot(pc['x0'][prof_domains[i]:prof_domains[i+1]]/1000.,
                 pc['z'][prof_domains[i]:prof_domains[i+1]], color='r')
    if ic[name]['comp'] is not None:
        ax2.plot(ic[name]['cp_x'], S[ii]-ic[name]['cp_depth'], color='r', linestyle='--')
    ax2.set_xlim(left=x_left, right=ic[name]['x'])
    ax2.set_xlabel('ice origin (km)', color='r', fontsize=15)
    ax2.spines['top'].set_color('r')
    ax2.tick_params(axis='x', colors='r')
    ax2.spines['bottom'].set_visible(False)

    ax3 = ax.twiny()
    for i in range(len(prof_domains)-1):
        ax3.plot(pc['tau'][prof_domains[i]:prof_domains[i+1]],
                 pc['z'][prof_domains[i]:prof_domains[i+1]], color='orange')
    if ic[name]['comp'] is not None:
        ax3.plot(ic[name]['cp_tau'], S[ii]-ic[name]['cp_depth'], color='orange', linestyle='--')
    ax3.set_xlabel('vertical thinning ratio (no unit)', color='orange', fontsize=15)
    ax3.spines['top'].set_color('orange')
    ax3.spines['top'].set_position(('axes', 1.1))
    ax3.tick_params(axis='x', colors='orange')
    ax3.set_xlim(left=0)

    ax4 = ax.twiny()
    for i in range(len(prof_domains)-1):
        if slope_diff:
            ax4.plot(pc['slope_diff'][prof_domains[i]:prof_domains[i+1]],
                     pc['z'][prof_domains[i]:prof_domains[i+1]], color='violet', linestyle='dotted')
        ax4.plot(pc['slope'][prof_domains[i]:prof_domains[i+1]],
                 pc['z'][prof_domains[i]:prof_domains[i+1]], color='violet')
    if ic[name]['comp'] is not None:
        ax4.plot(ic[name]['cp_slope'], S[ii]-ic[name]['cp_depth'], color='violet', linestyle='--')
    ax4.set_xlabel('layer slope (no unit)', color='violet', fontsize=15)
    ax4.spines['top'].set_color('violet')
    ax4.spines['top'].set_position(('axes', 1.2))
    ax4.tick_params(axis='x', colors='violet')

    savefig(name+'_ice_core')

# ---------------------------------------------------------
# 12. R(t) - Age
# ---------------------------------------------------------

if temp_factor is not None:
    fig, ax = plt.subplots()
    plt.stairs(R[:-2], age_R[:-1]/1000, baseline=None)
    plt.xlabel('time (kyr)', fontsize=15)
    plt.ylabel(r'$R(t) (relative unit)$', fontsize=15)
    savefig('temporal_factor')

# ---------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------

if output_fl:
    output = np.vstack((x/1000., Y, a*R[0], m*R[0], Q, Qm, surf_vel))
    np.savetxt(datadir+'flow_line_output.txt', np.transpose(output),
               header='x(km) tube_width PD_accu(m/yr) PD_melting(m/yr) total_flux melting_flux PD_surf_velocity(m/yr)')

for name in ic:
    ii = np.argmin(np.abs(x/1000. - ic[name]['x']))
    pc = prof_col(ii)
    output = np.vstack((S[ii]-pc['z'], pc['age'], pc['x0'], pc['tau'], pc['a0']))
    prof_domains = pc['unconf']
    for i in range(len(prof_domains)):
        output = np.insert(output, prof_domains[i]+i, np.nan, axis=1)
    np.savetxt(datadir+name+'_ice_core_output.txt', np.transpose(output)[::-1],
               header='depth(m) age(yr) origin(km) thinning accu(m/yr')

MESSAGE = 'Program execution time: '+str(time.perf_counter()-START_TIME)+' s.'
print(MESSAGE)
if os.name != 'nt':
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max memory usage: '+str(mem)+' kbytes')