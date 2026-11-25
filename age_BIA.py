# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# TODO: exact formula for the age increase
# TODO: Check why slope converges slowly at DC for the DC-BELDC flow line (even for age_flow_line)
# TODO: test case for a dome and check isochrones near the dome
# TODO: dip is singular at accu/ablation transitions. Can we do better?

import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm, Normalize, LogNorm
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
xstart_traj = None
Q_left = 1000.
Qm_left = 100.
fig_format = 'pdf'
fig_age_iso = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
step_traj = 5
scaling_age = 10.
fig_age_maxrange = None
fig_agegrad_maxrange = None
fig_invtau_maxrange = None
fig_slope_maxrange = None
fig_dip_maxrange = None
fig_thinning = True
scaling_q = 10.
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
comp_unconf = None
solid_fraction = None
output_fl = False
slope_diff = False
x_grid = {}
z_grid = {}
fig_ic_dip = True
fig_ic_thinning = 'thinning'
nb_zeta = 1001
solid_fraction_profiles = None
dzie_D = 1.0
tube_width_profiles = None
fig_tube_3d_view = [25., -60.]        # elevation and azimuth of the 3D tube view (deg)
fig_tube_3d_aspect = [1., 0.3, 0.3]   # box aspect (x, lateral, z) of the 3D tube view
fig_tube_3d_traj = True               # draw the trajectories in the 3D tube view
fig_tube_width_scale = 'auto'         # colour/axis scale of the tube width: 'auto', 'linear', 'log' or 'symlog'
fig_tube_width_maxrange = None        # optional upper cap of the tube width scale (m)
scaling_width = 100.                  # symlog: linear range below max width / scaling_width

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

def fct_y(zeta, i):
    # Tube width shape y = Y(x, zeta) / Y^S(x) at node i
    return np.interp(zeta, zeta_grid, y_tab[:, i])

def fct_dy_dzeta(zeta, i):
    # dy/dzeta at fixed x
    return np.interp(zeta, zeta_grid, dy_tab[:, i])

def mean_integrand(f1, f2):
    # Geometric mean of two values of an integrand when they have the same
    # sign (as in the original scheme), arithmetic mean otherwise: with a
    # vertically varying tube width the integrand of gamma can change sign.
    pos = (f1 >= 0) & (f2 >= 0)
    neg = (f1 <= 0) & (f2 <= 0)
    g = np.sqrt(np.abs(f1*f2))
    return np.where(pos, g, np.where(neg, -g, 0.5*(f1+f2)))
    


# -----------------------------------------------------
# Loading files for Geographic data, arrays creations
# -----------------------------------------------------

GEO_INPUTS = ('accumulation', 'melting', 'sliding', 'p_Lliboutry',
              'surface', 'thickness', 'tube_width')
geo = {name: np.loadtxt(datadir+name+'.txt', unpack=True)
       for name in GEO_INPUTS}
x_a, a_measure = geo['accumulation']
x_m, m_measure = geo['melting']
x_s, s_measure = geo['sliding']
x_p, p_measure = geo['p_Lliboutry']
x_S, S_measure = geo['surface']
x_H, H_measure = geo['thickness']
x_Y, Y_measure = geo['tube_width']

# Reading of observed bedrock when different from mechanical bedrock
if obs_bedrock is not None:
    readarray = np.loadtxt(datadir+obs_bedrock, unpack=True)
    obs_bed_x = readarray[0, :]
    obs_bed_depth = readarray[1, :]

# ---------------------------------------------------------------
# Reading of comparison isochrones and unconformities
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

if comp_unconf is not None:
    readarray = np.loadtxt(datadir+comp_unconf, unpack=True)
    cp_unc_x = readarray[0, :]              # x en km (cohérent avec l'axe des figures)
    cp_unc_depth = readarray[1:, :]          # une ligne par discontinuité observée
    cp_unc_nb = cp_unc_depth.shape[0]
    cp_unc_z = np.interp(cp_unc_x, x_S/1000., S_measure) - cp_unc_depth

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

# -----------------------------------------------------------------
# Loading tube width shape profiles y(x, zeta) = Y(x, zeta) / Y^S(x)
# -----------------------------------------------------------------
# Each anchor is a two-column file: zeta (0 = bed, 1 = surface) and the
# ratio Y/Y^S, which is 1 at the surface by definition. Y^S(x) itself is
# still given by tube_width.txt.
tw_anchors = []   # (x_km, zeta, ratio)
if tube_width_profiles is not None:
    xs_check = np.array([float(e['x']) for e in tube_width_profiles])
    if np.any(np.diff(xs_check) <= 0):
        raise ValueError('tube_width_profiles must be sorted by increasing x: '
                         + str(list(xs_check)))
    for e in tube_width_profiles:
        zz, yy = np.loadtxt(datadir+e['file'], unpack=True)
        if np.any(np.diff(zz) <= 0) or zz[0] > 1e-9 or zz[-1] < 1.-1e-9:
            raise ValueError(e['file']+': zeta must be increasing and span [0, 1].')
        if np.any(yy <= 0.):
            raise ValueError(e['file']+': Y/Y^S must be strictly positive.')
        if abs(np.interp(1., zz, yy)-1.) > 1e-6:
            raise ValueError(e['file']+': Y/Y^S must be 1 at the surface (zeta=1).')
        tw_anchors.append((float(e['x']), zz, yy))

# -----------------------------------------------------------
# Reading of comparison files for ice cores
# -----------------------------------------------------------
for name in ic:
    if ic[name]['comp'] is not None:
        df = pd.read_csv(datadir+ic[name]['comp'], sep=None, comment='#', engine='python')
        ic[name]['cp_depth'] = df['depth'].to_numpy(dtype=float)
        for opt_col in ('age', 'x', 'tau', 'slope', 'agegrad', 'invtau', 'dip'):
            if opt_col in df.columns:
                ic[name]['cp_'+opt_col] = df[opt_col].to_numpy()
            else:
                ic[name]['cp_'+opt_col] = np.full_like(ic[name]['cp_depth'], np.nan)
        if 'tau' in df.columns and 'invtau' not in df.columns:
            with np.errstate(divide='ignore'):
                ic[name]['cp_invtau'] = 1/ic[name]['cp_tau']
        if 'invtau' in df.columns and 'tau' not in df.columns:
            with np.errstate(divide='ignore'):
                ic[name]['cp_tau'] = 1/ic[name]['cp_invtau']
        if 'slope' in df.columns and 'dip' not in df.columns:
            ic[name]['cp_dip'] = np.degrees(np.arctan(ic[name]['cp_slope']))
        if 'dip' in df.columns and 'slope' not in df.columns:
            with np.errstate(divide='ignore'):
                ic[name]['cp_slope'] = np.degrees(np.tan(np.radians(ic[name]['cp_dip'])))
        

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
# Adding absissas of ice cores
ic_x = np.array([ic[n]['x'] for n in ic], dtype=float)
if ic_x.size:
    ic_x = ic_x[(ic_x >= x[0]) & (ic_x <= x[-1])]       # dans le domaine
    # pas de doublon avec un nœud existant (sinon dx=0 dans d_dx !)
    keep = np.min(np.abs(ic_x[:, None]-x[None, :]), axis=1) > 1e-9
    x = np.sort(np.concatenate([x, ic_x[keep]]))

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

# Tube width shape y(x, zeta) = Y/Y^S on the zeta grid: linear interpolation
# in zeta (inside each anchor) then in x between anchors (constant outside).
zeta_grid = np.linspace(0, 1, nb_zeta)
if len(tw_anchors) == 0:
    y_tab = np.ones((nb_zeta, len(x)))
else:
    xs_tw = np.array([e[0] for e in tw_anchors])
    y_anch = np.array([np.interp(zeta_grid, e[1], e[2]) for e in tw_anchors])
    if len(tw_anchors) == 1:
        y_tab = np.repeat(y_anch[0][:, None], len(x), axis=1)
    else:
        k_tw = np.clip(np.searchsorted(xs_tw, x/1000., side='right')-1, 0, len(xs_tw)-2)
        w_tw = np.clip((x/1000.-xs_tw[k_tw])/(xs_tw[k_tw+1]-xs_tw[k_tw]), 0., 1.)
        y_tab = ((1.-w_tw)[:, None]*y_anch[k_tw] + w_tw[:, None]*y_anch[k_tw+1]).T
dy_tab = np.gradient(y_tab, zeta_grid, axis=0)
# Y (from tube_width.txt) is the surface width Y^S; the width at the bed is:
Yb = Y * y_tab[0, :]

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
    (m[:-1] * Yb[:-1] +
     0.5 * ((m[1:]-m[:-1]) * Yb[:-1] + (Yb[1:]-Yb[:-1])
            * m[:-1]) +
     1./3 * (m[1:]-m[:-1]) * (Yb[1:]-Yb[:-1]))
dQmdx = np.insert(dQmdx, 0, Qm_left)
Qm = np.cumsum(dQmdx)

if Q[0] > 0:
    mu = Qm / (Q - Qm)
else:
    mu = np.zeros_like(Q)
    mu[1:] = Qm[1:] / (Q[1:] - Qm[1:])
    mu[0] = mu[1]
QH = Q - Qm

# The model assumes no reverse flow: the effective flux QH = Q - Qm must
# stay positive at every node where trajectories live. Node 0 is only
# used when there is an influx at the upstream border (Q[0] > 0); in the
# dome case (Q[0] = 0) it is unused and QH[0] = 0 is legitimate.
if np.any(QH[1:] <= 0) or (Q[0] > 0 and QH[0] <= 0):
    print('Warning: effective flux QH = Q - Qm becomes non-positive '
          'at some point! Exiting...')
    sys.exit()

print('Before firn correction calculations:', round(time.perf_counter()-START_TIME, 4), 's.')

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

# Common grid for ice equivalent depth (dzie_D step)
v_max = 1.5*np.max(H_ie)
nv = max(int(np.ceil(v_max/dzie_D))+1, 2)
v_grid = np.linspace(0., v_max, nv)
nanc = len(sf_anchors)
sf_single = (nanc == 1)
Dprof = np.empty((nanc, nv)); Rprof = np.empty((nanc, nv))
for k in range(nanc):
    Dd, DD, Die = sf_anchors[k][1], sf_anchors[k][2], sf_anchors[k][3]
    Dprof[k] = np.interp(v_grid, Die, DD)
    Rprof[k] = np.interp(v_grid, Die, Dd)
if sf_single:
    # Single, x-invariant profile: zero-copy views (no (nx, nv) copy),
    # and no varying-density term (d(1/D)/dx = 0, so TIE = 0).
    D_col = np.broadcast_to(Dprof[0], (len(x), nv))
    R_col = np.broadcast_to(Rprof[0], (len(x), nv))
    R_row = Rprof[0]
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

print('After firn correction calculations:', round(time.perf_counter()-START_TIME, 4), 's.')


# ----------------------------------------
# Calculation of surface velocity
# ----------------------------------------

surf_vel = np.ones_like(x)*np.nan
for i in range(len(x)): # FIXME: avoid this loop with a vector function
    surf_vel[i] = QH[i] / Y[i] * fct_domega_dzeta(1., i) / H_ie[i]


# ------------------------------------
# Determination of the trajectory start node
# ------------------------------------

# Fluxes are computed over the whole x grid, but trajectories (both
# upstream/border and surface ones) only start at node i_b. This allows
# skipping the near-divide zone, where the pseudo-steady horizontal
# scheme is singular.
if xstart_traj is None:
    # Default: upstream border if there is an influx, otherwise the dome
    # condition is applied on the first inner profile.
    if Q[0] > 0:
        i_b = 0
    else:
        i_b = 1
else:
    i_b = int(np.argmin(np.abs(x/1000. - xstart_traj)))

# Trajectories must start in an accumulation zone
if a[i_b] <= 0.:
    sys.exit('Trajectory start node (x = %.3f km) is not in an accumulation '
             'zone (a = %.4f m/yr). Check xstart_traj.'
             % (x[i_b]/1000., a[i_b]))
print('Trajectories start at x = %.3f km (node %d).' % (x[i_b]/1000., i_b))

# ------------------------------------
# Determination of accumulation areas
# ------------------------------------

accu_starts = [i_b]
accu_ends = []
for i in range(max(i_b, 1), len(x)-1):
    if Q[i] > Q[i-1] and Q[i] >= Q[i+1]:  # To be checked
        accu_ends.append(i)
    elif Q[i] <= Q[i-1] and Q[i] < Q[i+1]:  # To be checked
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


# --------------------------------------------
# Initialisation of trajectories (vectorized)
# --------------------------------------------

index_unconf = []
i0_l, q_l, a0_l, x0_l = [], [], [], []
in_l, z0_l, age0_l, gi0_l, tau0_l, slope0_l, zieb0_l = [], [], [], [], [], [], []

# From the upstream border (Q[0]>0) or the second x node (Q[0]=0)
# From the upstream border or dome, at node i_b
mu_b = Qm[i_b]/QH[i_b]
f_b = H_ie[i_b]*(1+mu_b)/(a[i_b]*(tab_omega[:, i_b]+mu_b))*y_tab[:, i_b]   # Eq. (S3.4)
# Trapezoidal primitive integrated FROM the surface (zeta=1) downwards:
# a possible singularity at zeta=0 stays confined to the last cells.
fs = f_b[::-1]
A_b = np.concatenate(([0.], np.cumsum(0.5*(fs[1:]+fs[:-1])*np.diff(zeta_grid))))
age_grid = A_b[::-1]
zg = grid({**z_grid, 'start': 0., 'end': 1.})
zeta_arr = zg[(zg < 1.) & (zg > 0.)]
if zeta_arr.size:
    omega = fct_omega(zeta_arr, i_b)
    qq = Qm[i_b] + (Q[i_b]-Qm[i_b])*omega
    y_arr = fct_y(zeta_arr, i_b)
    gi0 = H_ie[i_b]/a[i_b]*Q[i_b]/QH[i_b]/qq*\
          (y_arr/fct_domega_dzeta(zeta_arr, i_b)-1/fct_domega_dzeta(1., i_b))   # Eq. (S3.6)
    age0 = np.interp(zeta_arr, zeta_grid, age_grid)
    z_arr = B[i_b] + zeta_arr*H_ie[i_b]
    nb = z_arr.size
    i0_l += [i_b]*nb
    q_l += list(qq)
    in_l += ['upstream']*nb
    z0_l += list(z_arr)
    age0_l += list(age0)
    gi0_l += list(gi0)
    tau0_l += list(qq/Q[i_b]/y_arr)   # Eq. (S3.1)
    slope0_l += [0.]*nb
    zieb0_l += [np.nan]*nb



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
z0_arr = np.array(z0_l) 
z_ie = np.full((ntraj, len(x)), np.nan)
steady_age = np.full((ntraj, len(x)), np.nan)
gammaint = np.full((ntraj, len(x)), np.nan)
tau = np.full((ntraj, len(x)), np.nan)
invtau = np.full((ntraj, len(x)), np.nan)
invtau_x = np.full((ntraj, len(x)), np.nan)
slope = np.full((ntraj, len(x)), np.nan)
dip = np.full((ntraj, len(x)), np.nan)
zieb_iso = np.full((ntraj, len(x)), np.nan)
ziea_iso = np.full((ntraj, len(x)), np.nan)
st_agegrad = np.full((ntraj, len(x)), np.nan)
st_agegrad_x = np.full((ntraj, len(x)), np.nan)
gamma_mat = np.full((ntraj, len(x)), np.nan)

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
c_gamma = a[i0_arr] * domega1_i0 * QH0/H0 * q_arr/Q0
c_tr = QH0/Q0/H0 * domega1_i0
# Order factor at birth: gamma from the upstream initialization (Eq. S3.6),
# zero for surface-born trajectories
gamma_mat[rows, i0_arr] = c_gamma*np.array(gi0_l)
# Eq. (14) at birth: dchi/dz = -D / (a0 * tau)
tau0_arr = np.array(tau0_l)
D_birth = np.empty(ntraj)
for t in range(ntraj):
    D_birth[t] = np.interp(S_ie[i0_arr[t]]-z0_arr[t], v_grid, D_col[i0_arr[t]])
with np.errstate(invalid='ignore', divide='ignore'):
    st_agegrad[rows, i0_arr] = -D_birth/(a0_arr*tau0_arr)
invtau[rows, i0_arr] = 1/tau[rows, i0_arr]
# Initialization of the horizontal age gradient
# Eq. (17) evaluated at the nodes where the trajectories enter the domain,
# with the LOCAL derivatives -- the same formula as the batch, so that the
# stored field is continuous across the boundary.

# Surface-born (entry node = birth node, at the surface): at zeta=1 the melt,
# flow, geometric and firn terms vanish, and kappa_x cancels with the
# accumulation term (a = a0), leaving the level-set identity
# chi_x = -chi_z * dS/dx (the age-0 isochrone at the surface is the surface).
invtau_x[rows, i0_arr] = dS_dx[i0_arr]
with np.errstate(invalid='ignore', divide='ignore'):
    st_agegrad_x[rows, i0_arr] = invtau_x[rows, i0_arr]*D_birth/a0_arr
dip[rows, i0_arr] = np.degrees(np.arctan2(invtau_x[rows, i0_arr], 1./tau0_arr))

# Upstream-born (entry node = upstream boundary): the full Eq. (17) with the
# local derivatives at i_b, as the batch would compute it there.
up = (in_arr == 'upstream')
if up.any():
    om_b = fct_domega_dzeta(zeta_arr, i_b)
    kap_b = Y[i_b]*y_arr*H_ie[i_b]/(QH[i_b]*om_b)
    D_b = D_birth[up]
    T2_b = -(a[i_b]*qq*Y[i_b]*H_ie[i_b])/(Q[i_b]*QH[i_b]*D_b*om_b)
    T3_b = (-(m[i_b]*Yb[i_b]*Q[i_b]-a[i_b]*Y[i_b]*Qm[i_b])/QH[i_b]**2*(1-qq/Q[i_b])
            - fct_domega_dx(zeta_arr, i_b))*H_ie[i_b]/(om_b*D_b) \
           + (-(1-zeta_arr)*dHie_dx[i_b])/D_b
    if not sf_single:
        u_dep_b = S_ie[i_b] - z_arr
        T3_b = T3_b - np.interp(u_dep_b, v_grid, TIE[i_b])
    T3_b = T3_b + dS_dx[i_b]
    invtau_x_b = a0_arr[up]*kap_b/D_b + (1./tau0_arr[up])*(T2_b + T3_b)
    invtau_x[up, i_b] = invtau_x_b
    st_agegrad_x[up, i_b] = invtau_x_b*D_b/a0_arr[up]
    dip[up, i_b] = np.degrees(np.arctan2(invtau_x_b, 1./tau0_arr[up]))

print('Before following trajectories:', round(time.perf_counter()-START_TIME, 4), 's.')


# ---------------------------------------------
# Follow-up of trajectories, batch by column
# ---------------------------------------------

with np.errstate(invalid='ignore', divide='ignore'): 
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
            st_agegrad[idx[exits], j:] = np.nan
            gamma_mat[idx[exits], j:] = np.nan
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
        y_j = fct_y(zeta, j); y_jb = fct_y(zetab, j-1)
        dy_j = fct_dy_dzeta(zeta, j); dy_jb = fct_dy_dzeta(zetab, j-1)
        kappab = Y[j-1]*y_jb*H_ie[j-1]/QH[j-1]/dom_dzeta_b
        kappa = Y[j]*y_j*H_ie[j]/QH[j]/dom_dzeta
        dage = (x[j]-x[j-1]) * np.sqrt(kappa*kappab)
        steady_age[idx, j] = steady_age[idx, j-1] + dage
        # Integrand of gamma, Eq. (S1.15) with Y = Y^S y(x, zeta):
        # Y^S H_ie/QH^2 * [ y omega'' / omega'^3 - (dy/dzeta) / omega'^2 ]
        gi_j = Y[j]*H_ie[j]/QH[j]**2*(y_j*fct_d2omega_dzeta2(zeta, j)/dom_dzeta**3
                                        - dy_j/dom_dzeta**2)
        gi_jb = Y[j-1]*H_ie[j-1]/QH[j-1]**2*(y_jb*fct_d2omega_dzeta2(zetab, j-1)/dom_dzeta_b**3
                                             - dy_jb/dom_dzeta_b**2)
        gammaint[idx, j] = gammaint[idx, j-1] + (x[j]-x[j-1])*mean_integrand(gi_j, gi_jb)
        gam = c_gamma[idx]*gammaint[idx, j]
        gamma_mat[idx, j] = gam
        u_dep = S_ie[j] - z_ie[idx, j]
        D = np.interp(u_dep, v_grid, D_col[j])
        # Inverse thinning ratio (always defined)
        invtau_j = (dom_dzeta*QH[j]/H_ie[j])*(1+gam)/(qq*c_tr[idx])
        invtau[idx, j] = invtau_j
        # Vertical age gradient (Eq. 14): chi_z = -(D/a0) * tau^-1
        agegrad_j = -(D/a0_arr[idx])*invtau_j        # <-- la définition manquante
        st_agegrad[idx, j] = agegrad_j
        # Vertical thinning ratio
        tau_j = 1./invtau_j
        tau[idx, j] = tau_j
        # --- finite parts of the isochrone orientation ---
        # Accumulation term: T2 = -(a Y^S Omega / Q) (dOmega/dz)^-1
        T2 = -(a[j]*qq*Y[j]*H_ie[j]) / (Q[j]*QH[j]*D*dom_dzeta)
        # iso-Omega slope: T3 = (dz/dx)|_Omega (melting, flow, geometric, firn)
        T3 = (-(m[j]*Yb[j]*Q[j]-a[j]*Y[j]*Qm[j]) / QH[j]**2 * (1-qq/Q[j])
              - fct_domega_dx(zeta, j)) * H_ie[j]/(dom_dzeta*D) \
             + (-(1-zeta)*dHie_dx[j]) / D
        if not sf_single:
            T3 = T3 - np.interp(u_dep, v_grid, TIE[j])
        T3 = T3 + dS_dx[j]
        # --- layer slope: singular at vertical tangents (tau -> inf)
        slope[idx, j] = kappa*a0_arr[idx]*tau_j/D + T2 + T3
        # --- horizontal age gradient: everywhere defined (the (1+gamma)
        # factors cancel: -chi_z * (kappa a0 tau / D) = kappa)
        # --- rescaled horizontal age gradient (primary, no singularity):
        #     tau_x = (a0/D) * chi_x = (a0/D) kappa + tau^-1 * (T2 + T3)
        invtau_x_j = a0_arr[idx]*kappa/D + invtau_j*(T2 + T3)
        invtau_x[idx, j] = invtau_x_j
        # Horizontal age gradient (Eq. 17): chi_x = (D/a0) * tau_x.
        # Singular when a0 -> 0, but that is the physical chi_x.
        st_agegrad_x[idx, j] = invtau_x_j*D/a0_arr[idx]
        # --- layer dip (degrees), from the rescaled age gradient vector,
        # unwrapped along the trajectory for angle continuity:
        dip_new = np.degrees(np.arctan2(invtau_x_j, invtau_j))
        prev = dip[idx, j-1]
        delta = dip_new - prev
        delta = delta - 360.*np.round(delta/360.)
        dip[idx, j] = np.where(np.isnan(prev), dip_new, prev + delta)
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
# Real-depth conversion of the trajectories
if sf_single:
    # Single profile: one vectorized interpolation over the whole field
    Zmat = S[None, :] - np.interp((S_ie[None, :]-z_ie).ravel(),
                                  v_grid, R_row).reshape(z_ie.shape)
else:
    Zmat = np.full_like(z_ie, np.nan)
    for j in range(len(x)):
        Zmat[:, j] = S[j] - np.interp(S_ie[j]-z_ie[:, j], v_grid, R_col[j])
AgeMat = np.interp(steady_age.ravel(), steady_age_R, age_R).reshape(steady_age.shape)
Rfac = np.interp(steady_age.ravel(), steady_age_R, R).reshape(steady_age.shape)
with np.errstate(invalid='ignore'):
    agegrad = st_agegrad*Rfac
    agegrad_x = st_agegrad_x*Rfac
Xmat = np.where(np.isnan(z_ie), np.nan, x)
Qmat = np.where(np.isnan(z_ie), np.nan, q_arr[:, None])
# Layer dip (degrees), from the age gradient pair (always defined).
# The tangent direction is taken as (-chi_z, chi_x) — the same line as
# (chi_z, -chi_x) — so that ordered stratigraphy reads the conventional
# dip in (-90, 90) deg directly, with no folding. The angle is continuous
# through vertical tangents (+/-90 deg); it only branches at +/-180 deg
# (horizontal overturned layers), where a future per-trajectory
# unwrapping would make it a continuous rotation angle.
if slope_diff:
    ziea_iso[:, -1] = np.nan
    ZBmat = np.full((ntraj, len(x)), np.nan)
    ZAmat = np.full((ntraj, len(x)), np.nan)
    if sf_single:
        ZBmat[:, 1:] = S[None, :-1] - np.interp(
            (S_ie[None, :-1]-zieb_iso[:, 1:]).ravel(),
            v_grid, R_row).reshape(ntraj, len(x)-1)
        ZAmat[:, :-1] = S[None, 1:] - np.interp(
            (S_ie[None, 1:]-ziea_iso[:, :-1]).ravel(),
            v_grid, R_row).reshape(ntraj, len(x)-1)
    else:
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
    pc['z'] = S[ii]-np.interp(S_ie[ii]-pc['z_ie'], v_grid, R_col[ii])
    pc['steady_age'] = steady_age[sel, ii]
    pc['tau'] = tau[sel, ii]
    pc['invtau'] = invtau[sel, ii]
    pc['st_agegrad'] = st_agegrad[sel, ii]
    pc['agegrad'] = agegrad[sel, ii]
    pc['slope'] = slope[sel, ii]
    pc['dip'] = dip[sel, ii]
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

with np.errstate(divide='ignore', invalid='ignore'): 
    tfrac = np.where(ok, (S - z1)/np.where(dz == 0., 1., dz), 0.)
    def surf_extrap(Y):
        y1, y2 = Y[top, cols], Y[top2, cols]
        return np.where(ok, y1 + tfrac*(y1-y2), np.nan)

def surf_nearest(Y):
    # Surface value = value of the topmost alive trajectory: the newborn
    # (exactly at the surface) in accumulation areas, the emerging one
    # (just below the surface) in ablation areas. No extrapolation: avoids
    # the tfrac blow-up where the top two trajectories are nearly
    # coincident (near the accu/ablation transition, where a0 -> 0 keeps
    # the newborns at the surface).
    return np.where(alive.any(axis=0), Y[top, cols], np.nan)

surf_steady_age = surf_extrap(steady_age)
surf_tau        = surf_extrap(tau)
surf_invtau = surf_extrap(invtau)
surf_slope      = surf_extrap(slope)
surf_dip = surf_extrap(dip)
surf_agegrad = surf_nearest(agegrad)
surf_order_factor = surf_extrap(1.+gamma_mat)

surf_age = np.interp(surf_steady_age, steady_age_R, age_R)

print('Before creating figures:', round(time.perf_counter()-START_TIME, 4), 's.')

# ---------------------------------------------------------
# Preparing for figures (in 2D)
# ---------------------------------------------------------


cuts = [0] + list(index_unconf) + [ntraj]
domain_slices = [slice(cuts[k], cuts[k+1]) for k in range(len(cuts)-1)]
unconf_list = [Zmat[iu] for iu in index_unconf]

traj_sel = list(range(0, ntraj, step_traj))
if fig_age_maxrange is None:
    # Auto range from the sampled trajectories
    age_max = max(np.nanmax(AgeMat[t]/1000.) for t in traj_sel)
else:
    # Manual cap (kyr), to avoid a few barely visible deep points
    # stretching the whole color scale
    age_max = fig_age_maxrange
linthresh = age_max/scaling_age
norm_age = SymLogNorm(linthresh=linthresh, linscale=1, vmin=0., vmax=age_max)


order_max = max(np.nanpercentile(1.+gamma_mat[t], 99.) for t in traj_sel)
norm_order = SymLogNorm(linthresh=1., linscale=1, vmin=-order_max, vmax=order_max)
levels_order = norm_order.inverse(np.linspace(0., 1., 100))

if fig_agegrad_maxrange is None:
    # Auto range from the sampled trajectories
    grad_max = max(np.nanpercentile(np.abs(agegrad[t]), 99.) for t in traj_sel)
else:
    grad_max = fig_agegrad_maxrange
norm_grad = SymLogNorm(linthresh=grad_max/scaling_age, linscale=1,
                       vmin=-grad_max, vmax=grad_max)
levels_grad = norm_grad.inverse(np.linspace(0., 1., 100))

tau_max = max(np.nanmax(tau[t]) for t in traj_sel)
norm_tau = Normalize(vmin=0., vmax=tau_max)

# Range for the inverse thinning ratio: cap the upper end (near the bed,
# tau -> 0 makes invtau -> large), either manually or at the 99th
# percentile of the sampled trajectories.
if fig_invtau_maxrange is None:
    invtau_max = max(np.nanpercentile(invtau[t], 99.) for t in traj_sel)
else:
    invtau_max = fig_invtau_maxrange
# Linear between 0 and 1 (where the physics of the transition
# thinning/expansion lives), logarithmic above.
norm_invtau = SymLogNorm(linthresh=1., linscale=1, vmin=0., vmax=invtau_max)
levels_invtau = norm_invtau.inverse(np.linspace(0., 1., 100))

if fig_slope_maxrange is None:
    slope_min = min(np.nanmin(slope[t]) for t in traj_sel)
    slope_max = max(np.nanmax(slope[t]) for t in traj_sel)
    slope_max = max(-slope_min, slope_max)
else:
    slope_max = fig_slope_maxrange
levels_slope = np.arange(-slope_max, slope_max+0.000001, slope_max/50)


if fig_dip_maxrange is None:
    # Auto range from the sampled trajectories. The dip is bounded by
    # +/-90 deg, but gentle cases (e.g. DC-BELDC) deserve a finer scale.
    dip_max = max(np.nanmax(np.abs(dip[t])) for t in traj_sel)
else:
    dip_max = fig_dip_maxrange
levels_dip = np.arange(-dip_max, dip_max+1e-6, dip_max/25.)

for name in ic:
    ic[name]['XX'] = ic[name]['x'] * np.ones(2)
    ic[name]['S'] = np.interp(ic[name]['x'], x/1000., S)
    ic[name]['ZZ'] = np.array([ic[name]['S'], ic[name]['S']-ic[name]['max_depth']])
    ic[name]['DD'] = np.array([0., ic[name]['max_depth']])

# ---------------------------------------------------------
# Re-usable panels
# ---------------------------------------------------------

def top_accu(ax1, yred, ylabel, color='r', comp_x=None, comp_y=None,
             marker=None, ylim_fix=None, yscale_fix=None):
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
    if yscale_fix is not None:
        yscale_fix(ax1b)          # ex.: ax.set_yscale('symlog', linthresh=...)
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

def plot_comp_unconf(ax2, depthmode):
    if comp_unconf is None:
        return
    for i in range(cp_unc_nb):
        ax2.plot(cp_unc_x,
                 cp_unc_depth[i, :] if depthmode else cp_unc_z[i, :],
                 color='k', linewidth=1.5, linestyle='dotted')

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
    plot_comp_unconf(ax2, depthmode)
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
# Order factor parameter
# ---------------------------------------------------------

def order_factor_figure(depthmode, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,
                                   figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})
    top_accu(ax1, surf_order_factor, 'surf. order factor')
    bed_panel(ax2, depthmode)
    for sl in domain_slices:
        zt = (S[None, :]-Zmat[sl]) if depthmode else Zmat[sl]
        ax2.contourf(Xmat[sl]/1000., zt, 1.+gamma_mat[sl], levels=levels_order,
                     cmap='PuOr', norm=norm_order)
        # Zero contour: vertical isochrone tangent; negative: overturned
        ax2.contour(Xmat[sl]/1000., zt, 1.+gamma_mat[sl], levels=[0.],
                    colors='k', linewidths=2)
    plot_unconf(ax2, depthmode)
    plot_comp_unconf(ax2, depthmode)
    ax2.set_xlabel('horizontal distance (km)', fontsize=19)
    ax2.set_ylabel('depth (m)' if depthmode else 'elevation (m)', fontsize=19)
    if depthmode:
        ax2.set_ylim(ax2.get_ylim()[::-1])
    sm = plt.cm.ScalarMappable(cmap='PuOr', norm=norm_order)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
    cb.set_label(label='order factor (1+$\\gamma$)', size=18)
    plot_ic(ax2, depthmode)
    savefig(fname)

order_factor_figure(False, 'order_factor_x_z')
order_factor_figure(True, 'order_factor_x_depth')
print('After order figures:', round(time.perf_counter()-START_TIME, 4), 's.')

# ---------------------------------------------------------
# Vertical age gradient
# ---------------------------------------------------------

def agegrad_figure(depthmode, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,
                                   figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})
    top_accu(ax1, surf_agegrad, 'surf. age gr. (yr/m)')
    bed_panel(ax2, depthmode)
    for sl in domain_slices:
        zt = (S[None, :]-Zmat[sl]) if depthmode else Zmat[sl]
        ax2.contourf(Xmat[sl]/1000., zt, agegrad[sl], levels=levels_grad,
                     cmap='PuOr', norm=norm_grad)
        # Zero contour: vertical isochrone tangent (overturning diagnostic)
        ax2.contour(Xmat[sl]/1000., zt, agegrad[sl], levels=[0.],
                    colors='k', linewidths=2)
    plot_unconf(ax2, depthmode)
    plot_comp_unconf(ax2, depthmode)
    ax2.set_xlabel('horizontal distance (km)', fontsize=19)
    ax2.set_ylabel('depth (m)' if depthmode else 'elevation (m)', fontsize=19)
    if depthmode:
        ax2.set_ylim(ax2.get_ylim()[::-1])
    sm = plt.cm.ScalarMappable(cmap='PuOr', norm=norm_grad)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
    cb.set_label(label='vertical age gradient (yr/m)', size=18)
    plot_ic(ax2, depthmode)
    savefig(fname)

agegrad_figure(False, 'age_gradient_x_z')
agegrad_figure(True, 'age_gradient_x_depth')
print('After age gradient figures:', round(time.perf_counter()-START_TIME, 4), 's.')

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

if fig_thinning:
    thinning_figure(False, 'thinning_x_z')
    thinning_figure(True, 'thinning_x_depth')
print('After thinning figures:', round(time.perf_counter()-START_TIME, 4), 's.')

# ---------------------------------------------------------
# 7b-8b. Inverse vertical thinning ratio
# ---------------------------------------------------------

def invtau_figure(depthmode, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,
                                   figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})
    top_accu(ax1, surf_invtau, 'surface inverse thinning ratio',
             ylim_fix=lambda b, t: (min(b, 0.), max(t, 2.)))
    bed_panel(ax2, depthmode)
    for sl in domain_slices:
        zt = (S[None, :]-Zmat[sl]) if depthmode else Zmat[sl]
        ax2.contourf(Xmat[sl]/1000., zt, invtau[sl], levels=levels_invtau,
                     cmap='plasma_r', norm=norm_invtau)
        # Key contours: invtau = 1 (thinning/expansion transition),
        # invtau = 0 (vertical isochrone tangent, i.e. 1+gamma = 0)
        cp = ax2.contour(Xmat[sl]/1000., zt, invtau[sl],
                         levels=[0., 1.], colors='k', norm=norm_invtau)
    plot_unconf(ax2, depthmode)
    plot_comp_unconf(ax2, depthmode)
    ax2.set_xlabel('horizontal distance (km)', fontsize=19)
    ax2.set_ylabel('depth (m)' if depthmode else 'elevation (m)', fontsize=19)
    if depthmode:
        ax2.set_ylim(ax2.get_ylim()[::-1])
    sm = plt.cm.ScalarMappable(cmap='plasma_r', norm=norm_invtau)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
    cb.set_label(label='inverse vertical thinning ratio (no unit)', size=18)
    cb.add_lines(cp)
    plot_ic(ax2, depthmode)
    savefig(fname)

invtau_figure(False, 'inv_thinning_x_z')
invtau_figure(True, 'inv_thinning_x_depth')
print('After inverse thinning figures:', round(time.perf_counter()-START_TIME, 4), ' s.')


# ---------------------------------------------------------
# Layer slope
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
    cb = plt.colorbar(cp, pad=-0.02)
    cb.set_label(label='layer slope (no unit)', size=18)
    plot_ic(ax2, depthmode)
    savefig(fname)

slope_figure(False, 'layer_slope_x_z')
slope_figure(True, 'layer_slope_x_depth')



# ---------------------------------------------------------
# Layer dip
# ---------------------------------------------------------

def dip_figure(depthmode, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,
                                   figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})
    top_accu(ax1, surf_dip, 'surf. layer dip (°)')
    bed_panel(ax2, depthmode)
    for sl in domain_slices:
        zt = (S[None, :]-Zmat[sl]) if depthmode else Zmat[sl]
        cp = ax2.contourf(Xmat[sl]/1000., zt, dip[sl],
                          levels=levels_dip, cmap='PuOr')
    plot_unconf(ax2, depthmode)
    plot_comp_unconf(ax2, depthmode)
    ax2.set_xlabel('horizontal distance (km)', fontsize=19)
    ax2.set_ylabel('depth (m)' if depthmode else 'elevation (m)', fontsize=19)
    if depthmode:
        ax2.set_ylim(ax2.get_ylim()[::-1])
    cb = plt.colorbar(cp, pad=-0.02)
    cb.set_label(label='layer dip (°)', size=18)
    plot_ic(ax2, depthmode)
    savefig(fname)

dip_figure(False, 'layer_dip_x_z')
dip_figure(True, 'layer_dip_x_depth')
print('After layer dip figures:', round(time.perf_counter()-START_TIME, 4), 's.')


# ---------------------------------------------------------
# 11-12. Stream function
# ---------------------------------------------------------

q_min, q_max = np.nanmin(Qmat), np.nanmax(Qmat)
q_max = np.nanmax(Qmat)
q_lin = q_max/scaling_q

norm_q = SymLogNorm(linthresh=q_lin, linscale=1, vmin=0., vmax=q_max)
levels_q = norm_q.inverse(np.linspace(0., 1., 21))

def q_figure(depthmode, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,
                                   figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})
    top_accu(ax1, Q, 'Total flux (m$^3$/yr)',
             yscale_fix=lambda ax: ax.set_yscale('symlog', linthresh=q_lin))
    bed_panel(ax2, depthmode)
    for sl in domain_slices:
        zt = (S[None, :]-Zmat[sl]) if depthmode else Zmat[sl]
        ax2.contourf(Xmat[sl]/1000., zt, Qmat[sl], levels=levels_q,
                     cmap='viridis', norm=norm_q)
        ax2.contour(Xmat[sl]/1000., zt, Qmat[sl], levels=levels_q[1:-1:2],
                    colors='k', linewidths=0.5)
    plot_unconf(ax2, depthmode)
    ax2.set_xlabel('horizontal distance (km)', fontsize=19)
    ax2.set_ylabel('depth (m)' if depthmode else 'elevation (m)', fontsize=19)
    if depthmode:
        ax2.set_ylim(ax2.get_ylim()[::-1])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm_q)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
    cb.set_label(label='stream function q (m$^3$/yr)', size=18)
    plot_ic(ax2, depthmode)
    savefig(fname)

q_figure(False, 'stream_function_x_z')
q_figure(True, 'stream_function_x_depth')

# ---------------------------------------------------------
# Tube width
# ---------------------------------------------------------

# Width Y(x, zeta) = Y^S(x) y(x, zeta) along the trajectories
Ymat = np.full_like(z_ie, np.nan)
for j in range(len(x)):
    Ymat[:, j] = Y[j]*np.interp((z_ie[:, j]-B[j])/H_ie[j], zeta_grid, y_tab[:, j])

def tube_width_norm():
    # Scale from the whole tube (all x, from the bed to the surface), shared
    # by the 2D and 3D figures. 'auto': logarithmic when the width varies by
    # more than a factor 20 (e.g. exponential widening on a ridge), symlog
    # when it reaches zero, linear otherwise.
    vals = (Y[None, :]*y_tab).ravel()
    vals = vals[np.isfinite(vals)]
    vmin, vmax = vals.min(), vals.max()
    if fig_tube_width_maxrange is not None:
        vmax = fig_tube_width_maxrange
    scale = fig_tube_width_scale
    if scale == 'auto':
        if vmin <= 0.:
            scale = 'symlog'
        elif vmax/vmin >= 20.:
            scale = 'log'
        else:
            scale = 'linear'
    if scale == 'log' and vmin <= 0.:
        print('Tube width reaches zero: the logarithmic scale is replaced by symlog.')
        scale = 'symlog'
    linthresh = vmax/scaling_width
    if scale == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif scale == 'symlog':
        norm = SymLogNorm(linthresh=linthresh, linscale=1, vmin=min(vmin, 0.), vmax=vmax)
    else:
        if vmax > vmin:
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:   # constant width: centred colour scale
            norm = Normalize(vmin=0.5*vmin, vmax=1.5*vmin)
    return norm, scale, linthresh

norm_width, width_scale, width_linthresh = tube_width_norm()
if width_scale == 'linear':
    levels_width = np.linspace(norm_width.vmin, norm_width.vmax, 100)
else:
    levels_width = norm_width.inverse(np.linspace(0., 1., 100))

def tube_width_figure(depthmode, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True,
                                   figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})
    if width_scale == 'log':
        yfix = lambda ax: ax.set_yscale('log')
    elif width_scale == 'symlog':
        yfix = lambda ax: ax.set_yscale('symlog', linthresh=width_linthresh)
    else:
        yfix = None
    top_accu(ax1, Y, 'surface tube width (m)', yscale_fix=yfix)
    bed_panel(ax2, depthmode)
    for sl in domain_slices:
        zt = (S[None, :]-Zmat[sl]) if depthmode else Zmat[sl]
        ax2.contourf(Xmat[sl]/1000., zt, Ymat[sl], levels=levels_width,
                     cmap='viridis', norm=norm_width, extend='max')
    plot_unconf(ax2, depthmode)
    plot_comp_unconf(ax2, depthmode)
    ax2.set_xlabel('horizontal distance (km)', fontsize=19)
    ax2.set_ylabel('depth (m)' if depthmode else 'elevation (m)', fontsize=19)
    if depthmode:
        ax2.set_ylim(ax2.get_ylim()[::-1])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm_width)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax2, pad=-0.02)
    cb.set_label(label='tube width Y (m)', size=18)
    plot_ic(ax2, depthmode)
    savefig(fname)

tube_width_figure(False, 'tube_width_x_z')
tube_width_figure(True, 'tube_width_x_depth')

# ---------------------------------------------------------
# 3D view of the flow tube
# ---------------------------------------------------------

def tube_3d_figure(fname):
    # The model only knows the width Y(x, zeta) of the tube, not its lateral
    # position: the tube is drawn symmetric about the flow line (lateral
    # coordinate w = 0). The three axes are not to scale (see box aspect).
    ix = np.unique(np.round(np.linspace(0, len(x)-1, min(len(x), 150))).astype(int))
    nzp = 31
    zeta_p = np.linspace(0., 1., nzp)
    X3 = np.empty((nzp, ix.size)); Z3 = np.empty_like(X3); Y3 = np.empty_like(X3)
    for k, j in enumerate(ix):
        X3[:, k] = x[j]/1000.
        # Same ice-equivalent -> real elevation conversion as for Zmat
        Z3[:, k] = S[j] - np.interp(S_ie[j]-(B[j]+zeta_p*H_ie[j]), v_grid, R_col[j])
        Y3[:, k] = Y[j]*np.interp(zeta_p, zeta_grid, y_tab[:, j])
    norm_w = norm_width
    cmap_w = plt.cm.viridis
    fcol = cmap_w(norm_w(Y3))
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(projection='3d')
    kw = dict(rstride=1, cstride=1, edgecolor='none', linewidth=0, antialiased=True, shade=False, alpha=0.5)
    ax.plot_surface(X3, Y3/2., Z3, facecolors=fcol, **kw)     # right wall
    ax.plot_surface(X3, -Y3/2., Z3, facecolors=fcol, **kw)    # left wall
    # Surface and bed ribbons
    for row, col in ((-1, 'white'), (0, 'grey')):
        ax.plot_surface(X3[[row, row], :], np.vstack((-Y3[row]/2., Y3[row]/2.)),
                        Z3[[row, row], :], color=col, edgecolor='none', linewidth=0,
                        antialiased=True, shade=False, alpha=0.6)
    # Edges of the tube (surface and bed lines) and cross sections
    for row in (0, -1):
        for sgn in (-1, 1):
            ax.plot(X3[row], sgn*Y3[row]/2., Z3[row], color='k', linewidth=1.2)
    for k in np.linspace(0, ix.size-1, 9).astype(int):
        wl = np.concatenate((Y3[:, k]/2., -Y3[::-1, k]/2., [Y3[0, k]/2.]))
        zl = np.concatenate((Z3[:, k], Z3[::-1, k], [Z3[0, k]]))
        ax.plot(X3[0, k]*np.ones_like(wl), wl, zl, color='k', linewidth=0.8)
    if fig_tube_3d_traj:
        for t in traj_sel:
            ax.plot(Xmat[t]/1000., np.zeros_like(Xmat[t]), Zmat[t], linewidth=0.5,
                    color='k' if in_arr[t] == 'upstream' else 'b', alpha=0.6)
    for name in ic:
        ax.plot(ic[name]['XX'], np.zeros(2), ic[name]['ZZ'], linewidth=lw_core,
                color=color_core, linestyle=ls_core)
    ax.set_xlabel('horizontal distance (km)', fontsize=14, labelpad=14)
    ax.set_ylabel('lateral position (m)', fontsize=14, labelpad=10)
    ax.set_zlabel('elevation (m)', fontsize=14, labelpad=10)
    wmax = max(Y3.max()/2., 1e-9)*1.05
    ax.set_ylim(-wmax, wmax)
    ax.set_box_aspect(fig_tube_3d_aspect, zoom=1.05)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.view_init(elev=fig_tube_3d_view[0], azim=fig_tube_3d_view[1])
    sm = plt.cm.ScalarMappable(cmap=cmap_w, norm=norm_w)
    sm.set_array([])
    fig.subplots_adjust(left=0.0, right=0.72)
    cax = fig.add_axes([0.74, 0.3, 0.015, 0.4])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label='tube width Y (m)', size=16)
    savefig(fname)

tube_3d_figure('tube_3d')

# ---------------------------------------------------------
# 13. Ice ores
# ---------------------------------------------------------

for name in ic:
    ii = np.argmin(np.abs(x/1000. - ic[name]['x']))
    pc = prof_col(ii)
    # Mask values beyond max_depth, so that figure auto-scale correctly
    zmin = S[ii] - ic[name]['max_depth']
    visible = pc['z'] >= zmin
    def in_window(yv):
        return np.where(visible, yv, np.nan)


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
    ax2.set_xlim(left=x[0], right=ic[name]['x'])
    ax2.set_xlabel('ice origin (km)', color='r', fontsize=15)
    ax2.spines['top'].set_color('r')
    ax2.tick_params(axis='x', colors='r')
    ax2.spines['bottom'].set_visible(False)

    ax3 = ax.twiny()
    # Vertical thinning axis: tau (default), its inverse tau^-1 (primary,
    # singularity-free quantity), or the vertical age gradient. The
    # comparison curves from age_flow_line are converted accordingly.
    if fig_ic_thinning == 'invthinning':
        yval = pc['invtau']
        xlabel = 'inverse vertical thinning ratio (no unit)'
    elif fig_ic_thinning == 'agegrad':
        yval = -pc['agegrad']      # age density (positive in ordered regime)
        xlabel = 'age density (yr/m)'
    else:
        yval = pc['tau']
        xlabel = 'vertical thinning ratio (no unit)'
    for i in range(len(prof_domains)-1):
        ax3.plot(in_window(yval)[prof_domains[i]:prof_domains[i+1]],
                 in_window(pc['z'])[prof_domains[i]:prof_domains[i+1]], color='orange')
    if ic[name]['comp'] is not None:
        # The imported file contains the thinning ratio: convert it to
        # the plotted quantity when needed.
        if fig_ic_thinning == 'invthinning':
            cp_yval = ic[name]['cp_invtau']
        elif fig_ic_thinning == 'agegrad':
            cp_yval = ic[name]['cp_agegrad']
        else:
            cp_yval = ic[name]['cp_tau']
        ax3.plot(cp_yval, S[ii]-ic[name]['cp_depth'], color='orange', linestyle='--')
    if fig_ic_thinning == 'invthinning':
        ax3.set_xscale('symlog', linthresh=1., linscale=1.)
    elif fig_ic_thinning == 'agegrad':
        finite = yval[np.isfinite(yval)]
        if finite.size and finite[-1] > 0.:
            linthresh = finite[-1]*10
        else:
            linthresh = 1.
        ax3.set_xscale('symlog', linthresh=linthresh, linscale=1.)
    
    ax3.set_xlabel(xlabel, color='orange', fontsize=15)
    ax3.spines['top'].set_color('orange')
    ax3.spines['top'].set_position(('axes', 1.1))
    ax3.tick_params(axis='x', colors='orange')
    if fig_ic_thinning == 'thinning':
        ax3.set_xlim(left=0)

    ax4 = ax.twiny()
    if fig_ic_dip:
        for i in range(len(prof_domains)-1):
            if slope_diff:
                ax4.plot(in_window(np.degrees(np.arctan(pc['slope_diff'])))[prof_domains[i]:prof_domains[i+1]],
                         pc['z'][prof_domains[i]:prof_domains[i+1]], color='violet', linestyle='dotted')
            ax4.plot(in_window(pc['dip'])[prof_domains[i]:prof_domains[i+1]],
                     pc['z'][prof_domains[i]:prof_domains[i+1]], color='violet')
        if ic[name]['comp'] is not None:
            # Comparison dip converted from the imported slope
            ax4.plot(ic[name]['cp_dip'],
                     S[ii]-ic[name]['cp_depth'], color='violet', linestyle='--')
        ax4.set_xlabel('layer dip (°)', color='violet', fontsize=15)
    else:
        for i in range(len(prof_domains)-1):
            if slope_diff:
                ax4.plot(in_window(pc['slope_diff'])[prof_domains[i]:prof_domains[i+1]],
                         pc['z'][prof_domains[i]:prof_domains[i+1]], color='violet', linestyle='dotted')
            ax4.plot(in_window(pc['slope'])[prof_domains[i]:prof_domains[i+1]],
                     pc['z'][prof_domains[i]:prof_domains[i+1]], color='violet')
        if ic[name]['comp'] is not None:
            # Comparison slope: imported as such, no conversion needed
            ax4.plot(ic[name]['cp_slope'], S[ii]-ic[name]['cp_depth'],
                     color='violet', linestyle='--')
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
# OUTPUT Flow line
# ---------------------------------------------------------

if output_fl:
    output = np.vstack((x/1000., Y, a*R[0], m*R[0], Q, Qm, surf_vel))
    np.savetxt(datadir+'flow_line_output.txt', np.transpose(output),
               header='x(km) tube_width PD_accu(m/yr) PD_melting(m/yr) total_flux melting_flux PD_surf_velocity(m/yr)')

# ---------------------------------------------------------
# OUTPUT ice core
# ---------------------------------------------------------

for name in ic:
    ii = np.argmin(np.abs(x/1000. - ic[name]['x']))
    pc = prof_col(ii)
    output = np.vstack((S[ii]-pc['z'], pc['age'], pc['x0'], pc['tau'],
                        pc['a0'], pc['dip']))
    prof_domains = pc['unconf']
    for i in range(len(prof_domains)):
        output = np.insert(output, prof_domains[i]+i, np.nan, axis=1)
    np.savetxt(datadir+name+'_ice_core_output.txt', np.transpose(output)[::-1],
               header='depth(m) age(yr) origin(km) thinning accu(m/yr) dip(deg)')

# ---------------------------------------------------------
# Isochrones output
# ---------------------------------------------------------

# Isochrone ages: same as plotted on the age figures (fig_age_iso
# is in kyr, while profile ages are in yr).
ages_iso = np.array(fig_age_iso)*1000.
iso_depth = np.full((len(x), len(ages_iso)), np.nan)
# For each column, interpolate the depth of each isochrone within
# its profile segment (age is monotonic inside a segment, with jumps
# at the unconformities; an age falling inside a jump does not exist
# at this x and stays NaN).
for ii in range(i_b, len(x)):
    pc = prof_col(ii)
    prof_domains = [0] + list(pc['unconf']) + [len(pc['z'])]
    for k in range(len(prof_domains)-1):
        d1, d2 = prof_domains[k], prof_domains[k+1]
        seg_age = pc['age'][d1:d2]        # decreasing along rows (old -> young)
        if seg_age.size < 2 or seg_age[0] == seg_age[-1]:
            continue
        a_inc = seg_age[::-1]             # increasing, as required by np.interp
        z_inc = pc['z'][d1:d2][::-1]
        in_seg = (ages_iso >= a_inc[0]) & (ages_iso <= a_inc[-1])
        if in_seg.any():
            iso_depth[ii, in_seg] = S[ii] - np.interp(ages_iso[in_seg],
                                                      a_inc, z_inc)
out = np.vstack((x/1000., iso_depth.T))
header = 'x(km) ' + ' '.join('depth_age_%gkyr(m)' % A
                             for A in fig_age_iso)
np.savetxt(datadir+'isochrones_output.txt', np.transpose(out),
           header=header)

MESSAGE = 'Program execution time: '+str(time.perf_counter()-START_TIME)+' s.'
print(MESSAGE)
if os.name != 'nt':
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max memory usage: '+str(mem)+' kbytes')