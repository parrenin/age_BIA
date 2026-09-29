# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# TODO: plot Y
# TODO: exact formula for the age increase
# TODO: age observations for ice cores
# TODO: implement solid fraction
# TODO: implement real_thickness
# TODO: use depth for ice core figure instead of z

import sys
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d
import yaml
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm
import time
import math
import os
import resource
import warnings


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
temp_factor_file = None

yamls = open(datadir+'parameters.yml').read()
para = yaml.load(yamls, Loader=yaml.FullLoader)
globals().update(para)


# ---------------------------------------------------------
# Fonctions definition
# ---------------------------------------------------------

def fct_omega(zeta, p):
    return 1 - (p + 2) / (p + 1) * (1 - zeta) + 1 / (p + 1) * (1 - zeta) ** (p + 2)

def fct_zeta(omega, p):
    # Définir la fonction dont on cherche la racine
    def equation(zeta):
        return fct_omega(zeta, p) - omega
    def derivative(zeta):
        return fct_domega_dzeta(zeta, p)

    # Résoudre numériquement
    zeta_solution = fsolve(equation, omega, fprime=derivative)  # omega is an initial value
    return zeta_solution[0]

def fct_domega_dzeta(zeta, p):
    return (p + 2) / (p + 1) * (1 - (1 - zeta) ** (p + 1))

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


# ---------------------------------------------------------------
# Loading data from temporal_factor.txt
# ---------------------------------------------------------------

if temp_factor_file != None:
    age_R, R = np.loadtxt(datadir+temp_factor_file, unpack=True)
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


# --------------------------
# Interpolation and fluxes
# --------------------------

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

mu = Qm / (Q - Qm)
QH = Q - Qm

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    surf_vel = Q / Y * fct_domega_dzeta(1., p) / H

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
profs = []
# Trajectories from the upstream border
zz_start = B[0] + ((S[0] - B[0]) % z_step)
for z in np.arange(zz_start, S[0], z_step):
    xx = np.array([x[0]])
    zz = np.array([z])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        age = np.array([integrate_age((zz-B[0])/H[0], 1, p[0], H[0], a[0], mu[0])])
    zeta = (zz-B[0])/H[0]
    omega = fct_omega(zeta, p[0]) 
    qq = Qm[0] + (Q[0]-Qm[0]) * omega
    traj = {'x': xx, 'z': zz, 'steady_age': age, 'in': 'upstream', 'out': None, 'x0': x[0]}
    if z == zz_start:
        profs = [{'z': zz, 'steady_age': age, 'x0': x[0], 'unconf': []}]
    else:
        profs[0]['z'] = np.append(profs[0]['z'], z)
        profs[0]['steady_age'] = np.append(profs[0]['steady_age'], age[0])
    j = 1
    while j < len(x) :
        traj['x'] = np.append(traj['x'], x[j])  # FIXME: seems suboptimal, we might just use x
        if Q[j] >= qq and Qm[j] <= qq and traj['out'] == None: 
            omega = (qq-Qm[j])/(Q[j]-Qm[j])
            zeta = fct_zeta(omega, p[j])
            traj['z'] = np.append(traj['z'], B[j] + zeta * H[j])
            ux1 = QH[j-1]/Y[j-1]/H[j-1]*fct_domega_dzeta((traj['z'][-2]-B[j-1])/H[j-1], p[j-1])
            ux2 = QH[j]/Y[j]/H[j]*fct_domega_dzeta((traj['z'][-1]-B[j])/H[j], p[j])
            ux = (ux1 + ux2 ) / 2                           # Can we do better here?
            dage = 1000 * (traj['x'][-1]-traj['x'][-2]) / ux
            traj['steady_age'] = np.append(traj['steady_age'], traj['steady_age'][-1]+dage)
            if j == len(x)-1:
                traj['out'] = 'downstream'
            try:
                profs[j]['z'] = np.append(profs[j]['z'], traj['z'][-1])
                profs[j]['steady_age'] = np.append(profs[j]['steady_age'], traj['steady_age'][-1])
                profs[j]['x0'] = np.append(profs[j]['x0'], traj['x0'])
            except:
                profs.append({'z': traj['z'][-1], 'steady_age': traj['steady_age'][-1], 'x0': traj['x0'], 'unconf': []})
        elif Q[j] < qq or traj['out'] == 'surf':
            traj['z'] = np.append(traj['z'], np.nan)
            traj['steady_age'] = np.append(traj['steady_age'], np.nan)
            traj['out'] = 'surf'
        elif Qm[j] > qq or traj['out'] == 'bed':
            traj['z'] = np.append(traj['z'], np.nan)
            traj['steady_age'] = np.append(traj['steady_age'], np.nan)
            traj['out'] = 'bed'
        j += 1
    traj['age'] = np.interp(traj['steady_age'], steady_age_R, age_R)
    trajs.append(traj)
index_unconf = []
# Trajectories from the surface
for k in range(len(accu_starts)):
    if k>0:
        index_unconf.append(len(trajs))
    istart = accu_starts[k]
    try:
        iend = accu_ends[k]
    except:
        iend = len(x)-1
    print('Accu area n°', k, 'starts at x=',istart*x_step,'km and ends at x=', iend*x_step, 'km')
    for i in range(istart, iend+1):
        xx = np.concatenate([np.ones(i)*np.nan, np.array([x[i]])])
        zz = np.concatenate([np.ones(i)*np.nan, np.array([S[i]])])
        age = np.concatenate([np.ones(i)*np.nan, np.array([0.])])
        traj = {'x':xx, 'z':zz, 'steady_age': age, 'in': 'surf', 'out': None, 'x0': x[i]}
        profs[i]['z'] = np.append(profs[i]['z'], traj['z'][-1])
        profs[i]['steady_age'] = np.append(profs[i]['steady_age'], traj['steady_age'][-1])
        profs[i]['x0'] = np.append(profs[i]['x0'], traj['x0'])
        j = i+1
        while j < len(x) :
            traj['x'] = np.append(traj['x'], x[j])
            if Q[j] >= Q[i] and Qm[j] <= Q[i] and traj['out'] == None :
                omega = (Q[i]-Qm[j])/(Q[j]-Qm[j])
                zeta = fct_zeta(omega, p[j])
                traj['z'] = np.append(traj['z'], B[j] + zeta * H[j])
                ux1 = QH[j-1]/Y[j-1]/H[j-1]*fct_domega_dzeta((traj['z'][-2]-B[j-1])/H[j-1], p[j-1])
                ux2 = QH[j]/Y[j]/H[j]*fct_domega_dzeta((traj['z'][-1]-B[j])/H[j], p[j])
                ux = (ux1 + ux2 ) / 2                           # Can we do better here?
                dage = 1000 * (traj['x'][-1]-traj['x'][-2]) / ux
                traj['steady_age'] = np.append(traj['steady_age'], traj['steady_age'][-1]+dage)
                # if len(trajs) == 243:
                #     print(omega, zeta)
                if j == len(x)-1:
                    traj['out'] = 'downstream'
                try:
                    profs[j]['z'] = np.append(profs[j]['z'], traj['z'][-1])
                    profs[j]['steady_age'] = np.append(profs[j]['steady_age'], traj['steady_age'][-1])
                    profs[j]['x0'] = np.append(profs[j]['x0'], traj['x0'])
                    if k>0 and i == istart:
                        profs[j]['unconf'].append(len(profs[j]['z'])-1)
                except:
                    profs.append({'z': traj['z'][-1], 'steady_age': traj['steady_age'][-1], 'x0': traj['x0'], 'unconf': []})
            elif Q[j] < Q[i] or traj['out'] == 'surf':
                traj['z'] = np.append(traj['z'], np.nan)
                traj['steady_age'] = np.append(traj['steady_age'], np.nan)
                traj['out'] = 'surf'
            elif Qm[j] > Q[i] or traj['out'] == 'bed':
                traj['z'] = np.append(traj['z'], np.nan)
                traj['steady_age'] = np.append(traj['steady_age'], np.nan)
                traj['out'] = 'bed'
            j += 1
        traj['age'] = np.interp(traj['steady_age'], steady_age_R, age_R)
        trajs.append(traj)

for i in range(len(x)-1):
    profs[i]['age'] = np.interp(profs[i]['steady_age'], steady_age_R, age_R)

print('Before creating figures:',
      round(time.perf_counter()-START_TIME, 4), 's.')        
  
    
# ----------------------------------------
# Computation of surface ages
# ----------------------------------------

surf_steady_age = np.zeros_like(x)
for i in range(len(x)):
    try:
        surf_steady_age[i] = interp1d(profs[i]['z'], profs[i]['steady_age'], kind='linear', fill_value="extrapolate")(S[i])
    except:
        surf_steady_age[i] = np.nan
surf_age = np.interp(surf_steady_age, steady_age_R, age_R)        

# ----------------------------------------
# Preparing for figures
# ----------------------------------------

# Concatenation in lists for trajectories plots
x_list = []
z_list = []
age_list = []
in_list = []
for i in range(0, len(trajs), step_traj):
    traj = trajs[i]
    x_list.append(traj['x'])
    z_list.append(traj['z'])
    age_list.append(traj['age'])
    in_list.append(traj['in'])

# Creation of domains and unformities

mat_x = np.array([trajs[0]['x']])
mat_z = np.array([trajs[0]['z']])
mat_age = np.array([trajs[0]['age']])
if len(index_unconf) ==0:
    end = len(trajs)
else:
    end = index_unconf[0]
for i in range(1, end):
#    print(np.shape(mat_x), np.shape(np.array([trajs[i]['x']])), trajs[i]['z'][0])
    mat_x = np.concatenate([mat_x, np.array([trajs[i]['x']])])
    mat_z = np.concatenate([mat_z, np.array([trajs[i]['z']])])
    mat_age = np.concatenate([mat_age, np.array([trajs[i]['age']])])
domains_list = [{'mat_x': mat_x, 'mat_z': mat_z, 'mat_age': mat_age}]
unconf_list = []


for k in range(len(index_unconf)):
    traj_z = trajs[index_unconf[k]]['z']  # FIXME: average with traj below, but not so easy...
    unconf_list.append(traj_z)
    mat_x = np.array([trajs[index_unconf[k]]['x']])
    mat_z = np.array([trajs[index_unconf[k]]['z']])
    mat_age = np.array([trajs[index_unconf[k]]['age']])
    if k == len(index_unconf)-1:
        end = len(trajs)
    else:
        end = index_unconf[k+1]
    for i in range(index_unconf[k]+1, end):
    #    print(np.shape(mat_x), np.shape(np.array([trajs[i]['x']])), trajs[i]['z'][0])
        mat_x = np.concatenate([mat_x, np.array([trajs[i]['x']])])
        mat_z = np.concatenate([mat_z, np.array([trajs[i]['z']])])
        mat_age = np.concatenate([mat_age, np.array([trajs[i]['age']])])
    domains_list.append({'mat_x': mat_x, 'mat_z': mat_z, 'mat_age': mat_age})


# Quantities for ice cores
     
for name in ic:
    ic[name]['XX'] = ic[name]['x'] * np.ones(2)
    ic[name]['S'] = np.interp(ic[name]['x'], x, S)
    ic[name]['ZZ'] = np.array([ic[name]['S'], ic[name]['S']-ic[name]['max_depth']])

# ---------------------
# Trajectories figure
# ---------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x, a, color='k', linewidth=2)
ax1.plot(x, np.zeros_like(x), color='k', linewidth=2, linestyle='--')
ax1.set_ylabel('a (m/yr)', fontsize=19)

ax1b = ax1.twinx()
ax1b.plot(x, surf_vel, color='r', linewidth=2)
ax1b.set_ylabel('surface velocity (m/yr)', color='r')
ax1b.tick_params(axis='y', colors='r')

ax2.plot(x, S, label='Surface', color='k', linewidth=2.)
ax2.plot(x, B, label='Bedrock', color='k', linewidth=2.)
for xx, zz, iin in zip(x_list, z_list, in_list):
    if iin == 'upstream': 
        ax2.plot(xx, zz, color='k')
    elif iin == 'surf':
        ax2.plot(xx, zz, color='b')
    
for k in range(len(index_unconf)):
    traj = trajs[index_unconf[k]]
    ax2.plot(traj['x'], traj['z'], label='Discontinuities', color='k', linewidth=4., linestyle='--')

ax2.set_xlabel(r'$x$ (km)', fontsize=19)
ax2.set_ylabel(r'$z$ (m)', fontsize=19)

for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], ic[name]['S']+20), ha='center',
                 va='bottom', color=color_core)

plt.savefig(datadir+'trajectories_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()


print('After trajectories figure:',
      round(time.perf_counter()-START_TIME, 4), 's.')        

# ---------------------------
# Color trajectories figure
# ---------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, 
                               figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})
ax1.plot(x, a, color='k', linewidth=2)
ax1.plot(x, np.zeros_like(x), color='k', linewidth=2, linestyle='--')
ax1.set_ylabel('a (m/yr)', fontsize=19)

ax1b = ax1.twinx()
ax1b.plot(x, surf_vel, color='r', linewidth=2)
ax1b.set_ylabel('surface velocity (m/yr)', color='r')
ax1b.tick_params(axis='y', colors='r')

age_min = 0.
age_max = max(max(age/1000.) for age in age_list)
linthresh = age_max/scaling_age  # Seuil en dessous duquel l'échelle devient linéaire (évite log(0))
norm = SymLogNorm(linthresh=linthresh, linscale=1, vmin=age_min, vmax=age_max)

ax2.plot(x, S, label='Surface', color='k', linewidth=2.)
ax2.plot(x, B, label='Bedrock', color='k', linewidth=2.)

for xx, zz, aa in zip(x_list, z_list, age_list):
    # Créer des segments entre les points consécutifs
    points = np.array([xx, zz]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Créer une LineCollection pour cette ligne
    lc = LineCollection(segments, cmap='plasma', norm=norm)
    lc.set_array(aa/1000.)  # Associer les valeurs z aux segments
    lc.set_linewidth(2)
    ax2.add_collection(lc)

for k in range(len(index_unconf)):
    traj = trajs[index_unconf[k]]
    ax2.plot(traj['x'], traj['z'], label='Discontinuities', color='k', linewidth=4., linestyle='--')

ax2.set_xlabel(r'$x$ (km)', fontsize=19)
ax2.set_ylabel(r'$z$ (m)', fontsize=19)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax2, label='age (kyr)', pad=-0.02)

for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], ic[name]['S']+20), ha='center',
                 va='bottom', color=color_core)


plt.savefig(datadir+'trajectories_color_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()

print('After color trajectories figure:',
      round(time.perf_counter()-START_TIME, 4), 's.')        


# -----------
# Age figure
# -----------

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True, 
                               figsize=(15, 7), gridspec_kw={'height_ratios': [3, 7]})

ax1.plot(x, a, color='k', linewidth=2)
ax1.plot(x, np.zeros_like(x), color='k', linewidth=2, linestyle='--')
ax1.set_ylabel('a (m/yr)', fontsize=19)

ax1b = ax1.twinx()
ax1b.plot(x, surf_age/1000., color='r', linewidth=2)
ax1b.set_ylabel('surface age (kyr)', color='r')
ax1b.tick_params(axis='y', colors='r')
bottom, top = ax1b.get_ylim()
if top < 1.:
    top = 1.
ax1b.set_ylim(bottom, top)

age_min = 0.
age_max = max(max(age/1000.) for age in age_list)
linthresh = age_max/scaling_age  # Seuil en dessous duquel l'échelle devient linéaire (évite log(0))
norm = SymLogNorm(linthresh=linthresh, linscale=1, vmin=age_min, vmax=age_max)
levels_iso = np.array(fig_age_iso)

ax2.plot(x, S, label='Surface', color='k', linewidth=2.)
ax2.plot(x, B, label='Bedrock', color='k', linewidth=2.)


for dom in domains_list:
    ax2.contourf(dom['mat_x'], dom['mat_z'], dom['mat_age']/1000., levels=1000,
                 cmap='plasma', norm=norm)
    cp = ax2.contour(dom['mat_x'], dom['mat_z'], dom['mat_age']/1000., levels=levels_iso, colors='k',
                     norm=norm)

for unconf in unconf_list:
    ax2.plot(x, unconf, label='Discontinuities', color='k', linewidth=4.,
             linestyle='--')
    

ax2.set_xlabel(r'$x$ (km)', fontsize=19)
ax2.set_ylabel(r'$z$ (m)', fontsize=19)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, ax=ax2, label='age (kyr)', pad=-0.02)
cb.add_lines(cp)


for name in ic:
    ax2.plot(ic[name]['XX'], ic[name]['ZZ'], linewidth=lw_core,
             color=color_core, linestyle=ls_core)
    ax2.annotate(name, (ic[name]['x'], ic[name]['S']+20), ha='center',
                 va='bottom', color=color_core)

plt.savefig(datadir+'age_x_z.'+fig_format,
                format=fig_format, bbox_inches='tight')
plt.close()

print('After age figure:',
      round(time.perf_counter()-START_TIME, 4), 's.')        


# ----------------------------
# ice cores figures
# ----------------------------

for name in ic:
    ii = np.argmin(np.abs(x - ic[name]['x'])) # FIXME: interpolation is complicated...
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title('ice core '+name, fontsize=20)
    prof_domains = profs[ii]['unconf']
    prof_domains.insert(0, 0)
    prof_domains.append(len(profs[ii]['z']))
    for i in range(len(prof_domains)-1):
        ax.plot(profs[ii]['age'][prof_domains[i]: prof_domains[i+1]]/1000., profs[ii]['z'][prof_domains[i]: prof_domains[i+1]], color='b')
    ax.set_xlabel('age (kyr)', color='b')
    ax.set_ylabel('z (m)')
    ax.set_ylim(top=S[ii], bottom=S[ii]-ic[name]['max_depth'])
    jj = np.argmin(np.abs(profs[ii]['z']-(S[ii]-ic[name]['max_depth'])))
    ax.set_xlim(left=0, right=profs[ii]['age'][jj]/1000.)
    ax.spines['bottom'].set_color('b')
    ax.tick_params(axis='x', colors='b')
    x_low, x_up, y_low, y_up = ax.axis()
    ax.hlines(profs[ii]['z'][prof_domains[1:-1]], x_low, x_up, linestyles='--', linewidth=3., color = 'k')
    axb = ax.twinx()
    axb.set_ylim(top=0., bottom=S[ii]-B[ii])
    axb.set_ylabel('depth (m)')
    ax2 = ax.twiny()
    for i in range(len(prof_domains)-1):
        ax2.plot(profs[ii]['x0'][prof_domains[i]: prof_domains[i+1]], profs[ii]['z'][prof_domains[i]: prof_domains[i+1]], color='r')
    ax2.set_xlim(left=0)
    ax2.set_xlim(right=ic[name]['x'])
    ax2.set_xlabel('ice origin (km)', color='r')
    ax2.spines['top'].set_color('r')
    ax2.tick_params(axis='x', colors='r')
    ax2.spines['bottom'].set_visible(False)
    plt.savefig(datadir+name+'_ice_core.'+fig_format,
                    format=fig_format, bbox_inches='tight')
    plt.close()



# Program execution time
MESSAGE = 'Program execution time: '+str(time.perf_counter()-START_TIME)+' s.'
print(MESSAGE)
if os.name != 'nt':
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Max memory usage: '+str(mem)+' kbytes')