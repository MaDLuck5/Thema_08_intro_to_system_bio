import csv

import numpy as np
from IPython import get_ipython

np.set_printoptions(threshold=np.inf)
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette='Set2')
plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex=False)
ncolours = len(plt.rcParams['axes.prop_cycle'])
colours = [list(plt.rcParams['axes.prop_cycle'])[i]['color'] for i in range(ncolours)]





tau_p = 8.893084e+00
tau_b = 1.793090e+01

D = 6  # number of variables in the system

A = 1.15
B = 0.08
C = 0.1

parms = {'p': 6.459494e-05,
         'alpha': 1.904753e-03,
         'beta': 2.233049e+00,
         'gamma_b': 2.293953e-01,  # E to I
         'gamma_p': 2.863875e-01,  # E to I
         'delta_b': 2.626527e-01,  # recovery + death
         'delta_p': 3.413748e-01,  # recovery + death
         'epsilon': 3.204494e-02,  # I_b to I_p
         'kk': 1.105443e+00,  # Imperfect intervention
         'N': 25570895}  # population size

# Initial condition
S_b_0 = (int(parms['N']*parms["p"]))
S_p_0 = parms['N'] - S_b_0


def flea(t):
    x = A + B * np.sin((np.pi / 180.) * t) + C * np.cos((np.pi / 180.) * t)

    return (x)


def intervention_p(t):
    return 1. - 1. / ((parms["kk"]) + np.exp(tau_p - t))


def intervention_b(t):
    return 1. - 1. / ((parms["kk"]) + np.exp(tau_b - t))

state = {'Sb': S_b_0, 'Sy': S_p_0, 'Eb': 0, 'Ey': 0, 'Ib': 1, 'Ip': 1}
def deterministic_model(x, tijd):
    flea_t = flea(tijd)
    intervention_b1 = intervention_b(tijd)
    intervention_p1 = intervention_p(tijd)

    deltaSb = -parms['alpha'] * flea_t * intervention_b1 * x[0] - -parms[
        'beta'] * intervention_p1 * x[0] * x[5] / parms['N']
    deltaSy = -parms['beta'] * intervention_p1 * x[1] * x[5] / parms['N']
    deltaEb = parms['alpha'] * flea_t * intervention_b1 * x[0] - parms['gamma_b'] * x[2]
    deltaEy = parms['beta'] * intervention_p1 * (x[0] + x[1]) * x[5] / parms[
        'N'] - parms['gamma_p'] * x[3]
    deltaIb = parms['gamma_b'] * x[2] - parms['epsilon'] * x[4] - parms['delta_b'] *x[4]
    deltaIp = parms['gamma_p'] * x[3] + parms['epsilon'] * x[4] - parms['delta_p'] *x[5]

    return (deltaSb, deltaSy, deltaEb, deltaEy, deltaIb, deltaIp, tijd)

x_0 = [S_b_0, S_p_0, 0, 0, 1, 1,0]
dt = 0.1
lentime = 70 - 1
t = np.arange(0.0, lentime + dt, dt)


z = odeint(deterministic_model, x_0, t)
data_py = z

with open('C:/Users/matsp/Downloads/plague2017-master/plague2017-master/test2.csv', 'w',
          encoding='UTF8') as f:
    writer = csv.writer(f)

    for row in data_py:
        writer.writerow(row)

fig, axes = plt.subplots(1,2, figsize=(12,6))

axes[0].plot(t, z[:,4], color=colours[1], linestyle='-', linewidth=3., zorder=10000, label='Deterministic')
axes[1].plot(t, z[:,5], color=colours[1], linestyle='-', linewidth=3., zorder=10000)

axes[0].set_ylim(0,30)
axes[1].set_ylim(0,120)

axes[0].tick_params(labelsize=15)
axes[1].tick_params(labelsize=15)

axes[0].set_ylabel('Plague incidences', fontsize=18)
axes[1].set_ylabel('Plague incidences', fontsize=18)
axes[1].set_xlabel('Day from 22.09.2017', fontsize=16)
axes[0].set_xlabel('Day from 22.09.2017', fontsize=16)

axes[0].set_title('Bubonic', fontsize=18)
axes[1].set_title('Pneumonic', fontsize=18)

axes[0].legend(loc=(0.4,-0.3), ncol=3, fontsize=16)

plt.tight_layout()
plt.savefig('C:/Users/matsp/Downloads/plague2017-master/plague2017-master/Fig1.pdf', dpi=300, bbox_inches='tight')