
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette='Set2')
plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex=False)
ncolours = len(plt.rcParams['axes.prop_cycle'])
colours = [list(plt.rcParams['axes.prop_cycle'])[i]['color'] for i in range(ncolours)]

# parameters:

p       = 6.459494e-05 # fraction susceptible to bite
alpha   = 1.904753e-03 # rate of bites
beta    = 2.233049e+00 # between-host infection
gamma_b = 2.293953e-01 # E to I
gamma_p = 2.863875e-01 # E to I
delta_b = 2.626527e-01 # recovery + death
delta_p = 3.413748e-01 # recovery + death
epsilon = 3.204494e-02 # I_b to I_p
kk      = 1.105443e+00 # Imperfect intervention
N       = 25570895     # population size

# Initial condition
S_b_0 = int(N*p)
S_p_0 = N - S_b_0

tau_p = 8.893084e+00
tau_b = 1.793090e+01

D = 6 # number of variables in the system

A = 1.15
B = 0.08
C = 0.1

# base functions

def unit_vector(i, D):
    '''Returns (i+1)-th unit vector of dimension D'''
    out = np.zeros(D)
    out[i] = 1
    return out


def get_time(rates, Pk, Tk):
    '''Time step and index of reaction'''
    n = len(rates)
    dt = np.array([((Pk[i] - Tk[i])/rates[i]) if rates[i]>0.0 else np.inf for i in range(n)])
    idx = np.argmin(dt)
    return idx, dt[idx]

# rat flea functions

def flea(t):
    return A + B*np.sin((np.pi/180.)*t) + C*np.cos((np.pi/180.)*t)
def intervention_p(t):
    return 1. - 1./(kk + np.exp(tau_p - t))
def intervention_b(t):
    return 1. - 1./(kk + np.exp(tau_b - t))
v_flea  = np.vectorize(flea)
v_itv_p = np.vectorize(intervention_p)
v_itv_b = np.vectorize(intervention_b)

def transition_rates(x, f_irf, f_itv_p, f_itv_b):
    '''Transition rates for every possible reaction'''
    return np.array([alpha*f_irf*f_itv_b*x[0],
                     beta*f_itv_p*x[0]*x[5]/N,
                     beta*f_itv_p*x[1]*x[5]/N,
                     gamma_b*x[2],
                     gamma_p*x[3],
                     delta_b*x[4],
                     delta_p*x[5],
                     epsilon*x[4]])

# nu[i] corresponds to the i-th reaction
nu = np.array([unit_vector(2, D) - unit_vector(0, D), unit_vector(3, D) - unit_vector(0, D),
               unit_vector(3, D) - unit_vector(1, D), unit_vector(4, D) - unit_vector(2, D),
               unit_vector(5, D) - unit_vector(3, D), -unit_vector(4, D), -unit_vector(5, D),
               unit_vector(5, D) - unit_vector(4, D)], dtype=int)


def plague(x):
    t = 0.0
    times = [t]
    x_t = [x]
    rates = transition_rates(x, f_irf, f_itv_p, f_itv_b)
    n = len(rates)
    Pk = np.array([-np.log(np.random.random()) for _ in range(n)])
    Tk = np.zeros(n)
    while (x[0] + x[1]) > 0:
        reaction, time = get_time(rates, Pk, Tk)

        if time == np.inf:
            break

        t += time
        x = x + nu[reaction]
        x_t.append(x)
        times.append(t)
        Tk = np.array([Tk[k] + rates[k] * time for k in range(n)])
        Pk[reaction] -= np.log(np.random.random())
        rates = transition_rates(x)

    times = np.array(times)
    x_t = np.array(x_t)

    return times, x_t


def plague_tau(x, t, dt, f_irf, f_itv_p, f_itv_b):
    x_t = []
    for i in range(len(t)):
        x_t.append(x)
        rates = transition_rates(x, f_irf[i], f_itv_p[i], f_itv_b[i])
        n = len(rates)
        firings = np.random.poisson(np.maximum(rates, np.zeros(n)) * dt)
        change = np.sum([firings[i] * nu[i] for i in range(len(rates))], axis=0)
        x = x + change

    x_t = np.array(x_t)

    return x_t

# Data
data_raw = pd.read_csv("C://Users//matsp//Downloads//plague2017-master//plague2017-master//data//dataBy.csv", sep=',')
data_raw['date'] = pd.to_datetime(data_raw['date'])
start_date = data_raw[data_raw['date'] == pd.to_datetime('22/09/2017')].index[0]
data_raw = data_raw[start_date:]
data_b = np.array(data_raw['bubonic'])
data_p = np.array(data_raw['pneumonic'])

x_0 = [S_b_0, S_p_0, 0, 0, 1, 1]
dt = 0.1
lentime = data_b.size - 1
t = np.arange(0., lentime + dt, dt)
f_irf = v_flea(t)
f_itv_b = v_itv_b(t)
f_itv_p = v_itv_p(t)

# Stochastic runs
import random

random.seed(10042018)
runs = 20

I_b = []
I_p = []

for realisation in range(runs):
    y = plague_tau(x_0, t, dt, f_irf, f_itv_p, f_itv_b)

    I_b.append(y[:, 4][:: int(1. / dt)])
    I_p.append(y[:, 5][:: int(1. / dt)])

fig, axes = plt.subplots(1,2, figsize=(12,6))

axes[0].scatter(t[: : int(1./dt)], data_b, color=colours[2], zorder=10001)
axes[1].scatter(t[: : int(1./dt)], data_p, color=colours[2], zorder=10001)


axes[0].plot(t[: : int(1./dt)], I_b[0], linewidth=3., color=colours[0], alpha=0.3, label='Stochastic')
axes[1].plot(t[: : int(1./dt)], I_p[0], linewidth=1., color=colours[0], alpha=0.3)

for i in range(1,runs):
    axes[0].plot(t[: : int(1./dt)], I_b[i], linewidth=1., color=colours[0], alpha=0.3)
    axes[1].plot(t[: : int(1./dt)], I_p[i], linewidth=1., color=colours[0], alpha=0.3)

axes[0].set_ylim(0,30)
axes[1].set_ylim(0,120)

axes[0].plot(t[: : int(1./dt)], data_b, color=colours[2], linestyle='--', linewidth=1.5, label='Data')
axes[1].plot(t[: : int(1./dt)], data_p, color=colours[2], linestyle='--', linewidth=1.5)

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
plt.savefig('C://Users//matsp//Documents//Thema08//Thema_08_intro_to_system_bio//Project_madagascar//Fig2.pdf', dpi=300, bbox_inches='tight')