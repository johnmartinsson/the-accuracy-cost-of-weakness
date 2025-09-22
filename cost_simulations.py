import numpy as np
import os
import matplotlib.pyplot as plt

import theorems as thms


###############################################################################
# Cost analysis
###############################################################################
def cost(T, B, r):
    return (1-r) * T + r * B

def cost_fix(T, gamma, d_e, r):
    B = T/thms.Q_max(d_e, gamma)
    return cost(T, B, r)

def cost_orc(T, r, M):
    B = 2*M + 1
    return cost(T, B, r)

T = 100
d_e = 1
r = 0.5
gamma = 0.5
M = 1

# plot the relative cost of ORC and FIX as a function of M
Ms = np.arange(1, 100, 1)

max_relative_costs = []
for factor in [1, 2, 4, 8]:
    costs_orc = np.array([cost_orc(T, r, factor*M) for M in Ms])
    costs_fix = np.array([cost_fix(T, gamma, d_e, r) for M in Ms])
    relative_costs = costs_fix / costs_orc
    max_relative_costs.append(np.max(relative_costs))

    plt.plot(Ms, relative_costs, label=r's = {}'.format(factor))

plt.title(r'Default parameters: T={}, r={}, $\gamma$={}'.format(T, r, gamma))
plt.xlabel('M')
plt.ylabel(r'$C_{\text{FIX}} / C_{\text{ORC}}$')
# plot dashed horizontal line at y=1
plt.axhline(y=1, color='r', linestyle='--')
plt.ylim([0, np.max(relative_costs) + 0.5])
plt.legend()
plt.savefig('figures/cost_orc_fix_M.png')

# clear the plot and figure
plt.clf()
plt.cla()

# plot the relative cost of ORC and FIX as a function of T
Ts = np.arange(1, 100, 1)

max_relative_costs = []
for factor in [1, 2, 4, 8]:
    costs_orc = np.array([cost_orc(T, r, factor*M) for T in Ts])
    costs_fix = np.array([cost_fix(T, gamma, d_e, r) for T in Ts])
    relative_costs = costs_fix / costs_orc
    max_relative_costs.append(np.max(relative_costs))

    plt.plot(Ts, relative_costs, label=r's = {}'.format(factor))

plt.title(r'Default parameters: M={}, r={}, $\gamma$={}'.format(M, r, gamma))
plt.xlabel('T')
plt.ylabel(r'$C_{\text{FIX}} / C_{\text{ORC}}$')
# plot dashed horizontal line at y=1
plt.axhline(y=1, color='r', linestyle='--')
plt.ylim([0, np.max(max_relative_costs) + 0.5])
plt.legend()
plt.savefig('figures/cost_orc_fix_T.png')

# clear the plot and figure
plt.clf()
plt.cla()

# plot the relative cost of ORC and FIX as a function of r
rs = np.arange(0, 1, 0.01)

max_relative_costs = []
for factor in [1, 2, 4, 8]:
    costs_orc = np.array([cost_orc(T, r, factor*M) for r in rs])
    costs_fix = np.array([cost_fix(T, gamma, d_e, r) for r in rs])
    relative_costs = costs_fix / costs_orc
    max_relative_costs.append(np.max(relative_costs))

    plt.plot(rs, relative_costs, label=r's = {}'.format(factor))

plt.title(r'Default parameters: T={}, M={}, $\gamma$={}'.format(T, M, gamma))
plt.xlabel('r')
plt.ylabel(r'$C_{\text{FIX}} / C_{\text{ORC}}$')
# plot dashed horizontal line at y=1
plt.axhline(y=1, color='r', linestyle='--')
plt.ylim([0, np.max(max_relative_costs) + 0.5])
plt.legend()
plt.savefig('figures/cost_orc_fix_r.png')

# clear the plot and figure
plt.clf()
plt.cla()

# plot the relative cost of ORC and FIX as a function of gamma
gammas = np.arange(0.1, 1, 0.01)

max_relative_costs = []
for factor in [1, 2, 4, 8]:
    costs_orc = np.array([cost_orc(T, r, factor*M) for gamma in gammas])
    costs_fix = np.array([cost_fix(T, gamma, d_e, r) for gamma in gammas])
    relative_costs = costs_fix / costs_orc
    max_relative_costs.append(np.max(relative_costs))

    plt.plot(gammas, relative_costs, label=r's = {}'.format(factor))

plt.title(r'Default parameters: T={}, r={}, M={}'.format(T, r, M))
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$C_{\text{FIX}} / C_{\text{ORC}}$')
# plot dashed horizontal line at y=1
plt.axhline(y=1, color='r', linestyle='--')
plt.ylim([0, np.max(max_relative_costs) + 0.5])
plt.legend()
plt.savefig('figures/cost_orc_fix_gamma.png')

# clear the plot and figure
plt.clf()
plt.cla()