from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as net
from logbin import logbin
from main import BA_Model
import numpy as np
import time
from tqdm import tqdm
from itertools import chain
from scipy.stats import linregress

plt.rcParams["text.usetex"] = False
plt.rcParams["font.size"] = "14"


# Initialise
def theoretical(k, m):
    num = 2 * m * (m + 1)
    den = k * (k + 1) * (k + 2)
    return num/den


def linear(x, a, b):
    return a * x + b


n = 4
m = np.logspace(1, n, base=2, num=n)
c = np.arange(1, n+1)
N = 100000
M = 8

norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap.set_array([])

logbin_mean = []
logbin_std = []
logbin_x = []
for i in m:
    logbin_m = []
    logbinx = []
    for j in tqdm(range(M)):
        Model = BA_Model()
        g = Model.phase_1_preferential_attachment(N, int(i))
        degree_sequence = sorted(np.array([d for n, d in g.degree()]))
        x, y = logbin(degree_sequence, scale=1.1)
        logbin_m.append(list(y))
        logbinx.append(list(x))
    x_values = []
    # collect x values
    all_x = list(chain.from_iterable(logbinx))
    for i in all_x:
        if i not in x_values:
            x_values.append(i)
    logbin_all_y = []
    for k in x_values:
        prob_x = []
        for i in range(M):
            for j in range(len(logbinx[i])):
                if logbinx[i][j] == k:
                    prob_x.append(logbin_m[i][j])
        logbin_all_y.append(prob_x)
    logbin_mean_m = []
    logbin_std_m = []
    for i in logbin_all_y:
        logbin_mean_m.append(np.mean(i))
        logbin_std_m.append(np.std(i))
    logbin_x.append(x_values)
    logbin_mean.append(logbin_mean_m)
    logbin_std.append(logbin_std_m)

remove_i = []
remove_j = []
for i in range(n):
    data_unclean = logbin_mean[i]
    for j in range(len(data_unclean)-1, -1, -1):
        prob = theoretical(logbin_x[i][j], m[i])
        if abs(np.log(prob/data_unclean[j])) > 1:
            logbin_mean[i].remove(logbin_mean[i][j])
            logbin_x[i].remove(logbin_x[i][j])
            logbin_std[i].remove(logbin_std[i][j])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
for i in range(n):
    ax1.errorbar(logbin_x[i], logbin_mean[i], yerr=logbin_std[i], ls='',
                 marker='x', c=cmap.to_rgba(i+2), ms=0, capsize=2)
    ax1.scatter(logbin_x[i], logbin_mean[i], marker='x', c=cmap.to_rgba(i+2),
                s=13, label='$m=%s$' % (int(m[i])))
    x = np.linspace(logbin_x[i][0]*0.6, logbin_x[i][-1]*2, 10000)
    ax1.plot(x, theoretical(x, m[i]), c=cmap.to_rgba(i+2), ls='dashed')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title("Node Degree Distribution")
ax1.set_xlabel("Degree $k$")
ax1.set_ylabel(r"Degree Probability $\tilde{P}(k)$")
ax1.legend()

theoretical_prob = []
prob = []
for i in range(n):
    theory = []
    for j in range(len(logbin_x[i])):
        theory.append(theoretical(logbin_x[i][j], m[i]))
        theoretical_prob.append(theoretical(logbin_x[i][j], m[i]))
        prob.append(logbin_mean[i][j])
    ax2.errorbar(theory, logbin_mean[i], yerr=logbin_std[i], ls='',
                 marker='x', c=cmap.to_rgba(i+2), ms=0, capsize=2)
    ax2.scatter(theory, logbin_mean[i], marker='x', c=cmap.to_rgba(i+2),
                s=13)

slope, y_int, r_value, p_value, std_error = linregress(theoretical_prob, prob)
x = np.linspace(np.min(prob), np.max(prob), 10000)
ax2.plot(x, linear(x, slope, 0), color='red', linestyle='dashed',
         label='Linear Regression \n $R^2=%.4f$' % (r_value**2))
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Theoretical Probability P(k)')
ax2.set_ylabel(r"Numerical Probability $\tilde{P}(k)$")
ax2.set_title('Theoretical Accuracy')
ax2.legend()
plt.savefig('Figures/Degree_Distribution.png', dpi=600,
            bbox_inches='tight')
plt.show()
