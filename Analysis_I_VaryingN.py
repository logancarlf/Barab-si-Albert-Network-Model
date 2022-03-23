# =============================================================================
# Analysis: Degree Distirbution P(k) with Varying N
# =============================================================================
from itertools import chain
from main import BA_Model
from logbin import logbin
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#%%


def theoretical(k, m):
    num = 2 * m * (m + 1)
    den = k * (k + 1) * (k + 2)
    return num/den


n = 4
N = np.logspace(2, n+1, base=10, num=n)
c = np.arange(1, n+1)
m = 1
M = 100

norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap.set_array([])

logbin_mean = []
logbin_std = []
logbin_x = []
for i in range(n):
    logbin_m = []
    logbinx = []
    for j in tqdm(range(M)):
        Model = BA_Model()
        g = Model.phase_1_preferential_attachment(N[i], m)
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
        prob = theoretical(logbin_x[i][j], m)
        if abs(np.log(prob/data_unclean[j])) > 1:
            logbin_mean[i].remove(logbin_mean[i][j])
            logbin_x[i].remove(logbin_x[i][j])
            logbin_std[i].remove(logbin_std[i][j])

#%%


fig, ax1 = plt.subplots(1, 1, figsize=(6.2, 4.8))
for i in range(n):
    ax1.errorbar(logbin_x[i], logbin_mean[i], yerr=logbin_std[i], ls='',
                 marker='x', c=cmap.to_rgba(i+2), ms=0, capsize=2)
    ax1.scatter(logbin_x[i], logbin_mean[i], marker='x', c=cmap.to_rgba(i+2),
                s=13, label='$N=%s$' % (int(N[i])))
    x = np.linspace(logbin_x[i][0]*0.6, logbin_x[i][-1]*2, 10000)
    ax1.plot(x, theoretical(x, m), c=cmap.to_rgba(i+2), ls='dashed')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title("Node Degree Distribution")
ax1.set_xlabel("Degree $k$")
ax1.set_ylabel(r"Degree Probability $\tilde{P}(k)$")
ax1.legend()
plt.show()

logbin_mean_scaled = []
for i in range(n):
    d = []
    for j in range(len(logbin_x[i])):
        d.append(logbin_mean[i][j]/theoretical(logbin_x[i][j], m))
    logbin_mean_scaled.append(d)

fig, ax1 = plt.subplots(1, 1, figsize=(6.2, 4.8))
for i in range(n):
    ax1.errorbar(logbin_x[i], logbin_mean_scaled[i],
                 yerr=logbin_std[i], ls='', marker='x', c=cmap.to_rgba(i+2), ms=0, capsize=2)
    ax1.scatter(logbin_x[i], logbin_mean_scaled[i], marker='x', c=cmap.to_rgba(i+2),
                 s=13, label='$N=%s$' % (int(N[i])))
#    x = np.linspace(logbin_x[i][0]*0.6, logbin_x[i][-1]*2, 10000)
#    ax1.plot(x, theoretical(x, m), c=cmap.to_rgba(i+2), ls='dashed')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title("Node Degree Distribution")
ax1.set_xlabel("Degree $k$")
ax1.set_ylabel(r"Degree Probability $\tilde{P}(k)$")
ax1.legend()
plt.show()
