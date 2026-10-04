"""Econ 4850 Project Report #1 -- Question 3 (a) and (b): pooled class data.
Stand-alone script: only needs popData.csv (columns dj, nj, mj, tj)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

POP_PATH = r'C:\Users\azhao\PycharmProjects\CogMech_EconBehavior\popData.csv'          # <- change to your full path, e.g. r'C:\Users\azhao\...\popData.csv'

pop = pd.read_csv(POP_PATH)
pop.columns = ['d', 'n', 'm', 'T']                     # d_j, n_j (R chosen), m_j (L chosen), T_j (ms)
d, n, m_, T = (pop[c].to_numpy(dtype=float) for c in ['d', 'n', 'm', 'T'])
assert len(pop) == 15

def P(alpha, x):                                       # P_alpha(x) = 1/(1+exp(-2 alpha x))
    return 1.0 / (1.0 + np.exp(-2.0 * alpha * x))

def G(alpha, x):                                       # tanh(mu)/mu, mu = alpha*x  (=1 at x=0)
    mu = alpha * np.asarray(x, dtype=float)
    out = np.ones_like(mu)
    nz = np.abs(mu) > 1e-9
    out[nz] = np.tanh(mu[nz]) / mu[nz]
    return out

# ---------------- 3(a): MLE of alpha with Delta_i := d_j for every trial in bin j ----------------
# LL(alpha) = sum_j [ n_j log P(d_j) + m_j log P(-d_j) ]
def LL(alpha):
    return np.sum(-n * np.logaddexp(0, -2 * alpha * d) - m_ * np.logaddexp(0, 2 * alpha * d))

res = minimize_scalar(lambda a: -LL(a), bounds=(0, 10), method='bounded', options={'xatol': 1e-12})
alpha_pop = res.x
h = 1e-4
se_alpha = np.sqrt(-1 / ((LL(alpha_pop + h) - 2 * LL(alpha_pop) + LL(alpha_pop - h)) / h**2))
print(f"3(a): alpha_hat = {alpha_pop:.4f} (SE {se_alpha:.4f}),  LL = {LL(alpha_pop):.2f},  "
      f"trials = {int((n + m_).sum())}")

pop['N'] = n + m_
pop['p_j'] = n / (n + m_)                              # observed fraction choosing R
pop['e_j'] = P(alpha_pop, d)                           # predicted (Delta = d_j exactly)
pop['z'] = (pop['p_j'] - pop['e_j']) / np.sqrt(pop['e_j'] * (1 - pop['e_j']) / pop['N'])
chi2_stat = (pop['z'] ** 2).sum()
print(pop[['d', 'N', 'p_j', 'e_j', 'z']].round(4).to_string())
print(f"chi-square = {chi2_stat:.1f} (14 df), p = {1 - chi2.cdf(chi2_stat, 14):.2g};  "
      f"RMSE(p-e) = {np.sqrt(np.mean((pop.p_j - pop.e_j)**2)):.3f}")

dd = np.linspace(d.min(), d.max(), 400)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(dd, P(alpha_pop, dd), 'k-', lw=1.2, zorder=1, label=r'$e = P_{\hat\alpha}(d)$')
ax.scatter(d, pop['e_j'], s=28, c='k', zorder=3, label='DDM prediction $e_j$')
ax.scatter(d, pop['p_j'], s=60, facecolors='none', edgecolors='tab:blue', linewidths=1.5,
           zorder=2, label='Observed frequency $p_j$')
ax.set_xlabel(r'Value difference $d_j$'); ax.set_ylabel('Frequency of choosing R')
ax.set_title(rf'Pooled data: choice frequency ($\hat\alpha$ = {alpha_pop:.3f})')
ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.25); ax.legend(loc='upper left')
fig.tight_layout(); fig.savefig('fig_3a_pooled_choices.png', dpi=200); plt.show()

# ---------------- 3(b): response times, g_j = G_alpha(d_j), OLS of T_j on g_j ----------------
pop['g_j'] = G(alpha_pop, d)
A_pop, t0_pop = np.polyfit(pop['g_j'], T, 1)
pop['fit'] = A_pop * pop['g_j'] + t0_pop
resid = T - pop['fit']
r2 = 1 - (resid @resid) / ((T-T.mean()) @ (T-T.mean()))
se_A = np.sqrt(resid @ resid / (len(T) - 2) / ((pop['g_j'] - pop['g_j'].mean()) ** 2).sum())
print(f"\n3(b): A = {A_pop:.1f} ms (SE {se_A:.1f}),  t0 = {t0_pop:.1f} ms,  R^2 = {r2:.3f},  "
      f"RMSE = {np.sqrt(np.mean(resid**2)):.1f} ms")
print(pop[['d', 'T', 'g_j', 'fit']].round(3).to_string())

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(dd, A_pop * G(alpha_pop, dd) + t0_pop, 'k-', lw=1.2, zorder=1,
        label=r'$y = A\,G_{\hat\alpha}(d) + t_0$')
ax.scatter(d, pop['fit'], s=28, c='k', zorder=3, label=r'DDM prediction $A g_j + t_0$')
ax.scatter(d, T, s=60, facecolors='none', edgecolors='tab:blue', linewidths=1.5, zorder=2,
           label='Observed mean RT $T_j$')
ax.set_xlabel(r'Value difference $d_j$'); ax.set_ylabel('Mean response time (ms)')
ax.set_title(rf'Pooled data: mean RT ($\hat\alpha$={alpha_pop:.3f}, A={A_pop:.0f}, $t_0$={t0_pop:.0f})')
ax.grid(alpha=0.25); ax.legend(loc='upper right')
fig.tight_layout(); fig.savefig('fig_3b_pooled_rt.png', dpi=200); plt.show()