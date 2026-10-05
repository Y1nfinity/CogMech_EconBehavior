"""Econ 4850 Project Report #1 -- Question 1(c): MLE of alpha for the DDM.

Model (from part (a)):  P_alpha(D) = 1 / (1 + exp(-2*alpha*D)),  D = bidR - bidL
Log likelihood (part (b)):  LL(alpha) = sum_i log P_alpha(sigma_i * D_i)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib
import matplotlib.pyplot as plt

# ---------- 1. Load data (file has no header row) ----------
CSV_PATH = r'C:\Users\azhao\PycharmProjects\CogMech_EconBehavior\indv-Data.csv'

df = pd.read_csv(CSV_PATH, skipinitialspace=True)
df.columns = ['L', 'R', 'bidL', 'bidR', 'choice', 'rt', 'uni']
for c in ['L', 'R', 'choice', 'uni']:
    df[c] = df[c].astype(str).str.strip()


df['delta'] = df['bidR'] - df['bidL']                      # Delta_i = v_R - v_L
df['sigma'] = np.where(df['choice'] == df['R'], 1, -1)     # +1 if R chosen, -1 if L
assert len(df) == 435 and set(df['sigma']) <= {1, -1}

D = df['delta'].to_numpy()
S = df['sigma'].to_numpy()

# ---------- 2. Model functions ----------
def P(alpha, x):
    """P_alpha(x) = 1/(1+exp(-2*alpha*x))"""
    return 1.0 / (1.0 + np.exp(-2.0 * alpha * x))

def LL(alpha):
    """Log likelihood, written stably: log P(z) = -log(1+exp(-2*alpha*z))"""
    return -np.sum(np.logaddexp(0.0, -2.0 * alpha * S * D))

# ---------- 3. MLE via numerical optimizer ----------
res = minimize_scalar(lambda a: -LL(a), bounds=(0.0, 10.0),
                      method='bounded', options={'xatol': 1e-12})
alpha_hat = res.x
print(f"MLE alpha_hat = {alpha_hat:.4f}")
print(f"LL(alpha_hat) = {LL(alpha_hat):.4f}")
print(f"LL(0)         = {LL(0.0):.4f}   (= 435*ln(0.5))")

"""
# Standard error from curvature of LL
h = 1e-4
curv = (LL(alpha_hat + h) - 2 * LL(alpha_hat) + LL(alpha_hat - h)) / h**2
print(f"std. error    = {np.sqrt(-1.0 / curv):.4f}")

# ---------- 4. Grid-refinement check (the 'low-tech' method in the handout) ----------
lo, hi = 0.0, 2.0
grid = np.linspace(lo, hi, 11)
for stage in range(1, 12):
    ll = np.array([LL(a) for a in grid])
    k = int(np.argmax(ll))
    if k in (0, len(grid) - 1):
        raise RuntimeError('maximum on grid edge; widen the grid')
    a_m, a_k, a_p = grid[k - 1], grid[k], grid[k + 1]
    print(f"stage {stage:2d}: alpha_k-1={a_m:.5f}, alpha_k={a_k:.5f}, alpha_k+1={a_p:.5f} | "
          f"LL = {ll[k-1]:.4f}, {ll[k]:.4f}, {ll[k+1]:.4f}")
    grid = np.linspace(a_m, a_p, 11)
"""

# ---------- 5. Save for later parts (d), 2 ----------
df.to_csv('processed_choices.csv', index=False)
pd.Series({'alpha_hat': alpha_hat}).to_csv('alpha_hat.csv')

NBINS = 15
m = np.max(np.abs(D))  # max |Delta_i|
edges = np.linspace(-m, m, NBINS + 1)  # 15 equal-width bins on [-m, m]
df['bin'] = pd.cut(df['delta'], bins=edges, include_lowest=True, labels=False)

df['chose_R'] = (df['sigma'] == 1).astype(float)
df['P_i'] = P(alpha_hat, df['delta'])  # P_alpha(Delta_i) for each trial

g = df.groupby('bin')
tab = pd.DataFrame({
    'n': g.size(),
    'p_j': g['chose_R'].mean(),  # observed fraction choosing R
    'e_j': g['P_i'].mean(),  # mean predicted P_alpha(Delta_i)
}).reindex(range(NBINS))  # keep empty bins visible (NaN)
tab['lo'] = edges[:-1]
tab['hi'] = edges[1:]

# d_j such that P_alpha(d_j) = e_j  ->  d_j = ln(e_j / (1 - e_j)) / (2 alpha)
tab['d_j'] = np.log(tab['e_j'] / (1 - tab['e_j'])) / (2 * alpha_hat)
print(tab.round(4).to_string())
assert tab['n'].sum() == len(df)

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(7.5, 5))
dd = np.linspace(-m, m, 400)
ax.plot(dd, P(alpha_hat, dd), 'k-', lw=1.2, zorder=1, label=r'$e = P_{\hat\alpha}(d)$')
ax.scatter(tab['d_j'], tab['e_j'], s=28, c='k', zorder=3, label='DDM prediction $e_j$')
ax.scatter(tab['d_j'], tab['p_j'], s=60, facecolors='none', edgecolors='tab:blue',
           linewidths=1.5, zorder=2, label='Observed frequency $p_j$')
ax.set_xlabel(r'Value difference $d_j$  (bid$_R$ $-$ bid$_L$)')
ax.set_ylabel('Frequency of choosing R')
ax.set_title(rf'Choice frequency vs. value difference ($\hat\alpha$ = {alpha_hat:.3f})')
ax.set_ylim(-0.03, 1.03)
ax.grid(alpha=0.25)
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig('fig_1d_choice_frequency.png', dpi=200)
plt.show()

# ---------- optional fit diagnostics (for the written discussion) ----------
from scipy.stats import chi2

tab['z'] = (tab['p_j'] - tab['e_j']) / np.sqrt(tab['e_j'] * (1 - tab['e_j']) / tab['n'])
chi2_stat = (tab['z'] ** 2).sum()
print(tab[['n', 'p_j', 'e_j', 'd_j', 'z']].round(3).to_string())
print(f"chi-square = {chi2_stat:.2f} (14 df), p = {1 - chi2.cdf(chi2_stat, 14):.2f}")
print(f"RMSE(p_j - e_j) = {np.sqrt(np.mean((tab['p_j'] - tab['e_j']) ** 2)):.3f}")


# ======================= Question 2: response times =======================
# DDM mean decision time (barriers +/-1, drift mu = alpha*Delta, unit noise):
#   E[T] = tanh(mu)/mu  =: G_alpha(Delta)   (-> 1 as Delta -> 0);  predicted RT = A*G + t0
def G(alpha, x):                                       # tanh(alpha*x)/x  (= alpha at x=0)
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, alpha)                       # limit as x -> 0 is alpha
    nz = np.abs(x) > 1e-9
    out[nz] = np.tanh(alpha * x[nz]) / x[nz]
    return out


# ---- Step 1: discard RT outliers (mean/SD of the full, unbinned RT distribution) ----
rt = pd.to_numeric(df['rt'])
rt_mean, rt_sd = rt.mean(), rt.std()  # sample SD (ddof=1)
keep = (rt - rt_mean).abs() < 2 * rt_sd  # drop if |RT - mean| >= 2 SD
print(f"RT mean = {rt_mean:.1f} ms, SD = {rt_sd:.1f} ms; dropped {(~keep).sum()} of {len(df)} trials")
df2 = df[keep].copy()
df2['rt'] = rt[keep]

# ---- Step 2: T_j and g_j on the SAME 15 bins as part (d) (bins fixed by all 435 trials) ----
df2['G_i'] = G(alpha_hat, df2['delta'])
g2 = df2.groupby('bin')
tab2 = pd.DataFrame({
    'n_kept': g2.size(),
    'T_j': g2['rt'].mean(),  # mean RT, outliers removed
    'g_j': g2['G_i'].mean(),  # mean G_alpha(Delta_i), same trials
}).reindex(range(NBINS))
tab2['d_j'] = tab['d_j']  # d_j from part (d)

# ---- Step 3: OLS of T_j on g_j (15 observations):  T_j = t0 + A * g_j ----
ok = tab2[['T_j', 'g_j']].notna().all(axis=1)
x, y = tab2.loc[ok, 'g_j'].to_numpy(), tab2.loc[ok, 'T_j'].to_numpy()
A_hat, t0_hat = np.polyfit(x, y, 1)  # slope = A, intercept = t0
resid = y - (A_hat * x + t0_hat)
r2 = 1 - resid.var() / y.var()
n_ = len(x)
se_A = np.sqrt(resid @ resid / (n_ - 2) / ((x - x.mean()) ** 2).sum())
print(f"A = {A_hat:.1f} ms (SE {se_A:.1f}),  t0 = {t0_hat:.1f} ms,  R^2 = {r2:.3f}, bins used = {n_}")

tab2['fit'] = A_hat * tab2['g_j'] + t0_hat  # A*g_j + t0
print(tab2.round(3).to_string())

# ---- Step 4: figure ----
fig, ax = plt.subplots(figsize=(7.5, 5))
dd = np.linspace(-m, m, 400)
ax.plot(dd, A_hat * G(alpha_hat, dd) + t0_hat, 'k-', lw=1.2, zorder=1,
        label=r'$y = A\,G_{\hat\alpha}(d) + t_0$')
ax.scatter(tab2['d_j'], tab2['fit'], s=28, c='k', zorder=3, label=r'DDM prediction $A g_j + t_0$')
ax.scatter(tab2['d_j'], tab2['T_j'], s=60, facecolors='none', edgecolors='tab:blue',
           linewidths=1.5, zorder=2, label='Observed mean RT $T_j$')
ax.set_xlabel(r'Value difference $d_j$  (bid$_R$ $-$ bid$_L$)')
ax.set_ylabel('Mean response time (ms)')
ax.set_title(rf'Mean RT vs. value difference ($\hat\alpha$={alpha_hat:.3f}, A={A_hat:.0f}, $t_0$={t0_hat:.0f})')
ax.grid(alpha=0.25)
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig('fig_2_response_times.png', dpi=200)
plt.show()