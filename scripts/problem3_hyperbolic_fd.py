"""
Problem 3: Finite Difference Methods for Hyperbolic PDEs

One-way wave equation:  u_t + u_x = 0,  x > -2,  t > 0
  u(-2, t) = 0
  u(x, 0)  = u0(x) = (1-|x|)^2  for |x|<=1,  else 0

Exact solution: u(x,t) = u0(x - t) if x-t > -2, else 0.

Schemes tested (a=1):
  Explicit: FTFS, FTBS, FTCS, Leapfrog(CTCS), Lax-Friedrichs, Lax-Wendroff, Beam-Warming
  Implicit: BTFS, BTBS, BTCS, Crank-Nicolson

Domain: x in [-2, b],  b=3.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from scipy.linalg import solve_banded
from project_paths import ensure_problem_figure_dir, relative_to_root


FIGURES_DIR = ensure_problem_figure_dir("problem3")


# ─── Problem setup ────────────────────────────────────────────────────────────

def u0(x):
    return np.where(np.abs(x) <= 1.0, (1 - np.abs(x))**2, 0.0)

def u_exact(x, t):
    xt = x - t  # characteristic
    return np.where(xt > -2.0, u0(xt), 0.0)

a  = 1.0   # wave speed
b  = 3.0   # right boundary
x0 = -2.0  # left boundary


# ─── Grid construction ────────────────────────────────────────────────────────

def make_grid(h, lam, b_right=None):
    if b_right is None:
        b_right = b
    k = lam * h
    x = np.arange(x0, b_right + h/2, h)
    return x, k


# ─── Explicit schemes ─────────────────────────────────────────────────────────

def advance_explicit(u, lam, scheme, u_left=0.0):
    """
    Advance one time step for explicit schemes on interior+ghost.
    u[0] is the left ghost (boundary), u[-1] is the rightmost point.
    """
    N = len(u)
    v = np.zeros_like(u)
    v[0] = u_left  # left BC: u(-2,t) = 0

    if scheme == 'FTFS':
        for m in range(1, N - 1):
            v[m] = u[m] - lam * (u[m + 1] - u[m])
        v[-1] = u[-1] - lam * (u[-1] - u[-2])  # one-sided at right
    elif scheme == 'FTBS':
        for m in range(1, N):
            v[m] = u[m] - lam * (u[m] - u[m - 1])
    elif scheme == 'FTCS':
        for m in range(1, N - 1):
            v[m] = u[m] - 0.5 * lam * (u[m + 1] - u[m - 1])
        v[-1] = u[-1] - lam * (u[-1] - u[-2])
    elif scheme == 'LaxFriedrichs':
        for m in range(1, N - 1):
            v[m] = 0.5 * (u[m + 1] + u[m - 1]) - 0.5 * lam * (u[m + 1] - u[m - 1])
        v[-1] = u[-1] - lam * (u[-1] - u[-2])
    elif scheme == 'LaxWendroff':
        for m in range(1, N - 1):
            v[m] = (u[m]
                    - 0.5 * lam * (u[m + 1] - u[m - 1])
                    + 0.5 * lam**2 * (u[m + 1] - 2 * u[m] + u[m - 1]))
        v[-1] = u[-1] - lam * (u[-1] - u[-2])
    elif scheme == 'BeamWarming':
        v[1] = u[1] - lam * (u[1] - u[0])  # use FTBS for m=1
        for m in range(2, N):
            v[m] = (u[m]
                    - 0.5 * lam * (3 * u[m] - 4 * u[m - 1] + u[m - 2])
                    + 0.5 * lam**2 * (u[m] - 2 * u[m - 1] + u[m - 2]))
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    return v


def run_explicit(h, lam, T, scheme, b_right=None):
    x, k = make_grid(h, lam, b_right=b_right)
    u = u0(x)
    u[0] = 0.0  # left BC
    t = 0.0
    Nsteps = int(round(T / k))
    for _ in range(Nsteps):
        u = advance_explicit(u, lam, scheme)
        t += k
    return x, u


# ─── Leapfrog (CTCS) ─────────────────────────────────────────────────────────

def run_ctcs(h, lam, T, b_right=None):
    x, k = make_grid(h, lam, b_right=b_right)
    N = len(x)
    u_old = u0(x)
    u_old[0] = 0.0
    # Level 1: use FTBS (stable for a>0)
    u_cur = advance_explicit(u_old, lam, 'FTBS')
    t = k
    Nsteps = int(round(T / k))
    for n in range(1, Nsteps):
        v = np.zeros(N)
        v[0] = 0.0
        for m in range(1, N - 1):
            v[m] = u_old[m] - lam * (u_cur[m + 1] - u_cur[m - 1])
        v[-1] = u_cur[-1] - lam * (u_cur[-1] - u_cur[-2])
        u_old = u_cur
        u_cur = v
        t += k
    return x, u_cur


# ─── Implicit schemes ─────────────────────────────────────────────────────────

def run_implicit(h, lam, T, scheme, b_right=None):
    """
    Implicit schemes: solve tridiagonal system at each step.
    BTFS: (1+lam) v_m - lam v_{m+1} = u_m  => right-moving characteristics poorly handled
    BTBS: (1+lam) v_m - lam v_{m-1} = u_m  => L_lower=-lam, L_main=(1+lam), RHS=u
    BTCS: (1+lam/2) delta used differently...
    CN  : average of FTCS and BTCS
    """
    x, k = make_grid(h, lam, b_right=b_right)
    N = len(x)
    u = u0(x)
    u[0] = 0.0
    Nsteps = int(round(T / k))

    for _ in range(Nsteps):
        rhs = u.copy()
        # Set up tridiagonal: sub/main/super diags for interior points [1..N-1]
        M = N - 1  # number of unknowns (indices 1..N-1)
        sub  = np.zeros(M)
        main = np.ones(M)
        sup  = np.zeros(M)
        rhs_int = rhs[1:].copy()
        # Left BC contributes to rhs_int[0]

        if scheme == 'BTBS':
            # (1+lam) v_m - lam v_{m-1} = u_m
            main[:] = 1 + lam
            sub[1:] = -lam  # sub[i] multiplies v_{m-1} for m = i+1 (indices 1..M-1)
            # Left BC: v_0 = 0 => rhs_int[0] += lam*v_0 = 0 (nothing to add)
        elif scheme == 'BTFS':
            # v_m + lam*(v_{m+1}-v_m) = u_m => (1-lam)v_m + lam*v_{m+1} = u_m
            main[:] = 1 - lam
            sup[:-1] = lam  # sup[i] multiplies v_{m+1} for m = i+1 (indices 1..M-1)
        elif scheme == 'BTCS':
            # v_m + lam/2*(v_{m+1}-v_{m-1}) = u_m
            main[:] = 1.0
            sup[:-1] =  lam / 2.0
            sub[1:]  = -lam / 2.0
            # left BC: rhs_int[0] += lam/2 * v_0 = 0
        elif scheme == 'CrankNicolson':
            # v_m + lam/4*(v_{m+1}^{n+1}-v_{m-1}^{n+1}) = u_m - lam/4*(u_{m+1}-u_{m-1})
            main[:] = 1.0
            sup[:-1] =  lam / 4.0
            sub[1:]  = -lam / 4.0
            # Explicit part
            rhs_int[:-1] -= lam / 4.0 * (u[2:N] - u[0:N-2])
            rhs_int[-1]  -= lam / 4.0 * (u[N-1] - u[N-3])  # approximate at right
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        # Solve tridiagonal using banded solver
        # ab[0] = superdiag (shifted), ab[1] = main, ab[2] = subdiag (shifted)
        ab = np.zeros((3, M))
        ab[0, 1:]  = sup[:-1]   # superdiag: ab[0,j] for column j (j>0)
        ab[1, :]   = main
        ab[2, :-1] = sub[1:]    # subdiag: ab[2,j] for column j+1

        sol = solve_banded((1, 1), ab, rhs_int)
        u[1:] = sol
        u[0]  = 0.0  # left BC

    return x, u


# ─── Run and plot all schemes at T=1 ─────────────────────────────────────────

T  = 1.5
h  = 0.05
lm = 0.8

x_fine = np.linspace(x0, b, 1000)
u_ex   = u_exact(x_fine, T)

explicit_schemes = ['FTBS', 'FTCS', 'LaxFriedrichs', 'LaxWendroff', 'BeamWarming']
implicit_schemes = ['BTBS', 'BTCS', 'CrankNicolson']

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
axes = axes.flatten()

panel = 0
for sch in ['FTFS'] + explicit_schemes:
    try:
        xn, un = run_explicit(h, lm, T, sch)
    except Exception as e:
        print(f"{sch} failed: {e}")
        continue
    axes[panel].plot(x_fine, u_ex, 'b-', lw=2, label='Exact')
    axes[panel].plot(xn, un, 'r--', lw=1.5, label='FD')
    axes[panel].set_title(sch, fontsize=10)
    axes[panel].set_xlim(-2, b); axes[panel].set_ylim(-0.5, 1.3)
    axes[panel].legend(fontsize=7); axes[panel].grid(True)
    panel += 1

# CTCS
xn, un = run_ctcs(h, lm, T)
axes[panel].plot(x_fine, u_ex, 'b-', lw=2, label='Exact')
axes[panel].plot(xn, un, 'r--', lw=1.5, label='FD')
axes[panel].set_title('CTCS (Leapfrog)', fontsize=10)
axes[panel].set_xlim(-2, b); axes[panel].set_ylim(-0.5, 1.3)
axes[panel].legend(fontsize=7); axes[panel].grid(True)
panel += 1

for sch in implicit_schemes:
    xn, un = run_implicit(h, lm, T, sch)
    axes[panel].plot(x_fine, u_ex, 'b-', lw=2, label='Exact')
    axes[panel].plot(xn, un, 'r--', lw=1.5, label='FD')
    axes[panel].set_title(sch, fontsize=10)
    axes[panel].set_xlim(-2, b); axes[panel].set_ylim(-0.5, 1.3)
    axes[panel].legend(fontsize=7); axes[panel].grid(True)
    panel += 1

while panel < len(axes):
    axes[panel].set_visible(False)
    panel += 1

plt.suptitle(f'All FD schemes: $u_t+u_x=0$,  $h={h}$,  $\\lambda={lm}$,  $T={T}$',
             fontsize=12)
plt.tight_layout()
all_schemes_path = FIGURES_DIR / "all_schemes.png"
plt.savefig(all_schemes_path, dpi=100)
print(f"Saved: {relative_to_root(all_schemes_path)}")


# ─── Convergence study: FTBS and Lax-Wendroff ─────────────────────────────────

T_conv = 1.0
lm_conv = 0.8

for sch in ['FTBS', 'LaxWendroff']:
    print(f"\nConvergence: {sch}")
    hs_c = [0.1, 0.05, 0.025, 0.0125]
    errs_c = []
    for hc in hs_c:
        xn, un = run_explicit(hc, lm_conv, T_conv, sch)
        u_ref  = u_exact(xn, T_conv)
        err = np.max(np.abs(un - u_ref))
        errs_c.append(err)
        print(f"  h={hc:.4f}  err={err:.3e}")
    rates_c = np.log2(np.array(errs_c[:-1]) / np.array(errs_c[1:]))
    print("  Rates:", np.round(rates_c, 3))

print("\nConvergence: CTCS (Leapfrog)")
errs_ctcs = []
for hc in [0.1, 0.05, 0.025, 0.0125]:
    xn, un = run_ctcs(hc, lm_conv, T_conv)
    u_ref  = u_exact(xn, T_conv)
    errs_ctcs.append(np.max(np.abs(un - u_ref)))
    print(f"  h={hc:.4f}  err={errs_ctcs[-1]:.3e}")
rates_ctcs = np.log2(np.array(errs_ctcs[:-1]) / np.array(errs_ctcs[1:]))
print("  Rates:", np.round(rates_ctcs, 3))


# ─── Stability: FTFS with lam=0.8 and 1.1 ────────────────────────────────────

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
for col, lm_test in enumerate([0.8, 1.1]):
    xn, un = run_explicit(0.1, lm_test, T, 'FTBS')
    axes2[0, col].plot(x_fine, u_exact(x_fine, T), 'b-', lw=2, label='Exact')
    axes2[0, col].plot(xn, un, 'r--', lw=1.5, label='FD (FTBS)')
    axes2[0, col].set_title(f'FTBS  $\\lambda={lm_test}$  ({"stable" if lm_test<=1 else "unstable"})',
                            fontsize=11)
    axes2[0, col].set_xlim(-2, b)
    lim = 1.5 if lm_test <= 1 else max(1.5, np.max(np.abs(un)) * 1.1)
    axes2[0, col].set_ylim(-lim, lim)
    axes2[0, col].legend(); axes2[0, col].grid(True)

    xn, un = run_explicit(0.1, lm_test, T, 'FTCS')
    axes2[1, col].plot(x_fine, u_exact(x_fine, T), 'b-', lw=2, label='Exact')
    axes2[1, col].plot(xn, un, 'r--', lw=1.5, label='FD (FTCS)')
    axes2[1, col].set_title(f'FTCS  $\\lambda={lm_test}$  (always unstable)',
                            fontsize=11)
    axes2[1, col].set_xlim(-2, b)
    lim2 = 1.5 if abs(np.max(np.abs(un))) < 2 else 3
    axes2[1, col].set_ylim(-lim2, lim2)
    axes2[1, col].legend(); axes2[1, col].grid(True)

plt.suptitle('Stability comparison: FTBS vs FTCS', fontsize=12)
plt.tight_layout()
stability_path = FIGURES_DIR / "stability.png"
plt.savefig(stability_path, dpi=100)
print(f"Saved: {relative_to_root(stability_path)}")


# ─── Extended task sweep: vary b, h, and lambda ───────────────────────────────

all_schemes = [
    'FTFS', 'FTBS', 'FTCS', 'CTCS',
    'LaxFriedrichs', 'LaxWendroff', 'BeamWarming',
    'BTFS', 'BTBS', 'BTCS', 'CrankNicolson',
]

explicit_family = {
    'FTFS', 'FTBS', 'FTCS', 'LaxFriedrichs', 'LaxWendroff', 'BeamWarming',
}


def run_scheme(scheme, h, lam, T, b_right):
    if scheme == 'CTCS':
        return run_ctcs(h, lam, T, b_right=b_right)
    if scheme in explicit_family:
        return run_explicit(h, lam, T, scheme, b_right=b_right)
    return run_implicit(h, lam, T, scheme, b_right=b_right)


def scheme_error_and_size(scheme, h, lam, T, b_right):
    try:
        xn, un = run_scheme(scheme, h, lam, T, b_right)
        u_ref = u_exact(xn, T)
        err_inf = np.max(np.abs(un - u_ref))
        amp_inf = np.max(np.abs(un))
        finite = bool(np.all(np.isfinite(un)))
        status = 'stable-looking' if finite and amp_inf <= 5.0 else 'unstable-looking'
        return err_inf, amp_inf, status
    except Exception:
        return np.nan, np.nan, 'failed'


def observed_rates(errors, hs):
    errors = np.asarray(errors, dtype=float)
    hs = np.asarray(hs, dtype=float)
    rates = [np.nan]
    for old, new, h_old, h_new in zip(errors[:-1], errors[1:], hs[:-1], hs[1:]):
        if np.isfinite(old) and np.isfinite(new) and old > 0 and new > 0:
            rates.append(np.log(old / new) / np.log(h_old / h_new))
        else:
            rates.append(np.nan)
    return rates


def error_norms(x, u_num, T):
    h_local = x[1] - x[0]
    err = u_num - u_exact(x, T)
    e1 = h_local * np.sum(np.abs(err))
    e2 = np.sqrt(h_local * np.sum(err**2))
    einf = np.max(np.abs(err))
    return e1, e2, einf


T_sweep = 1.5
b_values = [3.0, 4.0, 5.0]
h_values = [0.1, 0.05, 0.01]
lambda_values = [0.8, 0.9, 1.0, 1.01, 1.05, 1.1]

print("\nExtended parameter sweep:")
print("  b =", b_values)
print("  h =", h_values)
print("  lambda =", lambda_values)

sweep_rows = []
for b_test in b_values:
    for h_test in h_values:
        for lam_test in lambda_values:
            for sch in all_schemes:
                err_inf, amp_inf, status = scheme_error_and_size(
                    sch, h_test, lam_test, T_sweep, b_test
                )
                sweep_rows.append({
                    'scheme': sch,
                    'b': b_test,
                    'h': h_test,
                    'lambda': lam_test,
                    'T': T_sweep,
                    'max_error': err_inf,
                    'max_abs_solution': amp_inf,
                    'status': status,
                })

sweep_csv = FIGURES_DIR / "parameter_sweep_summary.csv"
with open(sweep_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
    writer.writeheader()
    writer.writerows(sweep_rows)
print(f"Saved: {relative_to_root(sweep_csv)}")


# Convergence evidence for representative stable Courant numbers.
conv_hs = [0.1, 0.05, 0.025, 0.0125]
conv_lambdas = [0.8, 0.9]
conv_rows = []
for b_test in b_values:
    for lam_test in conv_lambdas:
        for sch in all_schemes:
            errs = []
            amps = []
            statuses = []
            for h_test in conv_hs:
                err_inf, amp_inf, status = scheme_error_and_size(
                    sch, h_test, lam_test, T_conv, b_test
                )
                errs.append(err_inf)
                amps.append(amp_inf)
                statuses.append(status)
            for h_test, err_inf, amp_inf, rate, status in zip(
                conv_hs, errs, amps, observed_rates(errs, conv_hs), statuses
            ):
                conv_rows.append({
                    'scheme': sch,
                    'b': b_test,
                    'lambda': lam_test,
                    'h': h_test,
                    'T': T_conv,
                    'max_error': err_inf,
                    'max_abs_solution': amp_inf,
                    'observed_rate': rate,
                    'status': status,
                })

conv_csv = FIGURES_DIR / "convergence_sweep_summary.csv"
with open(conv_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(conv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(conv_rows)
print(f"Saved: {relative_to_root(conv_csv)}")


# Norm-by-norm convergence plot requested in the task.
T_norm = 0.96
lam_norm = 0.8
norm_hs = [0.1, 0.05, 0.025, 0.0125, 0.00625]
norm_schemes = ['FTBS', 'CTCS', 'LaxFriedrichs', 'LaxWendroff',
                'BeamWarming', 'BTBS', 'BTCS', 'CrankNicolson']

norm_rows = []
norm_data = {sch: {'E1': [], 'E2': [], 'Einf': []} for sch in norm_schemes}
for sch in norm_schemes:
    for h_test in norm_hs:
        xn, un = run_scheme(sch, h_test, lam_norm, T_norm, 3.0)
        e1, e2, einf = error_norms(xn, un, T_norm)
        norm_data[sch]['E1'].append(e1)
        norm_data[sch]['E2'].append(e2)
        norm_data[sch]['Einf'].append(einf)
    rates = {
        key: observed_rates(values, norm_hs)
        for key, values in norm_data[sch].items()
    }
    for idx, h_test in enumerate(norm_hs):
        norm_rows.append({
            'scheme': sch,
            'b': 3.0,
            'lambda': lam_norm,
            'h': h_test,
            'T': T_norm,
            'E1': norm_data[sch]['E1'][idx],
            'E1_rate': rates['E1'][idx],
            'E2': norm_data[sch]['E2'][idx],
            'E2_rate': rates['E2'][idx],
            'Einf': norm_data[sch]['Einf'][idx],
            'Einf_rate': rates['Einf'][idx],
        })

norm_csv = FIGURES_DIR / "convergence_norms_summary.csv"
with open(norm_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(norm_rows[0].keys()))
    writer.writeheader()
    writer.writerows(norm_rows)
print(f"Saved: {relative_to_root(norm_csv)}")

fig_norm, axes_norm = plt.subplots(1, 3, figsize=(16, 4.6), sharex=True)
norm_specs = [('E1', r'$E_1$'), ('E2', r'$E_2$'), ('Einf', r'$E_\infty$')]
for ax, (key, label) in zip(axes_norm, norm_specs):
    for sch in norm_schemes:
        ax.loglog(norm_hs, norm_data[sch][key], 'o-', ms=3.5, label=sch)
    ref_h = np.asarray(norm_hs)
    if key == 'E1':
        ref = norm_data['FTBS'][key][-1] * (ref_h / ref_h[-1])
        ax.loglog(ref_h, ref, 'k--', lw=1.0, label=r'$O(h)$')
    if key in {'E2', 'Einf'}:
        ref = norm_data['LaxWendroff'][key][-1] * (ref_h / ref_h[-1])**2
        ax.loglog(ref_h, ref, 'k:', lw=1.2, label=r'$O(h^2)$')
    ax.invert_xaxis()
    ax.set_xlabel('$h$')
    ax.set_ylabel(label)
    ax.grid(True, which='both', alpha=0.35)
axes_norm[0].legend(fontsize=6.6, ncol=2)
fig_norm.suptitle(
    rf'Convergence norms at $T={T_norm:g}$, $\lambda={lam_norm:g}$, $b=3$',
    fontsize=12,
)
plt.tight_layout()
convergence_norms_path = FIGURES_DIR / "convergence_norms.png"
plt.savefig(convergence_norms_path, dpi=130)
print(f"Saved: {relative_to_root(convergence_norms_path)}")


# Plot lambda stability evidence at a fine mesh.
plot_b = 3.0
plot_h = 0.01

lambda_plot_groups = {
    "lambda_sweep_part1_cfl_limited.png": {
        "schemes": ['FTBS', 'CTCS', 'LaxFriedrichs', 'LaxWendroff'],
        "shape": (2, 2),
        "title": (
            rf'CFL-limited explicit schemes: $b={plot_b:g}$, '
            rf'$h={plot_h:g}$, $T={T_sweep:g}$'
        ),
        "note": r'Expected threshold: $\lambda=1$',
    },
    "lambda_sweep_part2_robust_unstable.png": {
        "schemes": ['FTFS', 'FTCS', 'BeamWarming', 'BTBS', 'BTCS', 'CrankNicolson'],
        "shape": (2, 3),
        "title": (
            rf'Always-unstable, wider-range, and implicit schemes: $b={plot_b:g}$, '
            rf'$h={plot_h:g}$, $T={T_sweep:g}$'
        ),
        "note": r'$\max |v^n|$ should remain $O(1)$ for a bounded physical solution',
    },
}

stability_notes = {
    'FTFS': r'downwind; unstable',
    'FTBS': r'stable for $0\leq\lambda\leq1$',
    'FTCS': r'centered FT; unstable',
    'CTCS': r'stable for $|\lambda|\leq1$',
    'LaxFriedrichs': r'stable for $|\lambda|\leq1$',
    'LaxWendroff': r'stable for $|\lambda|\leq1$',
    'BeamWarming': r'stable for $0\leq\lambda\leq2$',
    'BTBS': r'implicit upwind; stable',
    'BTCS': r'implicit centered; stable',
    'CrankNicolson': r'implicit CN; stable',
}


def plot_lambda_group(filename, config):
    nrows, ncols = config["shape"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.15 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    legend_handles = None
    legend_labels = None
    for ax, sch in zip(axes, config["schemes"]):
        errs = []
        amps = []
        for lam_test in lambda_values:
            err_inf, amp_inf, _ = scheme_error_and_size(sch, plot_h, lam_test, T_sweep, plot_b)
            errs.append(err_inf)
            amps.append(amp_inf)

        err_line, = ax.semilogy(
            lambda_values, np.maximum(errs, 1e-16), 'o-', lw=1.8, ms=5.2,
            label=r'max error $\|v^N-u(\cdot,T)\|_\infty$'
        )
        amp_line, = ax.semilogy(
            lambda_values, np.maximum(amps, 1e-16), 's--', lw=1.6, ms=4.8,
            label=r'max amplitude $\|v^N\|_\infty$'
        )
        ax.axvline(1.0, color='k', lw=1.0, ls=':', alpha=0.75)
        ax.set_title(f'{sch}\n{stability_notes.get(sch, "")}', fontsize=11)
        ax.set_xlabel(r'Courant number $\lambda=k/h$')
        ax.set_ylabel(r'log-scale magnitude')
        ax.grid(True, which='both', alpha=0.35)
        ax.text(
            0.02, 0.03, config["note"], transform=ax.transAxes,
            fontsize=8.5, va='bottom',
            bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.85, pad=2.5),
        )
        legend_handles = [err_line, amp_line]
        legend_labels = [handle.get_label() for handle in legend_handles]

    for ax in axes[len(config["schemes"]):]:
        ax.set_visible(False)

    fig.legend(
        legend_handles, legend_labels, loc='upper center', ncol=2,
        bbox_to_anchor=(0.5, 0.985), fontsize=10,
    )
    fig.suptitle(config["title"], fontsize=14, y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    print(f"Saved: {relative_to_root(output_path)}")


for filename, config in lambda_plot_groups.items():
    plot_lambda_group(filename, config)

fig3, axes3 = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
axes3 = axes3.flatten()
for ax, sch in zip(axes3, all_schemes):
    errs = []
    amps = []
    for lam_test in lambda_values:
        err_inf, amp_inf, _ = scheme_error_and_size(sch, plot_h, lam_test, T_sweep, plot_b)
        errs.append(err_inf)
        amps.append(amp_inf)
    ax.semilogy(lambda_values, np.maximum(errs, 1e-16), 'o-', label=r'$\|e\|_\infty$')
    ax.semilogy(lambda_values, np.maximum(amps, 1e-16), 's--', label=r'$\|u^n\|_\infty$')
    ax.axvline(1.0, color='k', lw=0.8, alpha=0.5)
    ax.set_title(sch, fontsize=9)
    ax.grid(True, which='both', alpha=0.35)
    ax.set_ylim(1e-4, 1e4)
    ax.set_xlabel(r'$\lambda=k/h$')
    ax.set_ylabel(r'log-scale magnitude')
    if ax is axes3[0]:
        ax.legend(fontsize=7, loc='upper left')
for ax in axes3[len(all_schemes):]:
    ax.set_visible(False)
fig3.suptitle(
    rf'Lambda sweep at $b={plot_b:g}$, $h={plot_h:g}$, $T={T_sweep:g}$',
    fontsize=12,
)
plt.tight_layout()
lambda_sweep_path = FIGURES_DIR / "lambda_sweep.png"
plt.savefig(lambda_sweep_path, dpi=120)
print(f"Saved: {relative_to_root(lambda_sweep_path)}")


# Plot right-boundary sensitivity for the main stable methods.
boundary_schemes = ['FTBS', 'LaxFriedrichs', 'LaxWendroff', 'BeamWarming', 'BTBS', 'BTCS', 'CrankNicolson']
fig4, ax4 = plt.subplots(figsize=(9, 5))
for sch in boundary_schemes:
    errs = []
    for b_test in b_values:
        err_inf, _, _ = scheme_error_and_size(sch, 0.01, 0.8, T_sweep, b_test)
        errs.append(err_inf)
    ax4.semilogy(b_values, np.maximum(errs, 1e-16), 'o-', label=sch)
ax4.set_xlabel(r'right boundary $b$')
ax4.set_ylabel(r'$\|e\|_\infty$')
ax4.set_title(r'Boundary sensitivity at $h=0.01$, $\lambda=0.8$')
ax4.grid(True, which='both', alpha=0.35)
ax4.legend(fontsize=8, ncol=2)
plt.tight_layout()
boundary_sweep_path = FIGURES_DIR / "boundary_sweep.png"
plt.savefig(boundary_sweep_path, dpi=120)
print(f"Saved: {relative_to_root(boundary_sweep_path)}")


print("\nSelected stability evidence at b=3, h=0.01:")
for sch in ['FTBS', 'FTCS', 'LaxFriedrichs', 'LaxWendroff', 'BeamWarming', 'BTBS', 'BTCS', 'CrankNicolson']:
    line = [sch]
    for lam_test in lambda_values:
        _, amp_inf, status = scheme_error_and_size(sch, plot_h, lam_test, T_sweep, plot_b)
        line.append(f"lambda={lam_test:g}: max|u|={amp_inf:.2e}, {status}")
    print("  " + " | ".join(line))
