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

def make_grid(h, lam):
    k = lam * h
    x = np.arange(x0, b + h/2, h)
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


def run_explicit(h, lam, T, scheme):
    x, k = make_grid(h, lam)
    u = u0(x)
    u[0] = 0.0  # left BC
    t = 0.0
    Nsteps = int(round(T / k))
    for _ in range(Nsteps):
        u = advance_explicit(u, lam, scheme)
        t += k
    return x, u


# ─── Leapfrog (CTCS) ─────────────────────────────────────────────────────────

def run_ctcs(h, lam, T):
    x, k = make_grid(h, lam)
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

def run_implicit(h, lam, T, scheme):
    """
    Implicit schemes: solve tridiagonal system at each step.
    BTFS: (1+lam) v_m - lam v_{m+1} = u_m  => right-moving characteristics poorly handled
    BTBS: (1+lam) v_m - lam v_{m-1} = u_m  => L_lower=-lam, L_main=(1+lam), RHS=u
    BTCS: (1+lam/2) delta used differently...
    CN  : average of FTCS and BTCS
    """
    x, k = make_grid(h, lam)
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
