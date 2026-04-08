"""
Problem 4: Numerical Study of the Burgers' Equations

Viscous Burgers' equation:  u_t + u u_x = eps u_xx,  x in R,  t > 0
Equivalently:               u_t + (1/2)(u^2)_x = eps u_xx

(a) Exact traveling wave solution:
    u(x,t) = omega*(1 - tanh[omega/(2*eps)*(x - omega*t - x0)])

(b) CNLF scheme (Crank-Nicolson-Leap-Frog):
    Time: Leapfrog for the full time derivative (3-level scheme)
    Nonlinear term at level n (from the problem statement):
      uu_x|(m,n) = (1/2)*d_x(u^2)|(m,n)
                 ≈ (1/2)*[(v_{m+1}^n)^2 - (v_{m-1}^n)^2]/(2h)
                 = [(v_{m+1}^n)^2 - (v_{m-1}^n)^2] / (4h)
    Diffusion: Crank-Nicolson between levels n-1 and n+1

    Full scheme (multiply through by 2k, mu = eps*k/h^2):
      v_m^{n+1} - mu*delta^2_x v^{n+1}
        = v_m^{n-1} + mu*delta^2_x v^{n-1}
          - k/(2h) * [(v_{m+1}^n)^2 - (v_{m-1}^n)^2]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


# ─── (a) Exact traveling wave solution ───────────────────────────────────────

def u_exact_tw(x, t, omega, eps, x0):
    """omega*(1 - tanh[omega/(2*eps)*(x - omega*t - x0)])"""
    xi = (omega / (2 * eps)) * (x - omega * t - x0)
    return omega * (1.0 - np.tanh(xi))


def verify_traveling_wave(omega=0.5, eps=0.1, x0=3.0):
    """Verify the exact solution numerically."""
    x  = np.linspace(-5, 5, 10000)
    t  = 1.0
    h  = x[1] - x[0]
    dt = 1e-8
    u  = u_exact_tw(x, t, omega, eps, x0)
    ut = (u_exact_tw(x, t + dt, omega, eps, x0) -
          u_exact_tw(x, t - dt, omega, eps, x0)) / (2 * dt)
    ux  = np.gradient(u, h)
    uxx = np.gradient(ux, h)
    res = np.max(np.abs((ut + u * ux - eps * uxx)[10:-10]))
    print(f"Verification: max|u_t + u*u_x - eps*u_xx| = {res:.3e}  (should be ~0)")

verify_traveling_wave()


# ─── CNLF scheme ─────────────────────────────────────────────────────────────

def _build_tridiag(M, mu):
    """
    Build the banded storage for the tridiagonal system:
      (I - mu*delta^2_x) v^{n+1} = rhs
    Diagonal = 1 + 2*mu,  off-diagonal = -mu.
    """
    ab = np.zeros((3, M))
    ab[0, 1:]  = -mu      # superdiagonal
    ab[1, :]   = 1 + 2*mu # main diagonal
    ab[2, :-1] = -mu      # subdiagonal
    return ab


def cnlf_step(u_old, u_cur, bc_l_new, bc_r_new, mu, r):
    """
    One CNLF step.  u_old = v^{n-1},  u_cur = v^n.
    mu = eps*k/h^2,  r = k/(2h)  (nonlinear coefficient).
    Returns v^{n+1}.
    """
    N = len(u_old)
    M = N - 2  # interior unknowns
    ab = _build_tridiag(M, mu)

    # RHS: v^{n-1} + mu*delta^2_x v^{n-1}  -  r*[(v_{m+1}^n)^2 - (v_{m-1}^n)^2]
    rhs = (u_old[1:-1]
           + mu * (u_old[2:] - 2 * u_old[1:-1] + u_old[:-2])
           - r  * (u_cur[2:]**2 - u_cur[:-2]**2))

    # Incorporate new-time BCs into RHS (moved from LHS)
    rhs[0]  += mu * bc_l_new
    rhs[-1] += mu * bc_r_new

    u_new = np.zeros(N)
    u_new[0]  = bc_l_new
    u_new[-1] = bc_r_new
    u_new[1:-1] = solve_banded((1, 1), ab, rhs)
    return u_new


def cn_startup_step(u0, bc_l1, bc_r1, mu_h, r_h):
    """
    Single Crank-Nicolson step to get v^1 from v^0.
    Uses mu_h = mu/2 = eps*k/(2h^2)  and  r_h = k/(4h) for the nonlinear term.
    Scheme: v^1 - (mu/2)*delta^2_x v^1 = v^0 + (mu/2)*delta^2_x v^0 - r_h*[...]
    """
    N = len(u0)
    M = N - 2
    ab = _build_tridiag(M, mu_h)   # uses mu_h = mu/2

    rhs = (u0[1:-1]
           + mu_h * (u0[2:] - 2 * u0[1:-1] + u0[:-2])
           - r_h  * (u0[2:]**2 - u0[:-2]**2))
    rhs[0]  += mu_h * bc_l1
    rhs[-1] += mu_h * bc_r1

    u1 = np.zeros(N)
    u1[0]  = bc_l1
    u1[-1] = bc_r1
    u1[1:-1] = solve_banded((1, 1), ab, rhs)
    return u1


def run_cnlf(x, T, u0_vals, bc_left, bc_right, eps, k,
             u1_exact=None):
    """
    Run the CNLF scheme.

    x        : spatial grid
    T        : final time
    u0_vals  : initial condition u(x,0)
    bc_left  : callable t -> u(x[0],t)
    bc_right : callable t -> u(x[-1],t)
    eps      : viscosity
    k        : time step
    u1_exact : if given, use as v^1 (exact startup); otherwise use CN startup
    """
    h  = x[1] - x[0]
    mu = eps * k / h**2
    r  = k / (2 * h)          # nonlinear coefficient for CNLF
    r_h = k / (4 * h)         # nonlinear coefficient for CN startup
    mu_h = mu / 2              # diffusion for CN startup

    # Level 0
    u_old = u0_vals.copy()
    u_old[0]  = bc_left(0.0)
    u_old[-1] = bc_right(0.0)

    # Level 1
    if u1_exact is not None:
        u_cur = u1_exact.copy()
    else:
        bc_l1 = bc_left(k)
        bc_r1 = bc_right(k)
        u_cur = cn_startup_step(u_old, bc_l1, bc_r1, mu_h, r_h)

    Nsteps = int(round(T / k))
    for n in range(1, Nsteps):
        t_new = (n + 1) * k
        u_new = cnlf_step(u_old, u_cur,
                          bc_left(t_new), bc_right(t_new),
                          mu, r)
        u_old = u_cur
        u_cur = u_new

    return u_cur


# ─── (b) Test: traveling wave solution ───────────────────────────────────────

omega = 0.5
eps   = 0.1
x0_w  = 3.0
xL, xR = -5.0, 5.0
T_end  = 2.0

k_ref = 1e-3
h_ref = 1e-3
x_ref = np.arange(xL, xR + h_ref / 2, h_ref)

u0_ref = u_exact_tw(x_ref, 0.0, omega, eps, x0_w)
u1_ref = u_exact_tw(x_ref, k_ref, omega, eps, x0_w)
bc_l = lambda t: u_exact_tw(np.array([xL]), t, omega, eps, x0_w).item()
bc_r = lambda t: u_exact_tw(np.array([xR]), t, omega, eps, x0_w).item()

print(f"\nRunning CNLF (traveling wave, h=k={k_ref})...")
u_num = run_cnlf(x_ref, T_end, u0_ref, bc_l, bc_r, eps, k_ref, u1_exact=u1_ref)
u_ex  = u_exact_tw(x_ref, T_end, omega, eps, x0_w)
err_ref = np.max(np.abs(u_num - u_ex))
print(f"  Max error at T={T_end}: {err_ref:.3e}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(x_ref, u_ex,  'b-',  lw=2,   label='Exact')
axes[0].plot(x_ref, u_num, 'r--', lw=1.5, label=f'CNLF (h=k={k_ref})')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$u$')
axes[0].set_title(f'Traveling wave: $T={T_end}$, $\\varepsilon={eps}$, $\\omega={omega}$')
axes[0].legend(); axes[0].grid(True)

axes[1].semilogy(x_ref, np.abs(u_num - u_ex) + 1e-16, 'r-')
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$|$error$|$')
axes[1].set_title(f'Pointwise error (max = {err_ref:.2e})')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('problem4_traveling_wave.png', dpi=120)
print("Saved: problem4_traveling_wave.png")


# ─── Convergence study ────────────────────────────────────────────────────────

T_conv = 1.0
hs = [4e-2, 2e-2, 1e-2, 5e-3]
errs_conv = []
print("\nConvergence (h=k coupled):")
for hi in hs:
    ki = hi
    xi = np.arange(xL, xR + hi / 2, hi)
    u0i = u_exact_tw(xi, 0.0, omega, eps, x0_w)
    u1i = u_exact_tw(xi, ki,  omega, eps, x0_w)
    bc_li = lambda t, _xL=xL: u_exact_tw(np.array([_xL]), t, omega, eps, x0_w).item()
    bc_ri = lambda t, _xR=xR: u_exact_tw(np.array([_xR]), t, omega, eps, x0_w).item()
    u_ni = run_cnlf(xi, T_conv, u0i, bc_li, bc_ri, eps, ki, u1_exact=u1i)
    u_ei = u_exact_tw(xi, T_conv, omega, eps, x0_w)
    err  = np.max(np.abs(u_ni - u_ei))
    errs_conv.append(err)
    print(f"  h=k={hi:.4f}  err={err:.3e}")

rates_c = np.log2(np.array(errs_conv[:-1]) / np.array(errs_conv[1:]))
print("  Convergence rates:", np.round(rates_c, 3))

fig3, ax3 = plt.subplots(figsize=(7, 5))
hs_arr = np.array(hs)
ax3.loglog(hs_arr, errs_conv, 'bo-', label='CNLF max-error')
ax3.loglog(hs_arr, errs_conv[1] * (hs_arr / hs_arr[1])**2, 'k--', label='$O(h^2)$ ref')
ax3.set_xlabel('$h = k$'); ax3.set_ylabel('Max error')
ax3.set_title('CNLF convergence: viscous Burgers\' (traveling wave)')
ax3.legend(); ax3.grid(True)
plt.tight_layout()
plt.savefig('problem4_convergence.png', dpi=120)
print("Saved: problem4_convergence.png")


# ─── (c) Burgers' on (-1,1) with u(±1,t)=0,  u(x,0)=-sin(πx) ────────────────

eps_c  = 0.02
T_c    = 0.995
k_c    = 1e-3
h_c    = 1e-3
x_c    = np.arange(-1.0, 1.0 + h_c / 2, h_c)
N_c    = len(x_c)

print(f"\nSimulating Burgers' on (-1,1): eps={eps_c}, k=h={k_c}, T={T_c}...")

u0_c = -np.sin(np.pi * x_c)

bc_lc = lambda t: 0.0
bc_rc = lambda t: 0.0

# Collect snapshots during time stepping for the plot
t_snaps = [0.0, 0.2, 0.4, 0.6, 0.8, T_c]

fig2, ax2 = plt.subplots(figsize=(9, 6))
ax2.plot(x_c, u0_c, label=f'$t=0$')

mu_c  = eps_c * k_c / h_c**2
r_c   = k_c / (2 * h_c)
r_h_c = k_c / (4 * h_c)
mu_h_c = mu_c / 2
ab_c   = _build_tridiag(N_c - 2, mu_c)

u_prev = u0_c.copy()
u_prev[0] = 0.0; u_prev[-1] = 0.0
# CN startup
u_cur = cn_startup_step(u_prev, 0.0, 0.0, mu_h_c, r_h_c)

Nsteps_c = int(round(T_c / k_c))
snap_idx  = 1

for n in range(1, Nsteps_c):
    t_new = (n + 1) * k_c
    u_new = cnlf_step(u_prev, u_cur, 0.0, 0.0, mu_c, r_c)
    u_prev = u_cur
    u_cur  = u_new
    if snap_idx < len(t_snaps) and t_new >= t_snaps[snap_idx] - k_c / 2:
        ax2.plot(x_c, u_cur, label=f'$t={t_snaps[snap_idx]:.1f}$')
        snap_idx += 1

ax2.set_xlabel('$x$'); ax2.set_ylabel('$u(x,t)$')
ax2.set_title(f"Burgers' on $(-1,1)$:  $\\varepsilon={eps_c}$,  $k=h={k_c}$,  $T={T_c}$")
ax2.legend(fontsize=9); ax2.grid(True)
plt.tight_layout()
plt.savefig('problem4_burgers_sine.png', dpi=120)
print("Saved: problem4_burgers_sine.png")
