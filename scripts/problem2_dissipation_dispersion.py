"""
Problem 2: Dissipation and Dispersion

PDE:  u_t + a u_x = alpha u_xx - beta u_xxx,  x in (-pi, pi), periodic BC
      u(x,0) = u0(x)

Fourier analysis:
  Write u(x,t) = sum_k v_k(t) e^{ikx}.
  Substituting: v_k'(t) = L_k v_k,  L_k = -iak - alpha k^2 + i beta k^3
  => v_k(t) = v_k(0) * exp(L_k t)
            = v_k(0) * exp(-alpha k^2 t) * exp(i(-ak + beta k^3) t)

  => u(x,t) = sum_k v_k(0) * exp(-alpha k^2 t)
                             * exp(i(kx - (ak - beta k^3)t))

DISSIPATION (alpha>0, beta=0):
  Amplitude of each mode decays as exp(-alpha k^2 t).
  High-frequency modes (large |k|) decay faster => SMOOTHING EFFECT.

DISPERSION (alpha=0, beta!=0):
  Phase speed c(k) = a - beta k^2 is FREQUENCY-DEPENDENT.
  Different wave numbers travel at different speeds => WAVE PACKET SPREADS.

Numerical scheme:
  Second-order Crank-Nicolson in time + second-order central differences in
  space.  Because the spatial operators with periodic BC are circulant, they
  are diagonalised by the DFT, and the scheme reduces to a scalar update for
  each Fourier mode.

  FD eigenvalues (central differences, 2nd order):
    D1 eigenvalue:  lambda1_k = i*sin(kh)/h   -> ik  as h->0
    D2 eigenvalue:  lambda2_k = 2*(cos(kh)-1)/h^2  -> -k^2
    D3 eigenvalue:  lambda3_k = i*(sin(2kh)-2*sin(kh))/h^3  -> -ik^3

  L_k (FD) = -a*lambda1_k + alpha*lambda2_k - beta*lambda3_k
           -> -iak - alpha k^2 + i beta k^3  = L_k (exact)

  CN update per step: U^{n+1}_k = [(1 + dt/2 L_k)/(1 - dt/2 L_k)] U^n_k
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from project_paths import ensure_problem_figure_dir, relative_to_root


FIGURES_DIR = ensure_problem_figure_dir("problem2")


# ─── Exact solution (FFT-based, exact for given periodic grid) ───────────────

def exact_solution_fft(x, t, u0, a, alpha, beta):
    """
    Exact solution for periodic u0 given on a uniform grid on (-pi,pi).
    Evolve each DFT mode exactly.
    """
    N = len(x)
    h = 2 * np.pi / N
    U0 = np.fft.fft(u0.astype(complex))
    kk = np.fft.fftfreq(N, d=h / (2 * np.pi))   # integer wavenumbers
    # L_k = -iak - alpha k^2 + i beta k^3
    Lk = -1j * a * kk - alpha * kk**2 + 1j * beta * kk**3
    return np.fft.ifft(U0 * np.exp(Lk * t)).real


# ─── FD Crank-Nicolson scheme (periodic, operated in Fourier space) ──────────

def fd_cn_periodic(x, T, u0, a, alpha, beta, dt):
    """
    Second-order Crank-Nicolson time integration with 2nd-order central-
    difference spatial operators on a periodic domain.

    FD eigenvalues of central-difference operators for mode k:
      D1: i*sin(kh)/h        (approximates  i*k = d/dx eigenvalue)
      D2: 2*(cos(kh)-1)/h^2  (approximates -k^2 = d^2/dx^2 eigenvalue)
      D3: i*(sin(2kh)-2*sin(kh))/h^3  (approximates -ik^3 = d^3/dx^3 eigenvalue)
    """
    N = len(x)
    h = 2 * np.pi / N
    kk = np.fft.fftfreq(N, d=h / (2 * np.pi))

    lam1 = 1j * np.sin(kk * h) / h                                     # D1
    lam2 = 2.0 * (np.cos(kk * h) - 1.0) / h**2                        # D2
    lam3 = 1j * (np.sin(2 * kk * h) - 2 * np.sin(kk * h)) / h**3      # D3

    # L_k (FD) = -a D1 + alpha D2 - beta D3
    Lk = -a * lam1 + alpha * lam2 - beta * lam3

    # Crank-Nicolson multiplier per mode per time step
    cn_mult = (1.0 + 0.5 * dt * Lk) / (1.0 - 0.5 * dt * Lk)

    U = np.fft.fft(u0.astype(complex))
    Nsteps = int(round(T / dt))
    for _ in range(Nsteps):
        U *= cn_mult
    return np.fft.ifft(U).real


# ─── Illustration: advection / dissipation / dispersion ─────────────────────

N     = 512
x     = np.linspace(-np.pi, np.pi, N, endpoint=False)
a_val = 1.0
T_ill = 2.0
dt    = 1e-3

# Smooth, truly periodic initial condition (sum of a few modes)
def u0_smooth(x):
    return np.sin(x) + 0.6 * np.cos(2 * x) + 0.3 * np.sin(3 * x)

u0 = u0_smooth(x)

cases = [
    (0.0,  0.0,  r'Pure advection: $\alpha=0,\,\beta=0$'),
    (0.15, 0.0,  r'Dissipation: $\alpha=0.15,\,\beta=0$'),
    (0.0,  0.05, r'Dispersion: $\alpha=0,\,\beta=0.05$'),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (alpha, beta, title) in zip(axes, cases):
    u_ex = exact_solution_fft(x, T_ill, u0, a_val, alpha, beta)
    u_fd = fd_cn_periodic(x, T_ill, u0, a_val, alpha, beta, dt)
    ax.plot(x, u0,   'g--', lw=1.5, label='Initial $u_0$')
    ax.plot(x, u_ex, 'b-',  lw=2,   label=f'Exact (Fourier), $t={T_ill}$')
    ax.plot(x, u_fd, 'r:',  lw=2,   label='FD (Crank–Nicolson)')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('$x$')
    ax.legend(fontsize=8)
    ax.grid(True)

plt.suptitle(r'$u_t + u_x = \alpha u_{xx} - \beta u_{xxx}$,  $a=1$', fontsize=13)
plt.tight_layout()
dissipation_dispersion_path = FIGURES_DIR / "dissipation_dispersion.png"
plt.savefig(dissipation_dispersion_path, dpi=120)
print(f"Saved: {relative_to_root(dissipation_dispersion_path)}")


# ─── Convergence: spatial (vary N, fixed tiny dt) ────────────────────────────

alpha_c, beta_c = 0.1, 0.05
T_c   = 0.5
dt_ref = 1e-6   # essentially exact in time

print("\nSpatial convergence test (fixed dt=1e-6, vary N):")
Ns_sp = [16, 32, 64, 128, 256]
errs_sp = []
for Ni in Ns_sp:
    xi  = np.linspace(-np.pi, np.pi, Ni, endpoint=False)
    u0i = u0_smooth(xi)
    u_ex = exact_solution_fft(xi, T_c, u0i, a_val, alpha_c, beta_c)
    u_fd = fd_cn_periodic(xi, T_c, u0i, a_val, alpha_c, beta_c, dt_ref)
    err  = np.max(np.abs(u_fd - u_ex))
    errs_sp.append(err)
    print(f"  N={Ni:4d}  h={2*np.pi/Ni:.4f}  err={err:.3e}")
rates_sp = np.log2(np.array(errs_sp[:-1]) / np.array(errs_sp[1:]))
print("  Convergence rates:", np.round(rates_sp, 3))

# ─── Convergence: temporal (vary dt, fixed large N) ──────────────────────────

N_t  = 512
x_t  = np.linspace(-np.pi, np.pi, N_t, endpoint=False)
u0_t = u0_smooth(x_t)
u_ex_t = exact_solution_fft(x_t, T_c, u0_t, a_val, alpha_c, beta_c)

print("\nTemporal convergence test (fixed N=512, vary dt):")
dts = [0.1, 0.05, 0.025, 0.0125]
errs_dt = []
for dti in dts:
    u_fd = fd_cn_periodic(x_t, T_c, u0_t, a_val, alpha_c, beta_c, dti)
    err  = np.max(np.abs(u_fd - u_ex_t))
    errs_dt.append(err)
    print(f"  dt={dti:.4f}  err={err:.3e}")
rates_dt = np.log2(np.array(errs_dt[:-1]) / np.array(errs_dt[1:]))
print("  Convergence rates:", np.round(rates_dt, 3))

# ─── Convergence plots ───────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

hs = np.array([2 * np.pi / Ni for Ni in Ns_sp])
axes2[0].loglog(hs, errs_sp, 'bo-', label='FD max-error')
axes2[0].loglog(hs, errs_sp[2] * (hs / hs[2])**2, 'k--', label='$O(h^2)$ ref')
axes2[0].set_xlabel('$h = 2\\pi/N$')
axes2[0].set_ylabel('Max error')
axes2[0].set_title('Spatial convergence (Crank–Nicolson FD)')
axes2[0].legend()
axes2[0].grid(True)

axes2[1].loglog(dts, errs_dt, 'ro-', label='FD max-error')
axes2[1].loglog(dts, errs_dt[1] * (np.array(dts) / dts[1])**2, 'k--', label='$O(\\Delta t^2)$ ref')
axes2[1].set_xlabel('$\\Delta t$')
axes2[1].set_ylabel('Max error')
axes2[1].set_title('Temporal convergence (Crank–Nicolson FD)')
axes2[1].legend()
axes2[1].grid(True)

plt.suptitle('Convergence of CN-FD for dissipation-dispersion PDE', fontsize=12)
plt.tight_layout()
convergence_fd_path = FIGURES_DIR / "convergence_fd.png"
plt.savefig(convergence_fd_path, dpi=120)
print(f"Saved: {relative_to_root(convergence_fd_path)}")


# ─── Additional: temporal convergence using spectral (exact) spatial operator

def spectral_cn(x, T, u0, a, alpha, beta, dt):
    """CN in time with the exact spectral spatial operator (no spatial FD error)."""
    N = len(x)
    h = 2 * np.pi / N
    kk = np.fft.fftfreq(N, d=h / (2 * np.pi))
    Lk = -1j * a * kk - alpha * kk**2 + 1j * beta * kk**3
    cn_mult = (1.0 + 0.5 * dt * Lk) / (1.0 - 0.5 * dt * Lk)
    U = np.fft.fft(u0.astype(complex))
    Nsteps = int(round(T / dt))
    for _ in range(Nsteps):
        U *= cn_mult
    return np.fft.ifft(U).real

N_t2 = 256
x_t2  = np.linspace(-np.pi, np.pi, N_t2, endpoint=False)
u0_t2 = u0_smooth(x_t2)
u_ex_t2 = exact_solution_fft(x_t2, T_c, u0_t2, a_val, alpha_c, beta_c)

print("\nTemporal convergence (spectral CN, N=256):")
dts2 = [0.1, 0.05, 0.025, 0.0125, 0.00625]
errs_dt2 = []
for dti in dts2:
    u_fd = spectral_cn(x_t2, T_c, u0_t2, a_val, alpha_c, beta_c, dti)
    err  = np.max(np.abs(u_fd - u_ex_t2))
    errs_dt2.append(err)
    print(f"  dt={dti:.5f}  err={err:.3e}")
rates_dt2 = np.log2(np.array(errs_dt2[:-1]) / np.array(errs_dt2[1:]))
print("  Convergence rates:", np.round(rates_dt2, 3))

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
hs2 = np.array([2 * np.pi / Ni for Ni in Ns_sp])
axes3[0].loglog(hs2, errs_sp, 'bo-', label='FD max-error')
axes3[0].loglog(hs2, errs_sp[2] * (hs2 / hs2[2])**2, 'k--', label='$O(h^2)$ ref')
axes3[0].set_xlabel('$h = 2\\pi/N$'); axes3[0].set_ylabel('Max error')
axes3[0].set_title('Spatial convergence (FD central diff)'); axes3[0].legend(); axes3[0].grid(True)

axes3[1].loglog(dts2, errs_dt2, 'rs-', label='Spectral-CN max-error')
axes3[1].loglog(dts2, errs_dt2[1] * (np.array(dts2) / dts2[1])**2, 'k--',
                label='$O(\\Delta t^2)$ ref')
axes3[1].set_xlabel('$\\Delta t$'); axes3[1].set_ylabel('Max error')
axes3[1].set_title('Temporal convergence (CN, 2nd order)'); axes3[1].legend(); axes3[1].grid(True)

plt.suptitle('Second-order convergence: space and time', fontsize=12)
plt.tight_layout()
convergence_spectral_path = FIGURES_DIR / "convergence_spectral.png"
plt.savefig(convergence_spectral_path, dpi=120)
print(f"Saved: {relative_to_root(convergence_spectral_path)}")
