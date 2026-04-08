"""
Problem 1: Finite Difference Method for the Elliptic PDE
-(a(x) u'(x))' + c(x) u(x) = f(x),  x in (0,1),  u(0)=u(1)=0

Second-order FD scheme using the approximation:
  -(a(x)u'(x))' ≈ -[ a_{j+1/2}*v_{j+1} - (a_{j+1/2}+a_{j-1/2})*v_j + a_{j-1/2}*v_{j-1} ] / h^2
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# ─── helper: assemble and solve the tridiagonal system ───────────────────────

def solve_fd(N, a_func, c_func, f_func):
    """
    Solve -(a u')' + c u = f on (0,1) with u(0)=u(1)=0.

    Interior nodes: j = 1, ..., N-1  (N intervals, h = 1/N).
    Returns (x_int, v) where x_int are the interior node positions and
    v is the FD solution.
    """
    h = 1.0 / N
    # interior nodes
    x = np.linspace(h, 1 - h, N - 1)   # x_1, ..., x_{N-1}

    # half-node values
    x_plus  = x + h / 2.0               # x_{j+1/2}
    x_minus = x - h / 2.0               # x_{j-1/2}
    a_plus  = a_func(x_plus)
    a_minus = a_func(x_minus)

    # diagonal entries of -L  (L is the discrete Laplacian-like operator)
    main_diag  = (a_plus + a_minus) / h**2 + c_func(x)
    upper_diag = -a_plus[:-1] / h**2      # connects j to j+1
    lower_diag = -a_minus[1:] / h**2      # connects j to j-1

    rhs = f_func(x)

    # solve using banded storage (scipy solve_banded)
    # ab[0] = upper diag (shifted), ab[1] = main diag, ab[2] = lower diag
    M = len(x)
    ab = np.zeros((3, M))
    ab[0, 1:]  = upper_diag   # upper: ab[0, j] refers to column j, but upper diagonal stored shifted
    ab[1, :]   = main_diag
    ab[2, :-1] = lower_diag

    v = solve_banded((1, 1), ab, rhs)
    return x, v


# ─── (c) Manufactured solution test ──────────────────────────────────────────

def a(x):    return 1 + x**2
def c(x):    return np.exp(x)

def u_exact(x):
    return x * np.sin(2 * np.pi * x)

def u_prime(x):
    return np.sin(2 * np.pi * x) + 2 * np.pi * x * np.cos(2 * np.pi * x)

def au_prime(x):
    return a(x) * u_prime(x)

def d_au_prime(x):
    """-(a u')'  computed analytically for the manufactured solution."""
    # a(x) = 1+x^2, u'(x) = sin(2πx) + 2πx cos(2πx)
    # (a u')' = a'u' + a u''
    # a'(x) = 2x
    # u''(x) = 4π cos(2πx) - 4π² x sin(2πx)
    ap  = 2 * x
    up  = np.sin(2 * np.pi * x) + 2 * np.pi * x * np.cos(2 * np.pi * x)
    upp = 4 * np.pi * np.cos(2 * np.pi * x) - 4 * np.pi**2 * x * np.sin(2 * np.pi * x)
    return -(ap * up + a(x) * upp)

def f_manufactured(x):
    return d_au_prime(x) + c(x) * u_exact(x)


# Convergence test
Ns = [10, 20, 40, 80, 160, 320]
errors = []
for N in Ns:
    x_int, v = solve_fd(N, a, c, f_manufactured)
    err = np.max(np.abs(v - u_exact(x_int)))
    errors.append(err)
    print(f"N={N:4d}  h={1/N:.5f}  max-error={err:.3e}")

errors = np.array(errors)
Ns_arr = np.array(Ns)
rates = np.log2(errors[:-1] / errors[1:])
print("\nConvergence rates (should be ≈ 2):")
for i, r in enumerate(rates):
    print(f"  N={Ns[i]}→{Ns[i+1]}: rate={r:.3f}")

# Plot solution at N=160
N_plot = 160
x_int, v = solve_fd(N_plot, a, c, f_manufactured)
x_full  = np.concatenate([[0], x_int, [1]])
v_full  = np.concatenate([[0], v,     [0]])
u_full  = u_exact(x_full)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(x_full, u_full,  'b-',  lw=2, label='Exact $u(x)=x\\sin(2\\pi x)$')
axes[0].plot(x_full, v_full,  'r--', lw=1.5, label=f'FD solution (N={N_plot})')
axes[0].set_xlabel('x')
axes[0].set_title('Manufactured solution test')
axes[0].legend()
axes[0].grid(True)

axes[1].loglog(1.0/Ns_arr, errors, 'bo-', label='Max error')
# reference slope-2 line
h_ref = 1.0 / Ns_arr
axes[1].loglog(h_ref, errors[0] * (h_ref / h_ref[0])**2, 'k--', label='$O(h^2)$ reference')
axes[1].set_xlabel('h = 1/N')
axes[1].set_ylabel('Max error')
axes[1].set_title('Convergence plot (manufactured solution)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('problem1_manufactured_solution.png', dpi=120)
print("\nSaved: problem1_manufactured_solution.png")


# ─── (d) f(x) = 1 ─────────────────────────────────────────────────────────────

def f_one(x):
    return np.ones_like(x)

N_compute = 400
x_int, v = solve_fd(N_compute, a, c, f_one)
x_full  = np.concatenate([[0], x_int, [1]])
v_full  = np.concatenate([[0], v,     [0]])

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(x_full, v_full, 'b-', lw=2)
ax2.set_xlabel('x')
ax2.set_ylabel('u(x)')
ax2.set_title(r"Solution of $-(a(x)u')' + c(x)u = 1$, $a=1+x^2$, $c=e^x$")
ax2.grid(True)
plt.tight_layout()
plt.savefig('problem1_f_equals_1.png', dpi=120)
print("Saved: problem1_f_equals_1.png")

# Basic sanity checks: solution should be non-negative (since f≥0, c≥0, a>0)
print(f"\nMin of numerical solution with f=1: {v_full.min():.6f}  (should be ≥ 0)")
print(f"Max of numerical solution with f=1: {v_full.max():.6f}")
