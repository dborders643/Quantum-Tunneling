import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy.optimize import root_scalar

# Enable interactive mode for dual-window rendering
plt.ion()

    
def solve_bound_states(a, V0, m, hbar, x_grid, dx):
  """Computes transcendental functions for even and odd parity to find discrete bound states (-V0 < E < 0)."""
  z0 = a * np.sqrt(2 * m * V0) / hbar
  bound_states = []

  # Transcendental functions
  f_even = lambda z: z * np.tan(z) - np.sqrt(z0**2 - z**2)
  f_odd = lambda z: -z / np.tan(z) - np.sqrt(z0**2 - z**2)

  n_max = int(np.ceil(2 * z0 / np.pi))
  for n in range(n_max):
    z_min = n * np.pi / 2 + 1e-4
    z_max = min((n + 1) * np.pi / 2 - 1e-4, z0 - 1e-4)
    if z_min >= z_max:
      continue

    is_even = n % 2 == 0
    f = f_even if is_even else f_odd
    if f(z_min) * f(z_max) < 0:
      res = root_scalar(f, bracket=[z_min, z_max])
      if res.converged:
        z_val = res.root
        q_val = z_val / a
        kappa_val = np.sqrt(z0**2 - z_val**2) / a
        E_val = -(hbar**2 * kappa_val**2) / (2 * m)

        # Spatial wavefunction construction
        psi_b = np.zeros_like(x_grid)
        region1 = x_grid < -a
        region2 = np.abs(x_grid) <= a
        region3 = x_grid > a

        if is_even:
          C = np.cos(q_val * a) * np.exp(kappa_val * a)
          psi_b[region1] = C * np.exp(kappa_val * x_grid[region1])
          psi_b[region2] = np.cos(q_val * x_grid[region2])
          psi_b[region3] = C * np.exp(-kappa_val * x_grid[region3])
        else:
          C = np.sin(q_val * a) * np.exp(kappa_val * a)
          psi_b[region1] = -C * np.exp(kappa_val * x_grid[region1])
          psi_b[region2] = np.sin(q_val * x_grid[region2])
          psi_b[region3] = C * np.exp(-kappa_val * x_grid[region3])

        # Normalize
        norm = np.sqrt(np.sum(np.abs(psi_b) ** 2) * dx)
        psi_b /= norm

        bound_states.append({
            'n': len(bound_states),
            'E': E_val,
            'psi': psi_b,
            'parity': 'even' if is_even else 'odd',
        })

  return bound_states


def get_psi_k(k, x_grid, a, V0, m, hbar):
  """Computes wavefunction psi_k(x) for E_k > 0 (scattering states)."""
  E_k = (hbar**2 * k**2) / (2 * m)
  q = np.sqrt(2 * m * (E_k + V0)) / hbar

  denom = np.cos(2 * q * a) - 1j * ((q**2 + k**2) / (2 * q * k)) * np.sin(
      2 * q * a
  )
  t_amp = np.exp(-2j * k * a) / denom

  C_amp = 0.5 * t_amp * (1 + k / q) * np.exp(1j * (k - q) * a)
  D_amp = 0.5 * t_amp * (1 - k / q) * np.exp(1j * (k + q) * a)

  r_amp = (
      C_amp * np.exp(-1j * q * a) + D_amp * np.exp(1j * q * a) - np.exp(-1j * k * a)
  ) * np.exp(-1j * k * a)

  psi_k = np.zeros(len(x_grid), dtype=complex)
  region1 = x_grid < -a
  region2 = np.abs(x_grid) <= a
  region3 = x_grid > a

  psi_k[region1] = np.exp(1j * k * x_grid[region1]) + r_amp * np.exp(
      -1j * k * x_grid[region1]
  )
  psi_k[region2] = C_amp * np.exp(1j * q * x_grid[region2]) + D_amp * np.exp(
      -1j * q * x_grid[region2]
  )
  psi_k[region3] = t_amp * np.exp(1j * k * x_grid[region3])

  return psi_k


def main():
  # =======================
  # 1. Physics Parameters
  # =======================
  hbar = 1.0
  m = 1.0
  a = 5.0
  V0 = 100.0

  # Spatial grid
  x_min, x_max, N_x = -30.0, 30.0, 1024
  x_grid = np.linspace(x_min, x_max, N_x)
  dx = x_grid[1] - x_grid[0]
  V_potential = np.where(np.abs(x_grid) <= a, -V0, 0.0)

  # Initial scattering packet
  x0 = -15.0
  sigma0 = 1.5
  k0 = 10.0

  # Momentum grid
  k_min = k0 - 4.0 / sigma0
  k_max = k0 + 4.0 / sigma0
  N_k = 250
  k_grid = np.linspace(k_min, k_max, N_k)
  dk = k_grid[1] - k_grid[0]
  k_grid_fft = fftshift(fftfreq(N_x, dx)) * 2 * np.pi

  # Compute states
  bound_states = solve_bound_states(a, V0, m, hbar, x_grid, dx)
  print(f'Found {len(bound_states)} bound states.')
  for b in bound_states:
    print(f"  n = {b['n']} ({b['parity']}): E{b['n']} = {b['E']:.4f}")

  phi_k = (
      (2 * sigma0**2 / np.pi) ** 0.25
      * np.exp(-sigma0**2 * (k_grid - k0) ** 2)
      * np.exp(-1j * k_grid * x0)
  )
  psi_k_matrix = np.array(
      [get_psi_k(k, x_grid, a, V0, m, hbar) for k in k_grid]
  )
  E_k = (hbar**2 * k_grid**2) / (2 * m)

  # =======================
  # 2. Figure Setup
  # =======================
  # Window 1: Bound States
  fig_bound, (ax_bound_x, ax_bound_k) = plt.subplots(2, 1, figsize=(9, 7))
  fig_bound.canvas.manager.set_window_title('Bound States (-V_0 < E < 0)')

  ax_bound_x.set_title(r'$|\Psi(x,t)|^2$ and $V(x)$')
  ax_bound_x.set_xlabel('Position x')
  ax_bound_x.set_ylabel('Probability Density')
  ax_bound_x.set_xlim(-10, 10)
  ax_bound_x.set_ylim(-(V0 + 1), 1)
  ax_bound_x.grid(True, alpha=0.3)
  ax_bound_x.plot(x_grid, V_potential, 'r-', alpha=0.4, label='V(x)')

  ax_bound_k.set_title(r'$|\Phi(k,t)|^2$')
  ax_bound_k.set_xlabel('Wavenumber k')
  ax_bound_k.set_ylabel('Probability Density')
  ax_bound_k.set_xlim(-4, 4)
  ax_bound_k.set_ylim(-0.1, 1.5)
  ax_bound_k.grid(True, alpha=0.3)

  (line_bound_x,) = ax_bound_x.plot([], [], 'b-', label=r'$|\Psi(x,t)|^2$')
  (line_bound_k,) = ax_bound_k.plot([], [], 'm-', label=r'$|\Phi(k,t)|^2$')
  ax_bound_x.legend(loc='upper right')
  ax_bound_k.legend(loc='upper right')
  fig_bound.tight_layout()

  # Window 2: Scattering States
  fig_scatt, (ax_scatt_x, ax_scatt_k) = plt.subplots(2, 1, figsize=(9, 7))
  fig_scatt.canvas.manager.set_window_title('Scattering States (E > 0)')

  ax_scatt_x.set_title(r'$|\Psi(x,t)|^2$ and $V(x)$')
  ax_scatt_x.set_xlabel('Position x')
  ax_scatt_x.set_ylabel('Probability Density')
  ax_scatt_x.set_xlim(-25, 25)
  ax_scatt_x.set_ylim(-(V0 + 1), 1)
  ax_scatt_x.grid(True, alpha=0.3)
  ax_scatt_x.plot(x_grid, V_potential, 'r-', alpha=0.4, label='V(x)')

  ax_scatt_k.set_title(r'$|\Phi(k,t)|^2$')
  ax_scatt_k.set_xlabel('Wavenumber k')
  ax_scatt_k.set_ylabel('Probability Density')
  ax_scatt_k.set_xlim(-15, 15)
  ax_scatt_k.set_ylim(-0.1, 1.5)
  ax_scatt_k.grid(True, alpha=0.3)

  (line_scatt_x,) = ax_scatt_x.plot([], [], 'b-', label=r'$|\Psi(x,t)|^2$')
  (line_scatt_k,) = ax_scatt_k.plot([], [], 'm-', label=r'$|\Phi(k,t)|^2$')
  ax_scatt_x.legend(loc='upper right')
  ax_scatt_k.legend(loc='upper right')
  fig_scatt.tight_layout()

  # Overlays
  text_bound = ax_bound_x.text(
      0.02,
      0.88,
      '',
      transform=ax_bound_x.transAxes,
      fontsize=10,
      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
  )
  text_scatt = ax_scatt_x.text(
      0.02,
      0.85,
      '',
      transform=ax_scatt_x.transAxes,
      fontsize=10,
      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
  )

  # Bound states selected
  psi0_b = bound_states[0]['psi']
  psi1_b = bound_states[1]['psi']
  E0_b = bound_states[0]['E']
  E1_b = bound_states[1]['E']

  regionR = x_grid < -a
  regionT = x_grid > a

  # Region masks
  regionR = x_grid < -a
  regionT = x_grid > a
  region_outside = np.abs(x_grid) > a  # <--- ADD THIS LINE

  t_max = 5.0
  num_frames = 100
  t_array = np.linspace(0, t_max, num_frames)

  print(
      'Running continuous animation loop... Close figures or press Ctrl+C to'
      ' exit.'
  )

  try:
    while True:
      for t in t_array:
        # Bound State Evolution
        Psi_bound = (1 / np.sqrt(2)) * psi0_b * np.exp(
            -1j * E0_b * t / hbar
        ) + (1 / np.sqrt(2)) * psi1_b * np.exp(-1j * E1_b * t / hbar)
        Phi_bound = (
            np.fft.fftshift(np.fft.fft(Psi_bound))
            * dx
            / np.sqrt(2 * np.pi * hbar)
        )

        # <--- ADD THESE CALCULATIONS --->
        P_outside = np.sum(np.abs(Psi_bound[region_outside]) ** 2) * dx
        P_inside = 1.0 - P_outside

        line_bound_x.set_data(x_grid, np.abs(Psi_bound) ** 2 * 4 + E0_b)
        line_bound_k.set_data(k_grid_fft, np.abs(Phi_bound) ** 2)
        text_bound.set_text(f't = {t:.2f} s')

        # Scattering State Evolution
        time_phases = np.exp(-1j * E_k * t / hbar)
        Psi_scatt = (
            np.dot(phi_k * time_phases, psi_k_matrix) * dk / np.sqrt(2 * np.pi)
        )
        Phi_scatt = (
            np.fft.fftshift(np.fft.fft(Psi_scatt))
            * dx
            / np.sqrt(2 * np.pi * hbar)
        )

        R_t = np.sum(np.abs(Psi_scatt[regionR]) ** 2) * dx
        T_t = np.sum(np.abs(Psi_scatt[regionT]) ** 2) * dx

        line_scatt_x.set_data(x_grid, np.abs(Psi_scatt) ** 2 * 4 - V0)
        line_scatt_k.set_data(k_grid_fft, np.abs(Phi_scatt) ** 2)
        text_bound.set_text(
            f"t = {t:.2f} s\n"
            f"Inside Well  (|x| <= {a}): {P_inside*100:.1f}%\n"
            f"Tunneling Out (|x| > {a}) : {P_outside*100:.1f}%"
        )
        text_scatt.set_text(
            f't = {t:.2f} s\nReflected  R(t): {R_t:.3f}\nTransmitted T(t):'
            f' {T_t:.3f}'
        )

        # Redraw
        fig_bound.canvas.draw_idle()
        fig_scatt.canvas.draw_idle()
        fig_bound.canvas.flush_events()
        fig_scatt.canvas.flush_events()
        plt.pause(0.001)

  except (KeyboardInterrupt, SystemExit):
    print('\nSimulation stopped.')


if __name__ == '__main__':
  main()