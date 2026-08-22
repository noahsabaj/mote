import math
import numpy as np
from typing import Dict, List, Tuple, Any


def simulate_linear_ode_system(
    A_val: float = -0.5,
    freq: float = 2.0,
    T_max: float = 10.0,
    dt_sweep: List[float] = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625],
) -> Dict[str, Any]:
    """
    Simulates continuous ODE: h'(t) = A * h(t) + cos(freq * t)
    Analytical solution:
    h(t) = (h_0 + A/(A^2+omega^2)) * e^(A*t) + (-A*cos(omega*t) + omega*sin(omega*t)) / (A^2 + omega^2)
    """
    def exact_h(t: np.ndarray, h0: float = 1.0) -> np.ndarray:
        denom = A_val**2 + freq**2
        transient = (h0 + (A_val / denom)) * np.exp(A_val * t)
        forced = (-A_val * np.cos(freq * t) + freq * np.sin(freq * t)) / denom
        return transient + forced

    results = {
        "dt": dt_sweep,
        "euler_error": [],
        "exp_euler_error": [],
        "exp_trapezoidal_error": [],
    }

    for dt in dt_sweep:
        t_grid = np.arange(0, T_max + dt, dt)
        u_grid = np.cos(freq * t_grid)
        h_exact = exact_h(t_grid, h0=1.0)
        n_steps = len(t_grid)

        # 1. Forward Euler
        h_fwd = np.zeros(n_steps)
        h_fwd[0] = 1.0
        for k in range(1, n_steps):
            h_fwd[k] = (1.0 + dt * A_val) * h_fwd[k - 1] + dt * u_grid[k - 1]

        # 2. Exponential-Euler (Mamba-1/2)
        h_ee = np.zeros(n_steps)
        h_ee[0] = 1.0
        alpha = math.exp(dt * A_val)
        for k in range(1, n_steps):
            h_ee[k] = alpha * h_ee[k - 1] + dt * u_grid[k]

        # 3. Exponential-Trapezoidal (Mamba-3)
        h_et = np.zeros(n_steps)
        h_et[0] = 1.0
        alpha = math.exp(dt * A_val)
        beta = 0.5 * dt * alpha
        gamma = 0.5 * dt
        for k in range(1, n_steps):
            h_et[k] = alpha * h_et[k - 1] + beta * u_grid[k - 1] + gamma * u_grid[k]

        # Compute Max Absolute Errors
        err_fwd = np.max(np.abs(h_fwd - h_exact))
        err_ee = np.max(np.abs(h_ee - h_exact))
        err_et = np.max(np.abs(h_et - h_exact))

        results["euler_error"].append(float(err_fwd))
        results["exp_euler_error"].append(float(err_ee))
        results["exp_trapezoidal_error"].append(float(err_et))

    return results


def run_ode_discretization_benchmark():
    """Runs the ODE discretization error comparison benchmark."""
    print("=" * 70)
    print("SCIENTIFIC DYNAMICAL SYSTEMS ODE DISCRETIZATION BENCHMARK")
    print("Mamba-3 Exponential-Trapezoidal vs Exponential-Euler vs Forward Euler")
    print("=" * 70)

    dt_sweep = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
    results = simulate_linear_ode_system(A_val=-0.8, freq=3.0, T_max=10.0, dt_sweep=dt_sweep)

    print(f"\n{'Step Size (dt)':<15} | {'Forward Euler':<15} | {'Exp-Euler (M-2)':<15} | {'Exp-Trapezoidal (M-3)':<20}")
    print("-" * 72)
    for i, dt in enumerate(dt_sweep):
        fe = results["euler_error"][i]
        ee = results["exp_euler_error"][i]
        et = results["exp_trapezoidal_error"][i]
        print(f"{dt:<15.5f} | {fe:<15.4e} | {ee:<15.4e} | {et:<20.4e}")

    # Compute empirical convergence order (slope of log(error) vs log(dt))
    log_dt = np.log(dt_sweep)
    slope_fe, _ = np.polyfit(log_dt, np.log(results["euler_error"]), 1)
    slope_ee, _ = np.polyfit(log_dt, np.log(results["exp_euler_error"]), 1)
    slope_et, _ = np.polyfit(log_dt, np.log(results["exp_trapezoidal_error"]), 1)

    print("\n--- Empirical Convergence Order (Theoretical: Euler=1st/2nd order, Trap=2nd/3rd order) ---")
    print(f"Forward Euler Order:         {slope_fe:.2f}")
    print(f"Exponential-Euler Order:     {slope_ee:.2f}")
    print(f"Exponential-Trapezoidal Order: {slope_et:.2f}")
    print(f"\n--> Mamba-3 Exponential-Trapezoidal achieves {results['exp_euler_error'][-1] / results['exp_trapezoidal_error'][-1]:.1f}x lower error at dt={dt_sweep[-1]}!")

    return results


if __name__ == "__main__":
    run_ode_discretization_benchmark()
