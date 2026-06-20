import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SimParams:
    zone_temps: List[float] = field(default_factory=lambda: [100.0, 140.0, 170.0, 210.0, 245.0, 250.0, 200.0, 150.0])
    zone_lengths: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    belt_speed: float = 0.015
    board_thickness: float = 0.0016
    board_density: float = 1900.0
    board_specific_heat: float = 1100.0
    board_thermal_conductivity: float = 0.35
    h_top: float = 25.0
    h_bottom: float = 15.0
    initial_temp: float = 25.0
    ambient_temp: float = 25.0
    num_nodes: int = 12
    dt: float = 0.5
    total_time: float = 300.0


class ThermalSimulator:
    def __init__(self, params: SimParams = None):
        self.params = params or SimParams()

    def _get_ambient_temp(self, t: float) -> float:
        pos = self.params.belt_speed * t
        total_length = sum(self.params.zone_lengths)

        if pos <= 0 or pos >= total_length:
            return self.params.ambient_temp

        cumulative = 0.0
        for i, length in enumerate(self.params.zone_lengths):
            if pos < cumulative + length:
                return self.params.zone_temps[i]
            cumulative += length

        return self.params.ambient_temp

    def _build_tridiagonal(self, n: int, r: float, h_top: float, h_bot: float,
                           dx: float, k: float, T_amb: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.zeros(n)
        b = np.zeros(n)
        c = np.zeros(n)

        Bi_top = h_top * dx / k
        Bi_bot = h_bot * dx / k

        b[0] = 1 + 2 * r * (1 + Bi_top)
        c[0] = -2 * r
        a[-1] = -2 * r
        b[-1] = 1 + 2 * r * (1 + Bi_bot)

        for i in range(1, n - 1):
            a[i] = -r
            b[i] = 1 + 2 * r
            c[i] = -r

        d = np.zeros(n)
        d[0] = 2 * r * Bi_top * T_amb
        d[-1] = 2 * r * Bi_bot * T_amb

        return a, b, c, d

    def _thomas_solve(self, a: np.ndarray, b: np.ndarray, c: np.ndarray,
                      d: np.ndarray, x_old: np.ndarray) -> np.ndarray:
        n = len(b)
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)

        c_prime[0] = c[0] / b[0]
        d_prime[0] = (d[0] + x_old[0]) / b[0]

        for i in range(1, n):
            m = b[i] - a[i] * c_prime[i - 1]
            c_prime[i] = c[i] / m
            d_prime[i] = (d[i] + x_old[i] - a[i] * d_prime[i - 1]) / m

        x = np.zeros(n)
        x[-1] = d_prime[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    def simulate(self) -> dict:
        params = self.params
        k = params.board_thermal_conductivity
        rho = params.board_density
        cp = params.board_specific_heat
        alpha = k / (rho * cp)

        L = params.board_thickness
        n = params.num_nodes
        dx = L / (n - 1)
        dt = params.dt
        total_time = params.total_time

        r = alpha * dt / (dx ** 2)

        T = np.ones(n) * params.initial_temp
        num_steps = int(total_time / dt)

        time_points = []
        top_temp = []
        mid_temp = []
        bot_temp = []
        amb_temp = []

        for step in range(num_steps):
            t = step * dt
            T_amb = self._get_ambient_temp(t)

            a, b, c, d = self._build_tridiagonal(n, r, params.h_top, params.h_bottom,
                                                 dx, k, T_amb)

            T = self._thomas_solve(a, b, c, d, T)

            time_points.append(t)
            top_temp.append(T[0])
            mid_temp.append(T[n // 2])
            bot_temp.append(T[-1])
            amb_temp.append(T_amb)

        return {
            "time": [float(t) for t in time_points],
            "top_temp": [float(t) for t in top_temp],
            "mid_temp": [float(t) for t in mid_temp],
            "bot_temp": [float(t) for t in bot_temp],
            "amb_temp": [float(t) for t in amb_temp],
            "peak_temp": float(max(mid_temp)),
            "time_above_200": float(self._calc_time_above(time_points, mid_temp, 200.0)),
            "ramp_up_rate": float(self._calc_ramp_rate(time_points, mid_temp, 150.0, 200.0)),
            "cooling_rate": float(self._calc_cooling_rate(time_points, mid_temp, 200.0, 150.0)),
        }

    def _calc_time_above(self, times: List[float], temps: List[float], threshold: float) -> float:
        above = 0.0
        for i in range(1, len(times)):
            if temps[i] > threshold and temps[i - 1] > threshold:
                above += times[i] - times[i - 1]
            elif temps[i] > threshold:
                ratio = (temps[i] - threshold) / (temps[i] - temps[i - 1])
                above += (times[i] - times[i - 1]) * ratio
            elif temps[i - 1] > threshold:
                ratio = (temps[i - 1] - threshold) / (temps[i - 1] - temps[i])
                above += (times[i] - times[i - 1]) * ratio
        return above

    def _calc_ramp_rate(self, times: List[float], temps: List[float],
                        t_low: float, t_high: float) -> float:
        idx_low = None
        idx_high = None
        for i in range(len(temps)):
            if idx_low is None and temps[i] >= t_low:
                idx_low = i
            if temps[i] >= t_high:
                idx_high = i
                break
        if idx_low is None or idx_high is None or idx_low >= idx_high:
            return 0.0
        return (t_high - t_low) / ((times[idx_high] - times[idx_low]) / 60.0)

    def _calc_cooling_rate(self, times: List[float], temps: List[float],
                           t_high: float, t_low: float) -> float:
        peak_idx = np.argmax(temps)
        idx_high = None
        idx_low = None
        for i in range(peak_idx, len(temps)):
            if idx_high is None and temps[i] <= t_high:
                idx_high = i
            if temps[i] <= t_low:
                idx_low = i
                break
        if idx_high is None or idx_low is None or idx_high >= idx_low:
            return 0.0
        return (t_high - t_low) / ((times[idx_low] - times[idx_high]) / 60.0)
