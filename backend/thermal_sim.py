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
        self._validate_params()

    def _validate_params(self):
        p = self.params
        p.belt_speed = float(max(p.belt_speed, 1e-6))
        p.board_thickness = float(max(p.board_thickness, 1e-6))
        p.board_density = float(max(p.board_density, 1e-6))
        p.board_specific_heat = float(max(p.board_specific_heat, 1e-6))
        p.board_thermal_conductivity = float(max(p.board_thermal_conductivity, 1e-6))
        p.h_top = float(max(p.h_top, 0.0))
        p.h_bottom = float(max(p.h_bottom, 0.0))
        p.initial_temp = float(p.initial_temp)
        p.ambient_temp = float(p.ambient_temp)
        p.num_nodes = int(max(p.num_nodes, 3))
        p.dt = float(max(p.dt, 0.01))
        p.total_time = float(max(p.total_time, p.dt))
        p.zone_temps = [float(t) for t in p.zone_temps]
        p.zone_lengths = [float(max(l, 1e-6)) for l in p.zone_lengths]

    def _safe_div(self, a: float, b: float, default: float = 0.0) -> float:
        b = float(b)
        if abs(b) < 1e-12:
            return float(default)
        return float(a) / b

    def _get_ambient_temp(self, t: float) -> float:
        t = float(t)
        pos = self.params.belt_speed * t
        total_length = float(sum(self.params.zone_lengths))

        if pos <= 0.0 or pos >= total_length:
            return float(self.params.ambient_temp)

        cumulative = 0.0
        for i, length in enumerate(self.params.zone_lengths):
            if pos < cumulative + length:
                return float(self.params.zone_temps[i])
            cumulative += length

        return float(self.params.ambient_temp)

    def _build_tridiagonal(self, n: int, r: float, h_top: float, h_bot: float,
                           dx: float, k: float, T_amb: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = int(n)
        r = float(r)
        h_top = float(h_top)
        h_bot = float(h_bot)
        dx = float(max(dx, 1e-12))
        k = float(max(k, 1e-12))
        T_amb = float(T_amb)

        a = np.zeros(n, dtype=float)
        b = np.zeros(n, dtype=float)
        c = np.zeros(n, dtype=float)
        d = np.zeros(n, dtype=float)

        Bi_top = self._safe_div(h_top * dx, k, 0.0)
        Bi_bot = self._safe_div(h_bot * dx, k, 0.0)

        b[0] = 1.0 + 2.0 * r * (1.0 + Bi_top)
        c[0] = -2.0 * r
        a[-1] = -2.0 * r
        b[-1] = 1.0 + 2.0 * r * (1.0 + Bi_bot)

        for i in range(1, n - 1):
            a[i] = -r
            b[i] = 1.0 + 2.0 * r
            c[i] = -r

        d[0] = 2.0 * r * Bi_top * T_amb
        d[-1] = 2.0 * r * Bi_bot * T_amb

        return a, b, c, d

    def _thomas_solve(self, a: np.ndarray, b: np.ndarray, c: np.ndarray,
                      d: np.ndarray, x_old: np.ndarray) -> np.ndarray:
        n = len(b)
        c_prime = np.zeros(n, dtype=float)
        d_prime = np.zeros(n, dtype=float)

        b0 = float(b[0])
        if abs(b0) < 1e-12:
            return x_old.copy()

        c_prime[0] = float(c[0]) / b0
        d_prime[0] = (float(d[0]) + float(x_old[0])) / b0

        for i in range(1, n):
            m = float(b[i]) - float(a[i]) * float(c_prime[i - 1])
            if abs(m) < 1e-12:
                m = 1e-12 if m >= 0 else -1e-12
            c_prime[i] = float(c[i]) / m
            d_prime[i] = (float(d[i]) + float(x_old[i]) - float(a[i]) * float(d_prime[i - 1])) / m

        x = np.zeros(n, dtype=float)
        x[-1] = float(d_prime[-1])
        for i in range(n - 2, -1, -1):
            x[i] = float(d_prime[i]) - float(c_prime[i]) * float(x[i + 1])

        return x

    def simulate(self) -> dict:
        params = self.params
        self._validate_params()

        k = float(params.board_thermal_conductivity)
        rho = float(params.board_density)
        cp = float(params.board_specific_heat)
        alpha = self._safe_div(k, rho * cp, 0.0)

        L = float(params.board_thickness)
        n = int(params.num_nodes)
        dx = self._safe_div(L, n - 1, L)
        dt = float(params.dt)
        total_time = float(params.total_time)

        dx_sq = dx * dx
        r = self._safe_div(alpha * dt, dx_sq, 0.0)

        T = np.ones(n, dtype=float) * float(params.initial_temp)
        num_steps = max(int(total_time / dt), 1)
        num_steps = min(num_steps, 100000)

        time_points = []
        top_temp = []
        mid_temp = []
        bot_temp = []
        amb_temp = []

        for step in range(num_steps):
            t = float(step) * dt
            T_amb = self._get_ambient_temp(t)

            a, b, c_vec, d_vec = self._build_tridiagonal(
                n, r, params.h_top, params.h_bottom, dx, k, T_amb
            )

            T = self._thomas_solve(a, b, c_vec, d_vec, T)

            time_points.append(float(t))
            top_temp.append(float(T[0]))
            mid_temp.append(float(T[n // 2]))
            bot_temp.append(float(T[-1]))
            amb_temp.append(float(T_amb))

        return {
            "time": time_points,
            "top_temp": top_temp,
            "mid_temp": mid_temp,
            "bot_temp": bot_temp,
            "amb_temp": amb_temp,
            "peak_temp": float(max(mid_temp)) if mid_temp else 0.0,
            "time_above_200": float(self._calc_time_above(time_points, mid_temp, 200.0)),
            "ramp_up_rate": float(self._calc_ramp_rate(time_points, mid_temp, 150.0, 200.0)),
            "cooling_rate": float(self._calc_cooling_rate(time_points, mid_temp, 200.0, 150.0)),
        }

    def _calc_time_above(self, times: List[float], temps: List[float], threshold: float) -> float:
        threshold = float(threshold)
        above = 0.0
        for i in range(1, len(times)):
            t_i = float(times[i])
            t_prev = float(times[i - 1])
            temp_i = float(temps[i])
            temp_prev = float(temps[i - 1])

            if temp_i > threshold and temp_prev > threshold:
                above += t_i - t_prev
            elif temp_i > threshold:
                denom = temp_i - temp_prev
                if abs(denom) < 1e-12:
                    continue
                ratio = (temp_i - threshold) / denom
                above += (t_i - t_prev) * ratio
            elif temp_prev > threshold:
                denom = temp_prev - temp_i
                if abs(denom) < 1e-12:
                    continue
                ratio = (temp_prev - threshold) / denom
                above += (t_i - t_prev) * ratio
        return float(above)

    def _calc_ramp_rate(self, times: List[float], temps: List[float],
                        t_low: float, t_high: float) -> float:
        t_low = float(t_low)
        t_high = float(t_high)
        idx_low = None
        idx_high = None
        for i in range(len(temps)):
            if idx_low is None and float(temps[i]) >= t_low:
                idx_low = i
            if float(temps[i]) >= t_high:
                idx_high = i
                break
        if idx_low is None or idx_high is None or idx_low >= idx_high:
            return 0.0
        time_diff = float(times[idx_high]) - float(times[idx_low])
        if abs(time_diff) < 1e-12:
            return 0.0
        return float((t_high - t_low) / (time_diff / 60.0))

    def _calc_cooling_rate(self, times: List[float], temps: List[float],
                           t_high: float, t_low: float) -> float:
        t_high = float(t_high)
        t_low = float(t_low)
        if not temps:
            return 0.0
        temps_float = [float(t) for t in temps]
        peak_idx = int(np.argmax(temps_float))
        idx_high = None
        idx_low = None
        for i in range(peak_idx, len(temps_float)):
            if idx_high is None and temps_float[i] <= t_high:
                idx_high = i
            if temps_float[i] <= t_low:
                idx_low = i
                break
        if idx_high is None or idx_low is None or idx_high >= idx_low:
            return 0.0
        time_diff = float(times[idx_low]) - float(times[idx_high])
        if abs(time_diff) < 1e-12:
            return 0.0
        return float((t_high - t_low) / (time_diff / 60.0))
