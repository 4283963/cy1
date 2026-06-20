import numpy as np
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from thermal_sim import ThermalSimulator, SimParams


@dataclass
class ProcessWindow:
    peak_temp_min: float = 235.0
    peak_temp_max: float = 250.0
    time_above_200_min: float = 30.0
    time_above_200_max: float = 90.0
    ramp_up_max: float = 3.0
    cooling_rate_min: float = 1.0
    cooling_rate_max: float = 6.0
    target_peak_temp: float = 245.0
    target_time_above_200: float = 60.0
    target_ramp_up: float = 2.0
    target_cooling_rate: float = 3.0


@dataclass
class OptimizerConfig:
    initial_temp: float = 100.0
    cooling_rate: float = 0.95
    num_iterations: int = 200
    step_size: float = 8.0
    step_decay: float = 0.98
    zone_temp_min: float = 50.0
    zone_temp_max: float = 300.0
    belt_speed_min: float = 0.008
    belt_speed_max: float = 0.025
    optimize_belt_speed: bool = False


class SimulatedAnnealingOptimizer:
    def __init__(self, base_params: SimParams,
                 process_window: ProcessWindow = None,
                 config: OptimizerConfig = None):
        self.base_params = base_params
        self.process_window = process_window or ProcessWindow()
        self.config = config or OptimizerConfig()
        self._validate_config()
        self.history = []

    def _validate_config(self):
        cfg = self.config
        cfg.initial_temp = float(max(cfg.initial_temp, 1e-6))
        cfg.cooling_rate = float(max(min(cfg.cooling_rate, 0.999), 1e-6))
        cfg.num_iterations = int(max(cfg.num_iterations, 1))
        cfg.step_size = float(max(cfg.step_size, 1e-6))
        cfg.step_decay = float(max(min(cfg.step_decay, 1.0), 1e-6))
        cfg.zone_temp_min = float(cfg.zone_temp_min)
        cfg.zone_temp_max = float(cfg.zone_temp_max)
        cfg.belt_speed_min = float(max(cfg.belt_speed_min, 1e-6))
        cfg.belt_speed_max = float(max(cfg.belt_speed_max, cfg.belt_speed_min))
        cfg.optimize_belt_speed = bool(cfg.optimize_belt_speed)

        pw = self.process_window
        pw.peak_temp_min = float(pw.peak_temp_min)
        pw.peak_temp_max = float(pw.peak_temp_max)
        pw.time_above_200_min = float(pw.time_above_200_min)
        pw.time_above_200_max = float(pw.time_above_200_max)
        pw.ramp_up_max = float(pw.ramp_up_max)
        pw.cooling_rate_min = float(pw.cooling_rate_min)
        pw.cooling_rate_max = float(pw.cooling_rate_max)
        pw.target_peak_temp = float(pw.target_peak_temp)
        pw.target_time_above_200 = float(pw.target_time_above_200)
        pw.target_ramp_up = float(pw.target_ramp_up)
        pw.target_cooling_rate = float(pw.target_cooling_rate)

    def _objective(self, result: dict) -> float:
        pw = self.process_window
        cost = 0.0

        peak = float(result["peak_temp"])
        if peak < pw.peak_temp_min:
            cost += ((pw.peak_temp_min - peak) * 5.0) ** 2
        elif peak > pw.peak_temp_max:
            cost += ((peak - pw.peak_temp_max) * 5.0) ** 2
        cost += ((peak - pw.target_peak_temp) * 0.3) ** 2

        t200 = float(result["time_above_200"])
        if t200 < pw.time_above_200_min:
            cost += ((pw.time_above_200_min - t200) * 0.5) ** 2
        elif t200 > pw.time_above_200_max:
            cost += ((t200 - pw.time_above_200_max) * 0.5) ** 2
        cost += ((t200 - pw.target_time_above_200) * 0.05) ** 2

        ramp = float(result["ramp_up_rate"])
        if ramp > pw.ramp_up_max:
            cost += ((ramp - pw.ramp_up_max) * 10.0) ** 2
        cost += ((ramp - pw.target_ramp_up) * 0.5) ** 2

        cool = float(result["cooling_rate"])
        if cool < pw.cooling_rate_min:
            cost += ((pw.cooling_rate_min - cool) * 5.0) ** 2
        elif cool > pw.cooling_rate_max:
            cost += ((cool - pw.cooling_rate_max) * 5.0) ** 2
        cost += ((cool - pw.target_cooling_rate) * 0.3) ** 2

        return float(cost)

    def _is_within_window(self, result: dict) -> bool:
        pw = self.process_window
        peak = float(result["peak_temp"])
        t200 = float(result["time_above_200"])
        ramp = float(result["ramp_up_rate"])
        cool = float(result["cooling_rate"])
        return bool(
            pw.peak_temp_min <= peak <= pw.peak_temp_max and
            pw.time_above_200_min <= t200 <= pw.time_above_200_max and
            ramp <= pw.ramp_up_max and
            pw.cooling_rate_min <= cool <= pw.cooling_rate_max
        )

    def _neighbor(self, params: SimParams, step_size: float) -> SimParams:
        step_size = float(step_size)
        new_params = copy.deepcopy(params)
        n_zones = len(new_params.zone_temps)
        idx = int(np.random.randint(0, n_zones))
        delta = float(np.random.uniform(-step_size, step_size))
        new_t = float(new_params.zone_temps[idx]) + delta
        new_t = max(self.config.zone_temp_min, min(self.config.zone_temp_max, new_t))
        new_params.zone_temps[idx] = float(new_t)

        if self.config.optimize_belt_speed and float(np.random.random()) < 0.3:
            delta_v = float(np.random.uniform(-0.001, 0.001))
            new_v = float(new_params.belt_speed) + delta_v
            new_v = max(self.config.belt_speed_min, min(self.config.belt_speed_max, new_v))
            new_params.belt_speed = float(new_v)

        return new_params

    def _metropolis_accept(self, delta_cost: float, T: float) -> bool:
        delta_cost = float(delta_cost)
        T = float(max(T, 1e-12))
        if delta_cost < 0:
            return True
        exponent = -delta_cost / T
        exponent = max(exponent, -500.0)
        exponent = min(exponent, 500.0)
        return float(np.random.random()) < float(np.exp(exponent))

    def optimize(self, progress_callback: Optional[Callable] = None) -> dict:
        self._validate_config()
        config = self.config
        current_params = copy.deepcopy(self.base_params)
        sim = ThermalSimulator(current_params)
        current_result = sim.simulate()
        current_cost = float(self._objective(current_result))

        best_params = copy.deepcopy(current_params)
        best_result = copy.deepcopy(current_result)
        best_cost = float(current_cost)

        T = float(config.initial_temp)
        step_size = float(config.step_size)

        self.history = []
        self.history.append({
            "iteration": 0,
            "cost": float(current_cost),
            "best_cost": float(best_cost),
            "temp": float(T),
            "zone_temps": [float(t) for t in current_params.zone_temps],
            "peak_temp": float(current_result["peak_temp"]),
        })

        num_iters = int(config.num_iterations)
        for i in range(1, num_iters + 1):
            new_params = self._neighbor(current_params, step_size)
            sim = ThermalSimulator(new_params)
            new_result = sim.simulate()
            new_cost = float(self._objective(new_result))

            delta_cost = new_cost - current_cost

            if self._metropolis_accept(delta_cost, T):
                current_params = new_params
                current_result = new_result
                current_cost = float(new_cost)

                if current_cost < best_cost:
                    best_params = copy.deepcopy(current_params)
                    best_result = copy.deepcopy(current_result)
                    best_cost = float(current_cost)

            T *= float(config.cooling_rate)
            T = float(max(T, 1e-12))
            step_size *= float(config.step_decay)
            step_size = float(max(step_size, 1e-6))

            self.history.append({
                "iteration": int(i),
                "cost": float(current_cost),
                "best_cost": float(best_cost),
                "temp": float(T),
                "zone_temps": [float(t) for t in current_params.zone_temps],
                "peak_temp": float(current_result["peak_temp"]),
            })

            if progress_callback and i % 10 == 0:
                progress_callback(i, num_iters, best_cost)

        return {
            "best_params": {
                "zone_temps": [float(t) for t in best_params.zone_temps],
                "belt_speed": float(best_params.belt_speed),
            },
            "best_result": best_result,
            "best_cost": float(best_cost),
            "within_window": bool(self._is_within_window(best_result)),
            "history": [
                {
                    "iteration": int(h["iteration"]),
                    "cost": float(h["cost"]),
                    "best_cost": float(h["best_cost"]),
                    "temp": float(h["temp"]),
                    "peak_temp": float(h["peak_temp"]),
                }
                for h in self.history
            ],
        }
