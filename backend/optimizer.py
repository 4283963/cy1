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
        self.history = []

    def _objective(self, result: dict) -> float:
        pw = self.process_window
        cost = 0.0

        peak = result["peak_temp"]
        if peak < pw.peak_temp_min:
            cost += ((pw.peak_temp_min - peak) * 5.0) ** 2
        elif peak > pw.peak_temp_max:
            cost += ((peak - pw.peak_temp_max) * 5.0) ** 2
        cost += ((peak - pw.target_peak_temp) * 0.3) ** 2

        t200 = result["time_above_200"]
        if t200 < pw.time_above_200_min:
            cost += ((pw.time_above_200_min - t200) * 0.5) ** 2
        elif t200 > pw.time_above_200_max:
            cost += ((t200 - pw.time_above_200_max) * 0.5) ** 2
        cost += ((t200 - pw.target_time_above_200) * 0.05) ** 2

        ramp = result["ramp_up_rate"]
        if ramp > pw.ramp_up_max:
            cost += ((ramp - pw.ramp_up_max) * 10.0) ** 2
        cost += ((ramp - pw.target_ramp_up) * 0.5) ** 2

        cool = result["cooling_rate"]
        if cool < pw.cooling_rate_min:
            cost += ((pw.cooling_rate_min - cool) * 5.0) ** 2
        elif cool > pw.cooling_rate_max:
            cost += ((cool - pw.cooling_rate_max) * 5.0) ** 2
        cost += ((cool - pw.target_cooling_rate) * 0.3) ** 2

        return cost

    def _is_within_window(self, result: dict) -> bool:
        pw = self.process_window
        return bool(
            pw.peak_temp_min <= result["peak_temp"] <= pw.peak_temp_max and
            pw.time_above_200_min <= result["time_above_200"] <= pw.time_above_200_max and
            result["ramp_up_rate"] <= pw.ramp_up_max and
            pw.cooling_rate_min <= result["cooling_rate"] <= pw.cooling_rate_max
        )

    def _neighbor(self, params: SimParams, step_size: float) -> SimParams:
        new_params = copy.deepcopy(params)
        n_zones = len(new_params.zone_temps)
        idx = np.random.randint(0, n_zones)
        delta = np.random.uniform(-step_size, step_size)
        new_t = new_params.zone_temps[idx] + delta
        new_t = max(self.config.zone_temp_min, min(self.config.zone_temp_max, new_t))
        new_params.zone_temps[idx] = new_t

        if self.config.optimize_belt_speed and np.random.random() < 0.3:
            delta_v = np.random.uniform(-0.001, 0.001)
            new_v = new_params.belt_speed + delta_v
            new_v = max(self.config.belt_speed_min, min(self.config.belt_speed_max, new_v))
            new_params.belt_speed = new_v

        return new_params

    def optimize(self, progress_callback: Optional[Callable] = None) -> dict:
        config = self.config
        current_params = copy.deepcopy(self.base_params)
        sim = ThermalSimulator(current_params)
        current_result = sim.simulate()
        current_cost = self._objective(current_result)

        best_params = copy.deepcopy(current_params)
        best_result = copy.deepcopy(current_result)
        best_cost = current_cost

        T = config.initial_temp
        step_size = config.step_size

        self.history = []
        self.history.append({
            "iteration": 0,
            "cost": current_cost,
            "best_cost": best_cost,
            "temp": T,
            "zone_temps": list(current_params.zone_temps),
            "peak_temp": current_result["peak_temp"],
        })

        for i in range(1, config.num_iterations + 1):
            new_params = self._neighbor(current_params, step_size)
            sim = ThermalSimulator(new_params)
            new_result = sim.simulate()
            new_cost = self._objective(new_result)

            delta_cost = new_cost - current_cost

            if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / T):
                current_params = new_params
                current_result = new_result
                current_cost = new_cost

                if current_cost < best_cost:
                    best_params = copy.deepcopy(current_params)
                    best_result = copy.deepcopy(current_result)
                    best_cost = current_cost

            T *= config.cooling_rate
            step_size *= config.step_decay

            self.history.append({
                "iteration": i,
                "cost": current_cost,
                "best_cost": best_cost,
                "temp": T,
                "zone_temps": list(current_params.zone_temps),
                "peak_temp": current_result["peak_temp"],
            })

            if progress_callback and i % 10 == 0:
                progress_callback(i, config.num_iterations, best_cost)

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
