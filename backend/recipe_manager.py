import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional


RECIPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes")


def _ensure_dir():
    if not os.path.exists(RECIPE_DIR):
        os.makedirs(RECIPE_DIR, exist_ok=True)


def _format_recipe_content(
    recipe_name: str,
    zone_temps: List[float],
    belt_speed: float,
    sim_result: Dict,
    process_window: Dict,
    notes: str = ""
) -> Dict:
    timestamp = time.time()
    dt_now = datetime.fromtimestamp(timestamp)
    time_str = dt_now.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "recipe_name": recipe_name,
        "created_at": time_str,
        "created_timestamp": float(timestamp),
        "notes": notes,
        "parameters": {
            "zone_temps": [float(t) for t in zone_temps],
            "belt_speed": float(belt_speed),
        },
        "process_window": {
            "peak_temp_min": float(process_window.get("peak_temp_min", 235.0)),
            "peak_temp_max": float(process_window.get("peak_temp_max", 250.0)),
            "time_above_200_min": float(process_window.get("time_above_200_min", 30.0)),
            "time_above_200_max": float(process_window.get("time_above_200_max", 90.0)),
            "ramp_up_max": float(process_window.get("ramp_up_max", 3.0)),
            "cooling_rate_min": float(process_window.get("cooling_rate_min", 1.0)),
            "cooling_rate_max": float(process_window.get("cooling_rate_max", 6.0)),
        },
        "simulation_result": {
            "peak_temp": float(sim_result.get("peak_temp", 0.0)),
            "time_above_200": float(sim_result.get("time_above_200", 0.0)),
            "ramp_up_rate": float(sim_result.get("ramp_up_rate", 0.0)),
            "cooling_rate": float(sim_result.get("cooling_rate", 0.0)),
        },
    }


def _format_recipe_text(recipe: Dict) -> str:
    zone_temps = recipe["parameters"]["zone_temps"]
    belt_speed = recipe["parameters"]["belt_speed"]
    sim = recipe["simulation_result"]
    pw = recipe["process_window"]

    peak_ok = pw["peak_temp_min"] <= sim["peak_temp"] <= pw["peak_temp_max"]
    t200_ok = pw["time_above_200_min"] <= sim["time_above_200"] <= pw["time_above_200_max"]
    ramp_ok = sim["ramp_up_rate"] <= pw["ramp_up_max"]
    cool_ok = pw["cooling_rate_min"] <= sim["cooling_rate"] <= pw["cooling_rate_max"]
    all_ok = peak_ok and t200_ok and ramp_ok and cool_ok

    zone_line1 = "  ".join([f"Z{i+1}:{t:>6.1f}C" for i, t in enumerate(zone_temps[:4])])
    zone_line2 = "  ".join([f"Z{i+5}:{t:>6.1f}C" for i, t in enumerate(zone_temps[4:])])

    lines = []
    lines.append("=" * 78)
    lines.append("  回 流 焊 工 艺 配 方 单")
    lines.append("=" * 78)
    lines.append(f"  配方名称: {recipe['recipe_name']}")
    lines.append(f"  创建时间: {recipe['created_at']}")
    if recipe.get("notes"):
        lines.append(f"  备注: {recipe['notes']}")
    lines.append("-" * 78)
    lines.append("  【温区设定】")
    lines.append(f"  {zone_line1}")
    lines.append(f"  {zone_line2}")
    lines.append(f"  链条速度: {belt_speed:.4f} m/s  ({belt_speed*60:.2f} m/min)")
    lines.append("-" * 78)
    lines.append("  【工艺窗口】")
    lines.append(f"    峰值温度:    {pw['peak_temp_min']:.1f} ~ {pw['peak_temp_max']:.1f} C")
    lines.append(f"    200C以上时间: {pw['time_above_200_min']:.1f} ~ {pw['time_above_200_max']:.1f} s")
    lines.append(f"    升温速率:    <= {pw['ramp_up_max']:.2f} C/min")
    lines.append(f"    降温速率:    {pw['cooling_rate_min']:.2f} ~ {pw['cooling_rate_max']:.2f} C/min")
    lines.append("-" * 78)
    lines.append("  【模拟结果】")
    lines.append(f"    峰值温度:    {sim['peak_temp']:>7.2f} C    {'✓' if peak_ok else '✗'}  要求 {pw['peak_temp_min']:.1f}~{pw['peak_temp_max']:.1f}")
    lines.append(f"    200C以上时间: {sim['time_above_200']:>7.2f} s    {'✓' if t200_ok else '✗'}  要求 {pw['time_above_200_min']:.1f}~{pw['time_above_200_max']:.1f}")
    lines.append(f"    升温速率:    {sim['ramp_up_rate']:>7.2f} C/min  {'✓' if ramp_ok else '✗'}  要求 <= {pw['ramp_up_max']:.2f}")
    lines.append(f"    降温速率:    {sim['cooling_rate']:>7.2f} C/min  {'✓' if cool_ok else '✗'}  要求 {pw['cooling_rate_min']:.2f}~{pw['cooling_rate_max']:.2f}")
    lines.append("-" * 78)
    lines.append(f"  综合评定: {'✅ 满足工艺窗口' if all_ok else '⚠️  未完全满足工艺窗口'}")
    lines.append("=" * 78)
    lines.append("")

    return "\n".join(lines)


def save_recipe(
    recipe_name: str,
    zone_temps: List[float],
    belt_speed: float,
    sim_result: Dict,
    process_window: Dict,
    notes: str = ""
) -> Dict:
    _ensure_dir()

    recipe_data = _format_recipe_content(
        recipe_name, zone_temps, belt_speed, sim_result, process_window, notes
    )

    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in recipe_name)
    timestamp_str = datetime.fromtimestamp(recipe_data["created_timestamp"]).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp_str}_{safe_name}.txt"
    filepath = os.path.join(RECIPE_DIR, filename)

    text_content = _format_recipe_text(recipe_data)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text_content)

    json_filename = f"{timestamp_str}_{safe_name}.json"
    json_filepath = os.path.join(RECIPE_DIR, json_filename)

    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(recipe_data, f, ensure_ascii=False, indent=2)

    return {
        "success": True,
        "filename": filename,
        "filepath": filepath,
        "json_filepath": json_filepath,
        "recipe_preview": text_content,
    }


def list_recipes() -> List[Dict]:
    _ensure_dir()
    recipes = []
    for filename in sorted(os.listdir(RECIPE_DIR), reverse=True):
        if filename.endswith(".json"):
            filepath = os.path.join(RECIPE_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                recipes.append({
                    "filename": filename,
                    "recipe_name": data.get("recipe_name", "未命名"),
                    "created_at": data.get("created_at", ""),
                    "peak_temp": data.get("simulation_result", {}).get("peak_temp", 0.0),
                    "belt_speed": data.get("parameters", {}).get("belt_speed", 0.0),
                })
            except Exception:
                continue
    return recipes


def get_recipe(filename: str) -> Optional[Dict]:
    _ensure_dir()
    filepath = os.path.join(RECIPE_DIR, filename)
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
