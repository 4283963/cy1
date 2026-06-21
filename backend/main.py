from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thermal_sim import ThermalSimulator, SimParams
from optimizer import SimulatedAnnealingOptimizer, ProcessWindow, OptimizerConfig
from recipe_manager import save_recipe, list_recipes, get_recipe

app = FastAPI(title="回流焊热传导模拟平台", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


class SimulateRequest(BaseModel):
    zone_temps: Optional[List[float]] = None
    zone_lengths: Optional[List[float]] = None
    belt_speed: Optional[float] = None
    board_thickness: Optional[float] = None
    board_density: Optional[float] = None
    board_specific_heat: Optional[float] = None
    board_thermal_conductivity: Optional[float] = None
    h_top: Optional[float] = None
    h_bottom: Optional[float] = None
    initial_temp: Optional[float] = None
    ambient_temp: Optional[float] = None
    num_nodes: Optional[int] = None
    dt: Optional[float] = None
    total_time: Optional[float] = None


class OptimizeRequest(BaseModel):
    base_zone_temps: Optional[List[float]] = None
    base_belt_speed: Optional[float] = None
    zone_lengths: Optional[List[float]] = None
    process_window: Optional[dict] = None
    optimizer_config: Optional[dict] = None


class SaveRecipeRequest(BaseModel):
    recipe_name: str
    zone_temps: List[float]
    belt_speed: float
    sim_result: dict
    process_window: dict
    notes: Optional[str] = ""


def build_sim_params(req: SimulateRequest) -> SimParams:
    params = SimParams()
    if req.zone_temps is not None:
        params.zone_temps = req.zone_temps
    if req.zone_lengths is not None:
        params.zone_lengths = req.zone_lengths
    if req.belt_speed is not None:
        params.belt_speed = req.belt_speed
    if req.board_thickness is not None:
        params.board_thickness = req.board_thickness
    if req.board_density is not None:
        params.board_density = req.board_density
    if req.board_specific_heat is not None:
        params.board_specific_heat = req.board_specific_heat
    if req.board_thermal_conductivity is not None:
        params.board_thermal_conductivity = req.board_thermal_conductivity
    if req.h_top is not None:
        params.h_top = req.h_top
    if req.h_bottom is not None:
        params.h_bottom = req.h_bottom
    if req.initial_temp is not None:
        params.initial_temp = req.initial_temp
    if req.ambient_temp is not None:
        params.ambient_temp = req.ambient_temp
    if req.num_nodes is not None:
        params.num_nodes = req.num_nodes
    if req.dt is not None:
        params.dt = req.dt
    if req.total_time is not None:
        params.total_time = req.total_time
    return params


@app.get("/")
async def root():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "回流焊热传导模拟平台 API 运行中"}


@app.get("/api/defaults")
async def get_defaults():
    params = SimParams()
    window = ProcessWindow()
    config = OptimizerConfig()
    return {
        "sim_params": {
            "zone_temps": params.zone_temps,
            "zone_lengths": params.zone_lengths,
            "belt_speed": params.belt_speed,
            "board_thickness": params.board_thickness,
            "board_density": params.board_density,
            "board_specific_heat": params.board_specific_heat,
            "board_thermal_conductivity": params.board_thermal_conductivity,
            "h_top": params.h_top,
            "h_bottom": params.h_bottom,
            "initial_temp": params.initial_temp,
            "ambient_temp": params.ambient_temp,
            "num_nodes": params.num_nodes,
            "dt": params.dt,
            "total_time": params.total_time,
        },
        "process_window": {
            "peak_temp_min": window.peak_temp_min,
            "peak_temp_max": window.peak_temp_max,
            "time_above_200_min": window.time_above_200_min,
            "time_above_200_max": window.time_above_200_max,
            "ramp_up_max": window.ramp_up_max,
            "cooling_rate_min": window.cooling_rate_min,
            "cooling_rate_max": window.cooling_rate_max,
            "target_peak_temp": window.target_peak_temp,
            "target_time_above_200": window.target_time_above_200,
            "target_ramp_up": window.target_ramp_up,
            "target_cooling_rate": window.target_cooling_rate,
        },
        "optimizer_config": {
            "initial_temp": config.initial_temp,
            "cooling_rate": config.cooling_rate,
            "num_iterations": config.num_iterations,
            "step_size": config.step_size,
            "step_decay": config.step_decay,
            "zone_temp_min": config.zone_temp_min,
            "zone_temp_max": config.zone_temp_max,
            "belt_speed_min": config.belt_speed_min,
            "belt_speed_max": config.belt_speed_max,
            "optimize_belt_speed": config.optimize_belt_speed,
        },
    }


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    try:
        params = build_sim_params(req)
        sim = ThermalSimulator(params)
        result = sim.simulate()
        return {
            "success": True,
            "params": {
                "zone_temps": params.zone_temps,
                "belt_speed": params.belt_speed,
            },
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize")
async def optimize(req: OptimizeRequest):
    try:
        base_params = SimParams()
        if req.base_zone_temps is not None:
            base_params.zone_temps = req.base_zone_temps
        if req.base_belt_speed is not None:
            base_params.belt_speed = req.base_belt_speed
        if req.zone_lengths is not None:
            base_params.zone_lengths = req.zone_lengths

        window = ProcessWindow()
        if req.process_window:
            pw = req.process_window
            for key, val in pw.items():
                if hasattr(window, key):
                    setattr(window, key, val)

        config = OptimizerConfig()
        if req.optimizer_config:
            oc = req.optimizer_config
            for key, val in oc.items():
                if hasattr(config, key):
                    setattr(config, key, val)

        optimizer = SimulatedAnnealingOptimizer(base_params, window, config)
        result = optimizer.optimize()

        return {
            "success": True,
            "result": result,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recipes/save")
async def save_recipe_api(req: SaveRecipeRequest):
    try:
        if not req.recipe_name or not req.recipe_name.strip():
            raise HTTPException(status_code=400, detail="配方名称不能为空")
        if not req.zone_temps or len(req.zone_temps) == 0:
            raise HTTPException(status_code=400, detail="温区温度参数不能为空")

        result = save_recipe(
            recipe_name=req.recipe_name.strip(),
            zone_temps=req.zone_temps,
            belt_speed=req.belt_speed,
            sim_result=req.sim_result,
            process_window=req.process_window,
            notes=req.notes or ""
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recipes")
async def list_recipes_api():
    try:
        recipes = list_recipes()
        return {
            "success": True,
            "recipes": recipes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recipes/{filename}")
async def get_recipe_api(filename: str):
    try:
        recipe = get_recipe(filename)
        if recipe is None:
            raise HTTPException(status_code=404, detail="配方文件不存在")
        return {
            "success": True,
            "recipe": recipe,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
