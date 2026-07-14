from __future__ import annotations

from twin_types.poses import TwinState
from twin_mujoco.driver import TwinMujocoDriver


def format_overlay(state: TwinState) -> str:
    obj = state.objects[0] if state.objects else None
    track = f"tracking_ok={obj.tracking_ok}" if obj else "tracking_ok=n/a"
    if state.grasp is None:
        grasp = "grasp=n/a"
    else:
        mm = state.grasp.distance_m * 1000.0
        flag = "NEAR" if state.grasp.near_contact else "far"
        grasp = f"dist={mm:.1f}mm ({flag})"
    lat = f"lat={state.latency_s*1000:.0f}ms" if state.latency_s is not None else "lat=n/a"
    return f"{track} | {grasp} | {lat}"


def run_viewer(driver: TwinMujocoDriver, state_source) -> None:
    """state_source: callable() -> TwinState | None"""
    import time
    import mujoco.viewer

    with mujoco.viewer.launch_passive(driver.model, driver.data) as viewer:
        while viewer.is_running():
            state = state_source()
            if state is not None:
                driver.apply(state)
                print(format_overlay(state), flush=True)
            viewer.sync()
            time.sleep(1 / 30)
