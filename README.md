# Franka MuJoCo Digital Twin

Online digital twin for Franka FR3: mirror robot state into MuJoCo and track known objects from camera (AprilTag MVP, swappable pose backend).

## Status

Design approved; MVP implementation on branch `feature/online-digital-twin-mvp`.

See [docs/superpowers/specs/2026-07-14-online-digital-twin-design.md](docs/superpowers/specs/2026-07-14-online-digital-twin-design.md).

## Dev install

```bash
pip install -e ".[dev]"
pytest
```

## Remote bringup

Deploy and run on `yao@10.229.20.125` (observation only, alongside existing teleop):

- [docs/bringup_remote.md](docs/bringup_remote.md) — rsync, install, extrinsics, launch, acceptance checklist
- `./scripts/sync_to_remote.sh` — push repo to the remote host
- `./scripts/run_twin_mvp.sh` or `python launch/twin_mvp.launch.py` — start `pose_node` + `twin_node`

## Constraints

- Observation only: the twin does not command the real robot.
- Does not modify robot-side network settings.
- Runs on the existing ROS host (`yao@10.229.20.125`) and reuses RealSense / joint topics.
