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

## Constraints

- Observation only: the twin does not command the real robot.
- Does not modify robot-side network settings.
- Runs on the existing ROS host (`yao@10.229.20.125`) and reuses RealSense / joint topics.
