# Online Franka–MuJoCo Digital Twin Design

**Date:** 2026-07-14  
**Project:** `franka-mujoco-digital-twin`  
**Branch for MVP:** `feature/online-digital-twin-mvp`  
**Status:** Approved; implementation plan at `docs/superpowers/plans/2026-07-14-online-digital-twin-mvp.md`

## Goal

Build an **online digital twin** that:

1. Mirrors Franka FR3 arm + gripper state into MuJoCo (read-only).
2. Tracks a **known** tabletop object from camera into the same MuJoCo scene.
3. Visualizes **grasp consistency** (fingertip/TCP–object distance and near-contact highlight).

MVP uses **AprilTag** for object pose, with a **swappable PoseBackend** so markerless CAD pose estimation can replace it later without changing twin messages.

## Non-goals (MVP)

- Markerless pose estimation (interface only).
- Multi-object scenes.
- Online scene geometry reconstruction (TSDF / heightfield).
- Twin-driven control of the real robot.
- Merging implementation into `franka_teleop_stable` as a monolith.
- Changing robot shopFloor / robot network settings.

## Context

Existing stack (reference only):

- Teleop stable tree: `~/Projects/franka_teleop_stable`
- Remote ROS host: `yao@10.229.20.125`
- Robot FCI host: `10.229.66.91` (wired; do not reconfigure via Desk network APIs)
- Cameras (ROS): `cam1` (D435I), `cam2` (D435) color topics already published
- Prior art to *reuse conceptually*: `gello_desk/mujoco_ros_mirror.py`

## Architecture

```
Remote host 10.229.20.125
─────────────────────────────────────────────────────────
RealSense RGB(+D) ──► PoseBackend (AprilTag v1)
                              │ object_poses
Franka joints/gripper ────────┤
                              ▼
                        TwinState aggregator
                              │ twin_state
                              ▼
              MuJoCo Viewer + grasp-consistency viz
              (observation only; no Franka commands)
```

### Packages

| Package | Responsibility |
|---------|----------------|
| `pose_backend` | Pluggable pose estimators; publish `ObjectPose[]` |
| `twin_core` | Aggregate arm + objects + grasp metrics → `TwinState` |
| `twin_mujoco` | MuJoCo model, viewer, distance / contact overlays |
| `twin_bringup` | Launch files, object library YAML, extrinsics |

## Interfaces

### PoseBackend

**Inputs:** RGB (optional depth), `camera_info`, optional robot TF  
**Outputs:** `ObjectPose[]` with fields:

- `object_id` (string)
- `pose_in_world` (pose in robot / MuJoCo world frame)
- `stamp`
- `confidence`
- `tracking_ok` (bool)

**Implementations:**

- v1: `AprilTagDetector`
- v2+: `CadPoseEstimator` (same message contract)

### TwinState

- `arm_qpos[7]`, `gripper_width`
- `objects[{id, T_world, ok}]`
- `grasp_metrics`: fingertip–object distance, near-contact / overlap flag
- stamps and latency diagnostics

### Object library YAML (per object)

- `object_id`, `mesh`, `mujoco_body`
- `tag_family`, `tag_id`, `tag_size_m`
- `T_object_tag` (object ← tag extrinsic)

## Runtime topology

- All twin processes run on **`yao@10.229.20.125`** alongside existing GELLO/ROS.
- Reuse existing camera and joint topics; do not bind into teleop control path.
- Deploy to an independent directory (e.g. `~/franka_mujoco_digital_twin`) with a colcon/venv overlay on the existing workspace.

## MuJoCo scene (MVP)

- FR3 + Hand + table + **one** known object mesh
- Drive qpos / free body (or mocap) from `TwinState`
- Overlays: `tracking_ok`, distance in mm, end-to-end latency estimate
- Optional bag recording of `TwinState` for offline replay

## Calibration (required for MVP)

Chain:

1. `tag` → `T_object_tag` → `object`
2. `camera` → `T_base_camera` (one-shot fixed extrinsics YAML or hand–eye) → `base`
3. Franka measured joints → MuJoCo under the same base convention

Missing extrinsics must **fail startup** with a checklist (no silent wrong frames).

## Failure / degradation

| Condition | Behavior |
|-----------|----------|
| Tag lost | Keep last pose; UI red / `tracking_ok=false`; optional freeze after timeout |
| Camera stream dead | Arm mirroring continues; object state = lost |
| Extrinsics missing | Abort bringup with clear errors |

## Acceptance (MVP)

1. Moving the tagged object updates MuJoCo smoothly; loss of track is visible.
2. Arm teleop mirroring remains at least as usable as current `mujoco_ros_mirror`.
3. Closing the gripper near the object yields continuous distance readings and near-contact highlight.
4. Killing the camera topic leaves arm updates alive and marks the object lost.

## Testing

- Unit: tag→object transform, distance metrics, YAML load
- Offline: recorded images + joint bags without the real robot
- On-robot: bringup script + checklist on the remote host

## Repository layout

```
franka-mujoco-digital-twin/
  README.md
  docs/superpowers/specs/2026-07-14-online-digital-twin-design.md
  src/
    pose_backend/
    twin_core/
    twin_mujoco/
    twin_bringup/
  configs/          # object library, camera extrinsics
  meshes/           # object meshes used by MuJoCo
```

## Git workflow

- `main`: skeleton + design docs
- `feature/online-digital-twin-mvp`: implementation workstream

## Relationship to `franka_teleop_stable`

- **Read-only dependency** on ROS topics and existing FCI/GELLO launches
- May **port ideas** from `mujoco_ros_mirror.py` into `twin_mujoco`
- Does **not** modify robot networking or replace the stable teleop launch path

## Decisions log

| Decision | Choice |
|----------|--------|
| Twin scope | Robot + known object poses |
| Pose method | Hybrid: AprilTag MVP, swappable backend |
| Runtime host | Remote ROS machine (A) |
| MVP success bar | Single object + grasp-consistency viz (B) |
| Implementation approach | Modular ROS2 packages in new repo (2) |
