# Remote bringup: online digital twin MVP

Deploy and run the observation-only twin on the existing ROS host **`yao@10.229.20.125`**, alongside the GELLO / Franka teleop stack. The twin **reads** camera and joint topics; it does **not** command the robot.

## Constraints

- **Observation only** — `pose_node` and `twin_node` never publish Franka motion commands.
- **Do not change robot network settings** — no Desk API edits to shopFloor / robot network. The FR3 is wired to FCI at `10.229.66.91`; leave networking as-is.
- **Independent directory** — deploy to `~/franka_mujoco_digital_twin` without modifying `franka_teleop_stable` launches.

## Prerequisites (remote host)

1. Existing teleop stack running (GELLO / Franka drivers publishing joint states).
2. RealSense cameras publishing (defaults in `configs/topics.yaml`):
   - `cam1` (D435I): `/cam1/cam1/color/image_raw`, `/cam1/cam1/color/camera_info`
3. ROS 2 Humble (or compatible) Python environment with `rclpy`, `sensor_msgs`, `std_msgs`.
4. Display / GLFW for the MuJoCo passive viewer (`MUJOCO_GL=glfw`).

## 1. Sync code from your dev machine

From the repo root on your laptop:

```bash
./scripts/sync_to_remote.sh
```

This rsyncs to `yao@10.229.20.125:~/franka_mujoco_digital_twin`, excluding `.git`, `.venv`, caches, and **local** `configs/camera_extrinsics.yaml` (calibration stays on the remote).

Override targets if needed:

```bash
REMOTE_USER=yao REMOTE_HOST=10.229.20.125 REMOTE_DIR=~/franka_mujoco_digital_twin ./scripts/sync_to_remote.sh
```

## 2. Install Python package on the remote

SSH to the remote host and use the **same Python** as your ROS overlay (system or teleop venv):

```bash
ssh yao@10.229.20.125
cd ~/franka_mujoco_digital_twin
source /opt/ros/humble/setup.bash          # plus any existing teleop workspace overlay
python3 -m pip install -e ".[vision,mujoco]"
```

`vision` pulls OpenCV for AprilTag; `mujoco` pulls the passive viewer.

## 3. Camera extrinsics (required for pose_node)

`pose_node` **aborts at startup** if extrinsics are missing or invalid.

```bash
cd ~/franka_mujoco_digital_twin
cp configs/camera_extrinsics.yaml.example configs/camera_extrinsics.yaml
# Edit T_base_camera: [x, y, z, qw, qx, qy, qz] in robot base frame (fr3_link0)
```

Confirm `frame_parent` / `frame_child` match your calibration convention (optical vs ROS frame).

## 4. Ensure teleop and cameras are already running

Before starting the twin, verify upstream topics exist:

```bash
ros2 topic list | grep -E 'cam1|franka_robot_state_broadcaster|franka_gripper'
ros2 topic hz /franka_robot_state_broadcaster/measured_joint_states
ros2 topic hz /cam1/cam1/color/image_raw
```

Do **not** stop or reconfigure the teleop launch. The twin only subscribes.

## 5. FR3 MuJoCo model (remote)

The MVP stub `src/twin_mujoco/scene_mvp.xml` mirrors joints into a TCP proxy only. On the ROS host, use the **real FR3 asset** already maintained in `franka_mujoco_sync` (same tree as legacy `mujoco_ros_mirror`).

### Canonical model path

```bash
export TWIN_MUJOCO_MODEL=/home/yao/franka_mujoco_sync/fr3.mujoco.urdf
```

`twin_node` does **not** read `TWIN_MUJOCO_MODEL` automatically — treat it as the canonical path when editing the wrapper or passing `--model`.

### Inject mocap `object_cube_a` (wrapper XML)

The synced FR3 model has `fr3_joint1`…`fr3_joint7` and gripper joints but **no** mocap body for AprilTag-tracked `cube_a`. Add one via a thin wrapper around the robot file:

```bash
cd ~/franka_mujoco_digital_twin
cp src/twin_mujoco/scene_fr3_wrapper.xml.example src/twin_mujoco/scene_fr3_wrapper.xml
# Edit scene_fr3_wrapper.xml: uncomment <include> and set file="$TWIN_MUJOCO_MODEL"
# (or an MJCF export if MuJoCo cannot <include> the URDF directly)
```

See comments in [src/twin_mujoco/scene_fr3_wrapper.xml.example](../src/twin_mujoco/scene_fr3_wrapper.xml.example). The wrapper adds:

- `floor` plane (optional visual ground)
- `object_cube_a` **mocap** body — must match `configs/objects/cube_a.yaml` (`mujoco_body: object_cube_a`) and `twin_node --object-body`

Quick load check (no ROS):

```bash
export TWIN_MUJOCO_MODEL=/home/yao/franka_mujoco_sync/fr3.mujoco.urdf
python -c "import mujoco; m=mujoco.MjModel.from_xml_path('src/twin_mujoco/scene_fr3_wrapper.xml'); print('bodies', m.nbody, 'mocap', m.nmocap)"
```

Expect `nmocap >= 1` and no `body not found: object_cube_a` error.

### Point `twin_node --model` at the wrapper

**Recommended** — wrapper (robot + mocap object):

```bash
python -m twin_ros.twin_node \
  --model src/twin_mujoco/scene_fr3_wrapper.xml \
  --object-body object_cube_a \
  ...
```

**Alternative** — raw FR3 file only if `object_cube_a` was added inside that model:

```bash
python -m twin_ros.twin_node --model "$TWIN_MUJOCO_MODEL" ...
```

Pass the same path to the launcher:

```bash
python launch/twin_mvp.launch.py --model src/twin_mujoco/scene_fr3_wrapper.xml
```

For grasp distance on the real mesh, set `--tcp-name` to a hand site/body in the FR3 model (defaults `tcp_proxy` / `tcp_site` are for `scene_mvp.xml` only).

## 6. Start pose_node and twin_node

### Option A — launcher (recommended)

```bash
cd ~/franka_mujoco_digital_twin
source /opt/ros/humble/setup.bash
./scripts/run_twin_mvp.sh
```

Or:

```bash
python launch/twin_mvp.launch.py
```

### Option B — two terminals (explicit CLI)

**Terminal 1 — object pose (AprilTag):**

```bash
cd ~/franka_mujoco_digital_twin
source /opt/ros/humble/setup.bash
python -m twin_ros.pose_node \
  --object-yaml configs/objects/cube_a.yaml \
  --extrinsics-yaml configs/camera_extrinsics.yaml \
  --topics-yaml configs/topics.yaml
```

`pose_node` flags (from `twin_ros.pose_node`):

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--object-yaml` | yes | — | Object library YAML (`ObjectSpec`) |
| `--extrinsics-yaml` | yes | — | `T_base_camera` YAML; fails loud if missing |
| `--topics-yaml` | no | built-in | Path to `configs/topics.yaml` |
| `--image-topic` | no | from topics | Override image topic |
| `--camera-info-topic` | no | from topics | Override `camera_info` topic |
| `--object-poses-topic` | no | from topics | Override `/twin/object_poses` |
| `--lose-track-timeout-s` | no | `0.5` | AprilTag hold timeout (reserved) |

**Terminal 2 — MuJoCo viewer mirror (FR3 wrapper on remote):**

```bash
cd ~/franka_mujoco_digital_twin
source /opt/ros/humble/setup.bash
export MUJOCO_GL=glfw
export TWIN_MUJOCO_MODEL=/home/yao/franka_mujoco_sync/fr3.mujoco.urdf
python -m twin_ros.twin_node \
  --model src/twin_mujoco/scene_fr3_wrapper.xml \
  --object-yaml configs/objects/cube_a.yaml \
  --object-body object_cube_a \
  --topics-yaml configs/topics.yaml
```

For local/offline tests without the FR3 asset, keep `--model src/twin_mujoco/scene_mvp.xml`.

`twin_node` flags (from `twin_ros.twin_node`):

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--model` | no | `src/twin_mujoco/scene_mvp.xml` | MuJoCo XML path; use `scene_fr3_wrapper.xml` on remote (see §5) |
| `--object-body` | no | `object_cube_a` | Mocap body name for tracked object |
| `--object-yaml` | no | — | Sets half-extent from `tag_size_m / 2` |
| `--object-half-extent-m` | no | `0.05` or from YAML | Grasp distance radius |
| `--topics-yaml` | no | built-in | Path to `configs/topics.yaml` |
| `--near-contact-m` | no | `0.02` | Near-contact threshold (m) |
| `--tcp-name` | no | `tcp_proxy` | MuJoCo site/body for TCP proxy |
| `--rate-hz` | no | `30.0` | Viewer update rate |

The viewer prints overlay lines such as `tracking_ok=True | dist=12.3mm (far) | lat=40ms`.

### Partial bringup

```bash
python launch/twin_mvp.launch.py --pose-only   # AprilTag publisher only
python launch/twin_mvp.launch.py --twin-only   # viewer only (joints + existing object_poses)
```

## 7. End-to-end dry run (first remote session)

Run this once on `yao@10.229.20.125` before marking acceptance items. No `docs/acceptance_log.md` until a real on-robot session is completed.

1. **Sync + install** — §1–§2; confirm `python -c "import twin_ros"` succeeds in the teleop Python.
2. **Extrinsics** — §3; `pose_node` must start without abort (missing YAML fails loud).
3. **Model** — §5; `scene_fr3_wrapper.xml` loads with `nmocap >= 1`; teleop topics still publishing (§4).
4. **Pose only** — `python launch/twin_mvp.launch.py --pose-only`; wave `cube_a` in `cam1` view; `ros2 topic echo /twin/object_poses` shows JSON with `tracking_ok` toggling.
5. **Twin only** — `python launch/twin_mvp.launch.py --twin-only --model src/twin_mujoco/scene_fr3_wrapper.xml`; arm teleop should mirror FR3 joints; mocap cube follows `/twin/object_poses`.
6. **Full stack** — `./scripts/run_twin_mvp.sh` with `--model src/twin_mujoco/scene_fr3_wrapper.xml`; overlay shows `tracking_ok`, distance mm, latency.
7. **Degrade** — stop camera driver briefly; arm mirroring continues, object marked lost (design §Failure).

If `<include>` of `fr3.mujoco.urdf` fails, export MJCF from MuJoCo simulate once, point the wrapper `<include>` at that `.xml`, and re-run step 3.

## 8. Acceptance checklist (MVP)

From [design spec](superpowers/specs/2026-07-14-online-digital-twin-design.md):

- [ ] **Object tracking** — Moving the tagged `cube_a` object updates the MuJoCo mocap body smoothly; tag loss shows `tracking_ok=False` / held pose degradation in the overlay.
- [ ] **Arm mirroring** — Teleop arm motion mirrors in MuJoCo at least as smoothly as the legacy `mujoco_ros_mirror` experience.
- [ ] **Grasp consistency** — Closing the gripper near the object shows continuous fingertip/TCP–object distance (mm) and a **NEAR** highlight when within `--near-contact-m`.
- [ ] **Camera failure** — Stopping the camera image topic (`ros2 topic pub` pause or kill cam driver) leaves arm joint mirroring alive while the object is marked lost.

## 9. Troubleshooting

| Symptom | Check |
|---------|--------|
| `body not found: object_cube_a` | Use `scene_fr3_wrapper.xml` (§5) or add mocap body to FR3 model |
| MuJoCo fails to load wrapper | `TWIN_MUJOCO_MODEL` path exists; try MJCF export if URDF `<include>` fails |
| `pose_node` exits immediately | `configs/camera_extrinsics.yaml` exists and `T_base_camera` has 7 numbers |
| No object in viewer | `ros2 topic echo /twin/object_poses` shows JSON; tag visible to `cam1` |
| No arm motion | `ros2 topic echo /franka_robot_state_broadcaster/measured_joint_states` |
| Viewer won't open | `echo $DISPLAY`, `MUJOCO_GL=glfw`, Mesa/GLFW on headless host |
| `rclpy` import error | Source ROS setup + install package in that same Python |

## Related

- Design: [2026-07-14-online-digital-twin-design.md](superpowers/specs/2026-07-14-online-digital-twin-design.md)
- Launch source: [launch/twin_mvp.launch.py](../launch/twin_mvp.launch.py)
- Sync helper: [scripts/sync_to_remote.sh](../scripts/sync_to_remote.sh)
