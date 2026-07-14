# Online Digital Twin MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an observation-only online twin on remote ROS that mirrors FR3 + gripper into MuJoCo, tracks one AprilTag object, and shows grasp-consistency distance/near-contact feedback.

**Architecture:** Pure-Python core libraries (types, transforms, YAML, metrics, AprilTag backend, MuJoCo driver) tested with pytest offline; thin ROS2 nodes wrap the core and run on `yao@10.229.20.125`. `PoseBackend` is an interface so AprilTag can later be swapped for CAD pose without changing `TwinState`.

**Tech Stack:** Python 3.10+, numpy, PyYAML, OpenCV (`cv2.aruco` AprilTag dict), MuJoCo 3.x + passive viewer, ROS2 Humble (`rclpy`, `sensor_msgs`, `geometry_msgs`), pytest

---

## File map

| Path | Responsibility |
|------|----------------|
| `pyproject.toml` | Root pytest / shared deps for local unit tests |
| `src/twin_types/__init__.py` | Public re-exports |
| `src/twin_types/poses.py` | `Pose3D`, `ObjectPose`, `TwinState`, `GraspMetrics` dataclasses |
| `src/twin_types/transforms.py` | SE3 helpers: compose, invert, apply point |
| `src/twin_types/object_library.py` | Load object YAML |
| `src/twin_types/grasp.py` | Finger–object distance + near-contact flag |
| `src/pose_backend/__init__.py` | Re-exports |
| `src/pose_backend/base.py` | `PoseBackend` Protocol / ABC |
| `src/pose_backend/apriltag_backend.py` | Image + intrinsics + extrinsics → `ObjectPose` |
| `src/twin_core/aggregator.py` | Fuse arm + object poses → `TwinState` |
| `src/twin_mujoco/scene_fr3_object.xml` | Table + free object; includes FR3 via path placeholder |
| `src/twin_mujoco/driver.py` | Apply `TwinState` to `MjModel`/`MjData` |
| `src/twin_mujoco/viewer_app.py` | Passive viewer + overlay text |
| `src/twin_ros/extrinsics.py` | Load camera extrinsics YAML; fail loud if missing |
| `src/twin_ros/pose_node.py` | Subscribe image/`camera_info` → publish object pose |
| `src/twin_ros/twin_node.py` | Subscribe joints + object pose → `TwinState` JSON topic + MuJoCo |
| `configs/objects/cube_a.yaml` | MVP single object |
| `configs/camera_extrinsics.yaml.example` | Required keys documented |
| `configs/topics.yaml` | Default ROS topic names |
| `launch/twin_mvp.launch.py` | Bringup |
| `tests/...` | Unit tests mirroring each pure module |
| `scripts/offline_replay.py` | Replay recorded twin frames without robot |
| `docs/bringup_remote.md` | Deploy + acceptance checklist |

Python packages are plain namespaces under `src/` (editable install via `pyproject.toml`) so Mac CI can run unit tests without ROS. ROS nodes import the same modules on the remote host after `pip install -e .` inside the ROS venv/overlay.

---

### Task 1: Scaffold project + pytest

**Files:**
- Create: `pyproject.toml`
- Create: `src/twin_types/__init__.py`
- Create: `tests/test_smoke.py`
- Modify: `README.md` (dev install one-liner)

- [ ] **Step 1: Add `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "franka-mujoco-digital-twin"
version = "0.1.0"
description = "Online Franka–MuJoCo digital twin (observation only)"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]
vision = ["opencv-python-headless>=4.8"]
mujoco = ["mujoco>=3.1"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Add empty package + smoke test**

```python
# src/twin_types/__init__.py
"""Shared twin datatypes and helpers."""

# tests/test_smoke.py
def test_import_twin_types():
    import twin_types  # noqa: F401
```

- [ ] **Step 3: Install editable and run smoke test**

Run:
```bash
cd /Users/xudazhong/Projects/franka-mujoco-digital-twin
python3 -m pip install -e ".[dev]" -q
python3 -m pytest tests/test_smoke.py -v
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/twin_types/__init__.py tests/test_smoke.py README.md
git commit -m "chore: scaffold Python package and pytest"
```

---

### Task 2: Pose types + SE3 transforms (TDD)

**Files:**
- Create: `src/twin_types/poses.py`
- Create: `src/twin_types/transforms.py`
- Create: `tests/test_transforms.py`
- Modify: `src/twin_types/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transforms.py
import numpy as np
from twin_types.poses import Pose3D
from twin_types.transforms import compose, invert, transform_point


def test_compose_invert_roundtrip():
    a = Pose3D.from_xyz_quat(0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    b = Pose3D.from_xyz_quat(0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0)
    ab = compose(a, b)
    recovered = compose(invert(a), ab)
    np.testing.assert_allclose(recovered.as_matrix(), b.as_matrix(), atol=1e-9)


def test_transform_point():
    pose = Pose3D.from_xyz_quat(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    out = transform_point(pose, np.array([0.5, 0.0, 0.0]))
    np.testing.assert_allclose(out, [1.5, 2.0, 3.0], atol=1e-9)
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python3 -m pytest tests/test_transforms.py -v`  
Expected: FAIL (`ModuleNotFoundError` or import error)

- [ ] **Step 3: Implement poses + transforms**

```python
# src/twin_types/poses.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Pose3D:
    """Rigid transform. Quaternion is (w, x, y, z)."""

    matrix: np.ndarray  # 4x4

    @staticmethod
    def identity() -> Pose3D:
        return Pose3D(np.eye(4))

    @staticmethod
    def from_xyz_quat(
        x: float, y: float, z: float, qw: float, qx: float, qy: float, qz: float
    ) -> Pose3D:
        n = np.array([qw, qx, qy, qz], dtype=float)
        n = n / np.linalg.norm(n)
        w, xq, yq, zq = n
        r = np.array(
            [
                [1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - zq * w), 2 * (xq * zq + yq * w)],
                [2 * (xq * yq + zq * w), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - xq * w)],
                [2 * (xq * zq - yq * w), 2 * (yq * zq + xq * w), 1 - 2 * (xq * xq + yq * yq)],
            ],
            dtype=float,
        )
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = [x, y, z]
        return Pose3D(m)

    def as_matrix(self) -> np.ndarray:
        return self.matrix.copy()

    def xyz_quat_wxyz(self) -> tuple[float, float, float, float, float, float, float]:
        r = self.matrix[:3, :3]
        t = self.matrix[:3, 3]
        trace = float(np.trace(r))
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (r[2, 1] - r[1, 2]) * s
            y = (r[0, 2] - r[2, 0]) * s
            z = (r[1, 0] - r[0, 1]) * s
        else:
            # stable branch for rare cases
            i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
            if i == 0:
                s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
                w = (r[2, 1] - r[1, 2]) / s
                x = 0.25 * s
                y = (r[0, 1] + r[1, 0]) / s
                z = (r[0, 2] + r[2, 0]) / s
            elif i == 1:
                s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
                w = (r[0, 2] - r[2, 0]) / s
                x = (r[0, 1] + r[1, 0]) / s
                y = 0.25 * s
                z = (r[1, 2] + r[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
                w = (r[1, 0] - r[0, 1]) / s
                x = (r[0, 2] + r[2, 0]) / s
                y = (r[1, 2] + r[2, 1]) / s
                z = 0.25 * s
        return (float(t[0]), float(t[1]), float(t[2]), float(w), float(x), float(y), float(z))


@dataclass(frozen=True)
class ObjectPose:
    object_id: str
    pose_in_world: Pose3D
    stamp_s: float
    confidence: float
    tracking_ok: bool


@dataclass(frozen=True)
class GraspMetrics:
    distance_m: float
    near_contact: bool


@dataclass
class TwinState:
    arm_qpos: list[float]  # len 7
    gripper_width: float
    objects: list[ObjectPose]
    grasp: GraspMetrics | None
    stamp_s: float
    latency_s: float | None = None
```

```python
# src/twin_types/transforms.py
from __future__ import annotations

import numpy as np
from twin_types.poses import Pose3D


def compose(a: Pose3D, b: Pose3D) -> Pose3D:
    """Return a ∘ b (apply b first, then a), i.e. T_a @ T_b."""
    return Pose3D(a.matrix @ b.matrix)


def invert(pose: Pose3D) -> Pose3D:
    r = pose.matrix[:3, :3]
    t = pose.matrix[:3, 3]
    m = np.eye(4)
    m[:3, :3] = r.T
    m[:3, 3] = -r.T @ t
    return Pose3D(m)


def transform_point(pose: Pose3D, point: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=float).reshape(3)
    r = pose.matrix[:3, :3]
    t = pose.matrix[:3, 3]
    return r @ p + t
```

```python
# src/twin_types/__init__.py
from twin_types.poses import GraspMetrics, ObjectPose, Pose3D, TwinState
from twin_types.transforms import compose, invert, transform_point

__all__ = [
    "Pose3D",
    "ObjectPose",
    "GraspMetrics",
    "TwinState",
    "compose",
    "invert",
    "transform_point",
]
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `python3 -m pytest tests/test_transforms.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/twin_types tests/test_transforms.py
git commit -m "feat: add Pose3D and SE3 transform helpers"
```

---

### Task 3: Object library YAML loader (TDD)

**Files:**
- Create: `src/twin_types/object_library.py`
- Create: `configs/objects/cube_a.yaml`
- Create: `tests/test_object_library.py`

- [ ] **Step 1: Write failing test + sample YAML**

```yaml
# configs/objects/cube_a.yaml
object_id: cube_a
mesh: meshes/cube_a.obj
mujoco_body: object_cube_a
tag_family: tag36h11
tag_id: 0
tag_size_m: 0.04
T_object_tag: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # x y z qw qx qy qz
```

```python
# tests/test_object_library.py
from pathlib import Path
from twin_types.object_library import load_object_spec


def test_load_cube_a():
    path = Path(__file__).resolve().parents[1] / "configs/objects/cube_a.yaml"
    spec = load_object_spec(path)
    assert spec.object_id == "cube_a"
    assert spec.tag_id == 0
    assert abs(spec.tag_size_m - 0.04) < 1e-9
    assert spec.mujoco_body == "object_cube_a"
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `python3 -m pytest tests/test_object_library.py -v`  
Expected: FAIL (import error)

- [ ] **Step 3: Implement loader**

```python
# src/twin_types/object_library.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml
from twin_types.poses import Pose3D


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    mesh: str
    mujoco_body: str
    tag_family: str
    tag_id: int
    tag_size_m: float
    T_object_tag: Pose3D


def load_object_spec(path: str | Path) -> ObjectSpec:
    raw = yaml.safe_load(Path(path).read_text())
    required = [
        "object_id",
        "mesh",
        "mujoco_body",
        "tag_family",
        "tag_id",
        "tag_size_m",
        "T_object_tag",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    t = raw["T_object_tag"]
    if len(t) != 7:
        raise ValueError(f"{path}: T_object_tag must be [x,y,z,qw,qx,qy,qz]")
    return ObjectSpec(
        object_id=str(raw["object_id"]),
        mesh=str(raw["mesh"]),
        mujoco_body=str(raw["mujoco_body"]),
        tag_family=str(raw["tag_family"]),
        tag_id=int(raw["tag_id"]),
        tag_size_m=float(raw["tag_size_m"]),
        T_object_tag=Pose3D.from_xyz_quat(*[float(v) for v in t]),
    )
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python3 -m pytest tests/test_object_library.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/twin_types/object_library.py configs/objects/cube_a.yaml tests/test_object_library.py
git commit -m "feat: load known-object library YAML"
```

---

### Task 4: Grasp metrics (TDD)

**Files:**
- Create: `src/twin_types/grasp.py`
- Create: `tests/test_grasp.py`
- Modify: `src/twin_types/__init__.py` (export `compute_grasp_metrics`)

- [ ] **Step 1: Write failing test**

```python
# tests/test_grasp.py
import numpy as np
from twin_types.grasp import compute_grasp_metrics
from twin_types.poses import Pose3D


def test_distance_and_near_contact():
    tcp = Pose3D.from_xyz_quat(0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)
    obj = Pose3D.from_xyz_quat(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    # object half-extent along z is 0.05 → surface at z=0.05, tcp at 0.1 → dist 0.05
    m = compute_grasp_metrics(tcp, obj, object_half_extent_m=0.05, near_contact_m=0.02)
    assert abs(m.distance_m - 0.05) < 1e-9
    assert m.near_contact is False

    m2 = compute_grasp_metrics(
        Pose3D.from_xyz_quat(0.0, 0.0, 0.055, 1, 0, 0, 0),
        obj,
        object_half_extent_m=0.05,
        near_contact_m=0.02,
    )
    assert m2.near_contact is True
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `python3 -m pytest tests/test_grasp.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# src/twin_types/grasp.py
from __future__ import annotations

import numpy as np
from twin_types.poses import GraspMetrics, Pose3D


def compute_grasp_metrics(
    tcp_pose_world: Pose3D,
    object_pose_world: Pose3D,
    object_half_extent_m: float,
    near_contact_m: float = 0.02,
) -> GraspMetrics:
    """Axis-aligned sphere approximation: center distance minus radius.

    MVP uses a single radius = object_half_extent_m. Replace later with
    mesh query inside twin_mujoco if needed.
    """
    tcp = tcp_pose_world.matrix[:3, 3]
    center = object_pose_world.matrix[:3, 3]
    center_dist = float(np.linalg.norm(tcp - center))
    distance = max(0.0, center_dist - float(object_half_extent_m))
    return GraspMetrics(distance_m=distance, near_contact=distance <= near_contact_m)
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python3 -m pytest tests/test_grasp.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/twin_types/grasp.py tests/test_grasp.py src/twin_types/__init__.py
git commit -m "feat: compute grasp distance metrics"
```

---

### Task 5: PoseBackend + AprilTag detector (TDD without live camera)

**Files:**
- Create: `src/pose_backend/__init__.py`
- Create: `src/pose_backend/base.py`
- Create: `src/pose_backend/apriltag_backend.py`
- Create: `tests/test_apriltag_backend.py`
- Create: `tests/fixtures/make_tag_image.py` (generator used by test)

- [ ] **Step 1: Write failing test that synthesizes a tag image**

```python
# tests/test_apriltag_backend.py
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pose_backend.apriltag_backend import AprilTagBackend
from twin_types.object_library import ObjectSpec
from twin_types.poses import Pose3D
from twin_types.transforms import compose


def _synthetic_intrinsics():
    fx = fy = 600.0
    cx = cy = 320.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def test_detect_single_tag_identity_extrinsics(tmp_path):
    # Generate a tag36h11 id0 board centered in image using OpenCV
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(dictionary, 0, 200)
    img = np.full((480, 640), 255, dtype=np.uint8)
    img[140:340, 220:420] = tag

    spec = ObjectSpec(
        object_id="cube_a",
        mesh="meshes/cube_a.obj",
        mujoco_body="object_cube_a",
        tag_family="tag36h11",
        tag_id=0,
        tag_size_m=0.04,
        T_object_tag=Pose3D.identity(),
    )
    backend = AprilTagBackend(
        object_spec=spec,
        T_base_camera=Pose3D.identity(),
        lose_track_timeout_s=0.5,
    )
    poses = backend.estimate(
        image_bgr=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
        camera_matrix=_synthetic_intrinsics(),
        dist_coeffs=np.zeros(5),
        stamp_s=1.0,
    )
    assert len(poses) == 1
    assert poses[0].object_id == "cube_a"
    assert poses[0].tracking_ok is True
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `python3 -m pip install 'opencv-python-headless>=4.8' -q && python3 -m pytest tests/test_apriltag_backend.py -v`  
Expected: FAIL (module missing)

- [ ] **Step 3: Implement backend**

```python
# src/pose_backend/base.py
from __future__ import annotations

from typing import Protocol
import numpy as np
from twin_types.poses import ObjectPose


class PoseBackend(Protocol):
    def estimate(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        stamp_s: float,
    ) -> list[ObjectPose]:
        ...
```

```python
# src/pose_backend/apriltag_backend.py
from __future__ import annotations

import numpy as np
import cv2
from twin_types.object_library import ObjectSpec
from twin_types.poses import ObjectPose, Pose3D
from twin_types.transforms import compose


_FAMILY = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


class AprilTagBackend:
    def __init__(
        self,
        object_spec: ObjectSpec,
        T_base_camera: Pose3D,
        lose_track_timeout_s: float = 0.5,
    ) -> None:
        if object_spec.tag_family not in _FAMILY:
            raise ValueError(f"Unsupported tag_family: {object_spec.tag_family}")
        self.spec = object_spec
        self.T_base_camera = T_base_camera
        self.lose_track_timeout_s = lose_track_timeout_s
        self._dictionary = cv2.aruco.getPredefinedDictionary(_FAMILY[object_spec.tag_family])
        self._params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._params)
        self._last: ObjectPose | None = None
        self._last_ok_stamp: float | None = None

    def estimate(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        stamp_s: float,
    ) -> list[ObjectPose]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        found = None
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                if int(marker_id) != self.spec.tag_id:
                    continue
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    marker_corners,
                    self.spec.tag_size_m,
                    camera_matrix,
                    dist_coeffs,
                )
                rmat, _ = cv2.Rodrigues(rvecs[0][0])
                T_cam_tag = np.eye(4)
                T_cam_tag[:3, :3] = rmat
                T_cam_tag[:3, 3] = tvecs[0][0]
                T_base_tag = compose(self.T_base_camera, Pose3D(T_cam_tag))
                # T_object_tag: points in tag frame → object frame.
                # Therefore T_base_object = T_base_tag ∘ invert(T_object_tag).
                from twin_types.transforms import invert

                T_base_object = compose(T_base_tag, invert(self.spec.T_object_tag))
                found = ObjectPose(
                    object_id=self.spec.object_id,
                    pose_in_world=T_base_object,
                    stamp_s=stamp_s,
                    confidence=1.0,
                    tracking_ok=True,
                )
                break

        if found is not None:
            self._last = found
            self._last_ok_stamp = stamp_s
            return [found]

        if (
            self._last is not None
            and self._last_ok_stamp is not None
            and (stamp_s - self._last_ok_stamp) <= self.lose_track_timeout_s
        ):
            held = ObjectPose(
                object_id=self._last.object_id,
                pose_in_world=self._last.pose_in_world,
                stamp_s=stamp_s,
                confidence=0.0,
                tracking_ok=False,
            )
            return [held]

        if self._last is not None:
            return [
                ObjectPose(
                    object_id=self._last.object_id,
                    pose_in_world=self._last.pose_in_world,
                    stamp_s=stamp_s,
                    confidence=0.0,
                    tracking_ok=False,
                )
            ]
        return []
```

```python
# src/pose_backend/__init__.py
from pose_backend.base import PoseBackend
from pose_backend.apriltag_backend import AprilTagBackend

__all__ = ["PoseBackend", "AprilTagBackend"]
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python3 -m pytest tests/test_apriltag_backend.py -v`  
Expected: PASS  
If OpenCV API differs on the remote host, adjust `ArucoDetector` vs `detectMarkers` to the installed OpenCV 4.x API in a follow-up commit—keep the same public `estimate(...)` signature.

- [ ] **Step 5: Commit**

```bash
git add src/pose_backend tests/test_apriltag_backend.py
git commit -m "feat: add AprilTag PoseBackend with track-hold"
```

---

### Task 6: TwinState aggregator (TDD)

**Files:**
- Create: `src/twin_core/__init__.py`
- Create: `src/twin_core/aggregator.py`
- Create: `tests/test_aggregator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_aggregator.py
from twin_core.aggregator import TwinAggregator
from twin_types.poses import ObjectPose, Pose3D


def test_aggregate_arm_and_object():
    agg = TwinAggregator(object_half_extent_m=0.05, near_contact_m=0.02)
    agg.update_arm([0.1] * 7, gripper_width=0.04, stamp_s=1.0)
    agg.update_objects(
        [
            ObjectPose(
                object_id="cube_a",
                pose_in_world=Pose3D.from_xyz_quat(0.5, 0.0, 0.05, 1, 0, 0, 0),
                stamp_s=1.0,
                confidence=1.0,
                tracking_ok=True,
            )
        ]
    )
    # TCP proxy: place TCP at same as a fake FK — for unit test inject tcp pose
    state = agg.build(
        stamp_s=1.05,
        tcp_pose_world=Pose3D.from_xyz_quat(0.5, 0.0, 0.15, 1, 0, 0, 0),
    )
    assert state.arm_qpos[0] == 0.1
    assert state.objects[0].tracking_ok is True
    assert state.grasp is not None
    assert state.grasp.distance_m > 0
```

- [ ] **Step 2: Run — expect FAIL**

Run: `python3 -m pytest tests/test_aggregator.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement aggregator**

```python
# src/twin_core/aggregator.py
from __future__ import annotations

from twin_types.grasp import compute_grasp_metrics
from twin_types.poses import GraspMetrics, ObjectPose, Pose3D, TwinState


class TwinAggregator:
    def __init__(self, object_half_extent_m: float, near_contact_m: float = 0.02) -> None:
        self.object_half_extent_m = object_half_extent_m
        self.near_contact_m = near_contact_m
        self._arm: list[float] | None = None
        self._gripper: float = 0.0
        self._arm_stamp: float | None = None
        self._objects: list[ObjectPose] = []

    def update_arm(self, qpos7: list[float], gripper_width: float, stamp_s: float) -> None:
        if len(qpos7) != 7:
            raise ValueError("arm qpos must have length 7")
        self._arm = [float(v) for v in qpos7]
        self._gripper = float(gripper_width)
        self._arm_stamp = stamp_s

    def update_objects(self, objects: list[ObjectPose]) -> None:
        self._objects = list(objects)

    def build(self, stamp_s: float, tcp_pose_world: Pose3D | None) -> TwinState:
        if self._arm is None:
            raise RuntimeError("arm state not yet received")
        grasp: GraspMetrics | None = None
        if tcp_pose_world is not None and self._objects:
            grasp = compute_grasp_metrics(
                tcp_pose_world,
                self._objects[0].pose_in_world,
                object_half_extent_m=self.object_half_extent_m,
                near_contact_m=self.near_contact_m,
            )
        latency = None
        if self._arm_stamp is not None:
            latency = max(0.0, stamp_s - self._arm_stamp)
        return TwinState(
            arm_qpos=list(self._arm),
            gripper_width=self._gripper,
            objects=list(self._objects),
            grasp=grasp,
            stamp_s=stamp_s,
            latency_s=latency,
        )
```

```python
# src/twin_core/__init__.py
from twin_core.aggregator import TwinAggregator

__all__ = ["TwinAggregator"]
```

- [ ] **Step 4: Run — expect PASS**

Run: `python3 -m pytest tests/test_aggregator.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/twin_core tests/test_aggregator.py
git commit -m "feat: add TwinState aggregator"
```

---

### Task 7: MuJoCo scene + driver (TDD with minimal XML)

**Files:**
- Create: `src/twin_mujoco/__init__.py`
- Create: `src/twin_mujoco/scene_mvp.xml`
- Create: `meshes/cube_a.obj` (unit cube scaled in XML)
- Create: `src/twin_mujoco/driver.py`
- Create: `tests/test_mujoco_driver.py`

MVP scene strategy: ship a **self-contained** XML with a simplified 7-DOF arm stub + free object for offline tests; on the remote host, `TWIN_FR3_MODEL` env can point to `/home/yao/franka_mujoco_sync/fr3.mujoco.urdf` and `driver.py` merges object mocap into that model via a wrapper XML that `<include>`s it when available.

- [ ] **Step 1: Minimal scene + failing test**

```xml
<!-- src/twin_mujoco/scene_mvp.xml -->
<mujoco model="twin_mvp">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.05" rgba="0.8 0.8 0.8 1"/>
    <body name="tcp_proxy" pos="0 0 0.3">
      <geom name="tcp_sphere" type="sphere" size="0.01" rgba="0 0 1 1"/>
      <site name="tcp_site" pos="0 0 0" size="0.005"/>
    </body>
    <body name="object_cube_a" pos="0.4 0 0.05" mocap="true">
      <geom name="object_cube_a_geom" type="box" size="0.05 0.05 0.05" rgba="0.9 0.4 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
```

```python
# tests/test_mujoco_driver.py
import pytest

mujoco = pytest.importorskip("mujoco")
from pathlib import Path
from twin_mujoco.driver import TwinMujocoDriver
from twin_types.poses import GraspMetrics, ObjectPose, Pose3D, TwinState


def test_apply_object_pose():
    xml = Path(__file__).resolve().parents[1] / "src/twin_mujoco/scene_mvp.xml"
    driver = TwinMujocoDriver(str(xml), object_body="object_cube_a")
    state = TwinState(
        arm_qpos=[0.0] * 7,
        gripper_width=0.04,
        objects=[
            ObjectPose(
                "cube_a",
                Pose3D.from_xyz_quat(0.2, 0.1, 0.05, 1, 0, 0, 0),
                stamp_s=1.0,
                confidence=1.0,
                tracking_ok=True,
            )
        ],
        grasp=GraspMetrics(0.1, False),
        stamp_s=1.0,
    )
    driver.apply(state)
    pos = driver.object_xpos()
    assert abs(pos[0] - 0.2) < 1e-6
    assert abs(pos[1] - 0.1) < 1e-6
```

- [ ] **Step 2: Run — expect FAIL**

Run: `python3 -m pip install 'mujoco>=3.1' -q && python3 -m pytest tests/test_mujoco_driver.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement driver**

```python
# src/twin_mujoco/driver.py
from __future__ import annotations

import numpy as np
import mujoco
from twin_types.poses import TwinState


class TwinMujocoDriver:
    def __init__(self, model_path: str, object_body: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.object_body = object_body
        self._body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if self._body_id < 0:
            raise RuntimeError(f"body not found: {object_body}")
        self._mocap_id = int(self.model.body_mocapid[self._body_id])
        if self._mocap_id < 0:
            raise RuntimeError(f"body {object_body} is not mocap")

        # Optional FR3 joints if present
        self._joint_qpos: dict[str, int] = {}
        for i in range(1, 8):
            name = f"fr3_joint{i}"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self._joint_qpos[name] = self.model.jnt_qposadr[jid]
        self._finger_qpos: list[int] = []
        for name in ("fr3_finger_joint1", "fr3_finger_joint2"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self._finger_qpos.append(self.model.jnt_qposadr[jid])

    def apply(self, state: TwinState) -> None:
        if len(self._joint_qpos) == 7:
            for idx, name in enumerate([f"fr3_joint{i}" for i in range(1, 8)]):
                self.data.qpos[self._joint_qpos[name]] = state.arm_qpos[idx]
        for adr in self._finger_qpos:
            half = max(0.0, min(0.04, state.gripper_width))
            self.data.qpos[adr] = half
        if state.objects:
            T = state.objects[0].pose_in_world.matrix
            self.data.mocap_pos[self._mocap_id] = T[:3, 3]
            # convert rotm to quat wxyz for mocap_quat
            x, y, z, qw, qx, qy, qz = state.objects[0].pose_in_world.xyz_quat_wxyz()
            self.data.mocap_quat[self._mocap_id] = [qw, qx, qy, qz]
        mujoco.mj_forward(self.model, self.data)

    def object_xpos(self) -> np.ndarray:
        return self.data.xpos[self._body_id].copy()
```

```python
# src/twin_mujoco/__init__.py
from twin_mujoco.driver import TwinMujocoDriver

__all__ = ["TwinMujocoDriver"]
```

- [ ] **Step 4: Run — expect PASS**

Run: `python3 -m pytest tests/test_mujoco_driver.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/twin_mujoco tests/test_mujoco_driver.py meshes
git commit -m "feat: MuJoCo driver applies TwinState to mocap object"
```

---

### Task 8: Viewer overlay (grasp consistency)

**Files:**
- Create: `src/twin_mujoco/viewer_app.py`
- Create: `tests/test_overlay_text.py`

- [ ] **Step 1: Unit-test overlay string (no GUI)**

```python
# tests/test_overlay_text.py
from twin_mujoco.viewer_app import format_overlay
from twin_types.poses import GraspMetrics, ObjectPose, Pose3D, TwinState


def test_format_overlay_near_contact():
    state = TwinState(
        arm_qpos=[0]*7,
        gripper_width=0.02,
        objects=[ObjectPose("cube_a", Pose3D.identity(), 1.0, 1.0, True)],
        grasp=GraspMetrics(0.01, True),
        stamp_s=1.0,
        latency_s=0.04,
    )
    text = format_overlay(state)
    assert "NEAR" in text
    assert "ok=True" in text or "tracking_ok=True" in text
```

- [ ] **Step 2: Implement `format_overlay` + viewer loop skeleton**

```python
# src/twin_mujoco/viewer_app.py
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
                # Optional: print overlay to stdout when near-contact toggles
                print(format_overlay(state), flush=True)
            viewer.sync()
            time.sleep(1 / 30)
```

- [ ] **Step 3: Run test — expect PASS**

Run: `python3 -m pytest tests/test_overlay_text.py -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/twin_mujoco/viewer_app.py tests/test_overlay_text.py
git commit -m "feat: grasp-consistency overlay formatting"
```

---

### Task 9: Extrinsics loader (fail loud)

**Files:**
- Create: `src/twin_ros/__init__.py`
- Create: `src/twin_ros/extrinsics.py`
- Create: `configs/camera_extrinsics.yaml.example`
- Create: `tests/test_extrinsics.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_extrinsics.py
from pathlib import Path
import pytest
from twin_ros.extrinsics import load_camera_extrinsics


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError) as ei:
        load_camera_extrinsics(tmp_path / "nope.yaml")
    assert "checklist" in str(ei.value).lower() or "Camera extrinsics" in str(ei.value)


def test_load_example_shape():
    # copy example into tmp and ensure keys work
    example = Path(__file__).resolve().parents[1] / "configs/camera_extrinsics.yaml.example"
    text = example.read_text().replace(".example", "")
    # example uses placeholder values — loader must accept them
    p = Path(__file__).resolve().parents[1] / "configs/camera_extrinsics.yaml.example"
    pose = load_camera_extrinsics(p)
    assert pose.matrix.shape == (4, 4)
```

```yaml
# configs/camera_extrinsics.yaml.example
# Copy to camera_extrinsics.yaml and set T_base_camera: [x,y,z,qw,qx,qy,qz]
frame_parent: fr3_link0
frame_child: cam1_color_optical_frame
T_base_camera: [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
```

- [ ] **Step 2: Implement**

```python
# src/twin_ros/extrinsics.py
from __future__ import annotations

from pathlib import Path
import yaml
from twin_types.poses import Pose3D

_CHECKLIST = (
    "Camera extrinsics missing or invalid. Checklist:\n"
    "1) Copy configs/camera_extrinsics.yaml.example → camera_extrinsics.yaml\n"
    "2) Fill T_base_camera [x,y,z,qw,qx,qy,qz] in robot base frame\n"
    "3) Confirm optical vs ROS frame convention\n"
)


def load_camera_extrinsics(path: str | Path) -> Pose3D:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(_CHECKLIST + f"Expected file: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if "T_base_camera" not in raw:
        raise ValueError(_CHECKLIST + f"Missing T_base_camera in {path}")
    t = raw["T_base_camera"]
    if len(t) != 7:
        raise ValueError(_CHECKLIST + "T_base_camera must have 7 numbers")
    return Pose3D.from_xyz_quat(*[float(v) for v in t])
```

- [ ] **Step 3: pytest PASS + commit**

```bash
python3 -m pytest tests/test_extrinsics.py -v
git add src/twin_ros configs/camera_extrinsics.yaml.example tests/test_extrinsics.py
git commit -m "feat: fail-loud camera extrinsics loader"
```

---

### Task 10: ROS nodes + topic config

**Files:**
- Create: `configs/topics.yaml`
- Create: `src/twin_ros/pose_node.py`
- Create: `src/twin_ros/twin_node.py`
- Create: `src/twin_ros/tcp_fk.py` (simple body site lookup from MuJoCo after apply, or stub)

Topic defaults (from existing teleop stack):

```yaml
# configs/topics.yaml
image: /cam1/cam1/color/image_raw
camera_info: /cam1/cam1/color/camera_info
joint_states: /franka_robot_state_broadcaster/measured_joint_states
gripper_joint_states: /franka_gripper/joint_states
object_poses: /twin/object_poses
```

- [ ] **Step 1: Implement `pose_node.py` (subscribe image → AprilTag → publish)**

Publish each detection as `geometry_msgs/PoseStamped` on `/twin/object_poses` with `frame_id = object_id` and `tracking_ok` encoded by setting `pose.orientation` unchanged and publishing a parallel `std_msgs/Bool` on `/twin/object_tracking_ok` **or** pack into `std_msgs/String` JSON for MVP simplicity:

Prefer **JSON on `std_msgs/String`** topic `/twin/object_poses` for MVP to avoid a custom msg package:

```json
{"object_id":"cube_a","xyz_quat_wxyz":[...],"stamp_s":1.0,"confidence":1.0,"tracking_ok":true}
```

Include full node code that:

1. Loads object YAML + extrinsics (abort on error).
2. Subscribes to image + camera_info.
3. Calls `AprilTagBackend.estimate`.
4. Publishes JSON string.

- [ ] **Step 2: Implement `twin_node.py`**

1. Subscribe joints (reuse name list from existing mirror: `fr3_joint1..7`).
2. Subscribe gripper.
3. Subscribe object JSON.
4. Update `TwinAggregator`.
5. For TCP: after applying arm to MuJoCo FR3 model, read site `tcp_site` or body `fr3_hand` xpos as TCP proxy; for MVP stub scene use `tcp_proxy` body pose from driver.
6. Call `run_viewer` **or** integrate viewer loop in-process like `mujoco_ros_mirror.py` (rclpy spin thread + viewer main thread).

Pattern to copy from `~/Projects/franka_teleop_stable/gello_desk/mujoco_ros_mirror.py`: spin thread + passive viewer.

- [ ] **Step 3: Manual syntax check**

Run: `python3 -m py_compile src/twin_ros/pose_node.py src/twin_ros/twin_node.py`  
Expected: exit 0

- [ ] **Step 4: Commit**

```bash
git add configs/topics.yaml src/twin_ros
git commit -m "feat: ROS pose and twin viewer nodes"
```

---

### Task 11: Launch + remote bringup docs

**Files:**
- Create: `launch/twin_mvp.launch.py`
- Create: `docs/bringup_remote.md`
- Create: `scripts/sync_to_remote.sh`
- Modify: `README.md`

- [ ] **Step 1: Launch file** starts `pose_node` and `twin_node` with CLI args for config paths.

- [ ] **Step 2: Write `docs/bringup_remote.md` with:**

1. rsync to `yao@10.229.20.125:~/franka_mujoco_digital_twin`
2. `pip install -e ".[vision,mujoco]"` under ROS Python
3. Copy extrinsics example → real values
4. Ensure teleop/cameras already running
5. `python3 -m twin_ros.pose_node ...` and `python3 -m twin_ros.twin_node ...`
6. Acceptance checklist from the design spec (4 items)
7. Explicit: do not change robot network; observation only

- [ ] **Step 3: Commit**

```bash
git add launch docs/bringup_remote.md scripts/sync_to_remote.sh README.md
git commit -m "docs: remote bringup and MVP launch"
```

---

### Task 12: Offline replay harness

**Files:**
- Create: `scripts/offline_replay.py`
- Create: `tests/test_offline_replay_roundtrip.py`

- [ ] **Step 1:** Script reads a JSONL of `TwinState`-like dicts and drives `TwinMujocoDriver` without ROS.

- [ ] **Step 2:** Unit test writes 3 fake frames to temp JSONL and asserts driver object x moves.

- [ ] **Step 3: Commit**

```bash
git add scripts/offline_replay.py tests/test_offline_replay_roundtrip.py
git commit -m "feat: offline TwinState JSONL replay"
```

---

### Task 13: Wire FR3 real model path + end-to-end dry run notes

**Files:**
- Create: `src/twin_mujoco/scene_fr3_wrapper.xml.example`
- Modify: `src/twin_mujoco/driver.py` (if needed for URDF joint naming)
- Modify: `docs/bringup_remote.md`

- [ ] **Step 1:** Document `TWIN_MUJOCO_MODEL=/home/yao/franka_mujoco_sync/fr3.mujoco.urdf` and how to inject a mocap object body (wrapper XML or programmatic body — prefer wrapper XML committed as example).

- [ ] **Step 2:** On remote (when available), run unit tests + launch with cameras; paste checklist results into a short `docs/acceptance_log.md` (optional, only after real run).

- [ ] **Step 3: Commit**

```bash
git add src/twin_mujoco/scene_fr3_wrapper.xml.example docs/bringup_remote.md
git commit -m "feat: FR3 model wrapper path for remote MuJoCo"
```

---

## Spec coverage checklist

| Spec requirement | Task(s) |
|------------------|---------|
| Mirror arm + gripper read-only | 7, 10 |
| Known object via AprilTag | 5, 10 |
| Swappable PoseBackend protocol | 5 (`PoseBackend`) |
| Grasp consistency viz | 4, 8 |
| Object library YAML | 3 |
| Extrinsics fail loud | 9 |
| Tag lost hold + tracking_ok | 5 |
| Camera death → arm continues | 10 (object lost path) |
| Remote-only topology + no network edits | 11 |
| Offline replay | 12 |
| Independent repo / feature branch | already on `feature/online-digital-twin-mvp` |

## Self-review notes

- No markerless / multi-object / reverse control tasks (YAGNI per non-goals).
- Types stay consistent: `Pose3D`, `ObjectPose`, `TwinState`, `GraspMetrics` named in Tasks 2–8.
- Transform convention documented in AprilTag backend: `T_base_object = T_base_tag ∘ invert(T_object_tag)`.
- OpenCV ArUco API may need a one-line adjust on Humble’s system OpenCV; public `estimate` API stays fixed.
