#!/usr/bin/env python3
"""CLI: generate scene_fr3_wrapper.xml from $TWIN_MUJOCO_MODEL."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from twin_mujoco.prepare_fr3_scene import write_fr3_wrapper

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO / "src" / "twin_mujoco" / "scene_fr3_wrapper.xml"
_DEFAULT_FR3 = "/home/yao/franka_mujoco_sync/fr3.mujoco.urdf"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write MuJoCo FR3+object wrapper XML")
    p.add_argument(
        "--fr3-model",
        default=os.environ.get("TWIN_MUJOCO_MODEL", _DEFAULT_FR3),
    )
    p.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--require-exists", action="store_true")
    args = p.parse_args(argv)
    try:
        out = write_fr3_wrapper(
            args.output, args.fr3_model, require_exists=args.require_exists
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote {out} (include={args.fr3_model})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
