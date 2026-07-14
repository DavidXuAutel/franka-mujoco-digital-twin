#!/usr/bin/env bash
# Start pose_node + twin_node with repo default config paths.
# Requires: ROS2 overlay sourced, package installed (pip install -e ".[vision,mujoco]").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec python launch/twin_mvp.launch.py "$@"
