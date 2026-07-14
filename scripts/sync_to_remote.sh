#!/usr/bin/env bash
# Rsync this repo to the remote ROS host for online twin bringup.
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-yao}"
REMOTE_HOST="${REMOTE_HOST:-10.229.20.125}"
REMOTE_DIR="${REMOTE_DIR:-~/franka_mujoco_digital_twin}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC_EXCLUDES=(
  --exclude '.git'
  --exclude '.venv'
  --exclude '__pycache__'
  --exclude '.pytest_cache'
  --exclude '*.pyc'
  --exclude '.mypy_cache'
  --exclude 'configs/camera_extrinsics.yaml'
)

echo "Syncing ${ROOT} -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
rsync -avz --delete "${RSYNC_EXCLUDES[@]}" "${ROOT}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
echo "Done. SSH: ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "Then: cd ${REMOTE_DIR} && see docs/bringup_remote.md"
