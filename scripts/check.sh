#!/usr/bin/env bash
# The fast gate. `make check` and CI run this script; keep it under 15 seconds.
# Make 3.81 on macOS has no .ONESHELL, so the parallel lanes live here.
set -euo pipefail

cd "$(dirname "$0")/.."

# A stale lockfile fails here, before any lane runs.
uv sync --locked --extra dev --extra inference --quiet

log_dir="$(mktemp -d)"
trap 'rm -rf "$log_dir"' EXIT

lane_names=()
lane_pids=()

start_lane() {
  local name="$1"
  shift
  lane_names+=("$name")
  "$@" >"$log_dir/$name.log" 2>&1 &
  lane_pids+=("$!")
}

start_lane ruff-check uv run --no-sync ruff check box_office tests scripts
start_lane ruff-format uv run --no-sync ruff format --check box_office tests scripts
start_lane fast-tests uv run --no-sync pytest tests/ -m "not integration and not slow"

failed=0
for i in "${!lane_names[@]}"; do
  if wait "${lane_pids[$i]}"; then
    echo "ok    ${lane_names[$i]}"
  else
    failed=1
    echo "FAIL  ${lane_names[$i]}" >&2
    cat "$log_dir/${lane_names[$i]}.log" >&2
  fi
done

exit "$failed"
