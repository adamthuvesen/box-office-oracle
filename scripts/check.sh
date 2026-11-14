#!/usr/bin/env bash
# The fast gate. `make check` and CI run this script; keep it under 15 seconds.
# Make 3.81 on macOS has no .ONESHELL, so the parallel lanes live here.
set -euo pipefail

cd "$(dirname "$0")/.."

# A stale lockfile fails here, before any lane runs.
uv sync --locked --extra dev --extra inference --extra train --quiet

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
start_lane boundaries scripts/check_boundaries.sh
start_lane types uv run --no-sync basedpyright
start_lane fast-tests uv run --no-sync pytest tests/ -m "not integration and not slow"

web_lane() {
  local name="$1"
  local script="$2"
  start_lane "$name" bash -c '[ -d web/node_modules ] || { echo "web/node_modules is missing. Run: pnpm --dir web install --frozen-lockfile" >&2; exit 1; }; pnpm --dir web "$0"' "$script"
}

web_lane web-lint lint
web_lane web-format format:check
web_lane web-typecheck typecheck
web_lane web-tests test

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
