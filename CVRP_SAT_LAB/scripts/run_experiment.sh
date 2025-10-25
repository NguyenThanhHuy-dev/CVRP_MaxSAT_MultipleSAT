#!/usr/bin/env bash
# Simple runner for the experiment pipeline
set -euo pipefail
python3 "$(dirname "$0")/../src/experiment.py" "$@"
