#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "$0")"
if [ -z "$PYTHON" ]; then PYTHON=python3; fi
"$PYTHON" setup_env.py
if [ "$#" -eq 0 ]; then
    set -- --n 1000 --reps 40 --workers 12 --trees 500 --run-id cpu_compare
fi
"$PYTHON" benchmark.py "$@"
