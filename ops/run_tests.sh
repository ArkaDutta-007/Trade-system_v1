#!/usr/bin/env bash
# Self-test for the out-of-repo trade-ops tooling. Runs in the daily pipeline so
# a broken helper shows up as a line in the brief instead of a silent wrong number.
set -uo pipefail
# Machine-independent: code lives in <repo>/ops, live state in $TS_OPS
# (~/trade-ops on the RIT box). Same precedence as ops/paths.py.
REPO="${TS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPS="${TS_OPS:-$HOME/trade-ops}"
CODE="$REPO/ops"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
source venv/bin/activate
export PYTHONPATH=$CODE/flags:$CODE/portfolio:$CODE/research:$CODE
cd $CODE || exit 1
python3 -m pytest tests/ -q --tb=line 2>&1 | tail -25
