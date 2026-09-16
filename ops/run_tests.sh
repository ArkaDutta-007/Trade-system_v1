#!/usr/bin/env bash
# Self-test for the out-of-repo trade-ops tooling. Runs in the daily pipeline so
# a broken helper shows up as a line in the brief instead of a silent wrong number.
set -uo pipefail
cd /home/ad2688/Desktop/Trade-system_v1 || exit 1
# shellcheck disable=SC1091
source venv/bin/activate
export PYTHONPATH=/home/ad2688/trade-ops/flags:/home/ad2688/trade-ops/portfolio:/home/ad2688/trade-ops/research
cd /home/ad2688/trade-ops || exit 1
python3 -m pytest tests/ -q --tb=line 2>&1 | tail -25
