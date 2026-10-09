#!/usr/bin/env bash
# Weekly model refresh (Saturdays via openclaw cron, alert-wrapped).
#
# Alpha engine v2 only, since 2026-10-09: SEC insider trades → refit the production
# forecasters on every matured label → extend the causal walk-forward backtest. ~10 min.
#
# The legacy 14-model ensemble retrain (research/deploy.py, 4-5 h on CPU) was retired
# with the legacy pipeline on 2026-10-09: it fed only the ml_raw/ml_v2/ml_v2_gp paper
# books and the "ML model backtest" digest block, all retired the same day. The script
# stays in the repo (ops/research/deploy.py) if it is ever needed again.
set -uo pipefail
# Machine-independent: code lives in <repo>/ops, live state in $TS_OPS
# (~/trade-ops on the RIT box). Same precedence as ops/paths.py.
REPO="${TS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPS="${TS_OPS:-$HOME/trade-ops}"
CODE="$REPO/ops"
LOG="$OPS/logs/retrain-$(date +%F).log"

cd "$REPO" || exit 1
# shellcheck disable=SC1091
source venv/bin/activate

echo "=== weekly retrain start $(date -Is) ===" >> "$LOG"
FAIL=0
step() {  # step <timeout-secs> <cmd...> — best effort, but any failure makes the job exit non-zero
  local tmo="$1"; shift
  echo "--- $* ($(date -Is)) ---" >> "$LOG"
  timeout "$tmo" "$@" >> "$LOG" 2>&1 || { echo "$* FAILED rc=$?" >> "$LOG"; FAIL=1; }
}
# SEC insider trades (quarterly data sets; only the latest quarters are refetched). Research
# candidate features today — kept fresh so the next clean test can use them. ~1 min.
step 1800 ts data insiders
# Refit the production forecasters on every matured label (~6 min on the GPU), then extend the
# causal walk-forward with the new dates only.
step 3600 ts alpha train
step 3600 ts alpha backtest --extend
echo "=== weekly retrain done fail=$FAIL $(date -Is) ===" >> "$LOG"
tail -4 "$LOG"
find $OPS/logs -name 'retrain-*.log' -mtime +60 -delete 2>/dev/null
exit $FAIL
