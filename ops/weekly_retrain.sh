#!/usr/bin/env bash
# Weekly full model retrain + deploy (Saturdays via openclaw cron, alert-wrapped).
#
# Runs ~/trade-ops/research/deploy.py: trains the repo's full 14-model ensemble
# on the research-validated config (purged walk-forward + temporal validation,
# winner by OOS IC t-stat from ~/trade-ops/research/out/summary.json), then
#   - saves it to reports/models/ (gitignored registry). The decision layer
#     auto-picks the newest ensemble → ts signals/analyze/future-predict all
#     use the fresh model from the next run onward.
#   - refreshes data/gold/predictions.parquet (feeds the daily ML backtest
#     block in the morning digest).
# Heavy (1-3 h) — which is why it's weekly and on a Saturday, not in the
# 05:15 weekday window.
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
timeout 14400 python3 $CODE/research/deploy.py >> "$LOG" 2>&1
RC=$?
# Alpha engine v2: refit the production forecasters on every matured label and
# extend the causal walk-forward (new dates only) so the backtest report and the
# ledger's calibration keep up with the live period. ~5 min on the GPU.
echo "--- ts alpha train + backtest --extend ($(date -Is)) ---" >> "$LOG"
timeout 3600 ts alpha train >> "$LOG" 2>&1 || echo "ts alpha train FAILED rc=$?" >> "$LOG"
timeout 3600 ts alpha backtest --extend >> "$LOG" 2>&1 || echo "ts alpha backtest FAILED rc=$?" >> "$LOG"
echo "=== weekly retrain done rc=$RC $(date -Is) ===" >> "$LOG"

# quick post-deploy sanity: newest registry entry + a one-line backtest health
tail -3 "$LOG"
ls -t "$REPO/reports/models" | head -3
timeout 300 python3 $CODE/research/daily_ml_backtest.py | head -4
find $OPS/logs -name 'retrain-*.log' -mtime +60 -delete 2>/dev/null
exit $RC
