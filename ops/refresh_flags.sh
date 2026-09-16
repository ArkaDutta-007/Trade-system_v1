#!/usr/bin/env bash
# Standalone daily flag refresh. Separate from daily_update.sh on purpose: the
# flag board is cheap (~30s) and must stay current even on days the heavy
# ingest/feature pipeline fails or times out.
set -uo pipefail
# Machine-independent: code lives in <repo>/ops, live state in $TS_OPS
# (~/trade-ops on the RIT box). Same precedence as ops/paths.py.
REPO="${TS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPS="${TS_OPS:-$HOME/trade-ops}"
CODE="$REPO/ops"
LOG="$OPS/logs/flags-$(date +%F).log"
cd $REPO || exit 1
# shellcheck disable=SC1091
source venv/bin/activate
{
  echo "=== flag refresh $(date -Is) ==="
  timeout 600 python3 $CODE/flags/auto_flags.py
} >> "$LOG" 2>&1
RC=$?
tail -8 "$LOG"
find $OPS/logs -name 'flags-*.log' -mtime +30 -delete 2>/dev/null
exit $RC
