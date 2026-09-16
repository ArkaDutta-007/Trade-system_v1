#!/usr/bin/env bash
# Standalone daily flag refresh. Separate from daily_update.sh on purpose: the
# flag board is cheap (~30s) and must stay current even on days the heavy
# ingest/feature pipeline fails or times out.
set -uo pipefail
LOG="/home/ad2688/trade-ops/logs/flags-$(date +%F).log"
cd /home/ad2688/Desktop/Trade-system_v1 || exit 1
# shellcheck disable=SC1091
source venv/bin/activate
{
  echo "=== flag refresh $(date -Is) ==="
  timeout 600 python3 /home/ad2688/trade-ops/flags/auto_flags.py
} >> "$LOG" 2>&1
RC=$?
tail -8 "$LOG"
find /home/ad2688/trade-ops/logs -name 'flags-*.log' -mtime +30 -delete 2>/dev/null
exit $RC
