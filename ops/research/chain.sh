#!/usr/bin/env bash
# Research chain: wait for harness → backtest → wait for baseline train → deploy.
set -uo pipefail
R=/home/ad2688/trade-ops/research
LOG=$R/out/chain.log
cd /home/ad2688/Desktop/Trade-system_v1 || exit 1
# shellcheck disable=SC1091
source venv/bin/activate
exec >> "$LOG" 2>&1

echo "=== chain start $(date -Is) ==="
# 1. wait for harness (max 4h). [h] trick: never match this script/loop itself.
for _ in $(seq 1 480); do
  grep -q "ALL VARIANTS DONE" "$R/out/harness.log" 2>/dev/null && break
  if ! pgrep -f "[h]arness.py" >/dev/null; then
    echo "harness not running and not done — aborting chain $(date -Is)"
    grep -E "Error|Traceback" "$R/out/run.log" | tail -3
    exit 1
  fi
  sleep 30
done
echo "--- harness done $(date -Is):"
grep "== " "$R/out/harness.log" || true

# 2. backtest all variants into P&L + DSR/PBO
echo "=== backtest.py $(date -Is) ==="
python3 "$R/backtest.py" || echo "backtest.py FAILED rc=$?"

# 3. wait for baseline ts train to release CPU (max 2h more)
for _ in $(seq 1 240); do
  pgrep -f "venv/bin/ts train" >/dev/null || break
  sleep 30
done
echo "--- baseline train finished by $(date -Is); registry:"
ls reports/models/ || true

# 4. deploy full 14-model ensemble on the winning config
echo "=== deploy.py $(date -Is) ==="
python3 "$R/deploy.py" || echo "deploy.py FAILED rc=$?"
echo "=== chain done $(date -Is) ==="
