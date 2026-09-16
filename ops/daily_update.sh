#!/usr/bin/env bash
# Daily trade-system update + morning digest builder (RIT machine).
# Invoked by openclaw cron on weekday mornings; safe to re-run manually anytime.
#
# Produces:
#   ~/trade-ops/briefs/digest-YYYY-MM-DD.md  (sections the brief agent reads)
#   ~/trade-ops/briefs/latest.md             (copy of the newest digest)
#   ~/trade-ops/logs/daily-YYYY-MM-DD.log    (full pipeline stderr/stdout)
set -uo pipefail  # deliberately no -e: every step is best-effort and recorded

REPO="/home/ad2688/Desktop/Trade-system_v1"
OPS="/home/ad2688/trade-ops"
TODAY="$(date +%F)"
DIGEST="$OPS/briefs/digest-$TODAY.md"
LOG="$OPS/logs/daily-$TODAY.log"
export COLUMNS=200  # let rich render tables wide instead of truncating cells

cd "$REPO" || exit 1
# shellcheck disable=SC1091
source venv/bin/activate

echo "=== daily_update.sh start $(date -Is) ===" >> "$LOG"
: > "$DIGEST"
{
  echo "# Trade-system morning digest — $TODAY"
  echo
  echo "_Generated $(date -Is) on the RIT machine._"
  echo
} >> "$DIGEST"

# ---- heavy steps: output goes to the log, only status lands in the digest ----
# Every step also prints ONE line to stdout. The openclaw cron runner enforces
# a no-output timeout, and this script sends all real output to $LOG — so with
# no stdout the runner saw a "silent" job and killed it at the 1h mark, every
# day (2026-09-11..15: "command produced no output before
# noOutputTimeoutSeconds"). These lines are the liveness signal.
say() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
  cp -f "$DIGEST" "$OPS/briefs/latest.md" 2>/dev/null   # keep latest.md live mid-run
}

heavy_step() {  # heavy_step <title> <timeout-secs> <cmd...>
  local title="$1" tmo="$2"; shift 2
  echo "--- $title: $* ($(date -Is)) ---" >> "$LOG"
  say "start: $title"
  local t0=$SECONDS
  if timeout "$tmo" "$@" >> "$LOG" 2>&1; then
    echo "- $title: OK" >> "$DIGEST"
    say "ok ($((SECONDS-t0))s): $title"
  else
    local rc=$?
    echo "- $title: FAILED or timed out rc=$rc (see $LOG)" >> "$DIGEST"
    say "FAILED rc=$rc ($((SECONDS-t0))s): $title"
  fi
}

# ---- digest steps: stdout is captured into the digest itself ----
digest_step() {  # digest_step <title> <timeout-secs> <cmd...>
  local title="$1" tmo="$2"; shift 2
  {
    echo
    echo "## $title"
    echo '```'
  } >> "$DIGEST"
  say "start: $title"
  local t0=$SECONDS
  if ! timeout "$tmo" "$@" >> "$DIGEST" 2>> "$LOG"; then
    echo "[step failed or timed out: $* — see $LOG]" >> "$DIGEST"
    say "FAILED ($((SECONDS-t0))s): $title"
  else
    say "ok ($((SECONDS-t0))s): $title"
  fi
  echo '```' >> "$DIGEST"
}

# ---- flags step: retry on transient Yahoo rate-limits ----
# `ts flags --refresh` exits 0 even when a flag (usually S, from ^NDX on Yahoo)
# comes back UNKNOWN with "Too Many Requests" — so the failure never trips the
# digest_step non-zero path and a rate-limited flag gets shipped into the brief.
# Detect the failure in the output text and back off instead. Runs at 05:45 ET,
# ~2h before the 07:45 brief, so a few 45s retries are free.
flags_step() {  # flags_step <tries> <delay-secs>
  local tries="${1:-4}" delay="${2:-45}" out i
  {
    echo
    echo "## Flag board + composite (ts flags --refresh)"
    echo '```'
  } >> "$DIGEST"
  for (( i=1; i<=tries; i++ )); do
    out="$(timeout 300 ts flags --config /home/ad2688/trade-ops/flags/local_config.yaml --refresh 2>> "$LOG")"
    if ! grep -qiE 'Too Many Requests|Rate limited|lookup failed' <<<"$out"; then
      break
    fi
    echo "[flags attempt $i/$tries rate-limited; sleeping ${delay}s] ($(date -Is))" >> "$LOG"
    (( i < tries )) && sleep "$delay"
  done
  printf '%s\n' "$out" >> "$DIGEST"
  echo '```' >> "$DIGEST"
}

cp -f "$DIGEST" "$OPS/briefs/latest.md"   # publish early; refreshed per step below
echo "## Pipeline run" >> "$DIGEST"
# Self-test the out-of-repo tooling first: a broken helper should be a
# visible line in the brief, not a silently wrong number downstream.
heavy_step "trade-ops self-test (pytest)" 600 /home/ad2688/trade-ops/run_tests.sh
heavy_step "ts daily (ingest→quality→features→predict→paper-trade→future-update)" 5400 ts daily
heavy_step "ts ledger --resolve (score matured predictions)" 900 ts ledger --resolve
# ts daily rebuilds gold features on the CORE universe (69 cols), but the
# committed models_store forecasters were trained on LIQUID features with the
# deep nonlinear tier (86 cols incl. lppls/lyapunov/rqa/entropy/chaos01) —
# ts picks / the Invest Planner break on the mismatch. Rebuild liquid+deep
# last so the gold matrix matches the models for picks and for the dashboard.
heavy_step "ts ingest -u liquid" 1800 ts ingest -u liquid
heavy_step "ts features -u liquid --deep (rebuild gold to match models_store)" 7200 ts features -u liquid --deep

# Derive ALL five flags from live data before the board is rendered, so the
# brief never ships a hand-typed override that went stale months ago (F and C
# sat at as_of 2026-06-10 for three months). Writes OUTSIDE the repo; the
# tracked configs/flag_overrides.yaml is never touched.
heavy_step "Auto-derive flags (O/F/I/S/C)" 600 python3 /home/ad2688/trade-ops/flags/auto_flags.py
flags_step 4 45
digest_step "Top long-horizon picks (ts picks --horizon 252 --top 10 -u liquid)" 900 ts picks --horizon 252 --top 10 -u liquid
digest_step "BUY signals today (ts signals --stance BUY --top 10)" 600 ts signals --stance BUY --top 10
digest_step "Future-predict sessions — MTM P&L + accuracy (ts future-status)" 600 ts future-status
digest_step "Paper portfolio (ts paper-status)" 600 ts paper-status
# ML model P&L: backtest of the deployed model's OOS predictions (top-20
# equal-weight, after costs) + trailing-63d decay check. Predictions are
# refreshed by the Saturday retrain (~/trade-ops/weekly_retrain.sh).
digest_step "ML model backtest (deployed ensemble, after costs)" 400 python3 /home/ad2688/trade-ops/research/daily_ml_backtest.py

# Picks v2: quality-gated (price/liquidity/vol), risk-adjusted (score/vol^0.5)
# and theme-diversified. Raw `ts picks` above ranks on unadjusted score, which
# measurably prefers volatile illiquid names (corr(score,vol)=+0.22,
# corr(score,liquidity)=-0.19) — see ~/trade-ops/portfolio/picks_v2.py.
digest_step "Top picks v2 (gated + diversified)" 900 python3 /home/ad2688/trade-ops/portfolio/picks_v2.py --top 10

# Competing $10k paper books (spy / ml_raw / ml_v2 / momentum / blend).
# Rebalance is monthly and self-guarding, so calling it daily is safe.
heavy_step "Dummy portfolios rebalance+mark" 1200 python3 /home/ad2688/trade-ops/portfolio/portfolios.py --rebalance --mark
digest_step "Dummy portfolios — long-run scoreboard" 300 python3 /home/ad2688/trade-ops/portfolio/portfolios.py --report

# Playbook briefing needs the broker snapshot, which is gitignored and may not
# exist on this machine yet — scp "portfolio and watchlist.json" from the laptop.
if [[ -f "$REPO/portfolio and watchlist.json" ]]; then
  digest_step "Playbook morning briefing (ts brief)" 600 ts brief --config /home/ad2688/trade-ops/flags/local_config.yaml
else
  {
    echo
    echo "## Playbook morning briefing"
    echo "_skipped: 'portfolio and watchlist.json' not present on this machine_"
  } >> "$DIGEST"
fi

cp -f "$DIGEST" "$OPS/briefs/latest.md"
find "$OPS/briefs" -name 'digest-*.md' -mtime +30 -delete 2>/dev/null
find "$OPS/logs" -name 'daily-*.log' -mtime +30 -delete 2>/dev/null
echo "=== daily_update.sh done $(date -Is) ===" >> "$LOG"
