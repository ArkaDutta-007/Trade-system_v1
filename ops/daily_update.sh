#!/usr/bin/env bash
# Daily trade-system update + morning digest builder (RIT machine).
# Invoked by openclaw cron on weekday mornings; safe to re-run manually anytime.
#
# Produces:
#   ~/trade-ops/briefs/digest-YYYY-MM-DD.md  (sections the brief agent reads)
#   ~/trade-ops/briefs/latest.md             (copy of the newest digest)
#   ~/trade-ops/logs/daily-YYYY-MM-DD.log    (full pipeline stderr/stdout)
set -uo pipefail  # deliberately no -e: every step is best-effort and recorded

# Machine-independent: code lives in <repo>/ops, live state in $TS_OPS
# (~/trade-ops on the RIT box). Same precedence as ops/paths.py.
REPO="${TS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPS="${TS_OPS:-$HOME/trade-ops}"
CODE="$REPO/ops"
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
    out="$(timeout 300 ts flags --config $OPS/flags/local_config.yaml --refresh 2>> "$LOG")"
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
heavy_step "trade-ops self-test (pytest)" 600 $CODE/run_tests.sh
# Massive (ex-Polygon) EOD feed: one grouped call brings yesterday's bar for the
# WHOLE market, plus recent splits/dividends and tagged news. Bounded to
# data.massive.update_max_calls (40 ≈ 8 min at the free 5 req/min). With no
# MASSIVE_API_KEY it prints a note and exits 0; `ts ingest` then falls back
# to yfinance on its own (data.source: auto).
heavy_step "ts massive update (EOD bars + corp actions + news, 5 req/min)" 2400 ts massive update -u liquid
# Alpha engine v2 (2026-09-21): point-in-time panel over the 1000-name Massive
# universe → tally the forecasts that matured → recalibrate on the tally →
# record today's forecasts. ~2 min; outputs are gitignored (data/gold, data/ledger,
# data/models). The book itself is the `alpha_v2` paper book below.
# Fundamentals from SEC EDGAR (Massive retired its free financials endpoint 2026-09-28):
# reads EDGAR's daily filing index, refetches only companies that filed a 10-Q/10-K. ~1 min.
heavy_step "ts data fundamentals (SEC EDGAR 10-Q/10-K, as first reported)" 1200 ts data fundamentals
heavy_step "ts alpha panel (1000-name point-in-time feature panel)" 900 ts alpha panel
heavy_step "ts alpha tally (score matured forecasts against prices)" 600 ts alpha tally
heavy_step "ts alpha forecast (record today's forecasts, recalibrate)" 900 ts alpha forecast --days 3
# Legacy engine RETIRED 2026-10-09 (`ts daily`, `ts ledger --resolve`, `ts features -u liquid --deep`,
# the sparse-date trim, BUY signals, future-predict/paper status, the ML-model backtest, picks_v2 and
# the raw `ts picks` ranking): ~34 of the run's 36 minutes, feeding only books that trailed SPY by 7-8%
# and diagnostic sections the alpha engine superseded on 2026-09-21. The code stays in the repo; restore
# the steps from git (commit before "Retire the legacy pipeline") if ever needed. What remains of it is
# the liquid-universe price file, which the paper books, the flag board and the regime layer read —
# rebuilt from the Massive tables in ~20 s, without the legacy news + LLM apprehension fetch.
heavy_step "ts ingest -u liquid --no-news (prices for books, flags, regime)" 900 ts ingest -u liquid --no-news

# Derive ALL five flags from live data before the board is rendered, so the
# brief never ships a hand-typed override that went stale months ago (F and C
# sat at as_of 2026-06-10 for three months). Writes OUTSIDE the repo; the
# tracked configs/flag_overrides.yaml is never touched.
heavy_step "Auto-derive flags (O/F/I/S/C)" 600 python3 $CODE/flags/auto_flags.py
flags_step 4 45
# Survivorship bias, measured not assumed (lab research 2026-09-15): equal
# weight of TODAY's universe vs RSP, a real survivorship-free index product.
digest_step "Survivorship bias check (ts bias-check) — context for every backtest number" 300 ts bias-check -u liquid

# THE picks for the brief (since 2026-09-21): the alpha engine v2 book —
# 1000-name point-in-time panel, 5/21/63d rank forecasters blended by realised
# IC from the tallied forecast ledger, gated, sector-capped, ½ equal +
# ½ inverse-vol weighted, 25% vol brake, 200-day trend overlay, GP partial
# trading against the live alpha_v2 paper book. --compact = phone-width card. The brief
# agent is told to use ONLY this.
digest_step "🎯 TOP PICKS — use THIS section for the brief" 600 ts alpha picks --top 20 --compact --prev $OPS/portfolio/books/alpha_v2.json
digest_step "Alpha book — full table (targets, calibrated E[r], 80% bands, drivers)" 600 ts alpha picks --top 20 --prev $OPS/portfolio/books/alpha_v2.json
digest_step "Data status — every dataset, sessions behind, crawler budget" 300 ts data status
digest_step "Alpha engine — realised forecast skill (ledger tally)" 300 ts alpha status
# Pre-registered live-evidence checkpoints (21d forecasts mature 2026-10-15, 63d mid-December):
# live IC vs the research expectation, paper book vs SPY/RSP, and the verdict the rules give. Emails
# Arka (ops alert channel) once per checkpoint reached and when the verdict turns RED/AMBER.
digest_step "Live evidence — model health vs pre-registered rules" 300 ts alpha live --alert-cmd "$HOME/ops/bin/alert"
# Macro/fragility background: oil, vol, credit, rates, AI concentration → closest
# historical episodes → how the signal and the book behaved in those backgrounds.
# The forecast step above already blended this into the calibration.
digest_step "Regime & fragility — where are we, when was it like this" 300 ts alpha regime --compact

# Competing $10k paper books (spy_benchmark / momentum / alpha_v2; the four legacy-model books are
# retired and shown frozen). Rebalance is monthly and self-guarding, so calling it daily is safe.
heavy_step "Dummy portfolios rebalance+mark" 1200 python3 $CODE/portfolio/portfolios.py --rebalance --mark
digest_step "Dummy portfolios — long-run scoreboard" 300 python3 $CODE/portfolio/portfolios.py --report

# Playbook briefing needs the broker snapshot, which is gitignored and may not
# exist on this machine yet — scp "portfolio and watchlist.json" from the laptop.
if [[ -f "$REPO/portfolio and watchlist.json" ]]; then
  digest_step "Playbook morning briefing (ts brief)" 600 ts brief --config $OPS/flags/local_config.yaml
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
