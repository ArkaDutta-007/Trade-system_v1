#!/usr/bin/env bash
# Email the trade brief. WEEKLY, to Arka + Om (requested 2026-09-17).
#
# History: this was a weekday-daily mail with a Mon-only second recipient, then
# disabled entirely on 2026-09-08 when the brief moved to Telegram. It is back
# on a WEEKLY cadence — the Telegram brief remains the daily channel, and this
# is the once-a-week copy that both of them get by mail.
#
# Prefers the agent-composed brief (briefs/brief-YYYY-MM-DD.md); falls back to
# the raw pipeline digest so an LLM failure never means "no email at all".
# Exits 0 quietly if smtp.env is not configured yet.
set -uo pipefail
# Machine-independent: code lives in <repo>/ops, live state in $TS_OPS
# (~/trade-ops on the RIT box). Same precedence as ops/paths.py.
REPO="${TS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPS="${TS_OPS:-$HOME/trade-ops}"
CODE="$REPO/ops"
TODAY="$(date +%F)"
LOG="$OPS/logs/email-$TODAY.log"

# Both recipients, every send. The cron runs weekly, so "who" and "how often"
# are no longer entangled: change the cron schedule to change the cadence.
RECIPIENTS="${BRIEF_RECIPIENTS:-arkadutta.cg@gmail.com,omsaha01@gmail.com}"

BODY="$OPS/briefs/brief-$TODAY.md"
SUBJECT="📈 Weekly trade brief — $TODAY"
if [[ ! -s "$BODY" ]]; then
  BODY="$OPS/briefs/latest.md"
  SUBJECT="📈 Weekly trade DIGEST (agent brief missing) — $TODAY"
fi
[[ -s "$BODY" ]] || { echo "$(date -Is) no brief or digest to send" >> "$LOG"; exit 1; }

# Flag a body that daily_update.sh failed to refresh (older than 20 hours) so
# a silently dead pipeline is obvious from the subject line alone.
if (( $(date +%s) - $(stat -c %Y "$BODY") > 72000 )); then
  SUBJECT="⚠️ STALE — pipeline may be dead — $SUBJECT"
fi

OUT=$("$OPS/bin/send-email" "$SUBJECT" "$BODY" "$RECIPIENTS" 2>&1)
RC=$?
echo "$(date -Is) rc=$RC to=[$RECIPIENTS] $OUT" >> "$LOG"
if [[ $RC -eq 3 ]]; then
  # unconfigured — not an error, just waiting for the app password
  exit 0
fi
exit $RC
