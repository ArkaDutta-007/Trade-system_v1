# TOOLS.md — Silas's operational playbook (RIT machine)

Machine-specific cheat sheet: how to actually run things when Arka asks in
natural language (Telegram or webchat). Safety rails in AGENTS.md always win.

## Trade-system_v1 (repo: ~/Desktop/Trade-system_v1)

Policy: **read + run = yes, mutate = no.** Run `ts` commands freely (they write
only gitignored artifacts like data/ and reports/). Never edit tracked files or
touch git state (add/commit/push/pull/reset/clean). Exception to "run freely":
**`ts train-forecast` writes the committed `models_store/` → dirties the repo —
only run it if Arka explicitly asks, and warn him first. (`ts train` writes only
gitignored `reports/models/` and is safe — verified 2026-07-14.)**

LAYOUT (since 2026-09-16): CODE lives in the repo at `~/Desktop/Trade-system_v1/ops/`
(machine-independent via ops/paths.py; pulls from the lab bring updates).
STATE lives in `~/trade-ops/` (briefs, logs, flags/*.yaml+json, portfolio/books,
private, secrets). Never run the old copies under ~/trade-ops — they were
retired to ~/trade-ops/private/retired-code-2026-09-16/.
New research commands: `ts wf-backtest` (strictly-causal walk-forward, heavy) and
`ts bias-check -u liquid` (survivorship bias, ~1 min — it is in the daily digest).

Always run via the wrapper (handles cd + venv + timeout + logging):

    ~/ops/bin/ts-run [--timeout secs] <ts-args>

What Arka says → what you run:
- "run the daily pipeline"        → `~/ops/bin/run-with-alert trade-daily-update ~/Desktop/Trade-system_v1/ops/daily_update.sh`
  (full pipeline + digest, ~1–2 h; or `openclaw cron run trade-daily-update`)
- "top picks"                     → `ts-run picks --horizon 252 --top 10 -u liquid`
- "buy signals"                   → `ts-run signals --stance BUY --top 10`
- "analyze NVDA"                  → `ts-run analyze NVDA`  (JSON+MD report)
- "flag board"                    → `ts-run flags --config ~/trade-ops/flags/local_config.yaml --refresh`
                                    (plain `ts-run flags --refresh` reads the STALE tracked overrides)
- "paper portfolio / how are we doing" → `ts-run paper-status` and `ts-run future-status`
- "today's digest/brief"          → read `~/trade-ops/briefs/latest.md`, summarize
  The 07:45 brief is DELIVERED to the Telegram group -5412027506 by the cron
  runner itself (delivery: announce → telegram → to=telegram:-5412027506).
  Before 2026-09-16 it went to agent:main:main, whose channel is WEBCHAT, so
  briefs never reached the phone. Do not "sessions_send" briefs to main.
- "email me/us the brief"         → `~/Desktop/Trade-system_v1/ops/send_brief_email.sh`
  Sends to Arka + Om. Automatic cadence is WEEKLY (cron `trade-brief-email`,
  Mondays 08:00 ET, re-enabled 2026-09-17). Telegram remains the DAILY channel.
  Override recipients for a one-off with BRIEF_RECIPIENTS="a@b,c@d".
- "how is the model doing / model backtest" → `~/Desktop/Trade-system_v1/venv/bin/python3 ~/Desktop/Trade-system_v1/ops/research/daily_ml_backtest.py`
  (also a section in the daily digest; watch the "last 63d" decay flag)
- "picks" / "good picks" → prefer **v2**:
  `~/Desktop/Trade-system_v1/venv/bin/python3 ~/Desktop/Trade-system_v1/ops/portfolio/picks_v2.py --top 10   (add --compact for the phone card)`
  Raw `ts picks` ranks on unadjusted score and measurably prefers volatile,
  illiquid names (corr(score,vol)=+0.22, corr(score,$vol)=−0.19) — it returned
  $1 stocks trading $92k/day. v2 adds price/liquidity/vol gates, ranks by
  score/vol^0.5, and caps names per theme. `--compare` shows v1 vs v2.
- "how are the paper books / which strategy is winning" →
  `… ~/Desktop/Trade-system_v1/ops/portfolio/portfolios.py --report`
  Six competing $10k books. Five seeded 2026-09-08 (spy_benchmark, ml_raw,
  ml_v2, momentum, blend, monthly full rebalance) plus **ml_v2_gp** seeded
  2026-09-16 — the research winner xgb63|gp35: 63d model, top-20, 10% cap,
  Garleanu-Pedersen partial trading at 0.35. Ignore the ranking until ~60
  sessions.
- "backtest the pick rules" → `… ~/Desktop/Trade-system_v1/ops/portfolio/backtest_v2.py --top 10`
- "flags" / "why is the composite X" → `… ~/Desktop/Trade-system_v1/ops/flags/auto_flags.py` prints
  all five with their scores; the board itself is
  `ts flags --config ~/trade-ops/flags/local_config.yaml --refresh`.
  ALL FIVE flags are now auto-derived daily (cron `trade-flags-refresh`, 04:30
  ET) into ~/trade-ops/flags/auto_overrides.yaml — OUTSIDE the repo, so the
  tracked configs/flag_overrides.yaml is never touched. Never hand-edit the
  auto file; it is regenerated every morning.
- "massive status / is the Massive feed ok" → `~/ops/bin/ts-run massive status`
  Massive (ex-Polygon) is the EOD price feed since 2026-09-19: whole-market bars
  for the last 2 years + splits/dividends/fundamentals/tagged news, capped at
  **5 requests/min** (enforced client-side; never work around it). Key
  `MASSIVE_API_KEY` is in the repo .env (added 2026-09-19).
- "is the Massive crawler running / crawler status" → `~/ops/bin/ts-run massive status`
  (bottom lines: crawler running/not, current tier+item, calls by tier, adjustment QA).
  It is the systemd user service `massive-crawler.service` (log
  `~/trade-ops/logs/massive-crawler.log`): `systemctl --user status|restart massive-crawler`.
  It runs FOREVER on purpose — tiers: newest bars → 2y backfill → directory/corp
  actions → universe depth → whole-market depth → QA — and it shares the 5/min
  limiter with every other Massive call, so never start a second crawler or a
  manual `ts massive backfill` while it runs (they only queue behind each other).
  The 05:15 pipeline's `ts massive update` is a no-op when the crawler is current.
- "are the tools healthy / run the tests" → `~/Desktop/Trade-system_v1/ops/run_tests.sh`
  (52 unit tests; also runs first in the daily pipeline).
- "retrain the model now"         → `~/ops/bin/run-with-alert trade-weekly-retrain ~/Desktop/Trade-system_v1/ops/weekly_retrain.sh`
  (heavy, 1–3 h; normally auto via cron `trade-weekly-retrain`, Sat 03:00 ET.
  Retrains the 14-model ensemble with the purged walk-forward config from
  ~/trade-ops/research/, deploys to reports/models — gitignored, safe)

Long jobs: warn that it takes a while, run it, report back when done — don't
block the conversation. GPU note: before delegating to local qwen, run
`bash ~/.openclaw/workspace/gpu_guard.sh` (never contend with a training run).

## Portfolio update from a Fidelity screenshot

Arka sends a Fidelity "Positions" screenshot on Telegram with a caption
containing **"update portfolio"**. The keyword is REQUIRED — never run this on an
uncaptioned or unrelated image (it overwrites his portfolio state). Then:

1. OCR the image yourself — vision routes to `gemma4:31b-cloud` (flash can't see
   images). Produce STRICT minified JSON and nothing else:
   `{"holdings":[{"symbol","quantity","last_price","current_value","average_cost","total_gain_loss","total_gain_loss_pct"}],"cash":[{"symbol","name","value"}]}`
   Money-market rows (SPAXX/FDRXX/"Cash") go in `cash`. Numbers plain — no `$` `,`
   or `%`. Use `null` where a field isn't legible. Never invent a ticker or row.
2. Write that JSON to `~/trade-ops/private/inbound_holdings.json` (write tool).
3. Run:
   `~/Desktop/Trade-system_v1/ops/bin/portfolio-from-screenshot --from-json ~/trade-ops/private/inbound_holdings.json --apply`
   (Alternatively, if the image is saved to a real path, run
   `~/Desktop/Trade-system_v1/ops/bin/portfolio-from-screenshot <path> --apply` — it OCRs via
   gemma4 itself. Drop `--apply` for a dry-run diff first if unsure.)
4. Reply with the script's decision report (regime + per-holding buy/sell/hold +
   12-month picks). It backs up the prior state to `~/trade-ops/private/backups/`
   and updates the private, gitignored "portfolio and watchlist.json" — this is
   the ONE sanctioned write to that file. Still never touch tracked files or git.

Privacy: that file + its backups are chmod 600 and gitignored. Never paste
holdings into git; never transmit them except via the normal brief/gist tools.

## Lab server (RIT, 129.21.34.40) — added 2026-09-08

Passwordless via `ssh lab` (ed25519 key `~/.ssh/id_ed25519_lab`). Arka does his
research there and pulls projects here to work on.

- "what's on the lab server"        → `~/ops/bin/lab-sync`  (lists projects + sizes)
- "pull/sync <project>"             → `~/ops/bin/lab-sync <project>`
- "what would change"               → `~/ops/bin/lab-sync <project> -n`  (dry run)
Lands in `~/Desktop/Works_ongoing/<project>`. Incremental and resumable; it does
NOT delete local-only files (pass `--mirror` only if Arka explicitly asks for an
exact mirror). It prints a verify line — remote files missing locally must be 0.

Caution: several projects are huge (Works_Done 381G, LLMs 193G, Exposure 146G).
Check `df -h` before pulling anything large; the disk is at 83%.

## Sending results

- Ad-hoc gist to Arka + Om (default): `~/ops/bin/send-gist "Subject" <file>`
  (or pipe text on stdin). Recipients are allowlist-enforced downstream; it
  refuses secrets/config paths.
- Ops alert to Arka only: `~/ops/bin/alert "Subject" "details"`.
- Never email any other way; never add recipients.

## Machine ops quick reference

- Backups: `~/backups/openclaw/` (nightly 02:30 via `rit-backup.timer`);
  run now: `systemctl --user start rit-backup.service`
- Weekly report: `openclaw cron run ops-weekly-maintenance`
- Cron: `openclaw cron list` / `openclaw cron runs <id>`
- Gateway: `systemctl --user status openclaw-gateway`;
  logs: `journalctl --user -u openclaw-gateway -n 50`
- Pipeline logs: `~/trade-ops/logs/`, wrapper logs: `~/ops/logs/`

## Local models (offline / free / private) — added 2026-07-23

Two strong local models now run on the 8 GB GPU (both 20–30B MoE, ~3B active,
~24 tok/s with CPU offload — see MODEL_ROUTING.md Tier 1.5):
- `ollama/qwen3-coder:30b` — coding: write/refactor/debug, code review.
- `ollama/gpt-oss:20b` — general reasoning, structured analysis, summaries.
  Also wired as failover #2 (qwen-plus → nemotron-3-ultra → **gpt-oss** → qwen3.5),
  so if the API and cloud are both down a *capable* local model still answers.

Prefer these over paid deepseek / cloud gemma for mid-tier coding & analysis
**when the GPU is FREE** — always gate on `bash ~/.openclaw/workspace/gpu_guard.sh`
first (it now works even while `nvidia-smi` is down; FREE ⇒ go, BUSY ⇒ use
nemotron-cloud or do it yourself). Delegate with an explicit model, e.g. spawn a
subagent with `model=ollama/qwen3-coder:30b`. Quick one-shot from the shell:
`openclaw agent --local --model ollama/qwen3-coder:30b --message "<task>"`
(add `--deliver` only if the reply should go to a channel).

Note: `nvidia-smi` currently fails ("NVML Driver/library version mismatch") after
a driver update — GPU compute is FINE (Ollama runs on it), only the monitoring
tool is broken; a reboot fixes it. `nvtop` still works.

## Other projects (the same pattern extends everywhere)

Any project under ~/Desktop: read and run freely (tests, scripts, notebooks,
builds), same rails — no git mutations, show diffs before touching tracked
files, secrets stay put. When a new project gets a routine (a runner, common
commands, a "what Arka says → what you run" map), add a section for it HERE so
the ability persists across sessions.
