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
- "picks" / "good picks" / "the book" → **the alpha engine v2 book** (since 2026-09-21):
  `~/ops/bin/ts-run alpha picks --top 20 --compact --prev ~/trade-ops/portfolio/books/alpha_v2.json`
  (drop `--compact` for the full table with calibrated expected returns, 80%
  bands and the model drivers per name; `--budget 10000` prints share counts).
  It is a 1000-name point-in-time panel → 5/21/63-day rank forecasters blended
  by their realised IC from the forecast ledger → gated (≥$5, ≥$20M/day, vol
  ≤110%), ≤5 per sector, inverse-vol weighted, 8% cap, 18% vol target, traded
  35% of the way toward target per month against the `alpha_v2` paper book.
  Explanations: docs/COMPENDIUM.md §13a. Older engines are diagnostics only:
  `picks_v2.py` (the 2026-09 gated version) and raw `ts picks` (prefers
  volatile illiquid names — never use it for the brief).
- "is the model any good / forecast skill / how accurate" → `~/ops/bin/ts-run alpha status`
  Prints the realised rank-IC / hit rate / decile spread / band coverage of
  every forecast the engine has made (live rows since 2026-09-21, backtest rows
  2003→), the calibrator's horizon weights, and the rolling 63-date IC. A
  rolling IC that sits below zero for weeks means the signal has decayed —
  say so in the brief; the calibrator already down-weights that horizon.
- "what data do we have / is the data fresh / data state" → `~/ops/bin/ts-run data status`
  One table: every dataset (prices, minute bars, fundamentals, news, short data,
  FRED, panel, ledger) with rows, tickers, span, sessions behind and ok/stale,
  plus crawler liveness, parked endpoint families and calls/day, and disk.
  Anything red = say so first in the reply.
- "try a model change / is X better" → `~/ops/bin/ts-run --timeout 7200 alpha experiment --variants base,<v>`
  Runs variants on identical causal folds; adoption needs paired-t ≥ 2 on IC
  OR a book-level excess with CI > 0 and ≥70% years won (see experiment.py).
  Never change the production model without that report.
- "what regime are we in / is this like 2008 / oil shock / AI bubble" → `~/ops/bin/ts-run alpha regime`
  Oil, vol, credit, rates, AI-concentration state (z-scores), the closest
  historical episodes (Gulf War, dot-com, 2008, 2022, tariff shock 2025 …)
  with similarity scores, and how the forecast signal and the book did in
  those backgrounds. The daily forecast already blends an analog-weighted
  calibration (50/50 with the trailing 3y) — so "recalibrated for the regime"
  is automatic; quote the analog-vs-unconditional IC when asked how much to
  trust the picks right now.
- "backtest the alpha engine" → `~/ops/bin/ts-run --timeout 3600 alpha backtest --extend`
  (weekly via cron `trade-weekly-retrain`; the report is
  `~/Desktop/Trade-system_v1/reports/alpha/backtest.md`). Read the
  "Read this before believing any of it" section before quoting numbers.
- "how are the paper books / which strategy is winning" →
  `… ~/Desktop/Trade-system_v1/ops/portfolio/portfolios.py --report`
  Seven competing $10k books. Five seeded 2026-09-08 (spy_benchmark, ml_raw,
  ml_v2, momentum, blend, monthly full rebalance), **ml_v2_gp** seeded
  2026-09-16 (research winner xgb63|gp35: 63d model, top-20, 10% cap,
  Garleanu-Pedersen partial trading at 0.35) and **alpha_v2** seeded
  2026-09-21 (the alpha engine book, same GP execution — the two differ only
  in the signal). Ignore the ranking until ~60 sessions.
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
  actions → universe depth (every article, full corp-action/short history) →
  universe 1-MINUTE bars (2y, `bronze/massive/bars_minute/<T>.parquet`) →
  extended top-1000 depth → whole-market depth → QA — and it shares the 5/min
  limiter with every other Massive call, so never start a second crawler or a
  manual `ts massive backfill` while it runs (they only queue behind each other).
  Two keys are set (MASSIVE_API_KEY, MASSIVE_API_KEY2 → 10 req/min).
  The 05:15 pipeline's `ts massive update` is a no-op when the crawler is current.
- "are the tools healthy / run the tests" → `~/Desktop/Trade-system_v1/ops/run_tests.sh`
  (52 unit tests; also runs first in the daily pipeline).
- "retrain the model now"         → `~/ops/bin/run-with-alert trade-weekly-retrain ~/Desktop/Trade-system_v1/ops/weekly_retrain.sh`
  (heavy, 1–3 h; normally auto via cron `trade-weekly-retrain`, Sat 03:00 ET.
  Retrains the 14-model ensemble with the purged walk-forward config from
  ~/trade-ops/research/, deploys to reports/models — gitignored, safe)

Long jobs: warn that it takes a while, run it, report back when done — don't
block the conversation.

## Portfolio update from a Fidelity screenshot

Arka sends a Fidelity "Positions" screenshot on Telegram with a caption
containing **"update portfolio"**. The keyword is REQUIRED — never run this on an
uncaptioned or unrelated image (it overwrites his portfolio state). Then:

1. OCR the image yourself — you (qwen3.7-flash) can read images directly (older note: flash couldn't see
   images). Produce STRICT minified JSON and nothing else:
   `{"holdings":[{"symbol","quantity","last_price","current_value","average_cost","total_gain_loss","total_gain_loss_pct"}],"cash":[{"symbol","name","value"}]}`
   Money-market rows (SPAXX/FDRXX/"Cash") go in `cash`. Numbers plain — no `$` `,`
   or `%`. Use `null` where a field isn't legible. Never invent a ticker or row.
2. Write that JSON to `~/trade-ops/private/inbound_holdings.json` (write tool).
3. Run:
   `~/Desktop/Trade-system_v1/ops/bin/portfolio-from-screenshot --from-json ~/trade-ops/private/inbound_holdings.json --apply`
   (Alternatively, if the image is saved to a real path, run
   `~/Desktop/Trade-system_v1/ops/bin/portfolio-from-screenshot <path> --apply` — it OCRs via
   qwen3.7-flash via the Qwen API itself. Drop `--apply` for a dry-run diff first if unsure.)
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

## Models — Qwen API only (2026-09-26)

Everything runs on `qwen/qwen3.7-flash` (text + tools + images), fallback
`qwen/qwen3.8-flash`. No Ollama, no local-GPU models, no gpu_guard. Don't
switch to pricier models unless Arka asks. See MODEL_ROUTING.md for prices.

## Other projects (the same pattern extends everywhere)

Any project under ~/Desktop: read and run freely (tests, scripts, notebooks,
builds), same rails — no git mutations, show diffs before touching tracked
files, secrets stay put. When a new project gets a routine (a runner, common
commands, a "what Arka says → what you run" map), add a section for it HERE so
the ability persists across sessions.
