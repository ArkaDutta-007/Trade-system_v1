#!/usr/bin/env bash
# "Integrate all models, retrain, backtest to the max, deploy" (2026-09-17).
#
# Sequential on purpose: the sequence models (lstm/gru/rnn), the cross-attention
# alpha and the PPO policy all want the same 8 GB GPU. Running them in parallel
# would thrash it and make the timings meaningless.
#
#   STAGE 1  ts train-forecast --all   -> models_store/ (TRACKED; this is the
#            production forecaster every live pick uses). All 4 horizons, all 7
#            families (4 tabular + 3 sequence) competing head-to-head under the
#            same purged+embargoed CV, ranked by ICIR, leakage-gated.
#   STAGE 2  ts wf-backtest            -> strictly-causal walk-forward over ALL
#            9 alphas incl. both attention variants and the RL execution policy.
#            Nothing here is scored by a model that saw the future.
#   STAGE 3  deploy: refit the live ensemble + reseed picks/books from whatever
#            stage 2 says actually won.
set -uo pipefail
REPO="${TS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPS="${TS_OPS:-$HOME/trade-ops}"
CODE="$REPO/ops"
LOG="$OPS/logs/full-retrain-$(date +%F).log"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
source venv/bin/activate

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; echo "[$(date -Is)] $*" >> "$LOG"; }
stage() {  # stage <name> <timeout> <cmd...>
  local n="$1" t="$2"; shift 2
  say "START $n"
  local t0=$SECONDS
  if timeout "$t" "$@" >> "$LOG" 2>&1; then
    say "OK    $n  ($(( (SECONDS-t0)/60 ))m)"
  else
    say "FAIL  $n  rc=$? ($(( (SECONDS-t0)/60 ))m) — see $LOG"
    return 1
  fi
}

say "=== full retrain+backtest start ==="
say "gpu: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"

# snapshot the current production metrics so the deploy decision is evidence-based
python3 - <<'PY' >> "$LOG" 2>&1
import json,pathlib
out={}
for h in ("21d","63d","126d","252d"):
    p=pathlib.Path(f"models_store/forecast/{h}/metrics.json")
    if p.exists():
        d=json.loads(p.read_text()); b=d.get("best_model")
        out[h]={"best":b,**{k:d["per_model"][b][k] for k in ("ic_mean","icir") if b in d.get("per_model",{})}}
pathlib.Path("reports/research").mkdir(parents=True,exist_ok=True)
pathlib.Path("reports/research/metrics_before.json").write_text(json.dumps(out,indent=1))
print("BEFORE:",json.dumps(out))
PY

# STAGE 1 — retrain every family on every horizon (6h cap)
stage "1/3 train-forecast --all (4 horizons x 7 families, purged CV, leakage-gated)" 21600 \
  ts train-forecast --all --neutralize --universe-weight 0.5 --priority-universe core \
                    --horizons 21,63,126,252 --n-splits 15

# STAGE 2 — strictly-causal walk-forward over every alpha (4h cap)
stage "2/3 wf-backtest ALL alphas (causal, per-name costs, DSR/PBO/SPA + PPO)" 14400 \
  ts wf-backtest --alphas momentum,random,xgb21,xgb63,xgb252,ens63,xattn63,xattn252,xattn63_e2e \
                 --exec-alpha xgb63 --oos-start 2005-01-01 --stage all --updates 120

# STAGE 3 — refit the live ensemble on the winning config, then reseed
stage "3/3 deploy: live ensemble refit" 14400 python3 "$CODE/research/deploy.py"
stage "3/3 deploy: rebuild predictions + books" 3600 \
  python3 "$CODE/portfolio/portfolios.py" --rebalance --mark

say "=== done ==="
tail -40 "$LOG" | grep -E "^\[" | tail -12
