#!/bin/bash
# Daily AI portfolio DRY run (compute + log + audit, NO orders).
#
# Installed as a weekday cron job to accumulate the forward-test record (scores,
# decisions, benchmark equity) before any paper trading. It NEVER places orders.
#
# cron runs with a minimal environment, so we cd into the project, load .env (for
# ALPACA_* and GEMINI_API_KEY), set the EDGAR User-Agent, and use an absolute python.
# Edit PROJECT_DIR / PYTHON / EDGAR_UA below if your paths or contact email differ.

set -euo pipefail

PROJECT_DIR="/Users/daniel/Downloads/trading_bot"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
EDGAR_UA="trading-bot imperialkoi9@gmail.com"

cd "$PROJECT_DIR"

# load credentials from .env if present (never hardcoded here)
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
export SEC_EDGAR_USER_AGENT="$EDGAR_UA"

mkdir -p results/ai
LOG="results/ai/dry_run_cron.log"
echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) daily dry run =====" >> "$LOG"
"$PYTHON" live_portfolio.py --mode dry --once --quiet >> "$LOG" 2>&1
echo "" >> "$LOG"
