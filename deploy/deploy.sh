#!/usr/bin/env bash
# Redeploy the trading-bot service after a code change:
#   git pull -> refresh venv/deps -> restart systemd -> show status.
# Run on the box as a sudo-capable user:  bash deploy/deploy.sh
set -euo pipefail

APP_DIR="${APP_DIR:-/home/tradingbot/trading_bot}"
APP_USER="${APP_USER:-tradingbot}"
VENV="${APP_DIR}/.venv"
BRANCH="${BRANCH:-main}"

echo "== git pull (${BRANCH}) =="
sudo -u "${APP_USER}" git -C "${APP_DIR}" fetch --all --prune
sudo -u "${APP_USER}" git -C "${APP_DIR}" checkout "${BRANCH}"
sudo -u "${APP_USER}" git -C "${APP_DIR}" pull --ff-only

echo "== refresh venv + pinned deps =="
[ -d "${VENV}" ] || sudo -u "${APP_USER}" python3 -m venv "${VENV}"
sudo -u "${APP_USER}" "${VENV}/bin/pip" install --upgrade pip
sudo -u "${APP_USER}" "${VENV}/bin/pip" install -r "${APP_DIR}/requirements-deploy.txt"

echo "== restart service =="
sudo systemctl restart tradingbot
sleep 2
sudo systemctl --no-pager status tradingbot | head -n 20

echo
echo "Follow logs with:  journalctl -u tradingbot -f"
