#!/usr/bin/env bash
# First-time setup for the trading-bot service on a fresh EC2 box
# (Amazon Linux 2023 or Ubuntu LTS). Run as a sudo-capable user (ec2-user/ubuntu).
#
#   REPO_URL=https://github.com/<you>/trading_bot.git bash deploy/bootstrap.sh
#
# Creates a non-root 'tradingbot' user, clones the repo, builds the venv, installs
# pinned deps, and installs (but does not start) the systemd unit. Idempotent.
set -euo pipefail

REPO_URL="${REPO_URL:?set REPO_URL to your git remote (https or ssh)}"
APP_USER="${APP_USER:-tradingbot}"
APP_HOME="/home/${APP_USER}"
APP_DIR="${APP_HOME}/trading_bot"

echo "== OS packages =="
if command -v dnf >/dev/null 2>&1; then
  sudo dnf -y update
  sudo dnf -y install git python3 python3-pip
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get -y update && sudo apt-get -y upgrade
  sudo apt-get -y install git python3 python3-pip python3-venv
else
  echo "Unsupported package manager (need dnf or apt-get)"; exit 1
fi

echo "== service user =="
id -u "${APP_USER}" >/dev/null 2>&1 || sudo useradd -m -s /bin/bash "${APP_USER}"

echo "== clone / update repo =="
if [ -d "${APP_DIR}/.git" ]; then
  sudo -u "${APP_USER}" git -C "${APP_DIR}" pull --ff-only
else
  sudo -u "${APP_USER}" git clone "${REPO_URL}" "${APP_DIR}"
fi

echo "== venv + pinned deps =="
sudo -u "${APP_USER}" python3 -m venv "${APP_DIR}/.venv"
sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install --upgrade pip
sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements-deploy.txt"
sudo -u "${APP_USER}" mkdir -p "${APP_DIR}/results"

echo "== install systemd unit =="
sudo cp "${APP_DIR}/deploy/tradingbot.service" /etc/systemd/system/tradingbot.service
sudo systemctl daemon-reload

cat <<EOF

Bootstrap done. NEXT:
  1) Edit AWS_DEFAULT_REGION in /etc/systemd/system/tradingbot.service
  2) Store secrets in SSM (see DEPLOY.md), confirm the instance IAM role is attached
  3) Enable + start:   sudo systemctl enable --now tradingbot
  4) Verify:           systemctl status tradingbot ; journalctl -u tradingbot -f
EOF
