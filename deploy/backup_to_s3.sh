#!/usr/bin/env bash
# OPTIONAL: sync the persistent state (inventory DB, forward-test history, audit logs)
# to S3 so a single instance isn't a single point of data loss. Free tier includes 5 GB.
#
#   BUCKET=my-tradingbot-state bash deploy/backup_to_s3.sh
#
# Requires the instance IAM role to allow s3:PutObject/ListBucket on the bucket
# (see DEPLOY.md). Pair with deploy/tradingbot-backup.timer for a daily run.
set -euo pipefail

APP_DIR="${APP_DIR:-/home/tradingbot/trading_bot}"
BUCKET="${BUCKET:?set BUCKET to your S3 bucket name}"
PREFIX="${PREFIX:-tradingbot}"

# results/ holds: portfolio/inventory.db, portfolio/history.csv, ai/*.csv, ai_audit/*.jsonl
aws s3 sync "${APP_DIR}/results" "s3://${BUCKET}/${PREFIX}/results" \
  --no-progress --only-show-errors
echo "backed up ${APP_DIR}/results -> s3://${BUCKET}/${PREFIX}/results"
