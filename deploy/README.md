# deploy/ — EC2 deployment artifacts

See **[DEPLOY.md](../DEPLOY.md)** for the full runbook. Files here:

| File | Purpose |
|---|---|
| `tradingbot.service` | systemd unit — runs `live_service.py --mode paper` as non-root `tradingbot`, `Restart=always`, starts on boot, logs to journald. |
| `iam-trust-policy.json` | Trust policy so EC2 can assume the instance role. |
| `iam-policy.json` | Least-privilege secret read: `ssm:GetParametersByPath`/`GetParameters` on `/tradingbot/*` + `kms:Decrypt` (via SSM). Edit `REGION`/`ACCOUNT_ID`/`KMS_KEY_ID`. |
| `bootstrap.sh` | First-time box setup: OS update, `tradingbot` user, clone, venv, pinned deps, install unit. |
| `deploy.sh` | Redeploy: `git pull` → refresh venv/deps → `systemctl restart` → status. |
| `backup_to_s3.sh` + `tradingbot-backup.{service,timer}` | Optional daily sync of `results/` state to S3 (free 5 GB). |

Notes:
- `iam-policy.json` / `iam-trust-policy.json` are **strict IAM documents** (no comment keys)
  so `aws iam` accepts them as-is.
- If your SSM SecureStrings use the default `alias/aws/ssm` key, you may drop the
  `DecryptSecretsWithCMK` statement; for a customer-managed key, set `KMS_KEY_ID`.
- Attach the AWS-managed `AmazonSSMManagedInstanceCore` to the role too (Session Manager).
- Secrets are loaded at startup by `service/secrets.py` via the instance IAM role
  (`TRADINGBOT_USE_SSM=1`); there is no plaintext `.env` on the box.
