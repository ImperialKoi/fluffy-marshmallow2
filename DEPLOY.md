# DEPLOY.md — Running the always-on service on free-tier EC2 (Step 2)

Deploy the Step-1 service (`live_service.py`) to a small EC2 box under **systemd**, with
secrets in **SSM Parameter Store** (loaded via the instance **IAM role** — no static AWS
keys, no plaintext `.env`). Python venv + systemd (no Docker — lighter on 1 GB).

> **You run the AWS commands.** This repo ships the artifacts; the steps below are exact
> commands for *your* account. Replace `REGION`, `ACCOUNT_ID`, `REPO_URL`, etc. Nothing
> here was run against your account.

---

## ⚠️ 0. BEFORE you launch (cost + safety — do this first)

1. **Verify your free-tier status.** Free-tier shape depends on account age:
   - Accounts older than the **12-month** intro period: t3.micro is **not** free.
   - Newer accounts use a **credit-based** free tier (e.g. $100–$200 credits) — different rules.
   Check **Billing → Free Tier** in the console and confirm *before* relying on "free".
2. **t3.micro only** (1 GB, free-tier eligible). **t3.small is NOT free-tier.** The service
   is built to stay under ~1 GB (it never loads the backtest CSV — Step 1).
3. **Public IPv4 costs ~$3.60/mo** now, even on free-tier instances, whenever a public IPv4
   is attached. Budget for it (or explore IPv6-only egress, advanced).
4. **Set a billing alarm at $5 BEFORE launch** (CloudWatch EstimatedCharges lives in
   `us-east-1`; enable "Receive Billing Alerts" in Billing preferences first):
   ```bash
   aws sns create-topic --name tradingbot-billing --region us-east-1
   aws sns subscribe --topic-arn arn:aws:sns:us-east-1:ACCOUNT_ID:tradingbot-billing \
     --protocol email --notification-endpoint you@example.com --region us-east-1
   # confirm the email subscription, then:
   aws cloudwatch put-metric-alarm --region us-east-1 \
     --alarm-name tradingbot-billing-5usd \
     --namespace "AWS/Billing" --metric-name EstimatedCharges \
     --dimensions Name=Currency,Value=USD \
     --statistic Maximum --period 21600 --evaluation-periods 1 \
     --threshold 5 --comparison-operator GreaterThanThreshold \
     --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:tradingbot-billing
   ```
   (AWS **Budgets** with a $5 monthly budget is an equally good alternative.)

---

## 1. Access model (pick one)

- **Recommended: SSM Session Manager** — shell access with **no inbound ports**, no SSH
  key to manage. The instance security group has **no inbound rules at all** (outbound
  only). Requires the `AmazonSSMManagedInstanceCore` managed policy on the role (below)
  and the SSM agent (preinstalled on AL2023 and Ubuntu LTS AMIs). This runbook assumes this.
- **Alternative: SSH** — open inbound TCP 22 **to your IP only** (`x.x.x.x/32`), attach an
  EC2 key pair. Less ideal (a public attack surface, a key to protect).

Either way: **outbound-only** otherwise, and keep the OS patched (`dnf upgrade` / `apt upgrade`).

---

## 2. Create the IAM role (instance profile)

```bash
# trust + permissions come from the repo's deploy/ JSON
aws iam create-role --role-name tradingbot-ec2 \
  --assume-role-policy-document file://deploy/iam-trust-policy.json

# least-privilege secret read (edit REGION/ACCOUNT_ID/KMS_KEY_ID in the file first)
aws iam put-role-policy --role-name tradingbot-ec2 \
  --policy-name tradingbot-ssm-read \
  --policy-document file://deploy/iam-policy.json

# Session Manager access (managed policy)
aws iam attach-role-policy --role-name tradingbot-ec2 \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# instance profile
aws iam create-instance-profile --instance-profile-name tradingbot-ec2
aws iam add-role-to-instance-profile \
  --instance-profile-name tradingbot-ec2 --role-name tradingbot-ec2
```
> If your SSM SecureStrings use the **default `alias/aws/ssm`** key, you can drop the KMS
> statement from `deploy/iam-policy.json`. If you use a **customer-managed key**, set
> `KMS_KEY_ID` to that key.

---

## 3. Launch the t3.micro

Console is fine; CLI example (Amazon Linux 2023, no inbound SG, instance profile attached):
```bash
# an outbound-only security group (no inbound rules)
aws ec2 create-security-group --group-name tradingbot-sg \
  --description "tradingbot outbound only" --vpc-id vpc-XXXX
# (do NOT add inbound rules if using Session Manager)

aws ec2 run-instances \
  --image-id ami-XXXXXXXX \                 # latest AL2023 x86_64 in your region
  --instance-type t3.micro \
  --iam-instance-profile Name=tradingbot-ec2 \
  --security-group-ids sg-XXXX \
  --subnet-id subnet-XXXX \                 # a public subnet
  --associate-public-ip-address \           # needed for outbound (note the IPv4 charge)
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tradingbot}]'
```

---

## 4. Connect (Session Manager)

```bash
aws ssm start-session --target i-XXXXXXXX --region REGION
```
(Or `ssh ec2-user@<ip>` if you chose the SSH model.)

---

## 5. Bootstrap the box

```bash
sudo dnf -y install git              # AL2023 (Ubuntu: sudo apt-get -y install git)
git clone REPO_URL /tmp/tb && cd /tmp/tb        # or clone anywhere you like
REPO_URL=REPO_URL bash deploy/bootstrap.sh
```
`bootstrap.sh` updates the OS, creates the non-root `tradingbot` user, clones to
`/home/tradingbot/trading_bot`, builds the venv, installs **pinned** deps
(`requirements-deploy.txt`), and installs the systemd unit.

Then set your region in the unit:
```bash
sudo sed -i 's/AWS_DEFAULT_REGION=us-east-1/AWS_DEFAULT_REGION=REGION/' \
  /etc/systemd/system/tradingbot.service
sudo systemctl daemon-reload
```

---

## 6. Store the secrets in SSM (SecureString)

Run these once (values are encrypted at rest; the instance role decrypts them at startup):
```bash
aws ssm put-parameter --name /tradingbot/ALPACA_KEY          --type SecureString --value 'PK...'        --region REGION
aws ssm put-parameter --name /tradingbot/ALPACA_SECRET       --type SecureString --value '...'          --region REGION
aws ssm put-parameter --name /tradingbot/GEMINI_API_KEY      --type SecureString --value '...'          --region REGION
aws ssm put-parameter --name /tradingbot/SEC_EDGAR_USER_AGENT --type SecureString --value 'bot you@example.com' --region REGION
# update later with --overwrite
```
Use **paper** Alpaca keys (the service runs `--mode paper`). Parameter leaf names map 1:1
to env vars (`service/secrets.py`).

**Smoke-test the SSM path with dummy params** (proves the role + loader work) before the
real run:
```bash
sudo -u tradingbot TRADINGBOT_USE_SSM=1 AWS_DEFAULT_REGION=REGION \
  /home/tradingbot/trading_bot/.venv/bin/python \
  -c "from service.secrets import load_secrets_from_ssm as L; print(sorted(L()))"
# -> prints the env vars loaded from SSM, e.g. ['ALPACA_KEY','ALPACA_SECRET',...]
```

---

## 7. Enable + start (on boot, paper mode)

```bash
sudo systemctl enable --now tradingbot
```

---

## 8. Verify

```bash
systemctl status tradingbot --no-pager
journalctl -u tradingbot -n 50 --no-pager      # recent logs
journalctl -u tradingbot -f                    # follow live
```
You should see the startup banner, `loaded N secret(s) from SSM`, the clock status, and
the two cadences (`[FAST ...]`, `[SLOW ...]`) firing during market hours.

**Reboot survival** (it's `enable`d + `WantedBy=multi-user.target`):
```bash
sudo reboot
# reconnect, then:
systemctl is-active tradingbot      # -> active
```

**Crash survival** (`Restart=always`, `RestartSec=10`):
```bash
sudo systemctl kill -s SIGKILL tradingbot
sleep 12 && systemctl status tradingbot --no-pager | grep -E "Active:|Main PID:"
# -> Active: active (running) again, new PID
```

---

## 9. State persistence, EBS billing & backup

Persistent state lives on the instance's **EBS** volume under
`/home/tradingbot/trading_bot/results/`:
- `portfolio/inventory.db` (metadata + snapshots), `portfolio/history.csv`
- `ai/decisions.csv`, `ai/equity.csv`, `ai/portfolio_state.json` (kill-switch high-water)
- `ai_audit/*.jsonl` (every LLM prompt/response)

**EBS keeps billing after instance *stop*, and even after *termination* if the volume isn't
deleted.** A t3.micro root volume (≤30 GB gp3) is within free-tier, but delete unused
volumes/snapshots to avoid charges.

Back it up (a single instance is a single point of failure):
```bash
# point-in-time EBS snapshot
aws ec2 create-snapshot --volume-id vol-XXXX --description "tradingbot state" --region REGION
```
**Optional S3 sync (free 5 GB):** add `s3:PutObject`/`s3:ListBucket` for your bucket to
`deploy/iam-policy.json`, set `BUCKET` in `deploy/tradingbot-backup.service`, then:
```bash
sudo cp deploy/tradingbot-backup.{service,timer} /etc/systemd/system/
sudo systemctl enable --now tradingbot-backup.timer    # daily sync of results/ -> S3
```

---

## 10. Update / redeploy

```bash
cd /home/tradingbot/trading_bot && bash deploy/deploy.sh
# git pull -> refresh venv/deps -> systemctl restart -> status
```

---

## 11. STOP / TEARDOWN (avoid charges)

```bash
# pause (stops compute + IPv4 charge; EBS still bills):
aws ec2 stop-instances --instance-ids i-XXXX --region REGION

# full teardown:
aws ec2 terminate-instances --instance-ids i-XXXX --region REGION
aws ec2 delete-volume --volume-id vol-XXXX --region REGION          # if not auto-deleted
aws ssm delete-parameters --names /tradingbot/ALPACA_KEY /tradingbot/ALPACA_SECRET \
  /tradingbot/GEMINI_API_KEY /tradingbot/SEC_EDGAR_USER_AGENT --region REGION
aws iam remove-role-from-instance-profile --instance-profile-name tradingbot-ec2 --role-name tradingbot-ec2
aws iam delete-instance-profile --instance-profile-name tradingbot-ec2
aws iam delete-role-policy --role-name tradingbot-ec2 --policy-name tradingbot-ssm-read
aws iam detach-role-policy --role-name tradingbot-ec2 --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam delete-role --role-name tradingbot-ec2
```
**Important:** stopping/terminating the instance does **not** cancel your **broker** orders.
Resting protective stop orders stay live at Alpaca and your paper positions persist — flatten
in the Alpaca dashboard if you want a clean slate.

---

## Optional: Docker (alternative, not recommended on 1 GB)

systemd + venv is lighter. If you prefer a container: base `python:3.12-slim`,
`pip install -r requirements-deploy.txt`, run `python live_service.py --mode paper`, pass
`AWS_REGION`/`TRADINGBOT_USE_SSM=1`, and grant the **task/instance role** the same SSM+KMS
permissions. Expect higher memory overhead than bare venv.

---

## Honest notes

- **Single instance = single point of failure.** Acceptable for **paper** testing. If the
  box dies mid-session, the **resting broker stop orders still protect open positions** at
  the exchange — that's the whole point of Step 1's protective-order model.
- **`--mode live` is intentionally not run under systemd**: live requires an interactive
  typed confirmation and must not run unattended. This deployment is paper-only by design.
- **Verify free-tier eligibility and set the $5 billing alarm before launching** (§0). The
  public-IPv4 charge (~$3.60/mo) applies regardless of free-tier compute.
- The drawdown kill switch persists in `results/ai/portfolio_state.json`; once tripped the
  service goes flat and stops opening. Clear it with `--reset-state` only deliberately.
