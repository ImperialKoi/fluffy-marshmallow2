"""
First thing to run after setting your keys. Confirms the connection works and
shows your account, before any trading logic is involved.

    export ALPACA_KEY=your_paper_key
    export ALPACA_SECRET=your_paper_secret
    python check_alpaca.py
"""

from broker.alpaca_broker import AlpacaBroker


def main():
    print("Connecting to Alpaca (paper)...")
    try:
        broker = AlpacaBroker(paper=True)
        acct = broker.account_summary()
    except Exception as e:
        print(f"\n  Connection failed: {e}")
        print("  Check that ALPACA_KEY / ALPACA_SECRET are set to your *paper* keys.")
        return

    print("\n  Connected.")
    for k, v in acct.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            print(f"  {label:14}: ${v:,.2f}")
        else:
            print(f"  {label:14}: {v}")

    if acct["blocked"]:
        print("\n  WARNING: trading is blocked on this account.")
    else:
        print("\n  Ready to trade (paper). Next: python live_trader.py --symbol AAPL --strategy sma")


if __name__ == "__main__":
    main()
