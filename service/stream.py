"""
Market-data stream consumer.

`AlpacaBarStream` subscribes to Alpaca's websocket (IEX feed, free) for minute bars
and writes them into the shared `BarBuffer`. It runs the SDK stream on a background
thread (the SDK manages the asyncio loop and websocket reconnects); a watchdog loop
restarts `run()` if it ever returns or raises, so the consumer self-heals.

Tests do NOT use this class (no live network) — they populate the BarBuffer directly,
which is the only integration point the rest of the service depends on.
"""

from __future__ import annotations

import logging
import os
import threading
import time

log = logging.getLogger("service.stream")


class AlpacaBarStream:
    def __init__(self, buffer, symbols, feed: str = "iex",
                 key: str = None, secret: str = None, reconnect_delay: float = 5.0):
        self.buffer = buffer
        self.symbols = [s.upper() for s in symbols]
        self.feed = feed
        self.key = key or os.environ.get("ALPACA_KEY") or os.environ.get("ALPACA_LIVE_KEY")
        self.secret = secret or os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_LIVE_SECRET")
        self.reconnect_delay = reconnect_delay
        self._stop = threading.Event()
        self._thread = None
        self._stream = None

    def start(self):
        if not (self.key and self.secret):
            raise EnvironmentError("ALPACA_KEY/ALPACA_SECRET required for the data stream.")
        self._thread = threading.Thread(target=self._run_forever, name="alpaca-bar-stream",
                                        daemon=True)
        self._thread.start()
        log.info("bar stream started for %d symbols (feed=%s)", len(self.symbols), self.feed)

    def _run_forever(self):
        from alpaca.data.live import StockDataStream
        from alpaca.data.enums import DataFeed

        feed = DataFeed.IEX if self.feed == "iex" else DataFeed.SIP
        while not self._stop.is_set():
            try:
                self._stream = StockDataStream(self.key, self.secret, feed=feed)

                async def _on_bar(bar):
                    self.buffer.add_bar(bar.symbol, bar.timestamp, bar.open, bar.high,
                                        bar.low, bar.close, bar.volume)

                self._stream.subscribe_bars(_on_bar, *self.symbols)
                log.info("connecting websocket ...")
                self._stream.run()   # blocking; SDK handles in-session reconnects
            except Exception as e:  # noqa: BLE001 — self-heal on any failure
                if self._stop.is_set():
                    break
                log.warning("stream error (%s); reconnecting in %.0fs", e, self.reconnect_delay)
                time.sleep(self.reconnect_delay)
        log.info("bar stream stopped")

    def stop(self):
        self._stop.set()
        try:
            if self._stream is not None:
                self._stream.stop()
        except Exception:  # noqa: BLE001
            pass
