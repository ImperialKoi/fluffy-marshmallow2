"""
Asyncio supervisor: one process coordinating the stream consumer and the two timed
tasks (fast scan + slow AI rebalance), plus periodic inventory sync.

Guarantees:
  * Each periodic task runs on its own cadence. An exception in one iteration is logged
    and swallowed — it never kills the task, the other tasks, or the process.
  * Market-hours gating: the fast and slow tasks only execute when the clock says open
    (the sync task runs regardless). When closed, tasks idle cheaply (just sleeping).
  * Blocking work (REST calls, LLM) runs in a thread executor so the event loop stays
    responsive.
  * Graceful shutdown: stop the timed tasks and the stream, but LEAVE resting protective
    orders in place at the exchange; do not open anything on the way out.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

log = logging.getLogger("service.supervisor")


class Supervisor:
    def __init__(self, *, clock, scan_interval, rebalance_interval, sync_interval,
                 fast_fn, slow_fn, sync_fn, stream=None, run_blocking=True,
                 discovery_fn=None, discovery_interval=None):
        self.clock = clock
        self.scan_interval = scan_interval
        self.rebalance_interval = rebalance_interval
        self.sync_interval = sync_interval
        self.fast_fn = fast_fn          # callable (sync or coroutine)
        self.slow_fn = slow_fn
        self.sync_fn = sync_fn
        # optional daily universe-discovery task (separate, slower cadence)
        self.discovery_fn = discovery_fn
        self.discovery_interval = discovery_interval
        self.stream = stream
        self.run_blocking = run_blocking
        self._stop = asyncio.Event()
        self._closed_logged = False

    def request_stop(self):
        self._stop.set()

    async def _call(self, fn):
        """Run a task fn, awaiting coroutines and offloading blocking fns to a DAEMON
        thread. Daemon (vs the default executor's non-daemon workers) means a slow
        network call in flight at shutdown is abandoned cleanly instead of wedging
        process exit via the executor's atexit join."""
        if asyncio.iscoroutinefunction(fn):
            await fn()
        elif not self.run_blocking:
            fn()
        else:
            loop = asyncio.get_event_loop()
            fut = loop.create_future()

            def _runner():
                try:
                    r = fn()
                    loop.call_soon_threadsafe(lambda r=r: fut.done() or fut.set_result(r))
                except Exception as e:  # noqa: BLE001
                    # bind `e` as a default arg: `except ... as e` UNBINDS e when the
                    # block exits, but this callback runs later on the loop thread.
                    loop.call_soon_threadsafe(lambda e=e: fut.done() or fut.set_exception(e))

            threading.Thread(target=_runner, daemon=True, name="svc-task").start()
            await fut

    async def _periodic(self, name, fn, interval, gate_market=True):
        log.info("task '%s' started (every %.0fs, market-gated=%s)", name, interval, gate_market)
        while not self._stop.is_set():
            start = time.monotonic()
            try:
                if (not gate_market) or self.clock.is_open():
                    self._closed_logged = False
                    await self._call(fn)
                elif not self._closed_logged:
                    log.info("market closed -> '%s' (and peers) idling", name)
                    self._closed_logged = True
            except Exception:  # noqa: BLE001 — isolate: log full trace, keep looping
                log.exception("task '%s' raised; continuing", name)
            # sleep the remainder of the interval, but wake immediately on shutdown
            elapsed = time.monotonic() - start
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=max(0.0, interval - elapsed))
            except asyncio.TimeoutError:
                pass
        log.info("task '%s' stopped", name)

    async def run(self):
        if self.stream is not None:
            try:
                self.stream.start()
            except Exception as e:  # noqa: BLE001
                log.warning("stream failed to start (%s); continuing without it", e)
        tasks = [
            asyncio.create_task(self._periodic("fast_scan", self.fast_fn, self.scan_interval)),
            asyncio.create_task(self._periodic("ai_rebalance", self.slow_fn, self.rebalance_interval)),
            asyncio.create_task(self._periodic("inventory_sync", self.sync_fn,
                                               self.sync_interval, gate_market=False)),
        ]
        if self.discovery_fn is not None and self.discovery_interval:
            tasks.append(asyncio.create_task(
                self._periodic("universe_discovery", self.discovery_fn,
                               self.discovery_interval)))
        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.stream is not None:
                self.stream.stop()
            log.info("supervisor shut down; resting protective orders left in place, "
                     "no new positions opened")

    async def run_for(self, seconds: float):
        """Run, then auto-stop after `seconds` (used for the dry demo / tests)."""
        async def _timer():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=seconds)
            except asyncio.TimeoutError:
                self.request_stop()
        await asyncio.gather(self.run(), _timer())
