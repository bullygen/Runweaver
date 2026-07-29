"""Signal handling layered on cooperative cancellation."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from types import FrameType

from runweaver.execution.context import CancellationToken


class SignalController:
    """First signal requests a safe point; a repeated signal terminates hard."""

    def __init__(self, token: CancellationToken, *, on_first: Callable[[], None] | None = None) -> None:
        self.token = token
        self.on_first = on_first
        self._count = 0
        self._previous: dict[int, object] = {}

    def __enter__(self) -> SignalController:
        if threading.current_thread() is not threading.main_thread():
            return self
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        self._count += 1
        if self._count == 1:
            self.token.cancel(f"signal {signal.Signals(signum).name}")
            if self.on_first:
                self.on_first()
            return
        raise KeyboardInterrupt(f"repeated signal {signal.Signals(signum).name}")
