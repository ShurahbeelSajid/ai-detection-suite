"""Dependency-free latest-frame worker primitives.

The worker owns and invokes a processor from exactly one thread.  While one
item is running, callers may retain at most one pending item; a newer
submission replaces that pending item.  This keeps latency and memory bounded
for real-time inference workloads.
"""

from __future__ import annotations

import threading
import time
import traceback as traceback_module
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class WorkerClosedError(RuntimeError):
    """Raised when work is submitted after shutdown or fatal termination."""


@dataclass(frozen=True, slots=True)
class WorkItem(Generic[InputT]):
    """A payload and the capture/submission metadata that travels with it."""

    sequence: int
    frame_id: int
    captured_at: float
    submitted_at: float
    payload: InputT


@dataclass(frozen=True, slots=True)
class CapturedError:
    """An exception snapshot that does not retain traceback frame objects."""

    module: str
    type_name: str
    message: str
    traceback: str

    @property
    def qualified_type(self) -> str:
        return f"{self.module}.{self.type_name}"


@dataclass(frozen=True, slots=True)
class WorkResult(Generic[OutputT]):
    """The value or captured error produced for one submitted frame."""

    sequence: int
    frame_id: int
    captured_at: float
    submitted_at: float
    started_at: float
    finished_at: float
    value: OutputT | None
    error: CapturedError | None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def queue_seconds(self) -> float:
        return max(0.0, self.started_at - self.submitted_at)

    @property
    def processing_seconds(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    @property
    def age_at_completion_seconds(self) -> float:
        return max(0.0, self.finished_at - self.captured_at)


@dataclass(frozen=True, slots=True)
class Submission(Generic[InputT]):
    """Submission metadata, including any pending frame it superseded."""

    item: WorkItem[InputT]
    replaced_sequence: int | None = None
    replaced_frame_id: int | None = None


@dataclass(frozen=True, slots=True)
class WorkerStats:
    submitted: int
    completed: int
    failed: int
    dropped: int
    running_frame_id: int | None
    pending_frame_id: int | None
    closed: bool
    terminated: bool


def _capture_exception(error: Exception) -> CapturedError:
    """Capture useful error text without retaining ``error.__traceback__``."""

    error_type = type(error)
    try:
        message = str(error)
    except Exception as formatting_error:
        message = f"<exception message unavailable: {type(formatting_error).__name__}>"

    try:
        rendered_traceback = "".join(
            traceback_module.format_exception(error_type, error, error.__traceback__)
        )
    except Exception:
        rendered_traceback = f"{error_type.__module__}.{error_type.__name__}: {message}\n"

    return CapturedError(
        module=error_type.__module__,
        type_name=error_type.__name__,
        message=message,
        traceback=rendered_traceback,
    )


class LatestFrameWorker(Generic[InputT, OutputT]):
    """Run a processor serially while retaining only the latest pending item.

    Normal ``Exception`` instances are represented in :class:`WorkResult` and
    the worker continues.  ``BaseException`` subclasses are deliberately not
    caught: the thread's normal ``threading.excepthook`` sees them, while the
    worker's ``finally`` block still closes its state and wakes waiters.
    """

    def __init__(
        self,
        processor: Callable[[InputT], OutputT],
        *,
        name: str = "latest-frame-worker",
        clock: Callable[[], float] = time.monotonic,
        autostart: bool = True,
        daemon: bool = False,
    ) -> None:
        if not callable(processor):
            raise TypeError("processor must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._processor = processor
        self._name = name
        self._clock = clock
        self._daemon = daemon

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._terminated = False
        self._drain_on_shutdown = False

        self._next_sequence = 0
        self._pending: WorkItem[InputT] | None = None
        self._running: WorkItem[InputT] | None = None
        self._latest_result: WorkResult[OutputT] | None = None

        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._dropped = 0

        if autostart:
            self.start()

    def start(self) -> LatestFrameWorker[InputT, OutputT]:
        """Start the owner thread; repeated calls are harmless."""

        with self._condition:
            if self._closed:
                raise WorkerClosedError("cannot start a closed worker")
            if self._started:
                return self

            thread = threading.Thread(
                target=self._run,
                name=self._name,
                daemon=self._daemon,
            )
            self._thread = thread
            self._started = True
            try:
                thread.start()
            except Exception:
                self._thread = None
                self._started = False
                self._closed = True
                self._terminated = True
                self._stop_event.set()
                self._condition.notify_all()
                raise
        return self

    def submit(
        self,
        payload: InputT,
        *,
        frame_id: int,
        captured_at: float | None = None,
    ) -> Submission[InputT]:
        """Submit a frame, replacing (but never interrupting) pending work."""

        submitted_at = self._clock()
        if captured_at is None:
            captured_at = submitted_at

        with self._condition:
            if self._closed or self._terminated or self._stop_event.is_set():
                raise WorkerClosedError("worker is closed")

            sequence = self._next_sequence
            self._next_sequence += 1
            item = WorkItem(
                sequence=sequence,
                frame_id=frame_id,
                captured_at=captured_at,
                submitted_at=submitted_at,
                payload=payload,
            )

            replaced = self._pending
            if replaced is not None:
                self._dropped += 1
            self._pending = item
            self._submitted += 1
            self._condition.notify_all()

            return Submission(
                item=item,
                replaced_sequence=None if replaced is None else replaced.sequence,
                replaced_frame_id=None if replaced is None else replaced.frame_id,
            )

    @property
    def latest_result(self) -> WorkResult[OutputT] | None:
        with self._condition:
            return self._latest_result

    @property
    def stop_event(self) -> threading.Event:
        """The read-only-by-convention shutdown event for cooperative processors."""

        return self._stop_event

    @property
    def stats(self) -> WorkerStats:
        with self._condition:
            return WorkerStats(
                submitted=self._submitted,
                completed=self._completed,
                failed=self._failed,
                dropped=self._dropped,
                running_frame_id=(None if self._running is None else self._running.frame_id),
                pending_frame_id=(None if self._pending is None else self._pending.frame_id),
                closed=self._closed,
                terminated=self._terminated,
            )

    def wait_for_result(
        self,
        *,
        after_sequence: int = -1,
        timeout: float | None = None,
    ) -> WorkResult[OutputT] | None:
        """Wait for a result newer than ``after_sequence``.

        ``None`` means the timeout elapsed or the worker terminated without a
        newer result.
        """

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                result = self._latest_result
                if result is not None and result.sequence > after_sequence:
                    return result
                if self._terminated:
                    return None

                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                self._condition.wait(remaining)

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Wait until neither running nor pending work remains."""

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._running is not None or self._pending is not None:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                self._condition.wait(remaining)
            return True

    def shutdown(
        self,
        *,
        wait: bool = True,
        timeout: float | None = None,
        cancel_pending: bool = True,
    ) -> bool:
        """Stop accepting work and optionally wait for the owner thread.

        Running work is never killed.  By default pending work is discarded;
        set ``cancel_pending=False`` to drain the single pending item.
        """

        with self._condition:
            self._closed = True
            self._drain_on_shutdown = not cancel_pending
            self._stop_event.set()
            if cancel_pending and self._pending is not None:
                self._pending = None
                self._dropped += 1
            self._condition.notify_all()

            if not self._started:
                self._terminated = True

        if not wait:
            return self._terminated
        return self.join(timeout)

    close = shutdown

    def join(self, timeout: float | None = None) -> bool:
        """Join the owner thread and return whether it has terminated."""

        with self._condition:
            thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            raise RuntimeError("a worker cannot join its own owner thread")
        thread.join(timeout)
        return not thread.is_alive()

    def __enter__(self) -> LatestFrameWorker[InputT, OutputT]:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.shutdown(wait=True)

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._stop_event.is_set():
                        self._condition.wait()

                    if self._stop_event.is_set() and (
                        self._pending is None or not self._drain_on_shutdown
                    ):
                        return

                    item = self._pending
                    if item is None:
                        continue
                    self._pending = None
                    self._running = item
                    self._condition.notify_all()

                started_at = self._clock()
                value: OutputT | None = None
                error: CapturedError | None = None
                try:
                    value = self._processor(item.payload)
                except Exception as processor_error:
                    error = _capture_exception(processor_error)
                finished_at = self._clock()

                result = WorkResult(
                    sequence=item.sequence,
                    frame_id=item.frame_id,
                    captured_at=item.captured_at,
                    submitted_at=item.submitted_at,
                    started_at=started_at,
                    finished_at=finished_at,
                    value=value,
                    error=error,
                )

                with self._condition:
                    self._latest_result = result
                    self._running = None
                    self._completed += 1
                    if error is not None:
                        self._failed += 1
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._running = None
                if self._pending is not None:
                    self._pending = None
                    self._dropped += 1
                self._closed = True
                self._terminated = True
                self._stop_event.set()
                self._condition.notify_all()
