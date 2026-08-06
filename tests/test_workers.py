import threading
import unittest

from ai_detection_suite.workers import LatestFrameWorker, WorkerClosedError


class LatestFrameWorkerTests(unittest.TestCase):
    def test_latest_pending_submission_replaces_intermediate_frame(self):
        first_started = threading.Event()
        release_first = threading.Event()
        processed = []

        def processor(value):
            processed.append(value)
            if value == 1:
                first_started.set()
                self.assertTrue(release_first.wait(2.0))
            return value * 10

        worker = LatestFrameWorker(processor, name="replacement-test")
        try:
            first = worker.submit(1, frame_id=101, captured_at=1.0)
            self.assertTrue(first_started.wait(2.0))
            second = worker.submit(2, frame_id=102, captured_at=2.0)
            third = worker.submit(3, frame_id=103, captured_at=3.0)

            self.assertIsNone(second.replaced_frame_id)
            self.assertEqual(third.replaced_frame_id, 102)
            release_first.set()

            result = worker.wait_for_result(
                after_sequence=first.item.sequence,
                timeout=2.0,
            )
            self.assertIsNotNone(result)
            self.assertEqual(result.frame_id, 103)
            self.assertEqual(result.value, 30)
            self.assertTrue(worker.wait_until_idle(2.0))
            self.assertEqual(processed, [1, 3])
            self.assertEqual(worker.stats.dropped, 1)
        finally:
            release_first.set()
            self.assertTrue(worker.shutdown(timeout=2.0))

    def test_exception_is_captured_and_worker_continues(self):
        def processor(value):
            if value == "bad":
                raise ValueError("broken frame")
            return value.upper()

        worker = LatestFrameWorker(processor, name="error-test")
        try:
            bad = worker.submit("bad", frame_id=7, captured_at=10.0)
            failed = worker.wait_for_result(
                after_sequence=bad.item.sequence - 1,
                timeout=2.0,
            )
            self.assertIsNotNone(failed)
            self.assertFalse(failed.succeeded)
            self.assertEqual(failed.error.type_name, "ValueError")
            self.assertEqual(failed.error.message, "broken frame")
            self.assertIn("ValueError: broken frame", failed.error.traceback)

            good = worker.submit("ok", frame_id=8, captured_at=11.0)
            succeeded = worker.wait_for_result(
                after_sequence=bad.item.sequence,
                timeout=2.0,
            )
            self.assertIsNotNone(succeeded)
            self.assertEqual(succeeded.sequence, good.item.sequence)
            self.assertTrue(succeeded.succeeded)
            self.assertEqual(succeeded.value, "OK")
            self.assertEqual(worker.stats.failed, 1)
        finally:
            self.assertTrue(worker.shutdown(timeout=2.0))

    def test_shutdown_cancels_pending_and_rejects_new_work(self):
        started = threading.Event()
        release = threading.Event()
        processed = []

        def processor(value):
            processed.append(value)
            started.set()
            self.assertTrue(release.wait(2.0))
            return value

        worker = LatestFrameWorker(processor, name="shutdown-test")
        worker.submit("running", frame_id=1)
        self.assertTrue(started.wait(2.0))
        worker.submit("pending", frame_id=2)

        self.assertFalse(worker.shutdown(wait=False, cancel_pending=True))
        with self.assertRaises(WorkerClosedError):
            worker.submit("too-late", frame_id=3)

        release.set()
        self.assertTrue(worker.join(2.0))
        self.assertEqual(processed, ["running"])
        self.assertTrue(worker.stats.terminated)
        self.assertEqual(worker.stats.dropped, 1)

    def test_shutdown_can_drain_the_single_pending_item(self):
        first_started = threading.Event()
        release_first = threading.Event()
        processed = []

        def processor(value):
            processed.append(value)
            if value == 1:
                first_started.set()
                self.assertTrue(release_first.wait(2.0))
            return value

        worker = LatestFrameWorker(processor, name="drain-test")
        try:
            worker.submit(1, frame_id=1)
            self.assertTrue(first_started.wait(2.0))
            worker.submit(2, frame_id=2)
            self.assertFalse(worker.shutdown(wait=False, cancel_pending=False))
            release_first.set()
            self.assertTrue(worker.join(2.0))
            self.assertEqual(processed, [1, 2])
        finally:
            release_first.set()
            worker.shutdown(wait=True, timeout=2.0)

    def test_base_exception_is_not_swallowed(self):
        class FatalProcessorSignal(BaseException):
            pass

        observed = []
        hook_called = threading.Event()
        previous_hook = threading.excepthook

        def exception_hook(arguments):
            observed.append(arguments.exc_value)
            hook_called.set()

        threading.excepthook = exception_hook
        worker = LatestFrameWorker(
            lambda _value: (_ for _ in ()).throw(FatalProcessorSignal("fatal")),
            name="base-exception-test",
        )
        try:
            worker.submit("frame", frame_id=1)
            self.assertTrue(hook_called.wait(2.0))
            self.assertTrue(worker.join(2.0))
            self.assertEqual(len(observed), 1)
            self.assertIsInstance(observed[0], FatalProcessorSignal)
            self.assertIsNone(worker.latest_result)
            self.assertTrue(worker.stats.terminated)
        finally:
            threading.excepthook = previous_hook
            worker.shutdown(wait=True, timeout=2.0)


if __name__ == "__main__":
    unittest.main()
