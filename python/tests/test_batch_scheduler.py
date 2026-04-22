"""Tests for BatchScheduler."""

import threading
import time
import pytest
from nve.serve.batch_scheduler import BatchRequest, BatchScheduler


def make_echo_dispatch(results_store: list):
    """Dispatch function that echoes the prompt back as result."""
    def dispatch(batch):
        for req in batch:
            req.result = {"text": f"echo: {req.prompt}", "generated_tokens": 5}
            req.done_event.set()
        results_store.append(len(batch))
    return dispatch


def test_single_request():
    batches = []
    scheduler = BatchScheduler(make_echo_dispatch(batches), max_batch_size=4, timeout_ms=20)
    scheduler.start()

    req = BatchRequest(
        request_id="r1",
        prompt="hello",
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.9,
    )
    scheduler.submit(req)
    req.done_event.wait(timeout=2.0)

    assert req.result is not None
    assert req.result["text"] == "echo: hello"
    scheduler.stop()


def test_batch_grouping():
    """Multiple requests submitted quickly should be grouped into one batch."""
    batches = []
    scheduler = BatchScheduler(make_echo_dispatch(batches), max_batch_size=8, timeout_ms=100)
    scheduler.start()

    reqs = []
    for i in range(4):
        req = BatchRequest(
            request_id=f"r{i}",
            prompt=f"prompt{i}",
            max_new_tokens=10,
            temperature=0.7,
            top_p=0.9,
        )
        scheduler.submit(req)
        reqs.append(req)

    for req in reqs:
        req.done_event.wait(timeout=2.0)

    assert all(r.result is not None for r in reqs)
    scheduler.stop()


def test_error_propagation():
    def failing_dispatch(batch):
        for req in batch:
            req.error = RuntimeError("backend exploded")
            req.done_event.set()

    scheduler = BatchScheduler(failing_dispatch, max_batch_size=4, timeout_ms=20)
    scheduler.start()

    req = BatchRequest(
        request_id="rerr",
        prompt="crash",
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.9,
    )
    scheduler.submit(req)
    req.done_event.wait(timeout=2.0)
    assert req.error is not None
    assert "exploded" in str(req.error)
    scheduler.stop()


def test_queue_depth():
    scheduler = BatchScheduler(lambda b: None, max_batch_size=4, timeout_ms=100)
    assert scheduler.queue_depth() == 0
    scheduler.start()
    scheduler.stop()
