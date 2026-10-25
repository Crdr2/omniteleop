# ruff: noqa: SLF001
"""The follower must publish a terminal ABORTED status frame when its loop dies.

Regression for the silent-stop failure: episode_0 (2026-07-03) died on the --record
camera-freshness RuntimeError, the process tore down, and the headset HUD kept showing
a stale "Stage: teleop" with no hint the follower was gone.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_wbc_vr_robot():
    path = Path(__file__).resolve().parents[1] / "scripts" / "wbc_vr_robot.py"
    spec = importlib.util.spec_from_file_location("wbc_vr_robot_abort_status_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakePublisher:
    def __init__(self) -> None:
        self.published: list = []

    def publish(self, data) -> None:
        self.published.append(data)


class _FakeNode:
    def __init__(self, publisher: _FakePublisher) -> None:
        self._publisher = publisher
        self.topics: list[str] = []

    def create_publisher(self, topic, encoder=None):
        self.topics.append(topic)
        return self._publisher


def test_publish_abort_status_sends_terminal_frame(monkeypatch):
    mod = _load_wbc_vr_robot()
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    pub = _FakePublisher()
    source = SimpleNamespace(node=_FakeNode(pub))

    exc = RuntimeError(
        "[wbc_vr_robot] --record: stale camera frame (publisher not delivering new "
        "frames at the record rate; WiFi clog / fps drop) -- head Δ=0 ns\n"
        "second line that must not reach the HUD"
    )
    mod._publish_abort_status(source, exc)

    assert len(pub.published) == 1
    frame = pub.published[0]
    assert frame["stage"] == mod.WBC_FOLLOWER_STAGE_ABORTED
    assert frame["hold"] is True
    assert frame["estop"] is True
    assert frame["success"] is False
    assert frame["safety_status"] == "ABORTED"
    assert "stale camera frame" in frame["hold_reason"]
    assert "second line" not in frame["hold_reason"]
    assert len(frame["hold_reason"]) <= 160
    assert frame["timestamp_ns"] > 0


def test_publish_abort_status_uses_exception_type_when_message_empty(monkeypatch):
    mod = _load_wbc_vr_robot()
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    pub = _FakePublisher()
    source = SimpleNamespace(node=_FakeNode(pub))

    mod._publish_abort_status(source, RuntimeError())

    assert pub.published[0]["hold_reason"] == "RuntimeError"


def test_publish_abort_status_is_silent_without_a_node():
    mod = _load_wbc_vr_robot()
    mod._publish_abort_status(SimpleNamespace(), RuntimeError("boom"))  # replay source


def test_publish_abort_status_never_masks_the_original_error(monkeypatch):
    mod = _load_wbc_vr_robot()
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    class _BrokenNode:
        def create_publisher(self, topic, encoder=None):
            raise ConnectionError("zenoh session already closed")

    mod._publish_abort_status(SimpleNamespace(node=_BrokenNode()), RuntimeError("boom"))


def test_recorder_save_wait_has_a_terminal_deadline():
    mod = _load_wbc_vr_robot()
    invalidated: list[str] = []
    recorder = SimpleNamespace(
        saving=True,
        invalidate_save=invalidated.append,
    )

    with pytest.raises(TimeoutError, match="refusing to wait forever"):
        mod._wait_for_recorder_save(
            recorder,
            path="/tmp/episode_0.hdf5",
            timeout_s=0.0,
        )
    assert len(invalidated) == 1
    assert "refusing to wait forever" in invalidated[0]
