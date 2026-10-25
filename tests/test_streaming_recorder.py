"""``StreamingEpisodeRecorder`` -- streams to HDF5 instead of buffering the take.

The contract that matters is that it is a drop-in for ``EpisodeRecorder``: feed both
the same frames and the same ``set_static`` tree, and the two files must be
indistinguishable. Everything else here guards the edges that differ because the file
now exists *during* the take (partial files, discard, a mid-take schema change).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from omniteleop.common.recorder import EpisodeRecorder
from omniteleop.common.streaming_recorder import StreamingEpisodeRecorder


def _frame(i: int) -> dict:
    """One frame shaped like what wbc_vr_robot_tmp.record_tick builds."""
    rng = np.random.default_rng(i)
    return {
        "timestamp_ns": np.int64(1_700_000_000_000_000_000 + i * 100_000_000),
        "action": {
            "joint": {
                "left_arm": rng.random(7).astype(np.float32),
                "torso": rng.random(3).astype(np.float32),
            },
            "gripper": {"left": np.float32(i % 3)},
        },
        "obs": {
            "images": {
                "head_left_rgb": (rng.random((6, 8, 3)) * 255).astype(np.uint8),
                "head_depth": (rng.random((6, 8)) * 1000).astype(np.uint16),
                "head_frame_ns": np.int64(i),
                "grab_wall_ns": np.int64(i * 2),
            },
            "base": {"pose": rng.random(3).astype(np.float32)},
        },
    }


_STATIC = {
    "obs": {"images": {"intrinsic": np.eye(3, dtype=np.float32) * 385.0}},
    "meta": {"ntp": {"offset_ns": np.int64(-24_470_000), "rtt_ns": np.int64(2_370_000)}},
}


def _dump(path) -> dict[str, tuple[tuple[int, ...], np.dtype, bytes]]:
    """Every dataset in the file as ``{path: (shape, dtype, raw bytes)}``."""
    out: dict = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            arr = np.asarray(obj[...])
            out[name] = (arr.shape, arr.dtype, arr.tobytes())

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return out


def _wait(rec, timeout: float = 20.0) -> None:
    t0 = time.time()
    while rec.saving and time.time() - t0 < timeout:
        time.sleep(0.01)
    assert not rec.saving, "recorder still saving after timeout"
    assert rec.last_save_error is None, rec.last_save_error


def test_matches_buffered_recorder(tmp_path):
    """Same frames + same static tree -> byte-identical datasets."""
    frames = [_frame(i) for i in range(37)]  # not a multiple of any chunk size

    buffered_dir = tmp_path / "buffered"
    streaming_dir = tmp_path / "streaming"

    ref = EpisodeRecorder(str(buffered_dir))
    ref.set_static(_STATIC)
    ref.start()
    for f in frames:
        ref.record(f)
    ref_path = ref.stop()
    _wait(ref)

    rec = StreamingEpisodeRecorder(str(streaming_dir))
    rec.set_static(_STATIC)
    rec.start()
    for f in frames:
        rec.record(f)
    new_path = rec.stop()
    _wait(rec)

    assert ref_path is not None and new_path is not None
    ref_dump, new_dump = _dump(ref_path), _dump(new_path)
    assert set(ref_dump) == set(new_dump), (
        f"only in buffered: {sorted(set(ref_dump) - set(new_dump))}; "
        f"only in streaming: {sorted(set(new_dump) - set(ref_dump))}"
    )
    for key in ref_dump:
        assert new_dump[key] == ref_dump[key], f"{key} differs"

    with h5py.File(new_path, "r") as f:
        assert f.attrs["complete"] is np.True_ or f.attrs["complete"]
        assert int(f.attrs["n_frames"]) == len(frames)
    assert rec.num_frames() == len(frames)
    assert not (streaming_dir / "episode_0.hdf5.partial").exists()


def test_episode_ids_and_multiple_takes(tmp_path):
    """Ids advance per saved take and resume from what is already on disk."""
    rec = StreamingEpisodeRecorder(str(tmp_path))
    for expected in (0, 1):
        assert rec.episode_id == expected
        rec.start()
        rec.record(_frame(0))
        assert rec.stop() == str(tmp_path / f"episode_{expected}.hdf5")
        _wait(rec)
    assert StreamingEpisodeRecorder(str(tmp_path)).episode_id == 2


def test_zero_frames_writes_nothing(tmp_path):
    """An engaged-but-empty take leaves no file and does not burn an id."""
    rec = StreamingEpisodeRecorder(str(tmp_path))
    rec.start()
    assert rec.stop() is None
    assert rec.episode_id == 0
    assert list(tmp_path.iterdir()) == []


def test_process_crash_keeps_checkpointed_states_and_metadata(tmp_path):
    import subprocess
    import sys

    code = """
import os,sys,time,numpy as np
from omniteleop.common.streaming_recorder import StreamingEpisodeRecorder
r=StreamingEpisodeRecorder(sys.argv[1]);r.set_static({'meta':{'run':np.bytes_('crash_probe')}});r.start()
for i in range(12):
 r.record({'timestamp_ns':np.int64(i),'obs':{'base':{'pose':np.array([i,0.,0.])},
           'images':{'rgb':np.zeros((750,1200,3),np.uint8)}}})
deadline=time.monotonic()+10
while r._n_written<12 and time.monotonic()<deadline:time.sleep(.01)
os._exit(0 if r._n_written==12 else 2)
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, timeout=15)
    with h5py.File(tmp_path / "episode_0.hdf5.partial") as f:
        assert not f.attrs["complete"]
        assert int(f.attrs["n_frames"]) >= 10
        assert f["meta/run"][()].decode() == "crash_probe"
        np.testing.assert_array_equal(f["obs/base/pose"][:10, 0], np.arange(10))


def test_discard_removes_partial(tmp_path):
    """discard() drops the take: no file survives, the id is reused."""
    rec = StreamingEpisodeRecorder(str(tmp_path))
    rec.start()
    for i in range(5):
        rec.record(_frame(i))
    rec.discard()
    assert rec.episode_id == 0
    assert list(tmp_path.iterdir()) == []


def test_abort_quarantines_incomplete_take_and_consumes_id(tmp_path):
    """Runtime failure preserves evidence but cannot publish a training input."""
    rec = StreamingEpisodeRecorder(str(tmp_path))
    rec.set_static(_STATIC)
    rec.start()
    for i in range(5):
        rec.record(_frame(i))

    path = rec.abort("camera alignment failed")
    _wait(rec)

    partial = tmp_path / "episode_0.hdf5.partial"
    assert path == str(partial)
    assert partial.exists()
    assert not (tmp_path / "episode_0.hdf5").exists()
    with h5py.File(partial, "r") as f:
        assert not bool(f.attrs["complete"])
        assert int(f.attrs["n_frames"]) == 5
        assert f.attrs["abort_reason"] == "camera alignment failed"
        assert int(f.attrs["abort_wall_ns"]) > 0
        assert "meta/ntp/offset_ns" in f  # static provenance is retained
    assert rec.episode_id == 1
    assert StreamingEpisodeRecorder(str(tmp_path)).episode_id == 1


def test_abort_before_first_frame_writes_nothing(tmp_path):
    rec = StreamingEpisodeRecorder(str(tmp_path))
    rec.start()

    assert rec.abort("startup cameras unavailable") is None
    assert rec.episode_id == 0
    assert list(tmp_path.iterdir()) == []


def test_schema_change_mid_episode_fails_loudly(tmp_path):
    """A frame with different keys aborts the take instead of writing a ragged file."""
    rec = StreamingEpisodeRecorder(str(tmp_path))
    rec.start()
    rec.record(_frame(0))
    bad = _frame(1)
    del bad["obs"]["images"]["head_depth"]
    rec.record(bad)
    rec.stop()
    t0 = time.time()
    while rec.saving and time.time() - t0 < 20.0:
        time.sleep(0.01)
    assert isinstance(rec.last_save_error, ValueError)
    assert "schema changed mid-episode" in str(rec.last_save_error)
    assert list(tmp_path.iterdir()) == []


def test_record_before_start(tmp_path):
    rec = StreamingEpisodeRecorder(str(tmp_path))
    with pytest.raises(RuntimeError, match="record\\(\\) before start\\(\\)"):
        rec.record(_frame(0))


def test_backpressure_raises_rather_than_dropping(tmp_path):
    """A writer that cannot keep up fails the take; it never silently loses a frame."""
    rec = StreamingEpisodeRecorder(str(tmp_path), queue_frames=2)
    release = threading.Event()
    writer = rec._writer

    def delayed_writer():
        release.wait(timeout=2.0)
        writer()

    rec._writer = delayed_writer
    rec.start()
    rec.record(_frame(0))
    rec.record(_frame(1))
    with pytest.raises(RuntimeError, match="frames behind"):
        rec.record(_frame(2))

    # The exception path must be able to quarantine even though the queue remains full.
    started = time.monotonic()
    path = rec.abort("backpressure")
    assert time.monotonic() - started < 0.2
    release.set()
    _wait(rec)
    assert path == str(tmp_path / "episode_0.hdf5.partial")


@pytest.mark.parametrize("operation", ["stop", "abort"])
def test_full_queue_close_never_blocks_and_writer_drains(operation, tmp_path):
    """A full live queue needs no sentinel slot; _closing provides bounded shutdown."""
    rec = StreamingEpisodeRecorder(str(tmp_path), queue_frames=1)
    release = threading.Event()
    writer = rec._writer

    def delayed_writer():
        release.wait(timeout=2.0)
        writer()

    rec._writer = delayed_writer
    rec.start()
    rec.record(_frame(0))  # fills the only slot while the writer is held back

    started = time.monotonic()
    path = rec.stop() if operation == "stop" else rec.abort("injected failure")
    assert time.monotonic() - started < 0.2
    release.set()
    _wait(rec)

    suffix = "" if operation == "stop" else ".partial"
    expected = tmp_path / f"episode_0.hdf5{suffix}"
    assert path == str(expected)
    assert expected.exists()
    with h5py.File(expected, "r") as f:
        assert bool(f.attrs["complete"]) is (operation == "stop")
        assert int(f.attrs["n_frames"]) == 1


@pytest.mark.parametrize("operation", ["stop", "abort"])
def test_dead_writer_with_orphaned_full_queue_cannot_wedge_close(operation, tmp_path):
    """Fault injection for the review's confirmed dead-consumer hang."""
    rec = StreamingEpisodeRecorder(str(tmp_path), queue_frames=1)
    rec._writer = lambda: None
    rec.start()
    assert rec._thread is not None
    rec._thread.join(timeout=1.0)
    assert not rec._thread.is_alive()

    # Model one frame accepted just before the worker died. The orphaned queue has no
    # free sentinel slot—the old blocking put(_STOP) hung forever here.
    assert rec._queue is not None
    rec._queue.put_nowait(_frame(0))
    rec._n_recorded = 1

    started = time.monotonic()
    path = rec.stop() if operation == "stop" else rec.abort("writer died")
    assert time.monotonic() - started < 0.2
    assert path is not None
    assert not rec.saving
    assert isinstance(rec.last_save_error, RuntimeError)
    assert "writer thread exited" in str(rec.last_save_error)


def test_invalidate_slow_clean_save_can_only_publish_partial(tmp_path):
    rec = StreamingEpisodeRecorder(str(tmp_path))
    release = threading.Event()
    writer = rec._writer

    def delayed_writer():
        release.wait(timeout=2.0)
        writer()

    rec._writer = delayed_writer
    rec.start()
    rec.record(_frame(0))
    final = rec.stop()

    partial = rec.invalidate_save("save wait timed out")
    release.set()
    _wait(rec)

    assert final == str(tmp_path / "episode_0.hdf5")
    assert partial == str(tmp_path / "episode_0.hdf5.partial")
    assert not (tmp_path / "episode_0.hdf5").exists()
    with h5py.File(partial, "r") as f:
        assert not bool(f.attrs["complete"])
        assert f.attrs["abort_reason"] == "save wait timed out"
        assert int(f.attrs["abort_wall_ns"]) > 0


def test_invalidation_racing_after_streaming_rename_rolls_final_back_to_partial(
    tmp_path, monkeypatch
):
    """The timeout can win after rename but before ``saving`` is cleared."""
    final = tmp_path / "episode_0.hdf5"
    staging = tmp_path / "episode_0.hdf5.partial"
    renamed = threading.Event()
    release = threading.Event()
    original_rename = Path.rename

    def delayed_rename(path, target):
        result = original_rename(path, target)
        if path == staging and Path(target) == final:
            renamed.set()
            assert release.wait(timeout=2.0)
        return result

    monkeypatch.setattr(Path, "rename", delayed_rename)
    rec = StreamingEpisodeRecorder(str(tmp_path))
    rec.start()
    rec.record(_frame(0))
    returned_final = rec.stop()
    assert renamed.wait(timeout=2.0)

    returned_partial = rec.invalidate_save("timeout raced with rename")
    release.set()
    _wait(rec)

    assert returned_final == str(final)
    assert returned_partial == str(staging)
    assert not final.exists()
    with h5py.File(staging, "r") as f:
        assert not bool(f.attrs["complete"])
        assert f.attrs["abort_reason"] == "timeout raced with rename"
