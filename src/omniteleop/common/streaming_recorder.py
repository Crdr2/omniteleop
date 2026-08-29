#!/usr/bin/env python3
"""Episode recorder that streams to HDF5 instead of buffering the take in RAM.

Drop-in for :class:`omniteleop.common.recorder.EpisodeRecorder` -- same
``start/record/set_static/stop/discard/num_frames`` API and the same
``recording`` / ``saving`` / ``episode_id`` attributes -- so a caller swaps the
class and nothing else changes. Its output schema matches the buffered recorder;
only the path to disk differs.

Why
---
``EpisodeRecorder`` appends every frame to a list and writes on ``stop()``, then
``_recursive_np_stack`` makes a second full copy -- so peak RAM is ~2x the take.
At the 600x960 head geometry that is 3.3 MiB/frame with depth, i.e. ~16 GiB peak
for a 5-minute take at 10 Hz, and back-to-back takes stack (the previous take's
buffer lives until its save thread finishes, which over a slow save target can
outlast the next take's start).

Here the frames go through a bounded queue to a writer thread that appends them
to resizable, chunked datasets. Peak RAM is the queue, not the take: ~200 MiB at
the default 64 frames, regardless of whether the demo runs 5 seconds or 50
minutes.

Two consequences worth knowing:

* The record tick never touches h5py -- ``record()`` is a ``put_nowait`` -- so a
  slow disk cannot stall the control loop. If the writer falls behind for long
  enough to fill the queue, ``record()`` raises rather than silently dropping a
  frame: a gap in a take is worse than a failed take.
* A crash mid-take leaves ``episode_<N>.hdf5.partial`` holding everything
  written so far, instead of losing the whole take. It is renamed to
  ``episode_<N>.hdf5`` only on a clean close, so a partial file can never be
  picked up by ``port_wbc_mobile_*.py`` as if it were complete.

The schema is fixed by the FIRST frame: every later frame must carry the same
keys, shapes and dtypes, or ``record()`` raises. That matches how the callers
already behave (the key set is decided once per run) and it is what makes the
resizable datasets possible.
"""

from __future__ import annotations

import atexit
import pathlib
import queue
import threading
from typing import Any, Optional

import numpy as np
from loguru import logger

from omniteleop.common.recorder import peek_next_episode_id

# Frames in flight between the record tick and the writer thread. 64 frames of
# 600x960 RGB+depth is ~210 MiB -- the whole RAM budget of a take, whatever its
# length. Raise it only if a genuinely bursty save target needs more slack.
_DEFAULT_QUEUE_FRAMES = 64

# Datasets are chunked along axis 0 so each chunk lands in this ballpark. Frames
# are also written in batches of the chunk length, so h5py never does a
# read-modify-write on a partially filled chunk.
_TARGET_CHUNK_BYTES = 4 * 1024 * 1024
_MAX_CHUNK_FRAMES = 256

_STOP = object()


def _flatten(tree: dict, prefix: str = "") -> dict[str, np.ndarray]:
    """Nested dict -> ``{"a/b/c": ndarray}``, mirroring the HDF5 layout."""
    flat: dict[str, np.ndarray] = {}
    for key, val in tree.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            flat.update(_flatten(val, path + "/"))
        else:
            flat[path] = np.asarray(val)
    return flat


class StreamingEpisodeRecorder:
    """Accumulates per-frame data and streams it to HDF5 as the take runs."""

    def __init__(
        self,
        save_dir: str,
        queue_frames: int = _DEFAULT_QUEUE_FRAMES,
        compression: Optional[str] = None,
    ) -> None:
        if queue_frames < 1:
            raise ValueError(f"queue_frames must be >= 1, got {queue_frames}")
        self._save_dir = pathlib.Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._queue_frames = int(queue_frames)
        self._compression = compression
        self._static: dict = {}
        self.recording = False
        self.saving = False
        self.save_progress: float = 0.0
        self.last_save_error: BaseException | None = None
        self.episode_id = peek_next_episode_id(save_dir)

        self._queue: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._n_recorded = 0
        self._n_written = 0
        self._abort = False
        self._closing = False
        self._keep_partial = False
        self._path: Optional[pathlib.Path] = None
        self._schema: Optional[dict[str, tuple[tuple[int, ...], np.dtype]]] = None

    # ── caller-facing API (matches EpisodeRecorder) ───────────────────────────

    def start(self) -> None:
        """Start a new episode; the file is created when the first frame lands."""
        if self.saving:
            raise RuntimeError(
                "StreamingEpisodeRecorder.start() while the previous episode is still "
                "closing; wait for .saving to go False."
            )
        self._queue = queue.Queue(maxsize=self._queue_frames)
        self._n_recorded = 0
        self._n_written = 0
        self._abort = False
        self._closing = False
        self._keep_partial = False
        self._schema = None
        self.last_save_error = None
        self.save_progress = 0.0
        self._path = self._save_dir / f"episode_{self.episode_id}.hdf5"
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
        # Without this, an exception in the record loop (or a Ctrl-C) leaves the writer
        # parked on get() with the HDF5 file open, and the interpreter hangs on exit.
        atexit.register(self._shutdown)
        self.recording = True
        logger.info("StreamingEpisodeRecorder: recording started")

    def record(self, frame: dict) -> None:
        """Hand one frame to the writer thread. Never blocks; raises if backed up."""
        if self._queue is None:
            raise RuntimeError("record() before start()")
        try:
            self._queue.put_nowait(frame)
        except queue.Full as exc:
            # The take is already ruined -- one frame was refused -- so stop the writer and
            # keep what reached disk as .partial (never renamed, so no porter can mistake a
            # truncated take for a complete one).
            self._keep_partial = True
            self._closing = True
            raise RuntimeError(
                f"StreamingEpisodeRecorder: writer is {self._queue_frames} frames behind "
                f"-- the save target cannot keep up with the record rate. Recorded "
                f"{self._n_recorded} frames, wrote {self._n_written}. Point --save-dir at "
                f"a local disk (not sshfs/NFS) or lower --record-rate."
            ) from exc
        self._n_recorded += 1

    def set_static(self, static: dict) -> None:
        """Register fields stored ONCE per episode; written when the file closes.

        Same contract as :meth:`EpisodeRecorder.set_static` -- a static leaf that
        collides with a per-frame path raises rather than shadowing it.
        """
        self._static = static

    def stop(self) -> Optional[str]:
        """Stop recording and close the file in the background.

        Returns the target path immediately (``None`` if the take had no frames,
        in which case no file is left behind and the episode id is not consumed).
        """
        if not self.recording:
            return None
        self.recording = False
        self._closing = True
        if self._n_recorded == 0:
            self._abort = True
            assert self._queue is not None
            self._queue.put(_STOP)  # the writer is parked on get(); wake it to exit
            self._finish_thread()
            logger.warning("StreamingEpisodeRecorder: 0 frames — skipping save")
            return None
        path = str(self._path)
        self.saving = True
        self.episode_id += 1
        assert self._queue is not None
        self._queue.put(_STOP)
        return path

    def discard(self) -> None:
        """Stop and drop the take: the partial file is deleted, the id is reused."""
        if not self.recording:
            return
        self.recording = False
        self._abort = True
        self._closing = True
        if self._queue is not None:
            self._queue.put(_STOP)
        self._finish_thread()

    def num_frames(self) -> int:
        """Frames handed to :meth:`record` for the current/just-finished episode."""
        return self._n_recorded

    # ── writer thread ─────────────────────────────────────────────────────────

    def _finish_thread(self, timeout: float = 30.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise RuntimeError(
                    f"StreamingEpisodeRecorder: writer thread did not finish within {timeout:g}s"
                )
            self._thread = None

    def _writer(self) -> None:
        """Drain the queue into resizable datasets; rename off .partial on success."""
        import h5py  # noqa: PLC0415 -- optional dep, only needed while saving

        partial = pathlib.Path(f"{self._path}.partial")
        h5file = None
        dsets: dict[str, Any] = {}
        chunk_frames = 1
        batch: list[dict[str, np.ndarray]] = []

        def flush() -> None:
            if not batch:
                return
            for path, dset in dsets.items():
                block = np.stack([f[path] for f in batch])
                start = dset.shape[0]
                dset.resize(start + len(batch), axis=0)
                dset[start:] = block
            self._n_written += len(batch)
            batch.clear()

        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if self._closing:
                        break
                    continue
                if item is _STOP:
                    break
                flat = _flatten(item)
                if self._schema is None:
                    self._schema = {k: (v.shape, v.dtype) for k, v in flat.items()}
                    h5file = h5py.File(partial, "w")
                    h5file.attrs["complete"] = False
                    frame_bytes = max(1, sum(v.nbytes for v in flat.values()))
                    chunk_frames = int(
                        min(_MAX_CHUNK_FRAMES, max(1, _TARGET_CHUNK_BYTES // frame_bytes))
                    )
                    for path, arr in flat.items():
                        dsets[path] = h5file.create_dataset(
                            path,
                            shape=(0,) + arr.shape,
                            maxshape=(None,) + arr.shape,
                            dtype=arr.dtype,
                            chunks=(chunk_frames,) + arr.shape,
                            compression=self._compression,
                        )
                else:
                    got = {k: (v.shape, v.dtype) for k, v in flat.items()}
                    if got != self._schema:
                        added = sorted(set(got) - set(self._schema))
                        dropped = sorted(set(self._schema) - set(got))
                        changed = sorted(
                            k for k in set(got) & set(self._schema) if got[k] != self._schema[k]
                        )
                        raise ValueError(
                            "StreamingEpisodeRecorder: frame schema changed mid-episode "
                            f"(added={added}, dropped={dropped}, changed={changed}). Every "
                            "frame in a take must carry the same keys, shapes and dtypes."
                        )
                batch.append(flat)
                if len(batch) >= chunk_frames:
                    flush()

            flush()
            if h5file is not None and not self._abort:
                self._write_static(h5file)
                h5file.attrs["n_frames"] = self._n_written
                h5file.attrs["complete"] = True
        except Exception as exc:  # noqa: BLE001 -- surfaced via last_save_error
            self.last_save_error = exc
            self._abort = True
            logger.exception("StreamingEpisodeRecorder: save failed")
        finally:
            if h5file is not None:
                h5file.close()
            if self._abort:
                partial.unlink(missing_ok=True)
            elif self._keep_partial:
                logger.error(
                    f"StreamingEpisodeRecorder: take truncated at {self._n_written} frames; "
                    f"kept as {partial} (NOT renamed -- inspect before using)"
                )
            elif partial.exists():
                partial.rename(self._path)
                logger.info(f"StreamingEpisodeRecorder: {self._n_written} frames → {self._path}")
            self.save_progress = 1.0
            self.saving = False

    def _shutdown(self) -> None:
        """atexit hook: wake the writer so an open HDF5 file cannot hang the exit."""
        self._closing = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def _write_static(self, h5file) -> None:
        """Write the ``set_static`` tree, refusing to shadow a per-frame path."""
        for path, arr in _flatten(self._static).items():
            if path in h5file:
                raise ValueError(f"static leaf {path!r} collides with a per-frame field")
            dset = h5file.create_dataset(path, shape=arr.shape, dtype=arr.dtype)
            dset[...] = arr
