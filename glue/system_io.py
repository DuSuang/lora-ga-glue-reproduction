"""Filesystem primitives used by asset preparation and experiment writers."""
from contextlib import contextmanager
import json
import os


@contextmanager
def file_lock(path):
    """A nonblocking kernel lock. Keep the inode in place across successive owners."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Closing the descriptor releases either lock, including on process exit.
        yield


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    tmp.replace(path)
