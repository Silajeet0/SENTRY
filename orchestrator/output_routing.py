"""
output_routing.py — routes print()/logging output from a specific thread to
a log file instead of the console.
"""
import sys
import threading
from contextlib import contextmanager
from pathlib import Path


class _ThreadRoutedStream:
    def __init__(self, default_stream):
        self._default = default_stream
        self._local = threading.local()

    def set_target(self, stream):
        self._local.target = stream

    def clear_target(self):
        if hasattr(self._local, "target"):
            del self._local.target

    def write(self, s):
        (getattr(self._local, "target", None) or self._default).write(s)

    def flush(self):
        (getattr(self._local, "target", None) or self._default).flush()

    def isatty(self):
        return False


_ROUTED_STDOUT = None
_ROUTED_STDERR = None
_INSTALL_LOCK = threading.Lock()


def _install():
    global _ROUTED_STDOUT, _ROUTED_STDERR
    with _INSTALL_LOCK:
        if _ROUTED_STDOUT is None:
            _ROUTED_STDOUT = _ThreadRoutedStream(sys.stdout)
            sys.stdout = _ROUTED_STDOUT
        if _ROUTED_STDERR is None:
            _ROUTED_STDERR = _ThreadRoutedStream(sys.stderr)
            sys.stderr = _ROUTED_STDERR


@contextmanager
def route_output_to_file(log_path: str):
    """
    Use inside a background worker thread:

    Every print()/logging call made from *this thread* while inside the
    block goes to that file. Calls from any other thread are unaffected.
    """
    _install()
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a", encoding="utf-8")
    _ROUTED_STDOUT.set_target(f)
    _ROUTED_STDERR.set_target(f)
    try:
        yield path
    finally:
        _ROUTED_STDOUT.clear_target()
        _ROUTED_STDERR.clear_target()
        f.close()
