"""Keep POSIX CLI cancellation responsive while the trainer runs native code."""
import os
import signal
import subprocess
import sys
import time


def supervise(command, *, env=None, grace_seconds=5):
    """Run an isolated process group; forward stops and bound shutdown time."""
    received = []
    previous = {}
    process = None

    def request_stop(signum, frame):
        received.append(signum)

    def send_group(signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    def wait_for_exit(timeout):
        deadline = time.monotonic() + timeout
        while process.poll() is None and len(received) < 2:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                process.wait(timeout=min(remaining, 0.1))
            except subprocess.TimeoutExpired:
                pass

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_stop)
        # The parent owns terminal signals; the child and all its workers form
        # a separate group, so cancellation never targets the shell or TensorBoard.
        process = subprocess.Popen(command, env=env, start_new_session=True)
        while process.poll() is None and not received:
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        if received:
            print('\nStopping training; unsaved updates will be lost.', flush=True)
            send_group(received[0])
            wait_for_exit(grace_seconds)
            if process.poll() is None:
                send_group(signal.SIGTERM)
                wait_for_exit(grace_seconds)
            return 128 + received[0]
        code = process.returncode
        return code if code >= 0 else 128 - code
    finally:
        try:
            if process is not None:
                # Also remove compiler/data-loader workers left after parent exit.
                send_group(signal.SIGKILL)
                process.wait(timeout=5)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def guard_cli():
    """Re-enter the CLI once, preserving interpreter flags, arguments and stdio."""
    if os.name != 'posix' or os.environ.pop('_AITK_CLI_CHILD', None) == '1':
        return
    env = dict(os.environ, _AITK_CLI_CHILD='1')
    command = [sys.executable, *sys.orig_argv[1:]]
    raise SystemExit(supervise(command, env=env))
