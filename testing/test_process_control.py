import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class CliQueueTest(unittest.TestCase):
    def test_ctrl_c_stops_recover_queue_even_during_job_creation(self):
        import run
        with patch.object(sys, 'argv', ['run.py', '--recover', 'first', 'second']), \
                patch.object(run, 'get_job', side_effect=KeyboardInterrupt) as get_job, \
                patch.object(run, 'print_acc'):
            with self.assertRaises(SystemExit) as stopped:
                run.main()
        self.assertEqual(stopped.exception.code, 130)
        self.assertEqual(get_job.call_count, 1)


@unittest.skipUnless(os.name == 'posix', 'POSIX process groups and terminal signals')
class ProcessControlTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def launch(self, body, *, grace=0.25, terminal=False):
        worker = self.folder / 'worker.py'
        worker.write_text(textwrap.dedent(body), encoding='utf-8')
        wrapper = self.folder / 'wrapper.py'
        wrapper.write_text(
            'import sys\n'
            f'sys.path.insert(0, {str(ROOT)!r})\n'
            'from toolkit.process_control import supervise\n'
            f'raise SystemExit(supervise([sys.executable, "-u", {str(worker)!r}], grace_seconds={grace!r}))\n',
            encoding='utf-8',
        )
        options = dict(cwd=self.folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if terminal:
            import fcntl
            import pty
            import termios
            master, slave = pty.openpty()
            self.addCleanup(os.close, master)
            self.addCleanup(os.close, slave)
            def own_terminal():
                os.setsid()
                fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
                os.tcsetpgrp(slave, os.getpgrp())
            options.update(stdin=slave, preexec_fn=own_terminal)
        else:
            master = None
            options.update(start_new_session=True)
        process = subprocess.Popen([sys.executable, '-u', str(wrapper)], **options)
        def cleanup():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            process.stdout.close()
        self.addCleanup(cleanup)
        return process, master

    def ready(self, name='ready'):
        path = self.folder / name
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if path.exists():
                return path.read_text()
            time.sleep(0.02)
        self.fail(f'Worker did not become ready: {name}')

    def assert_stopped(self, pid):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = Path('/proc') / str(pid) / 'stat'
            if not status.exists() or status.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                return
            time.sleep(0.02)
        self.fail(f'Worker {pid} is still running')

    def test_exit_codes_and_normal_exit_cleans_workers(self):
        for code in (0, 7):
            with self.subTest(code=code):
                process, _ = self.launch(f'''
                    import subprocess, sys
                    from pathlib import Path
                    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
                    Path('ready').write_text(str(child.pid))
                    raise SystemExit({code})
                ''')
                self.assertEqual(process.wait(timeout=5), code)
                self.assert_stopped(int(self.ready()))
                (self.folder / 'ready').unlink()

    def test_terminal_ctrl_c_cleans_unresponsive_tree_and_preserves_other_group(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        self.addCleanup(unrelated.wait)
        self.addCleanup(unrelated.terminate)
        process, terminal = self.launch('''
            import os, signal, subprocess, sys, time
            from pathlib import Path
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
            Path('ready').write_text(f'{os.getpid()} {child.pid}')
            while True: time.sleep(60)
        ''', terminal=True)
        pids = [int(pid) for pid in self.ready().split()]
        os.write(terminal, b'\x03')
        self.assertEqual(process.wait(timeout=5), 130)
        for pid in pids:
            self.assert_stopped(pid)
        self.assertIsNone(unrelated.poll())

    def test_sigint_reaches_responsive_worker(self):
        process, _ = self.launch('''
            import signal, time
            from pathlib import Path
            def stop(signum, frame):
                Path('stopped').write_text(str(signum))
                raise SystemExit(0)
            signal.signal(signal.SIGINT, stop)
            Path('ready').write_text('yes')
            while True: time.sleep(60)
        ''')
        self.ready()
        process.send_signal(signal.SIGINT)
        self.assertEqual(process.wait(timeout=5), 130)
        self.assertEqual(self.ready('stopped'), str(signal.SIGINT))

    def test_repeated_interrupt_skips_grace_period(self):
        process, _ = self.launch('''
            import signal, time
            from pathlib import Path
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            Path('ready').write_text('yes')
            while True: time.sleep(60)
        ''', grace=20)
        self.ready()
        process.send_signal(signal.SIGINT)
        time.sleep(0.15)
        process.send_signal(signal.SIGINT)
        self.assertEqual(process.wait(timeout=3), 130)

    def test_sigterm_cleans_training_group(self):
        process, _ = self.launch('''
            import os, time
            from pathlib import Path
            Path('ready').write_text(str(os.getpid()))
            while True: time.sleep(60)
        ''')
        pid = int(self.ready())
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=5), 143)
        self.assert_stopped(pid)


if __name__ == '__main__':
    unittest.main()
