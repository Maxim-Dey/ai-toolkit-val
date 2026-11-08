from datetime import timedelta
import multiprocessing
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch.distributed as dist
from toolkit.validation import main_process_call


def validation_worker(rank, rendezvous, queue):
    try:
        dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
        accelerator = SimpleNamespace(is_main_process=rank == 0)
        result = main_process_call(accelerator, lambda: 'published')
        queue.put((rank, result))
        def fail():
            raise ValueError('injected failure')
        try:
            main_process_call(accelerator, fail)
        except RuntimeError as exc:
            queue.put((rank, str(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class DistributedTest(unittest.TestCase):
    def test_two_ranks_receive_failure_without_hanging_and_publish_once(self):
        with tempfile.TemporaryDirectory() as root:
            context = multiprocessing.get_context('spawn')
            queue = context.Queue()
            rendezvous = (Path(root) / 'rendezvous').as_uri()
            processes = [context.Process(target=validation_worker, args=(rank, rendezvous, queue)) for rank in range(2)]
            for process in processes:
                process.start()
            try:
                for process in processes:
                    process.join(timeout=45)
                    self.assertFalse(process.is_alive(), 'DDP validation hung')
                    self.assertEqual(process.exitcode, 0)
                results = [queue.get(timeout=2) for _ in range(4)]
                self.assertEqual(sum(value == 'published' for _, value in results), 1)
                self.assertEqual(sum('injected failure' in str(value) for _, value in results), 2)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join()
                queue.close()


if __name__ == '__main__':
    unittest.main()
