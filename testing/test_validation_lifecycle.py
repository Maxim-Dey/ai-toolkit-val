from contextlib import nullcontext
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import patch

import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from safetensors.torch import save_file, load_file

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from jobs.process.BaseTrainProcess import BaseTrainProcess
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.validation_checkpoint import (save_checkpoint, read_checkpoint, restore_training_state,
                                           select_checkpoint, completed_checkpoints, rng_state, restore_rng)
from toolkit.ema import ExponentialMovingAverage
from toolkit.config_modules import SampleConfig


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float32))
        self.multiplier = .5

    def save_weights(self, path, dtype, metadata):
        save_file({k: v.detach().cpu().to(dtype) for k, v in self.state_dict().items()}, path, metadata=metadata)


def tiny_trainer(root):
    trainer = object.__new__(SDTrainer)
    trainer.network = TinyNetwork()
    trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=.1, momentum=.9)
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1, gamma=.9)
    trainer.train_config = SimpleNamespace(optimizer='sgd', gradient_accumulation_steps=1, max_grad_norm=100)
    trainer.uses_dataset_split = True
    trainer._accumulation_pending = False
    trainer.completed_optimizer_steps = trainer.step_num = trainer.epoch_num = 0
    trainer._validation_inputs = SimpleNamespace(manifest={'version': 1, 'hash': 'fixed'})
    trainer.ema = trainer.snr_gos = trainer.adapter = trainer.embedding = None
    trainer.save_root = root
    trainer.name = 'test'
    trainer.save_config = SimpleNamespace(max_step_saves_to_keep=2)
    trainer.sd = SimpleNamespace(is_multistage=False)
    trainer.params = list(trainer.network.parameters())
    trainer.timer = lambda *args: nullcontext()
    trainer.accelerator = SimpleNamespace(sync_gradients=True, optimizer_step_was_skipped=False,
                                         is_main_process=True, clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
    trainer.is_grad_accumulation_step = False
    trainer.steps_this_boundary = 0
    trainer.model_config = SimpleNamespace(low_vram=False)
    trainer.end_of_training_loop = lambda: None
    def train_batch(self, value, accum_scale=1):
        loss = ((self.network.weight - value) ** 2).mean()
        (loss * accum_scale).backward()
        return loss.detach()
    trainer.train_single_accumulation = MethodType(train_batch, trainer)
    return trainer


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name

    def test_both_accumulation_forms_match_and_skip_does_not_advance(self):
        a, b = tiny_trainer(self.root), tiny_trainer(self.root)
        a.hook_train_loop([2., 4.])
        b.train_config.gradient_accumulation_steps = 2
        b.is_grad_accumulation_step = True
        b.hook_train_loop(2.)
        self.assertEqual(b.completed_optimizer_steps, 0)
        self.assertEqual(b.lr_scheduler.last_epoch, 0)
        b.is_grad_accumulation_step = False
        b.hook_train_loop(4.)
        torch.testing.assert_close(a.network.weight, b.network.weight)
        self.assertEqual(b.completed_optimizer_steps, 1)
        b.accelerator.optimizer_step_was_skipped = True
        with patch.object(b.optimizer, 'step'):
            b.hook_train_loop(5.)
        self.assertEqual(b.completed_optimizer_steps, 1)
        self.assertEqual(b.lr_scheduler.last_epoch, 1)
        b.accelerator.optimizer_step_was_skipped = False
        with patch.object(b, 'train_single_accumulation', side_effect=torch.cuda.OutOfMemoryError('injected')):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                b.hook_train_loop(5.)
        self.assertEqual(b.completed_optimizer_steps, 1)
        self.assertEqual(b.lr_scheduler.last_epoch, 1)

    def test_checkpoint_dtype_roundtrip_selected_step_samples_and_rotation(self):
        trainer = tiny_trainer(self.root)
        trainer.hook_train_loop(3.)
        samples = Path(self.root) / 'test-step000001/samples'
        samples.mkdir(parents=True)
        (samples / 'example.txt').write_text('sample')
        first = save_checkpoint(trainer)
        first_weight = trainer.network.weight.detach().clone()
        first_opt = copy.deepcopy(trainer.optimizer.state_dict())
        self.assertTrue((samples / 'example.txt').is_file())
        self.assertTrue(all(v.dtype == torch.float32 for v in load_file(first / 'model.safetensors').values()))
        self.assertEqual(save_checkpoint(trainer), first)
        trainer.hook_train_loop(4.)
        second = save_checkpoint(trainer)
        self.assertEqual(select_checkpoint(self.root, 'test'), second)
        self.assertEqual(select_checkpoint(self.root, 'test', first), first)
        state = read_checkpoint(first)
        trainer.network.load_state_dict(load_file(first / 'model.safetensors'))
        restore_training_state(trainer, state)
        torch.testing.assert_close(first_weight, trainer.network.weight)
        self.assertEqual(trainer.completed_optimizer_steps, 1)
        torch.testing.assert_close(trainer.optimizer.state_dict()['state'][0]['momentum_buffer'], first_opt['state'][0]['momentum_buffer'])
        restore_rng(state['rng'])
        expected_noise = torch.rand(3)
        restore_rng(state['rng'])
        torch.testing.assert_close(expected_noise, torch.rand(3))
        trainer.completed_optimizer_steps = 3
        save_checkpoint(trainer)
        self.assertFalse(first.exists())
        self.assertTrue(second.exists())

    def test_interrupted_save_and_samples_only_are_not_resumable(self):
        trainer = tiny_trainer(self.root)
        trainer.hook_train_loop(3.)
        previous = save_checkpoint(trainer)
        trainer.hook_train_loop(4.)
        (Path(self.root) / 'test-step000002/samples').mkdir(parents=True)
        with patch('toolkit.validation_checkpoint.torch.save', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                save_checkpoint(trainer)
        self.assertEqual(select_checkpoint(self.root, 'test'), previous)
        self.assertEqual(len(completed_checkpoints(self.root, 'test')), 1)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            select_checkpoint(self.root, 'test', Path(self.root) / 'test-step000002')

    def test_rollback_replaces_old_future_step_but_deduplicates_current_save(self):
        trainer = tiny_trainer(self.root)
        trainer.hook_train_loop(3.)
        first = save_checkpoint(trainer)
        trainer.hook_train_loop(4.)
        future = save_checkpoint(trainer)
        old_id = read_checkpoint(future)['id']
        samples = future / 'samples'
        samples.mkdir()
        (samples / 'old.txt').write_text('old weights')
        trainer.network.load_state_dict(load_file(first / 'model.safetensors'))
        restore_training_state(trainer, read_checkpoint(first))
        trainer.hook_train_loop(9.)
        expected = trainer.network.weight.detach().clone()
        save_checkpoint(trainer)
        new_id = read_checkpoint(future)['id']
        self.assertNotEqual(old_id, new_id)
        self.assertFalse(samples.exists())
        torch.testing.assert_close(load_file(future / 'model.safetensors')['weight'], expected)
        samples.mkdir()
        (samples / 'new.txt').write_text('current weights')
        save_checkpoint(trainer)
        self.assertEqual(read_checkpoint(future)['id'], new_id)
        self.assertTrue((samples / 'new.txt').is_file())

    def test_legacy_lookup_and_pretrained_initialization_remain_available(self):
        trainer = tiny_trainer(self.root)
        trainer.config = {}
        trainer.job = SimpleNamespace(name='test')
        legacy = Path(self.root) / 'test_000000050.safetensors'
        trainer.network.save_weights(str(legacy), torch.float32, {})
        trainer.network_config = SimpleNamespace(pretrained_lora_path=str(legacy))
        trainer.uses_dataset_split = False
        self.assertEqual(trainer.get_latest_save_path(), str(legacy))
        trainer.uses_dataset_split = True
        self.assertEqual(trainer.get_latest_save_path(), str(legacy))
        trainer.config['resume_from'] = str(legacy)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            trainer.get_latest_save_path()
        trainer.config.clear()
        (Path(self.root) / 'optimizer.pt').write_bytes(b'legacy state')
        with self.assertRaisesRegex(ValueError, 'Legacy optimizer'):
            trainer.get_latest_save_path()

    def test_sharded_execution_is_rejected_before_model_loading(self):
        trainer = tiny_trainer(self.root)
        trainer.accelerator.distributed_type = SimpleNamespace(name='FSDP')
        with self.assertRaisesRegex(ValueError, 'FSDP'):
            trainer.run()

    def test_ema_save_and_failed_sampling_restore_training_weights(self):
        trainer = tiny_trainer(self.root)
        trainer.ema = ExponentialMovingAverage(trainer.network.parameters(), decay=.9)
        trainer.ema.shadow_params[0].fill_(9.)
        before = trainer.network.weight.detach().clone()
        path = save_checkpoint(trainer)
        self.assertEqual(load_file(path / 'model.safetensors')['weight'].item(), 9.)
        torch.testing.assert_close(before, trainer.network.weight)
        trainer.network.weight.data.fill_(10.)
        restore_training_state(trainer, read_checkpoint(path))
        torch.testing.assert_close(before, trainer.network.weight)
        trainer.network.can_merge_in = True
        trainer.sd.unet = trainer.network
        trainer.sd.vae = torch.nn.Linear(1, 1)
        trainer.sd.text_encoder = torch.nn.Linear(1, 1)
        trainer.adapter_config = None
        trainer.sample_config = trainer.first_sample_config = SampleConfig(prompts=['test'])
        trainer.trigger_word = None
        trainer.logger = None
        trainer.progress_bar = None
        captured = []
        def generate(*args, **kwargs):
            self.assertEqual(Path(args[0][0].output_path).parent, Path(self.root) / 'test-step000000/samples')
            captured.append(trainer.network.weight.item())
            raise RuntimeError('sample failure')
        trainer.sd.generate_images = generate
        with self.assertRaisesRegex(RuntimeError, 'sample failure'):
            trainer.sample(0)
        self.assertEqual(captured, [9.])
        torch.testing.assert_close(before, trainer.network.weight)
        self.assertTrue(trainer.network.can_merge_in)

    def test_automatic_writer_paths_and_shared_train_loss(self):
        for explicit in (None, str(Path(self.root) / 'explicit')):
            trainer = tiny_trainer(self.root)
            trainer.config = {'val_datasets': [{}]}
            trainer.log_dir = explicit
            trainer.writer = None
            BaseTrainProcess.setup_tensorboard(trainer)
            self.assertIsNone(trainer.writer)
            BaseTrainProcess.setup_tensorboard(trainer, purge_step=0)
            expected = explicit or str(Path(self.root) / 'tensorboard')
            self.assertEqual(Path(trainer.writer.log_dir), Path(expected))
            trainer.writer.add_scalar('loss', 1.5, 1)
            trainer.deterministic_validation_config = SimpleNamespace(val_every_n_steps=10)
            trainer._last_validation_step = None
            with patch('toolkit.validation.evaluate_validation', return_value={
                    'val_loss_mean': 2., 'val_loss_low_noise': 1., 'val_loss_high_noise': 3.}):
                trainer.validate()
            events = EventAccumulator(expected).Reload()
            self.assertIn('loss', events.Tags()['scalars'])
            self.assertEqual(len(events.Tags()['scalars']), 4)
            trainer.writer.close()

    def test_validation_schedule_flush_and_failure_publish_nothing(self):
        trainer = tiny_trainer(self.root)
        trainer.deterministic_validation_config = SimpleNamespace(val_every_n_steps=10)
        trainer._last_validation_step = None
        trainer.writer = SummaryWriter(self.root)
        metrics = {'val_loss_mean': 6, 'val_loss_low_noise': 3, 'val_loss_high_noise': 9}
        trainer.train_config.gradient_accumulation_steps = 2
        with patch('toolkit.validation.evaluate_validation', return_value=metrics):
            trainer.maybe_validate(initial=True)
            for microstep in range(50):
                trainer.is_grad_accumulation_step = microstep % 2 == 0
                trainer.hook_train_loop(2.)
                trainer.maybe_validate()
            trainer.maybe_validate(final=True)
            trainer.maybe_validate(final=True)
        events = EventAccumulator(self.root).Reload()
        for tag in metrics:
            self.assertEqual([e.step for e in events.Scalars(tag)], [0, 10, 20, 25])
        trainer.completed_optimizer_steps = 26
        with patch('toolkit.validation.evaluate_validation', side_effect=ValueError('manifest changed')):
            with self.assertRaisesRegex(RuntimeError, 'manifest changed'):
                trainer.validate()
        trainer.writer.close()
        for tag in metrics:
            self.assertEqual(len(EventAccumulator(self.root).Reload().Scalars(tag)), 4)
        trainer.writer = SummaryWriter(self.root, purge_step=20)
        trainer.completed_optimizer_steps = 20
        trainer._last_validation_step = None
        with patch('toolkit.validation.evaluate_validation', return_value=metrics):
            trainer.validate()
            trainer.completed_optimizer_steps = 30
            trainer.validate()
            trainer.validate()
        trainer.writer.close()
        self.assertEqual([e.step for e in EventAccumulator(self.root).Reload().Scalars('val_loss_mean')], [0, 10, 20, 30])


if __name__ == '__main__':
    unittest.main()
