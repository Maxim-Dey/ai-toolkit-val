import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from safetensors.torch import load_file

from testing.test_validation_lifecycle import tiny_trainer
from toolkit.config_modules import TrainConfig
from toolkit.scheduler import get_lr_scheduler
from toolkit.validation_checkpoint import save_checkpoint, read_checkpoint, restore_training_state


class WarmupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(__file__).resolve().parents[1] / 'config/qwen_image_21_kibm.yaml'
        self.config = TrainConfig(**yaml.safe_load(path.read_text(encoding='utf-8'))['config']['process'][0]['train'])

    def trainer(self):
        trainer = tiny_trainer(self.temp.name)
        for group in trainer.optimizer.param_groups:
            group['lr'] = group['initial_lr'] = self.config.lr
        trainer.lr_scheduler = get_lr_scheduler(
            self.config.lr_scheduler, trainer.optimizer, total_iters=self.config.steps,
            **self.config.lr_scheduler_params,
        )
        return trainer

    def test_main_config_100_updates_then_constant_and_legacy_default(self):
        self.assertEqual(self.config.lr_scheduler, 'constant_with_warmup')
        self.assertEqual(self.config.lr_scheduler_params, {'num_warmup_steps': 100})
        self.assertEqual(TrainConfig().lr_scheduler, 'constant')
        trainer = self.trainer()
        self.assertEqual(trainer.optimizer.param_groups[0]['lr'], 0)
        for step in range(1, 111):
            trainer.hook_train_loop([2., 4.])
            self.assertEqual(trainer.completed_optimizer_steps, step)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], 0.0001 * min(step / 100, 1))

    def test_microbatches_skipped_update_and_oom_do_not_advance_warmup(self):
        trainer = self.trainer()
        trainer.train_config.gradient_accumulation_steps = 2
        trainer.is_grad_accumulation_step = True
        trainer.hook_train_loop(2.)
        self.assertEqual(trainer.lr_scheduler.last_epoch, 0)
        self.assertEqual(trainer.optimizer.param_groups[0]['lr'], 0)
        trainer.is_grad_accumulation_step = False
        trainer.hook_train_loop(4.)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], 0.000001)
        before = copy.deepcopy(trainer.lr_scheduler.state_dict())
        trainer.accelerator.optimizer_step_was_skipped = True
        with patch.object(trainer.optimizer, 'step'):
            trainer.hook_train_loop(5.)
        self.assertEqual(trainer.lr_scheduler.state_dict(), before)
        trainer.accelerator.optimizer_step_was_skipped = False
        with patch.object(trainer, 'train_single_accumulation', side_effect=torch.cuda.OutOfMemoryError('injected')):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                trainer.hook_train_loop(5.)
        self.assertEqual(trainer.lr_scheduler.state_dict(), before)

    def test_checkpoint_resume_mid_warmup_matches_uninterrupted_updates(self):
        original = self.trainer()
        for _ in range(50):
            original.hook_train_loop([2., 4.])
        checkpoint = save_checkpoint(original)
        resumed = self.trainer()
        resumed.network.load_state_dict(load_file(checkpoint / 'model.safetensors'))
        restore_training_state(resumed, read_checkpoint(checkpoint))
        self.assertEqual(resumed.completed_optimizer_steps, 50)
        self.assertAlmostEqual(resumed.optimizer.param_groups[0]['lr'], 0.00005)
        for _ in range(60):
            original.hook_train_loop([2., 4.])
            resumed.hook_train_loop([2., 4.])
            self.assertEqual(original.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
            self.assertEqual(original.optimizer.param_groups[0]['lr'], resumed.optimizer.param_groups[0]['lr'])
            torch.testing.assert_close(original.network.weight, resumed.network.weight, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
