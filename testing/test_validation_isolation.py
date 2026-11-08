import copy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from testing.test_validation_inputs import TinyModel
from testing import test_validation_loss
from toolkit.prompt_utils import PromptEmbeds
from toolkit.validation import ValidationRecord, evaluate_validation


class IsolationTest(unittest.TestCase):
    def make_trainer(self):
        trainer = test_validation_loss.LossTest().trainer()
        sd = TinyModel()
        sd.unet = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.Dropout(.4))
        sd.unet[0].eval()
        sd.network = torch.nn.Linear(1, 1)
        sd.network.multiplier = .7
        sd.network.is_active = False
        sd.noise_scheduler = trainer.sd.noise_scheduler
        sd.prediction_type = 'epsilon'
        sd.is_flow_matching = True
        sd.do_masked_loss = True
        sd.x0_pred = False
        sd.get_loss_target = lambda **kw: kw['noise'] - kw['batch'].latents
        sd.scale_loss = lambda loss: loss * 1.3
        sd.predict_noise = lambda **kw: sd.unet(kw['latents']) + sd.network.weight.view(1, 1, 1, 1)
        trainer.sd = sd
        trainer.network = sd.network
        trainer.additional_logs = {'training': 5}
        trainer.optimizer = torch.optim.AdamW(sd.unet.parameters(), lr=.01)
        batch = SimpleNamespace(latents=torch.zeros(1, 1, 4, 4), mask_tensor=torch.ones(1, 1, 4, 4),
                                loss_multiplier_list=[1.], get_is_reg_list=lambda: [False],
                                get_network_weight_list=lambda: [1.])
        trainer.inputs = SimpleNamespace(verify=lambda: None, scheduler=copy.deepcopy(sd.noise_scheduler),
                                         config=SimpleNamespace(val_level_noise_n=2, val_seed_noise_n=2, val_seed_noise=42),
                                         records=[ValidationRecord('a', 'ab' * 32, batch, PromptEmbeds(torch.zeros(1, 2, 4)))])
        return trainer

    def test_incomplete_and_nonfinite_passes_publish_no_metrics(self):
        for last in (torch.tensor(float('nan')), torch.tensor(float('inf')), RuntimeError('interrupted'), None):
            with self.subTest(last=last), tempfile.TemporaryDirectory() as root:
                trainer = self.make_trainer()
                trainer.accelerator = SimpleNamespace(is_main_process=True)
                trainer.completed_optimizer_steps = 0
                trainer._last_validation_step = None
                trainer._validation_inputs = trainer.inputs
                trainer.writer = SummaryWriter(root)
                with patch.object(trainer, 'validation_loss', side_effect=[torch.tensor(1.)] * 3 + [last]):
                    with self.assertRaises(RuntimeError):
                        trainer.validate()
                trainer.writer.close()
                self.assertEqual(EventAccumulator(root).Reload().Tags()['scalars'], [])

    def test_success_exception_and_next_update_preserve_training_state(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                trainer = self.make_trainer()
                trainer.sd.unet(torch.ones(1, 1, 4, 4)).sum().backward()
                params = list(trainer.sd.unet.parameters()) + list(trainer.network.parameters())
                values = [p.clone() for p in params]
                gradients = [p.grad.clone() if p.grad is not None else None for p in params]
                rng = torch.get_rng_state().clone()
                modes = [m.training for m in trainer.sd.unet.modules()]
                opt_state = copy.deepcopy(trainer.optimizer.state_dict())
                if fail:
                    with patch.object(trainer, 'validation_loss', side_effect=RuntimeError('injected')):
                        with self.assertRaises(RuntimeError):
                            evaluate_validation(trainer, trainer.inputs)
                else:
                    a = evaluate_validation(trainer, trainer.inputs)
                    b = evaluate_validation(trainer, trainer.inputs)
                    self.assertEqual(a, b)
                torch.testing.assert_close(rng, torch.get_rng_state())
                self.assertEqual(modes, [m.training for m in trainer.sd.unet.modules()])
                self.assertEqual(trainer.additional_logs, {'training': 5})
                self.assertEqual(trainer.network.multiplier, .7)
                self.assertFalse(trainer.network.is_active)
                self.assertEqual(opt_state, trainer.optimizer.state_dict())
                for before, grad, after in zip(values, gradients, params):
                    torch.testing.assert_close(before, after)
                    if grad is None:
                        self.assertIsNone(after.grad)
                    else:
                        torch.testing.assert_close(grad, after.grad)
                def update():
                    trainer.optimizer.zero_grad()
                    trainer.sd.unet(torch.randn(1, 1, 4, 4)).square().mean().backward()
                    trainer.optimizer.step()
                    return [p.detach().clone() for p in params]
                with_val = update()
                with torch.no_grad():
                    for p, value in zip(params, values):
                        p.copy_(value)
                trainer.optimizer.load_state_dict(opt_state)
                torch.set_rng_state(rng)
                without_val = update()
                for a, b in zip(with_val, without_val):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
