import copy
from types import SimpleNamespace
import unittest

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import TrainConfig
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.train_tools import LearnableSNRGamma, apply_learnable_snr_gos, apply_snr_weight
from toolkit.validation import aggregate_validation, interpolate_weights, mask_multiplier, preflight_loss


class LossTest(unittest.TestCase):
    def trainer(self, loss='mse'):
        trainer = object.__new__(SDTrainer)
        trainer.train_config = TrainConfig(loss_type=loss, dtype='float32', noise_scheduler='flowmatch')
        trainer.device_torch = torch.device('cpu')
        trainer.uses_dataset_split = True
        trainer.adapter = trainer.assistant_adapter = trainer.dfe = None
        trainer.additional_logs = {}
        trainer.sd = SimpleNamespace(is_flow_matching=True, prediction_type='epsilon',
                                     noise_scheduler=CustomFlowMatchEulerDiscreteScheduler(),
                                     scale_loss=lambda loss: loss * 1.7, x0_pred=False, do_masked_loss=True,
                                     get_loss_target=lambda **kw: kw['noise'] - kw['batch'].latents)
        trainer.deterministic_validation_config = SimpleNamespace(val_level_noise_n=8)
        return trainer

    def test_train_val_loss_agree_for_mask_scaling_and_options(self):
        for loss in ('mse', 'mae', 'pseudo_huber', 'wavelet', 'band'):
            for masked in (False, True):
                for target in ('noise', 'source'):
                    with self.subTest(loss=loss, masked=masked, target=target):
                        trainer = self.trainer(loss)
                        trainer.train_config.loss_target = target
                        latent, noise, pred = [torch.randn(1, 4, 8, 8) for _ in range(3)]
                        batch = SimpleNamespace(latents=latent, mask_tensor=torch.rand(1, 1, 16, 16).clamp_min(.1) if masked else None,
                                                sigmas=torch.tensor(.4), loss_multiplier_list=[1.5], get_is_reg_list=lambda: [False])
                        mask = mask_multiplier(batch, latent)
                        args = (pred, noise, latent * .6 + noise * .4, torch.tensor([400.]), batch)
                        train = trainer.calculate_loss(*args, mask_multiplier=mask)
                        val = trainer.calculate_loss(*args, mask_multiplier=mask, validation=True)
                        torch.testing.assert_close(train, val, rtol=0, atol=0)

    def test_existing_loss_options_and_table_weights_match(self):
        for option in ('t0_loss_target', 'do_fft_loss', 'target_norm_std', 'match_noise_norm', 'linear_timesteps', 'linear_timesteps2'):
            with self.subTest(option=option):
                trainer = self.trainer()
                setattr(trainer.train_config, option, True)
                trainer.sd.get_x0_prediction = lambda pred, noisy, t: noisy - t * pred
                latent, noise, pred = [torch.randn(1, 4, 8, 8) for _ in range(3)]
                batch = SimpleNamespace(latents=latent, mask_tensor=torch.ones(1, 1, 8, 8),
                                        loss_multiplier_list=[1.], get_is_reg_list=lambda: [False])
                args = (pred, noise, .6 * latent + .4 * noise, torch.tensor([400.]), batch)
                torch.testing.assert_close(trainer.calculate_loss(*args), trainer.calculate_loss(*args, validation=True))

    def test_fractional_weights_preserve_pairs_and_nodes(self):
        x = torch.tensor([1000., 100., 1.])
        y = torch.tensor([10., 1., 0.])
        torch.testing.assert_close(interpolate_weights(x, x, y), y)
        value = interpolate_weights(torch.tensor([106.25]), x, y)
        torch.testing.assert_close(value, torch.tensor([1.0625]))
        for coordinates, query in [(torch.tensor([1., 1.]), 1.), (x, 0.), (x, 1001.)]:
            with self.assertRaises(ValueError):
                interpolate_weights(torch.tensor([query]), coordinates, torch.ones_like(coordinates))
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        before = scheduler.timesteps.clone()
        query = scheduler.timesteps[20:23]
        torch.testing.assert_close(scheduler.get_weights_for_timesteps(query), scheduler.get_weights_for_timesteps(query, interpolate=True))
        scheduler.get_weights_for_timesteps(torch.tensor([106.25]), interpolate=True)
        torch.testing.assert_close(before, scheduler.timesteps)

    def test_snr_read_only_does_not_train_coefficients(self):
        scheduler = SimpleNamespace(all_snr=torch.linspace(1, 10, 1001), timesteps=torch.arange(1000., -1., -1.))
        trainer = LearnableSNRGamma(scheduler, device='cpu')
        before = copy.deepcopy(trainer.optimizer.state_dict())
        params = [p.clone() for p in (trainer.gamma, trainer.offset_1, trainer.offset_2, trainer.scale)]
        result = apply_learnable_snr_gos(torch.tensor([2.]), torch.tensor([106.25]), trainer, read_only=True)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(before, trainer.optimizer.state_dict())
        self.assertEqual(trainer.buffer, [])
        for expected, actual in zip(params, (trainer.gamma, trainer.offset_1, trainer.offset_2, trainer.scale)):
            torch.testing.assert_close(expected, actual)
            self.assertIsNone(actual.grad)
        query = torch.tensor([106.])
        torch.testing.assert_close(apply_snr_weight(torch.ones(1), query, scheduler, 2.),
                                   apply_snr_weight(torch.ones(1), query, scheduler, 2., interpolate=True))

    def test_unknown_loss_and_unavailable_snr_fail_preflight(self):
        for option in ('adapter_assist_name_or_path', 'diffusion_feature_extractor_path'):
            trainer = self.trainer()
            setattr(trainer.train_config, option, '/not-loaded-yet')
            with self.assertRaisesRegex(ValueError, 'read-only'):
                preflight_loss(trainer)
        with self.assertRaisesRegex(ValueError, 'loss_type'):
            preflight_loss(self.trainer('typo'))
        trainer = self.trainer()
        trainer.train_config.snr_gamma = 5
        with self.assertRaisesRegex(ValueError, 'SNR'):
            preflight_loss(trainer)

    def test_hierarchical_mean_and_full_pass(self):
        records = [SimpleNamespace(target_hash='a', key='1', loss=2), SimpleNamespace(target_hash='a', key='2', loss=4),
                   SimpleNamespace(target_hash='b', key='3', loss=9)]
        config = SimpleNamespace(val_level_noise_n=8, val_seed_noise_n=2)
        calls = []
        def evaluate(record, t, i, j):
            calls.append((record.key, t, i, j))
            return record.loss
        result = aggregate_validation(records, config, evaluate)
        self.assertEqual(list(result.values()), [6, 6, 6])
        self.assertEqual(len(calls), 48)
        self.assertEqual(sum(t < .5 for _, t, _, _ in calls), 24)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            aggregate_validation(records, config, lambda *args: float('nan'))


if __name__ == '__main__':
    unittest.main()
