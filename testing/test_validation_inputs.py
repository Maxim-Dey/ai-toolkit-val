import copy
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from toolkit.config_modules import normalize_dataset_configs
from toolkit.prompt_utils import PromptEmbeds
from toolkit.validation import ValidationInputs, isolated_modules, isolated_rng, fixed_noise, noise_levels, stable_seed


class TinyModel:
    def __init__(self):
        self.model_config = SimpleNamespace(arch='qwen_image_2', name_or_path='tiny', extras_name_or_path='tiny', model_kwargs={})
        self.vae = torch.nn.Linear(1, 1)
        self.unet = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(.5))
        self.unet[0].eval()
        self.text_encoder = torch.nn.Linear(1, 1)
        self.device = 'cpu'
        self.device_torch = torch.device('cpu')
        self.torch_dtype = torch.float32
        self.load_rgba = False
        self.encode_control_in_text_embeddings = True
        self.use_raw_control_images = True
        self.te_padding_side = 'right'

    def get_bucket_divisibility(self):
        return 16

    def get_latent_space_version(self):
        return 'tiny-v1'

    def get_text_embedding_space_version(self):
        return 'tiny-v1'

    def encode_images(self, images, **kwargs):
        self.vae.eval().requires_grad_(False)
        return images[:, :1, ::8, ::8] + torch.randn_like(images[:, :1, ::8, ::8])

    def encode_prompt(self, captions, **kwargs):
        return PromptEmbeds(torch.randn(1, 3, 4) + len(captions[0]))


class InputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ('val', 'mask', 'ctrl'):
            (self.root / folder).mkdir()
        for name, color in [('a', 'red'), ('b', 'blue')]:
            for folder in ('val', 'ctrl'):
                Image.new('RGB', (64, 80), color).save(self.root / folder / f'{name}.png')
            Image.fromarray(np.tile(np.arange(64, dtype=np.uint8) * 4, (80, 1))).save(self.root / 'mask' / f'{name}.png')
        _, self.config, _ = normalize_dataset_configs({'val_datasets': [{
            'val_path': str(self.root / 'val'), 'mask_path': str(self.root / 'mask'),
            'control_path': str(self.root / 'ctrl'), 'resolution': 64,
        }]})

    def test_preparation_is_reproducible_and_isolates_rng_modes_grad_flags(self):
        model = TinyModel()
        modes = [module.training for module in model.unet.modules()]
        torch_state = torch.get_rng_state().clone()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        prepared = ValidationInputs(self.config, model, [])
        torch.testing.assert_close(torch_state, torch.get_rng_state())
        self.assertEqual(python_state, random.getstate())
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        self.assertEqual(modes, [module.training for module in model.unet.modules()])
        self.assertTrue(model.vae.weight.requires_grad)
        original_walk = os.walk
        def reverse_walk(*args, **kwargs):
            for root, dirs, files in original_walk(*args, **kwargs):
                yield root, dirs, list(reversed(files))
        with patch('os.walk', reverse_walk):
            repeated = ValidationInputs(self.config, model, [], expected_manifest=prepared.manifest)
        for left, right in zip(prepared.records, repeated.records):
            torch.testing.assert_close(left.batch.latents, right.batch.latents, rtol=0, atol=0)
            torch.testing.assert_close(left.embeds.text_embeds, right.embeds.text_embeds, rtol=0, atol=0)
            torch.testing.assert_close(left.batch.mask_tensor, right.batch.mask_tensor, rtol=0, atol=0)
            self.assertEqual(left.batch.mask_tensor.shape[-2:], left.batch.tensor.shape[-2:])
        (self.root / 'val/a.txt').write_text('new caption')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            prepared.verify()

    def test_automatic_control_is_generated_once_and_frozen(self):
        self.config.datasets[0].controls = ['depth']
        calls = []
        def generate(generator, path, role):
            calls.append(path)
            output = Path(generator.control_save_path(path, role))
            output.parent.mkdir(exist_ok=True)
            Image.new('RGB', (64, 80), (random.randint(0, 255), 0, 0)).save(output)
            return str(output)
        with patch('toolkit.control_generator.ControlGenerator._generate_control', generate):
            first = ValidationInputs(self.config, TinyModel(), [])
            second = ValidationInputs(self.config, TinyModel(), [])
            self.assertEqual(len(calls), 2)
            for path in (self.root / 'val/_controls').iterdir():
                path.unlink()
            regenerated = ValidationInputs(self.config, TinyModel(), [])
        self.assertEqual(len(calls), 4)
        self.assertEqual(first.manifest, second.manifest)
        self.assertEqual(first.manifest, regenerated.manifest)
        for a, b in zip(first.records, second.records):
            for x, y in zip(a.batch.control_tensor_list[0], b.batch.control_tensor_list[0]):
                torch.testing.assert_close(x, y)

    def test_exception_restores_state(self):
        model = TinyModel()
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with isolated_modules([model.vae, model.unet]):
                with isolated_rng(123):
                    model.vae.requires_grad_(False)
                    torch.randn(5)
                    raise RuntimeError('injected')
        torch.testing.assert_close(before, torch.get_rng_state())
        self.assertTrue(model.vae.weight.requires_grad)
        self.assertTrue(model.unet.training)
        self.assertFalse(model.unet[0].training)

    def test_resume_rejects_changed_preparation_precision(self):
        model = TinyModel()
        prepared = ValidationInputs(self.config, model, [])
        model.vae_torch_dtype = torch.bfloat16
        with patch.object(model, 'encode_images', side_effect=AssertionError('must fail before encoding')):
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                ValidationInputs(self.config, model, [], expected_manifest=prepared.manifest)

    def test_grid_and_noise(self):
        self.assertEqual(noise_levels(2), [.275, .7250000000000001])
        self.assertAlmostEqual(noise_levels(8)[0] * 1000, 106.25)
        np.testing.assert_allclose(noise_levels(10), [.095, .185, .275, .365, .455, .545, .635, .725, .815, .905])
        state = torch.get_rng_state().clone()
        latent = torch.zeros(1, 4, 8, 8)
        a = fixed_noise(latent, 42, 'ab' * 32, 1, 2)
        torch.testing.assert_close(a, fixed_noise(latent.clone(), 42, 'AB' * 32, 1, 2))
        self.assertFalse(torch.equal(a, fixed_noise(latent, 42, 'ab' * 32, 2, 2)))
        torch.testing.assert_close(state, torch.get_rng_state())
        self.assertEqual(stable_seed(42, 'ab' * 32, 1, 2), stable_seed(42, 'ab' * 32, 1, 2))


if __name__ == '__main__':
    unittest.main()
