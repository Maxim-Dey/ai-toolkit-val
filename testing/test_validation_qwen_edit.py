"""Edit-2511 wrapper and full validation path with small CPU encoder stand-ins."""
import copy
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from extensions_built_in.diffusion_models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusModel
from toolkit.models.base_model import BaseModel
from toolkit.config_modules import normalize_dataset_configs, validate_validation_model
from toolkit.validation import ValidationInputs, evaluate_validation, mask_multiplier
from testing.test_validation_inputs import TinyModel
from testing import test_validation_loss


MODULE = 'extensions_built_in.diffusion_models.qwen_image.qwen_image_edit_plus'


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.3))
        self.device, self.dtype = torch.device('cpu'), torch.float32
        self.calls = []

    def forward(self, hidden_states, timestep, encoder_hidden_states, img_shapes, **kwargs):
        self.calls.append((hidden_states.detach().clone(), copy.deepcopy(img_shapes), timestep.clone()))
        return (hidden_states * self.weight + hidden_states.mean(dim=1, keepdim=True)
                + encoder_hidden_states.mean(),)


def edit_model(stochastic=True, match_target_res=True):
    model = TinyModel()
    model.model_config.arch = 'qwen_image_edit_plus'
    model.model_config.model_kwargs = {'match_target_res': match_target_res}
    model.has_multiple_control_images = True
    model.vae.device = model.device_torch
    model.text_encoder.device = model.device_torch
    model.unet = model.transformer = TinyTransformer()
    model.vae_calls, model.text_calls = [], []
    model.is_flow_matching, model.x0_pred, model.do_masked_loss = True, False, True
    model.prediction_type = 'epsilon'

    def encode_images(images, **kwargs):
        model.vae_calls.append(images.detach().clone())
        latent = images[:, :1, ::8, ::8]
        return latent + torch.randn_like(latent) * .01 if stochastic else latent

    def encode_text(prompts, image, **kwargs):
        model.text_calls.append([value.clone() for value in image])
        return torch.stack([value.mean() for value in image]).view(1, -1, 1), None

    model.encode_images = encode_images
    model.pipeline = SimpleNamespace(vae_scale_factor=8, text_encoder=model.text_encoder, encode_prompt=encode_text)
    for name in ('_encode_control_image', 'prepare_validation_conditions', 'get_prompt_embeds',
                 'get_noise_prediction', 'get_loss_target'):
        setattr(model, name, MethodType(getattr(QwenImageEditPlusModel, name), model))
    for name in ('encode_prompt', 'predict_noise', 'scale_loss'):
        setattr(model, name, MethodType(getattr(BaseModel, name), model))
    return model


class QwenEditValidationTest(unittest.TestCase):
    def setUp(self):
        # Keep the wrapper's real resizing and packing, using small image areas.
        for name in ('VAE_IMAGE_SIZE', 'CONDITION_IMAGE_SIZE'):
            mock = patch(f'{MODULE}.{name}', 128 * 128)
            mock.start()
            self.addCleanup(mock.stop)

    def test_cached_controls_match_training_order_shapes_and_masked_loss(self):
        for count in (1, 2):
            for match in (False, True):
                with self.subTest(controls=count, match_target_res=match):
                    model = edit_model(stochastic=False, match_target_res=match)
                    controls = [torch.full((3, 64, 64), .25), torch.full((3, 32, 64), .75)][:count]
                    latent = torch.arange(64, dtype=torch.float32).view(1, 1, 8, 8) / 64
                    batch = SimpleNamespace(latents=latent, tensor=torch.zeros(1, 3, 64, 64),
                                            control_tensor=controls[0][None] if count == 1 else None,
                                            control_tensor_list=[controls] if count > 1 else None,
                                            fixed_condition_latents=None, fixed_condition_metadata=None,
                                            mask_tensor=torch.linspace(.1, 1, 64).view(1, 1, 8, 8),
                                            loss_multiplier_list=[1.3], get_is_reg_list=lambda: [False])
                    rng = torch.get_rng_state().clone()
                    model.prepare_validation_conditions(batch, 42)
                    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                    self.assertEqual(len(model.vae_calls), count)
                    self.assertEqual(model.vae_calls[0].shape[-2:], (64, 64) if match else (128, 128))
                    embeds = model.encode_prompt(['edit'], control_images=[controls], target_size=(64, 64))
                    self.assertEqual([round(image.mean().item(), 2) for image in model.text_calls[0]], [.25, .75][:count])
                    t = torch.tensor([106.25])
                    for _ in range(20):
                        cached_prediction = model.get_noise_prediction(latent, t, embeds, batch)
                    self.assertEqual(len(model.vae_calls), count)
                    packed, shapes, received_t = model.transformer.calls[-1]
                    torch.testing.assert_close(received_t, t / 1000, rtol=0, atol=0)
                    self.assertEqual(shapes, [[(1, 4, 4), *batch.fixed_condition_metadata]])
                    offset = 16
                    for index, (_, height, width) in enumerate(batch.fixed_condition_metadata):
                        tokens = height * width
                        torch.testing.assert_close(packed[:, offset:offset + tokens],
                                                   torch.full_like(packed[:, offset:offset + tokens], [-.5, .5][index]))
                        offset += tokens
                    self.assertEqual(offset, packed.shape[1])
                    training_batch = copy.copy(batch)
                    training_batch.fixed_condition_latents = None
                    raw_prediction = model.get_noise_prediction(latent, t, embeds, training_batch)
                    self.assertEqual(len(model.vae_calls), count * 2)
                    torch.testing.assert_close(cached_prediction, raw_prediction, rtol=0, atol=0)
                    torch.testing.assert_close(packed, model.transformer.calls[-1][0], rtol=0, atol=0)
                    trainer = test_validation_loss.LossTest().trainer()
                    model.noise_scheduler = trainer.sd.noise_scheduler
                    trainer.sd = model
                    mask = mask_multiplier(batch, latent)
                    args = (cached_prediction, torch.ones_like(latent), latent, t, batch)
                    train_loss = trainer.calculate_loss(*args, mask_multiplier=mask)
                    val_loss = trainer.calculate_loss(*args, mask_multiplier=mask, validation=True)
                    torch.testing.assert_close(train_loss, val_loss, rtol=0, atol=0)
                    raw_prediction.sum().backward()
                    self.assertIsNotNone(model.transformer.weight.grad)

    def test_preparation_failure_restores_rng_and_does_not_publish_cache(self):
        model = edit_model()
        batch = SimpleNamespace(latents=torch.zeros(1, 1, 8, 8), control_tensor=None,
                                control_tensor_list=[[torch.ones(3, 64, 64)]],
                                fixed_condition_latents=None, fixed_condition_metadata=None)
        before = torch.get_rng_state().clone()
        def fail(*args, **kwargs):
            torch.randn(3)
            raise RuntimeError('encoder failed')
        model.encode_images = fail
        with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
            model.prepare_validation_conditions(batch, 42)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        self.assertIsNone(batch.fixed_condition_latents)
        batch.control_tensor_list = None
        with self.assertRaisesRegex(ValueError, 'at least one control'):
            model.prepare_validation_conditions(batch, 42)

    def test_training_batch_keeps_each_samples_control_tokens(self):
        model = edit_model(stochastic=False)
        images = [[torch.full((3, 64, 64), value)] for value in (.2, .8)]
        batch = SimpleNamespace(control_tensor_list=images, control_tensor=None, fixed_condition_latents=None)
        embeds = model.encode_prompt(['first', 'second'], control_images=images)
        model.get_noise_prediction(torch.zeros(2, 1, 8, 8), torch.tensor([100., 500.]), embeds, batch)
        packed, shapes, _ = model.transformer.calls[-1]
        self.assertEqual(len(model.vae_calls), 2)
        self.assertEqual(shapes, [[(1, 4, 4), (1, 4, 4)]] * 2)
        for index, value in enumerate((-.6, .6)):
            torch.testing.assert_close(packed[index, 16:], torch.full_like(packed[index, 16:], value))

    def test_full_pass_resume_alternatives_automatic_controls_and_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for subdir in ('val', 'first', 'second', 'mask'):
                (root / subdir).mkdir()
            for name, value in [('a', 80), ('b', 160)]:
                for subdir, color in [('val', value), ('first', 40), ('second', 210)]:
                    Image.new('RGB', (64, 64), (color,) * 3).save(root / subdir / f'{name}.png')
                Image.fromarray(np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))).save(root / 'mask' / f'{name}.png')
            raw = dict(val_path=str(root / 'val'), resolution=64, mask_path=str(root / 'mask'),
                       control_path=[str(root / 'first'), str(root / 'second')], controls=['depth'],
                       val_level_noise_n=8, val_seed_noise_n=2)
            alternative = {**raw, 'control_path': list(reversed(raw['control_path']))}
            _, config, _ = normalize_dataset_configs({'val_datasets': [raw, alternative]})
            model = edit_model()
            validate_validation_model(model.model_config, config.datasets)
            trainer = test_validation_loss.LossTest().trainer()
            model.noise_scheduler = trainer.sd.noise_scheduler
            trainer.sd, trainer.network = model, None
            before = torch.get_rng_state().clone()
            generated = []
            def generate(generator, path, role):
                generated.append(path)
                output = Path(generator.control_save_path(path, role))
                output.parent.mkdir(exist_ok=True)
                color = random.randint(10, 240)
                Image.new('RGB', (64, 64), (color,) * 3).save(output)
                return str(output)
            with patch('toolkit.control_generator.ControlGenerator._generate_control', generate):
                inputs = ValidationInputs(config, model, [])
                torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
                self.assertEqual(len(generated), 2)
                self.assertEqual(len(inputs.records), 4)
                prepared_calls = len(model.vae_calls), len(model.text_calls)
                optimizer = torch.optim.AdamW(model.unet.parameters(), lr=.01)
                def update():
                    optimizer.zero_grad()
                    (model.unet.weight * torch.randn(5)).square().mean().backward()
                    optimizer.step()
                    return model.unet.weight.detach().clone()
                update()
                weights = model.unet.weight.detach().clone()
                grad = model.unet.weight.grad.clone()
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                evaluation_rng = torch.get_rng_state().clone()
                metrics = evaluate_validation(trainer, inputs)
                self.assertEqual(metrics, evaluate_validation(trainer, inputs))
                self.assertEqual(prepared_calls, (len(model.vae_calls), len(model.text_calls)))
                self.assertEqual(len(model.transformer.calls), 4 * 16 * 2)
                for record in inputs.records:
                    self.assertEqual(len(record.batch.fixed_condition_metadata), 3)
                    self.assertIsNotNone(record.batch.mask_tensor)
                torch.testing.assert_close(grad, model.unet.weight.grad, rtol=0, atol=0)
                torch.testing.assert_close(weights, model.unet.weight, rtol=0, atol=0)
                torch.testing.assert_close(evaluation_rng, torch.get_rng_state(), rtol=0, atol=0)
                self.assertTrue(model.unet.training)
                with patch.object(model, 'get_noise_prediction', side_effect=RuntimeError('prediction failed')):
                    with self.assertRaisesRegex(RuntimeError, 'prediction failed'):
                        evaluate_validation(trainer, inputs)
                torch.testing.assert_close(evaluation_rng, torch.get_rng_state(), rtol=0, atol=0)
                self.assertTrue(model.unet.training)
                with_val = update()
                with torch.no_grad():
                    model.unet.weight.copy_(weights)
                optimizer.load_state_dict(optimizer_state)
                torch.set_rng_state(evaluation_rng)
                torch.testing.assert_close(with_val, update(), rtol=0, atol=0)
                # Rebuild inputs from the serialized manifest with reversed traversal.
                model.unet.weight.data.copy_(weights)
                restored_config = copy.deepcopy(config)
                restored_config.datasets.reverse()
                walk = os.walk
                def reverse_walk(*args, **kwargs):
                    for parent, dirs, files in walk(*args, **kwargs):
                        yield parent, dirs, list(reversed(files))
                with patch('os.walk', reverse_walk):
                    resumed = ValidationInputs(restored_config, model, [],
                                               expected_manifest=json.loads(json.dumps(inputs.manifest)))
                self.assertEqual(len(generated), 2)
                for left, right in zip(inputs.records, resumed.records):
                    for name in ('latents', 'fixed_condition_latents', 'mask_tensor'):
                        torch.testing.assert_close(getattr(left.batch, name), getattr(right.batch, name), rtol=0, atol=0)
                    self.assertEqual(left.batch.fixed_condition_metadata, right.batch.fixed_condition_metadata)
                    torch.testing.assert_close(left.embeds.text_embeds, right.embeds.text_embeds, rtol=0, atol=0)
                self.assertEqual(metrics, evaluate_validation(trainer, resumed))
                Image.new('RGB', (64, 64), 'black').save(root / 'second/a.png')
                calls = len(model.transformer.calls)
                with self.assertRaisesRegex(ValueError, 'manifest changed'):
                    evaluate_validation(trainer, inputs)
                self.assertEqual(calls, len(model.transformer.calls))


if __name__ == '__main__':
    unittest.main()
