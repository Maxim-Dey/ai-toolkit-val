from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import patch

import torch

from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import QwenImage2Model
from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel


class ConditionsTest(unittest.TestCase):
    def test_qwen21_fp32_vae_accepts_bf16_inputs_and_preserves_output_dtype(self):
        vae = torch.nn.Conv3d(4, 4, 1).float()
        vae.device, vae.dtype = torch.device('cpu'), torch.float32
        vae.config = SimpleNamespace(latents_mean=[0.] * 4, latents_std=[1.] * 4, z_dim=4)
        vae.encode = lambda images: SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: vae(images)))
        vae.decode = lambda latents: SimpleNamespace(sample=vae(latents))
        model = SimpleNamespace(vae=vae, vae_device_torch=torch.device('cpu'), vae_torch_dtype=torch.float32,
                                vae_scale_factor=16, model_config=SimpleNamespace(low_vram=False))
        model._latent_stats = MethodType(QwenImage2Model._latent_stats, model)
        images = torch.zeros(1, 3, 16, 16, dtype=torch.bfloat16)
        latents = QwenImage2Model.encode_images(model, images, dtype=torch.bfloat16)
        decoded = QwenImage2Model._decode_rgba(model, latents, dtype=torch.bfloat16)
        self.assertEqual(latents.dtype, torch.bfloat16)
        self.assertEqual(decoded.dtype, torch.bfloat16)
        self.assertEqual(tuple(decoded.shape), (1, 4, 16, 16))
        self.assertEqual(vae.weight.dtype, torch.float32)

    def test_edit_sampling_encodes_controls_in_vae_compute_dtype(self):
        from extensions_built_in.diffusion_models.qwen_image.qwen_image_pipelines import QwenImageEditPlusCustomPipeline
        vae = torch.nn.Conv3d(3, 3, 1).float()
        vae.dtype = torch.float32
        vae.config = SimpleNamespace(latents_mean=[0.] * 3, latents_std=[1.] * 3, z_dim=3)
        vae.encode = lambda images: SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: vae(images)))
        pipeline = QwenImageEditPlusCustomPipeline.__new__(QwenImageEditPlusCustomPipeline)
        pipeline.vae = vae
        pipeline.latent_channels = 3
        image = torch.zeros(1, 3, 1, 16, 16, dtype=torch.bfloat16)
        latents = pipeline._encode_vae_image(image, generator=None)
        self.assertEqual(latents.dtype, torch.bfloat16)
        self.assertEqual(vae.weight.dtype, torch.float32)

    def test_qwen_vae_uses_its_compute_dtype_and_returns_requested_dtype(self):
        from toolkit.models.v2.vae.qwen_image import QwenImageVAEHolderMixin
        vae = torch.nn.Conv3d(3, 3, 1).float()
        vae.device, vae.dtype = torch.device('cpu'), torch.float32
        vae.config = SimpleNamespace(latents_mean=[0.] * 3, latents_std=[1.] * 3, z_dim=3)
        vae.encode = lambda images: SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: vae(images)))
        vae.decode = lambda latents: SimpleNamespace(sample=vae(latents))
        model = SimpleNamespace(vae=vae, vae_device_torch=torch.device('cpu'), vae_torch_dtype=torch.float32,
                                vae_decode_tiled_on_low_vram=False)
        images = torch.zeros(1, 3, 16, 16, dtype=torch.bfloat16)
        latents = QwenImageVAEHolderMixin.encode_images(model, images, dtype=torch.bfloat16)
        decoded = QwenImageVAEHolderMixin.decode_latents(model, latents, dtype=torch.bfloat16)
        self.assertEqual(latents.dtype, torch.bfloat16)
        self.assertEqual(decoded.dtype, torch.bfloat16)
        self.assertEqual(vae.weight.dtype, torch.float32)

    def test_qwen_reuses_prepared_controls_in_order_for_all_checks(self):
        calls = []
        images = [torch.full((1, 3, 32, 32), float(index)) for index in (1, 2)]
        def encode(controls):
            value = controls[0][0].mean().item()
            calls.append(value)
            return torch.full((1, 4, 8), value), [(2, 2)]
        model = SimpleNamespace(device_torch=torch.device('cpu'), torch_dtype=torch.float32,
                                model=SimpleNamespace(device=torch.device('cpu'), to=lambda device: None), transformer=None,
                                _normalize_control_images=lambda controls, n: controls,
                                _prepare_control_images=lambda controls, **kwargs: controls,
                                _target_pixels=lambda size: size[0] * size[1], encode_condition_images=encode,
                                pad_prompt_embeds=lambda embeds: (torch.zeros(1, 3, 8), torch.ones(1, 3), torch.ones(1, 3, dtype=torch.bool)))
        batch = SimpleNamespace(control_tensor_list=[images], tensor=torch.zeros(1, 3, 32, 32),
                                fixed_condition_latents=None, fixed_condition_metadata=None)
        QwenImage2Model.prepare_validation_conditions(model, batch, 42)
        self.assertEqual(calls, [1, 2])
        with patch('extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2.run_transformer') as run:
            for _ in range(20):
                QwenImage2Model.get_noise_prediction(model, torch.zeros(1, 64, 2, 2), torch.tensor([106.25]), None, batch)
                fixed = run.call_args.kwargs['condition_latents']
                torch.testing.assert_close(fixed[:, :4], torch.ones(1, 4, 8))
                torch.testing.assert_close(fixed[:, 4:], torch.full((1, 4, 8), 2.))
        self.assertEqual(calls, [1, 2])

    def test_moss_reuses_prepared_references(self):
        calls = []
        def encode(images, **kwargs):
            calls.extend(image.mean().item() for image in images)
            return [image[:1] for image in images]
        model = SimpleNamespace(is_edit=True, device_torch=torch.device('cpu'), torch_dtype=torch.float32,
                                _encode_ref_latents=encode)
        batch = SimpleNamespace(control_tensor_list=[[torch.ones(3, 32, 32), torch.full((3, 32, 32), 2.)]],
                                tensor=torch.zeros(1, 3, 32, 32), fixed_condition_latents=None)
        MossImageModel.prepare_validation_conditions(model, batch, 42)
        for _ in range(20):
            refs = MossImageModel._batch_ref_latents(model, batch, 1, 1024)
            self.assertEqual([x.mean().item() for x in refs[0]], [1, 2])
        self.assertEqual(calls, [1, 2])


if __name__ == '__main__':
    unittest.main()
