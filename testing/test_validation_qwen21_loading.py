"""Полные HF-веса Qwen-2.1/Turbo должны загрузиться без пропущенных параметров."""
import json
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from extensions_built_in.diffusion_models.qwen_image_2.src.transformer import QwenImage21Transformer2DModel


class Qwen21LoadingTest(unittest.TestCase):
    def fixture(self, folder, sharded):
        model = QwenImage21Transformer2DModel(
            in_channels=4, out_channels=4, num_layers=1, num_attention_heads=2,
            attention_head_dim=8, context_in_dim=16, axes_dims_rope=(2, 2, 4), mlp_ratio=2,
        )
        model.save_config(folder)
        state = dict(model.state_dict())
        gate_up = state.pop('transformer_blocks.0.img_mlp.gate_up.weight')
        gate, up = gate_up.chunk(2, dim=0)
        state['transformer_blocks.0.img_mlp.gate_layer.weight'] = gate.clone()
        up_key = 'transformer_blocks.0.img_mlp.proj.weight'
        if sharded:
            first, second = 'first.safetensors', 'second.safetensors'
            save_file(state, folder / first)
            save_file({up_key: up.clone()}, folder / second)
            mapping = {key: first for key in state}
            mapping[up_key] = second
            (folder / 'diffusion_pytorch_model.safetensors.index.json').write_text(
                json.dumps({'weight_map': mapping}))
        else:
            state[up_key] = up.clone()
            save_file(state, folder / 'diffusion_pytorch_model.safetensors')
        return model

    def test_local_hf_weights_and_split_shards_are_exact(self):
        for sharded in (False, True):
            with self.subTest(sharded=sharded), tempfile.TemporaryDirectory() as path:
                folder = Path(path) / 'transformer'
                folder.mkdir()
                original = self.fixture(folder, sharded)
                loaded = QwenImage21Transformer2DModel.load_model(
                    path, dtype=torch.float32, device=torch.device('cpu'))
                self.assertFalse(any(p.is_meta for p in loaded.parameters()))
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)

    def test_missing_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            folder = Path(path)
            model = self.fixture(folder, False)
            state = dict(model.state_dict())
            state.pop('img_in.weight')
            save_file(state, folder / 'diffusion_pytorch_model.safetensors')
            with self.assertRaisesRegex(RuntimeError, 'img_in.weight'):
                QwenImage21Transformer2DModel.load_model(path, dtype=torch.float32)


if __name__ == '__main__':
    unittest.main()
