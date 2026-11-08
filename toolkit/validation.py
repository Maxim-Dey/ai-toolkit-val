"""Fixed validation inputs and reproducible flow-matching evaluation."""
import hashlib
import json
import os
import random
import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace, MethodType
from contextlib import contextmanager
from pathlib import Path

from PIL import Image
import numpy as np
import torch


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')
MANIFEST_VERSION = 2  # VAE preparation honors its configured compute dtype.
NOISE_VERSION = 1


def stable_seed(*values):
    data = json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big') & 0x7fffffffffffffff


@contextmanager
def isolated_rng(seed=None):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def isolated_modules(modules):
    """Restore modes, placement and grad flags changed by existing encoders."""
    modules = [m for m in modules if isinstance(m, torch.nn.Module)]
    modes = {child: child.training for module in modules for child in module.modules()}
    params = {p: (p.device, p.dtype, p.requires_grad, p.grad,
                  (p.grad.device, p.grad.dtype) if p.grad is not None else None)
              for module in modules for p in module.parameters()}
    buffers = [(m, name, value, value.clone()) for m in modes for name, value in m._buffers.items() if value is not None]
    with isolated_rng(), torch.no_grad():
        try:
            for module in modules:
                module.eval()
            yield
        finally:
            for param, (device, dtype, requires_grad, grad, grad_placement) in params.items():
                if param.device != device or param.dtype != dtype:
                    param.data = param.data.to(device=device, dtype=dtype)
                if grad is not None:
                    grad.data = grad.data.to(device=grad_placement[0], dtype=grad_placement[1])
                param.grad = grad
                param.requires_grad_(requires_grad)
            for module, name, original, value in buffers:
                original.copy_(value)
                module._buffers[name] = original
            for module, training in modes.items():
                module.training = training


def model_modules(sd):
    encoders = sd.text_encoder if isinstance(sd.text_encoder, list) else [sd.text_encoder]
    return [sd.unet, sd.vae, getattr(sd, 'network', None), getattr(sd, 'adapter', None)] + encoders


def noise_levels(count):
    if type(count) is not int or count < 2 or count % 2:
        raise ValueError('val_level_noise_n must be even and >= 2')
    return [0.05 + (i - 0.5) * 0.9 / count for i in range(1, count + 1)]


def fixed_noise(latent, seed, target_hash, i, j):
    generator = torch.Generator(device='cpu').manual_seed(stable_seed(seed, target_hash.lower(), i, j))
    return torch.randn(latent.shape, generator=generator, dtype=torch.float32).to(latent)


def interpolate_weights(timesteps, coordinates, weights):
    x = torch.as_tensor(coordinates, dtype=torch.float64, device=timesteps.device).flatten()
    y = torch.as_tensor(weights, dtype=torch.float64, device=timesteps.device).flatten()
    query = timesteps.to(torch.float64)
    if len(x) != len(y) or len(x) < 2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Invalid timestep weight table')
    x, order = x.sort()
    y = y[order]
    if (x[1:] <= x[:-1]).any():
        raise ValueError('Ambiguous duplicate timestep weight coordinates')
    if not torch.isfinite(query).all() or (query < x[0]).any() or (query > x[-1]).any():
        raise ValueError('Validation timestep outside weight table range')
    right = torch.searchsorted(x, query).clamp(1, len(x) - 1)
    left = right - 1
    value = y[left] + (query - x[left]) / (x[right] - x[left]) * (y[right] - y[left])
    return value.to(dtype=torch.float32)


def mask_multiplier(batch, latents, enabled=True):
    mask = torch.ones((latents.shape[0], 1, 1, 1), device=latents.device, dtype=latents.dtype)
    if batch.mask_tensor is not None and enabled:
        mask = batch.mask_tensor.to(latents.device, dtype=torch.float16).detach()
        mask = torch.nn.functional.interpolate(mask, size=latents.shape[-2:])
        mask = mask.expand(-1, latents.shape[1], -1, -1).to(latents.dtype)
        mask = mask / mask.mean()
    return mask


def preflight_loss(trainer):
    config = trainer.train_config
    if config.loss_type not in ('mse', 'mae', 'pseudo_huber', 'wavelet', 'band'):
        raise ValueError(f'Unsupported deterministic validation loss_type: {config.loss_type}')
    if config.loss_target not in ('noise', 'source'):
        raise ValueError(f'Unsupported deterministic validation loss_target: {config.loss_target}')
    if config.timestep_type == 'next_sample':
        raise ValueError('next_sample changes the fixed validation noise through an extra scheduler step')
    # These branches perform extra teacher/stepped passes or train other
    # networks; they do not currently expose a read-only fixed-input loss.
    for name in ('train_turbo', 'diff_output_preservation', 'blank_prompt_preservation',
                 'inverted_mask_prior', 'correct_pred_norm', 'do_prior_divergence',
                 'do_guidance_loss', 'do_differential_guidance', 'do_random_cfg',
                 'do_cfg', 'do_next_sample_prediction'):
        if getattr(config, name, False):
            raise ValueError(f'{name} has no fixed-input validation path')
    if (trainer.adapter is not None or trainer.assistant_adapter is not None or trainer.dfe is not None
            or config.adapter_assist_name_or_path is not None or config.diffusion_feature_extractor_path is not None):
        raise ValueError('This adapter/feature loss has no read-only validation path')
    timesteps = torch.tensor(noise_levels(trainer.deterministic_validation_config.val_level_noise_n)) * 1000
    if config.linear_timesteps or config.linear_timesteps2 or config.timestep_type == 'weighted':
        if config.timestep_type not in ('linear', 'weighted', 'shift', 'flux_shift', 'lumina2_shift'):
            raise ValueError('Validation weighting requires a deterministic timestep schedule')
        trainer.sd.noise_scheduler.get_weights_for_timesteps(
            timesteps, v2=config.linear_timesteps2, timestep_type=config.timestep_type,
            x0_pred=trainer.sd.x0_pred, interpolate=True)
    if config.learnable_snr_gos or (config.snr_gamma or 0) > 0 or (config.min_snr_gamma or 0) > 0:
        from toolkit.train_tools import get_all_snr
        try:
            table = get_all_snr(trainer.sd.noise_scheduler, 'cpu')
            offset = 0 if config.learnable_snr_gos else int(trainer.sd.noise_scheduler.timesteps[0] == 1000)
            interpolate_weights(timesteps, torch.arange(len(table)) + offset, table)
        except (AttributeError, ValueError) as exc:
            raise ValueError('SNR weighting requires an unambiguous table covering all validation timesteps') from exc


def aggregate_validation(records, config, evaluate):
    groups = {}
    levels = noise_levels(config.val_level_noise_n)
    if not records:
        raise ValueError('Validation has no prepared inputs')
    for record in sorted(records, key=lambda r: (r.target_hash, r.key)):
        values, low, high = [], [], []
        for i, level in enumerate(levels, 1):
            for j in range(1, config.val_seed_noise_n + 1):
                value = float(evaluate(record, level, i, j))
                if not math.isfinite(value):
                    raise ValueError(f'Non-finite validation loss: target={record.target_hash}, i={i}, j={j}')
                values.append(value)
                (low if level < .5 else high).append(value)
        groups.setdefault(record.target_hash, []).append(tuple(math.fsum(v) / len(v) for v in (values, low, high)))
    return dict(zip(('val_loss_mean', 'val_loss_low_noise', 'val_loss_high_noise'),
                    [math.fsum(math.fsum(v[k] for v in alternatives) / len(alternatives)
                               for alternatives in groups.values()) / len(groups) for k in range(3)]))


def evaluate_validation(trainer, inputs):
    inputs.verify()
    sd, network = trainer.sd, trainer.network
    logs, trainer.additional_logs = trainer.additional_logs, {}
    original_scheduler = sd.noise_scheduler
    multiplier = network.multiplier if network is not None else None
    active = getattr(network, 'is_active', None)
    wrapped_unet = sd.unet
    try:
        from toolkit.accelerator import unwrap_model
        sd.unet = unwrap_model(sd.unet)
        with isolated_modules(model_modules(sd)):
            if network is not None:
                network.is_active = True

            def evaluate(record, level, i, j):
                batch = copy.copy(record.batch)
                batch.latents = record.batch.latents.to(trainer.device_torch, dtype=sd.torch_dtype)
                batch.sigmas = torch.tensor([level], device=trainer.device_torch, dtype=sd.torch_dtype).view(1, 1, 1, 1)
                if i == 1 and j == 1:
                    sd.noise_scheduler = copy.deepcopy(inputs.scheduler)
                    config = trainer.train_config
                    if config.linear_timesteps or config.linear_timesteps2 or config.timestep_type == 'weighted':
                        patch_size = getattr(getattr(sd.unet, 'config', None), 'patch_size', 1)
                        sd.noise_scheduler.set_train_timesteps(config.num_train_timesteps, device=trainer.device_torch,
                                                              timestep_type=config.timestep_type, latents=batch.latents,
                                                              patch_size=patch_size)
                noise = fixed_noise(batch.latents, inputs.config.val_seed_noise, record.target_hash, i, j)
                noisy = (1 - level) * batch.latents + level * noise
                timesteps = torch.tensor([level * 1000], device=trainer.device_torch, dtype=torch.float32)
                if network is not None:
                    network.multiplier = batch.get_network_weight_list()
                embeds = record.embeds.clone().to(trainer.device_torch, dtype=sd.torch_dtype)
                return trainer.validation_loss(batch, noisy, noise, timesteps, embeds).item()

            result = aggregate_validation(inputs.records, inputs.config, evaluate)
            inputs.verify()
            return result
    finally:
        if network is not None:
            network.multiplier = multiplier
            network.is_active = active
        trainer.additional_logs = logs
        sd.unet = wrapped_unet
        sd.noise_scheduler = original_scheduler


def main_process_call(accelerator, callback):
    """All ordinary DDP ranks participate, including on an exception."""
    result, error = None, None
    if accelerator.is_main_process:
        try:
            result = callback()
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        payload = [error]
        torch.distributed.broadcast_object_list(payload, src=0)
        error = payload[0]
    if error is not None:
        raise RuntimeError(f'Validation failed: {error}')
    return result


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def content_hash(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def model_source_identity(source):
    if source is None:
        return None
    path = Path(source)
    if path.is_file():
        return {'sha256': content_hash(path)}
    if path.is_dir():
        return {str(file.relative_to(path)).replace('\\', '/'): content_hash(file)
                for file in sorted(path.rglob('*')) if file.is_file()
                and file.suffix in ('.json', '.safetensors', '.bin', '.model', '.txt')}
    # Record the immutable revision already used by the loader, without a
    # network request or selecting a newer revision during validation.
    from huggingface_hub import try_to_load_from_cache
    for filename in ('model_index.json', 'config.json', 'tokenizer_config.json'):
        cached = try_to_load_from_cache(source, filename)
        if isinstance(cached, str) and 'snapshots' in Path(cached).parts:
            parts = Path(cached).parts
            return {'repository': source, 'revision': parts[parts.index('snapshots') + 1]}
    return {'repository': source}


def dataset_files(config):
    """Use the loader's folder/JSON conventions, excluding generated controls."""
    path = config.dataset_path or config.folder_path
    captions = None
    if os.path.isdir(path):
        files = []
        for root, dirs, names in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            if os.path.basename(root) == '_controls':
                continue
            files.extend(os.path.join(root, name) for name in names
                         if not name.startswith('.') and name.lower().endswith(IMAGE_EXTENSIONS))
    else:
        with open(path, encoding='utf-8') as stream:
            captions = json.load(stream)
        files = list(captions)
    if not files:
        raise ValueError(f'Empty validation dataset: {path}')
    return sorted(files), captions


def matching_image(folder, target, role):
    stem = Path(target).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = Path(folder) / (stem + ext)
        if candidate.is_file():
            return str(candidate)
    raise ValueError(f'Missing {role} for {target} in {folder}')


def check_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f'Invalid validation image {path}: {exc}') from exc


def input_manifest(configs, model_identity, train_configs=(), require_generated=True):
    """Content and associations, without absolute directory prefixes or order."""
    train_hashes = {content_hash(path) for config in train_configs
                    for path in dataset_files(config)[0]}
    records = {}
    paths = {}
    path_fields = {'folder_path', 'dataset_path', 'control_path', 'mask_path',
                   'inpaint_path', 'clip_image_path', 'unconditional_path'}
    for config in configs:
        files, captions = dataset_files(config)
        settings = {k: v for k, v in vars(config).items() if k not in path_fields}
        for target in files:
            check_image(target)
            image_hash = content_hash(target)
            if image_hash in train_hashes:
                raise ValueError(f'Train/val target content overlap: {target}')
            if captions is not None:
                caption = captions[target]
                caption_hash = hashlib.sha256(canonical_json(caption).encode()).hexdigest()
            else:
                caption_path = str(Path(target).with_suffix(config.caption_ext))
                caption = Path(caption_path).read_text(encoding='utf-8') if os.path.isfile(caption_path) else config.default_caption
                caption_hash = content_hash(caption_path) if os.path.isfile(caption_path) else None
            from toolkit.dataloader_mixins import CaptionProcessingDTOMixin
            caption_item = SimpleNamespace(path=target, dataset_config=config, trigger_word=config.trigger_word,
                                           is_reg=config.is_reg, raw_caption=None, raw_caption_short=None)
            caption_item.get_caption = MethodType(CaptionProcessingDTOMixin.get_caption, caption_item)
            with isolated_rng():
                CaptionProcessingDTOMixin.load_caption(caption_item, captions)
            control_folders = config.control_path or []
            if isinstance(control_folders, str):
                control_folders = [control_folders]
            controls = [matching_image(folder, target, 'control') for folder in control_folders]
            mask = matching_image(config.mask_path, target, 'mask') if config.mask_path else None
            if config.alpha_mask:
                with Image.open(target) as image:
                    if 'A' not in image.getbands():
                        raise ValueError(f'alpha_mask requires an alpha channel: {target}')
                mask = target
            generated = []
            for role in config.controls:
                folder = Path(target).parent / '_controls'
                generated_path = next((folder / (Path(target).stem + '.' + role + ext)
                                       for ext in IMAGE_EXTENSIONS
                                       if (folder / (Path(target).stem + '.' + role + ext)).is_file()), None)
                if generated_path is None:
                    if not require_generated:
                        generated.append({'role': role, 'hash': None})
                        continue
                    raise ValueError(f'Missing generated {role} for {target}')
                generated.append({'role': role, 'hash': content_hash(generated_path)})
                check_image(generated_path)
            for path in controls + ([mask] if mask else []):
                check_image(path)
            record = dict(target=image_hash, caption=caption, caption_hash=caption_hash,
                          prepared_caption=caption_item.caption,
                          controls=[content_hash(path) for path in controls],
                          mask=content_hash(mask) if mask else None,
                          generated=generated, settings=settings)
            key = hashlib.sha256(canonical_json(record).encode()).hexdigest()
            records[key] = record
            paths[(target, config.resolution)] = key
    manifest = dict(version=MANIFEST_VERSION, noise_version=NOISE_VERSION, model=model_identity,
                    records=[records[key] for key in sorted(records)])
    manifest['hash'] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    return manifest, paths


def verify_manifest(expected, actual):
    if expected != actual:
        raise ValueError('Validation manifest changed: targets, captions, masks, controls or preparation settings differ')


@dataclass
class ValidationRecord:
    key: str
    target_hash: str
    batch: object
    embeds: object


class ValidationInputs:
    def __init__(self, config, sd, train_configs, expected_manifest=None):
        from toolkit.data_loader import AiToolkitDataset
        from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
        self.config, self.sd, self.train_configs = config, sd, train_configs
        self.scheduler = copy.deepcopy(getattr(sd, 'noise_scheduler', None))
        self.model_identity = {
            'arch': sd.model_config.arch,
            'model': model_source_identity(sd.model_config.name_or_path),
            'extras': model_source_identity(sd.model_config.extras_name_or_path),
            'kwargs': sd.model_config.model_kwargs,
            'latent_space': sd.get_latent_space_version(),
            'text_space': sd.get_text_embedding_space_version(),
            'dtypes': {name: str(getattr(sd, name, sd.torch_dtype))
                       for name in ('torch_dtype', 'vae_torch_dtype', 'te_torch_dtype')},
            'torch': str(torch.__version__), 'numpy': np.__version__,
            'bucket_divisibility': sd.get_bucket_divisibility(),
        }
        self.records = []
        if expected_manifest is not None:
            current, _ = input_manifest(config.datasets, self.model_identity, train_configs)
            verify_manifest(expected_manifest, current)
        with isolated_modules(model_modules(sd)):
            datasets = [AiToolkitDataset(copy.deepcopy(c), batch_size=1, sd=sd) for c in config.datasets]
            self.manifest, _ = input_manifest(config.datasets, self.model_identity, train_configs)
            if expected_manifest is not None:
                verify_manifest(expected_manifest, self.manifest)
            seen = set()
            for dataset, original in zip(datasets, config.datasets):
                _, keys = input_manifest([original], self.model_identity)
                for index, unloaded in enumerate(dataset.file_list):
                    key = keys[(unloaded.path, original.resolution)]
                    if key in seen:
                        continue
                    seen.add(key)
                    with isolated_rng(stable_seed('prepare-v1', key, 'pixels')):
                        item = dataset._get_single_item(index)
                        batch = DataLoaderBatchDTO(file_items=[item])
                    with isolated_rng(stable_seed('prepare-v1', key, 'target')):
                        batch.latents = sd.encode_images(batch.tensor, device=sd.device_torch, dtype=sd.torch_dtype).detach().cpu()
                    if hasattr(sd, 'prepare_validation_conditions'):
                        sd.prepare_validation_conditions(batch, stable_seed('prepare-v1', key))
                    controls = batch.control_tensor_list if batch.control_tensor_list is not None else batch.control_tensor
                    with isolated_rng(stable_seed('prepare-v1', key, 'text')):
                        embeds = sd.encode_prompt(batch.get_caption_list(), control_images=controls,
                                                  target_size=(batch.tensor.shape[-1], batch.tensor.shape[-2]))
                    embeds = embeds.to('cpu').detach()
                    self.records.append(ValidationRecord(key, content_hash(item.path), batch, embeds))
        self.records.sort(key=lambda record: (record.target_hash, record.key))
        self.verify()

    def verify(self):
        current, _ = input_manifest(self.config.datasets, self.model_identity, self.train_configs)
        verify_manifest(self.manifest, current)
