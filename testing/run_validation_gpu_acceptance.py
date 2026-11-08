"""Короткий штатный train/resume с проверками неизменности и повторного val на GPU."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import faulthandler
import signal
from types import MethodType
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from safetensors import safe_open
from safetensors.torch import load_file

from toolkit.job import get_job
from toolkit import validation
from toolkit.validation_checkpoint import rng_state, restore_rng, read_checkpoint, completed_checkpoints


def digest(value):
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            result.update(str((item.dtype, tuple(item.shape))).encode())
            result.update(item.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            result.update(repr(item).encode())
    visit(value)
    return result.hexdigest()


def inputs_digest(inputs):
    return digest([(r.key, r.batch.latents, r.batch.mask_tensor, r.batch.fixed_condition_latents,
                    r.batch.fixed_condition_metadata, vars(r.embeds)) for r in inputs.records])


def state_digest(process):
    modules = [m for m in validation.model_modules(process.sd) if isinstance(m, torch.nn.Module)]
    return digest(dict(rng=rng_state(), weights=process.network.state_dict(),
                       grads=[p.grad for p in process.network.parameters()],
                       optimizer=process.optimizer.state_dict(), scheduler=process.lr_scheduler.state_dict(),
                       modes=[[m.training for m in module.modules()] for module in modules],
                       placement=[[(str(p.device), p.dtype, p.requires_grad) for p in module.parameters()] for module in modules],
                       multiplier=process.network.multiplier, active=process.network.is_active,
                       logs=process.additional_logs))


def main():
    faulthandler.enable()
    if hasattr(signal, 'SIGUSR1'):
        faulthandler.register(signal.SIGUSR1)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--report', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    report_path = Path(args.report)
    baseline_path = report_path.with_suffix('.initial.json')
    report = {'status': 'running', 'resume': args.resume, 'validation': [],
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def publish():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = yaml.safe_load(Path(args.config).read_text())
    raw = config['config']['process'][0]
    name = config['config']['name']
    if args.resume:
        raw['resume_from'] = str(Path(raw['training_folder']) / name / f'{name}-step000002')
    job = get_job(config)
    process = job.process[0]
    original_validate = process.validate
    original_setup = process.setup_validation
    original_hook = process.hook_train_loop
    original_sample = process.sample
    raw_evaluate = validation.evaluate_validation
    update_checked = False
    # Failure publication is exercised once in the initial run; resume checks
    # the restored state, inputs and next update independently.
    failure_checked = args.resume

    def checked_setup(self):
        if any(p.is_floating_point() and p.dtype != self.sd.vae_torch_dtype for p in self.sd.vae.parameters()):
            raise AssertionError('VAE precision differs from configured preparation precision')
        encoders = self.sd.text_encoder if isinstance(self.sd.text_encoder, list) else [self.sd.text_encoder]
        report['loaded_model_dtypes'] = {name: dict(Counter(str(p.dtype) for p in module.parameters()))
                                         for name, module in [('transformer', self.sd.unet), ('vae', self.sd.vae)]}
        report['loaded_text_encoder_dtypes'] = [dict(Counter(str(p.dtype) for p in module.parameters())) for module in encoders]
        report['quantization'] = dict(transformer=self.model_config.quantize, text_encoder=self.model_config.quantize_te)
        publish()
        return original_setup()

    def checked_validate(self):
        nonlocal failure_checked
        if self._last_validation_step == self.completed_optimizer_steps:
            return
        print(f'GPU audit: validation at step {self.completed_optimizer_steps}', flush=True)
        before = state_digest(self)
        captured = {}
        def evaluate(trainer, inputs):
            checks = []
            original_loss = trainer.validation_loss
            def audited_loss(batch, noisy, noise, timesteps, embeds):
                checks.append(digest((batch.latents, batch.fixed_condition_latents,
                                      batch.fixed_condition_metadata, vars(embeds), noise, noisy, timesteps)))
                return original_loss(batch, noisy, noise, timesteps, embeds)
            with patch.object(trainer, 'validation_loss', audited_loss), patch.object(trainer.sd.vae, 'encode', wraps=trainer.sd.vae.encode) as encode:
                first = raw_evaluate(trainer, inputs)
                first_checks = checks[:]
                checks.clear()
                second = raw_evaluate(trainer, inputs)
                if encode.call_count:
                    raise AssertionError('Validation re-encoded cached target/control latents')
            if first_checks != checks:
                raise AssertionError('Repeated validation changed latents, embeddings or noise')
            captured['fixed_checks_digest'] = digest(checks)
            captured['vae_encodes_during_validation'] = encode.call_count
            captured.update(first)
            delta = max(abs(first[key] - second[key]) for key in first)
            if delta > 1e-6:
                raise AssertionError(f'Repeated validation differs: {delta}')
            captured['repeat_max_abs_delta'] = delta
            return first
        with patch.object(validation, 'evaluate_validation', evaluate):
            original_validate()
        if before != state_digest(self):
            raise AssertionError('Validation changed training state or RNG')
        report['validation'].append({'step': self.completed_optimizer_steps, **captured})
        report['input_digest'] = inputs_digest(self._validation_inputs)
        report['manifest'] = self._validation_inputs.manifest
        report['model_dtypes'] = {name: dict(Counter(str(p.dtype) for p in module.parameters()))
                                  for name, module in [('transformer', self.sd.unet), ('vae', self.sd.vae)]}
        encoders = self.sd.text_encoder if isinstance(self.sd.text_encoder, list) else [self.sd.text_encoder]
        report['text_encoder_dtypes'] = [dict(Counter(str(p.dtype) for p in module.parameters())) for module in encoders]
        report['gpu'] = torch.cuda.get_device_name()
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        import diffusers, transformers, accelerate
        report['environment'] = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                                     diffusers=diffusers.__version__, transformers=transformers.__version__,
                                     accelerate=accelerate.__version__, cpu_threads=torch.get_num_threads())
        publish()
        print(f'GPU audit: repeated metrics {captured}', flush=True)
        if not failure_checked:
            failure_checked = True
            total = len(self._validation_inputs.records) * self.deterministic_validation_config.val_level_noise_n * self.deterministic_validation_config.val_seed_noise_n
            count = 0
            original_loss = self.validation_loss
            def failing_loss(*a, **kw):
                nonlocal count
                count += 1
                if count == total:
                    raise RuntimeError('acceptance: last validation check failed')
                return original_loss(*a, **kw)
            previous = self._last_validation_step
            self._last_validation_step = None
            try:
                with patch.object(self, 'validation_loss', failing_loss), patch.object(self.writer, 'add_scalar', wraps=self.writer.add_scalar) as writes:
                    try:
                        original_validate()
                    except RuntimeError as exc:
                        if 'last validation check failed' not in str(exc):
                            raise
                    else:
                        raise AssertionError('Injected validation failure was ignored')
                    if writes.call_count:
                        raise AssertionError('Failed pass published partial metrics')
            finally:
                self._last_validation_step = previous
            if before != state_digest(self):
                raise AssertionError('Failed validation changed training state')
            report['last_check_failure_published_nothing'] = True
            publish()
            print('GPU audit: last-check failure published no metrics', flush=True)
            source = Path(self._validation_inputs.records[0].batch.file_items[0].path)
            contents = source.read_bytes()
            self._last_validation_step = None
            try:
                source.write_bytes(contents + b'acceptance: changed input')
                with patch.object(self.writer, 'add_scalar', wraps=self.writer.add_scalar) as writes:
                    try:
                        original_validate()
                    except (ValueError, RuntimeError) as exc:
                        if 'manifest changed' not in str(exc):
                            raise
                    else:
                        raise AssertionError('Changed validation input was accepted')
                    if writes.call_count:
                        raise AssertionError('Input mismatch published metrics')
            finally:
                source.write_bytes(contents)
                self._last_validation_step = previous
            if before != state_digest(self):
                raise AssertionError('Input mismatch changed training state')
            report['input_mismatch_published_nothing'] = True
            publish()

    def checked_update(self, batches):
        nonlocal update_checked
        if update_checked or self.is_grad_accumulation_step:
            return original_hook(batches)
        update_checked = True
        print('GPU audit: comparing next update with and without validation', flush=True)
        if self.ema or self.snr_gos:
            raise ValueError('Acceptance update comparison expects EMA and learnable SNR disabled')
        network = copy.deepcopy(self.network.state_dict())
        gradients = [None if p.grad is None else p.grad.detach().clone() for p in self.network.parameters()]
        report['pending_gradients_at_update'] = sum(g is not None for g in gradients)
        optimizer = copy.deepcopy(self.optimizer.state_dict())
        scheduler = copy.deepcopy(self.lr_scheduler.state_dict())
        noise_scheduler = copy.deepcopy(self.sd.noise_scheduler)
        random_state = rng_state()
        names = ('completed_optimizer_steps', 'step_num', '_accumulation_pending', 'steps_this_boundary')
        attributes = {key: getattr(self, key) for key in names}
        logs = copy.deepcopy(self.additional_logs)
        # Flash backward on this GPU can differ even without validation.
        # Compare identical training kernels with deterministic math attention;
        # the regular validation and other training updates keep native SDPA.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        report['next_update_attention_backend'] = 'torch SDPA math'
        with sdpa_kernel(SDPBackend.MATH):
            original_hook(copy.deepcopy(batches))
        expected = {key: value.detach().cpu().clone() for key, value in self.network.state_dict().items()}
        expected_rng = digest(rng_state())
        self.network.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.lr_scheduler.load_state_dict(scheduler)
        for parameter, gradient in zip(self.network.parameters(), gradients):
            parameter.grad = gradient
        self.sd.noise_scheduler = noise_scheduler
        self.additional_logs = logs
        for key, value in attributes.items():
            setattr(self, key, value)
        restore_rng(random_state)
        before_validation = state_digest(self)
        raw_evaluate(self, self._validation_inputs)
        if before_validation != state_digest(self):
            raise AssertionError('Validation changed the pending optimizer update state')
        with sdpa_kernel(SDPBackend.MATH):
            result = original_hook(batches)
        delta = max((value.detach().cpu().float() - expected[key].float()).abs().max().item()
                    for key, value in self.network.state_dict().items())
        report['next_update_max_abs_delta'] = delta
        report['next_update_rng_equal'] = expected_rng == digest(rng_state())
        if delta > 1e-6 or not report['next_update_rng_equal']:
            raise AssertionError(f'Next update changed with validation: delta={delta}')
        publish()
        print(f'GPU audit: next update delta={delta}, RNG equal={report["next_update_rng_equal"]}', flush=True)
        return result

    def checked_sample(self, *a, **kw):
        step = self.completed_optimizer_steps
        checkpoint = Path(self.save_root) / f'{name}-step{step:06d}'
        weights = digest(self.network.get_state_dict(dtype=torch.float32))
        if (checkpoint / 'model.safetensors').exists():
            if weights != digest(load_file(checkpoint / 'model.safetensors')):
                raise AssertionError('Sample weights differ from saved checkpoint')
        result = original_sample(*a, **kw)
        if weights != digest(self.network.get_state_dict(dtype=torch.float32)):
            raise AssertionError('Sampling changed training weights')
        report.setdefault('sample_weights_verified_at_steps', []).append(step)
        publish()
        return result

    process.validate = MethodType(checked_validate, process)
    process.setup_validation = MethodType(checked_setup, process)
    process.hook_train_loop = MethodType(checked_update, process)
    process.sample = MethodType(checked_sample, process)
    try:
        job.run()
        events = EventAccumulator(str(Path(process.save_root) / 'tensorboard')).Reload()
        report['tensorboard'] = {tag: [{'step': item.step, 'value': item.value} for item in events.Scalars(tag)]
                                for tag in ('val_loss_mean', 'val_loss_low_noise', 'val_loss_high_noise')}
        report['checkpoints'] = []
        for step, path in completed_checkpoints(process.save_root, name):
            state = read_checkpoint(path)
            with safe_open(path / 'model.safetensors', framework='pt') as weights:
                dtypes = sorted({str(weights.get_tensor(key).dtype) for key in weights.keys()})
            if dtypes != ['torch.float32']:
                raise AssertionError(f'LoRA is not FP32: {dtypes}')
            samples = [p for p in (path / 'samples').glob('*') if p.is_file() and p.suffix in ('.jpg', '.png', '.webp')]
            if len(samples) != len(raw['sample']['samples']):
                raise AssertionError(f'Unexpected samples for checkpoint {step}: {len(samples)}')
            report['checkpoints'].append(dict(step=step, dtypes=dtypes, optimizer=state['optimizer_type'], samples=len(samples)))
        expected_steps = [2, 3] if args.resume else [0, 2, 3]
        if [item['step'] for item in report['validation']] != expected_steps:
            raise AssertionError('Unexpected validation schedule')
        if args.resume:
            baseline = json.loads(baseline_path.read_text())
            initial_step = next(item for item in baseline['validation'] if item['step'] == 2)
            resumed_step = report['validation'][0]
            report['resume_input_digest_equal'] = baseline['input_digest'] == report['input_digest']
            report['resume_fixed_checks_equal'] = initial_step['fixed_checks_digest'] == resumed_step['fixed_checks_digest']
            report['resume_metric_max_abs_delta'] = max(abs(initial_step[key] - resumed_step[key]) for key in ('val_loss_mean', 'val_loss_low_noise', 'val_loss_high_noise'))
            if not report['resume_input_digest_equal'] or not report['resume_fixed_checks_equal'] or report['resume_metric_max_abs_delta'] > 1e-6:
                raise AssertionError('Resume did not reproduce inputs and validation metrics')
        report['status'] = 'passed'
        publish()
        if not args.resume:
            baseline_path.write_text(report_path.read_text())
        job.cleanup()
    except BaseException as exc:
        report['status'], report['error'] = 'failed', f'{type(exc).__name__}: {exc}'
        publish()
        raise


if __name__ == '__main__':
    main()
