"""Completed-step LoRA checkpoints. A final marker publishes both files."""
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import uuid

import numpy as np
import torch
from safetensors import safe_open

from toolkit.validation import isolated_rng

VERSION = 1


def step_path(root, name, step):
    return Path(root) / f'{name}-step{step:06d}'


def rng_state():
    numpy = np.random.get_state()
    return dict(python=random.getstate(), numpy=(numpy[0], numpy[1].tolist(), *numpy[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def read_checkpoint(path):
    path = Path(path)
    if not all((path / name).is_file() for name in ('model.safetensors', 'optimizer.bin', 'complete.json')):
        raise ValueError(f'Incomplete checkpoint: {path}. Legacy LoRA can only initialize a new run via network.pretrained_lora_path.')
    marker = json.loads((path / 'complete.json').read_text())
    state = torch.load(path / 'optimizer.bin', map_location='cpu', weights_only=True)
    with safe_open(path / 'model.safetensors', framework='pt') as weights:
        metadata = weights.metadata()
    if state['version'] != VERSION or marker != {'version': VERSION, 'step': state['step'], 'id': state['id']}:
        raise ValueError(f'Inconsistent checkpoint state: {path}')
    if metadata.get('completed_step') != str(state['step']) or metadata.get('state_id') != state['id']:
        raise ValueError(f'Weights and optimizer do not match: {path}')
    if not path.name.endswith(f'-step{state["step"]:06d}'):
        raise ValueError(f'Checkpoint directory step disagrees with state: {path}')
    return state


def completed_checkpoints(root, name):
    result = []
    pattern = re.compile(re.escape(name) + r'-step(\d{6,})$')
    if not Path(root).exists():
        return result
    for path in Path(root).iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_dir() and (path / 'complete.json').is_file():
            read_checkpoint(path)
            result.append((int(match.group(1)), path))
    return sorted(result)


def select_checkpoint(root, name, selected=None):
    if selected is not None:
        path = Path(selected)
        read_checkpoint(path)
        return path
    checkpoints = completed_checkpoints(root, name)
    return checkpoints[-1][1] if checkpoints else None


def save_checkpoint(trainer):
    root = Path(trainer.save_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    step = trainer.completed_optimizer_steps
    target = step_path(root, trainer.name, step)
    replacing = (target / 'complete.json').exists()
    if replacing:
        read_checkpoint(target)
        if getattr(trainer, '_last_saved_checkpoint_step', None) == step:
            return target
    identity = uuid.uuid4().hex
    network = trainer.network
    state = dict(version=VERSION, id=identity, step=step, epoch=trainer.epoch_num,
                 optimizer=trainer.optimizer.state_dict(), scheduler=trainer.lr_scheduler.state_dict(),
                 optimizer_type=trainer.train_config.optimizer, rng=rng_state(),
                 manifest=trainer._validation_inputs.manifest if trainer._validation_inputs else None,
                 ema=trainer.ema.state_dict() if trainer.ema else None,
                 training_weights={k: v.detach().cpu().clone() for k, v in network.state_dict().items()} if trainer.ema else None,
                 scaler=trainer.accelerator.scaler.state_dict() if getattr(trainer.accelerator, 'scaler', None) else None,
                 snr=None)
    if trainer.snr_gos is not None:
        snr = trainer.snr_gos
        state['snr'] = dict(parameters={k: getattr(snr, k).detach().cpu().clone() for k in ('gamma', 'offset_1', 'offset_2', 'scale')},
                           optimizer=snr.optimizer.state_dict(), buffer=snr.buffer)
    temporary = Path(tempfile.mkdtemp(prefix='.checkpoint-', dir=root))
    multiplier = network.multiplier
    ema_train = trainer.ema._is_train_mode if trainer.ema else None
    try:
        with isolated_rng():
            # Serialize the training state before EMA changes the live parameters.
            torch.save(state, temporary / 'optimizer.bin')
            if trainer.ema:
                trainer.ema.eval()
            network.multiplier = 1.0
            network.save_weights(str(temporary / 'model.safetensors'), dtype=torch.float32,
                                 metadata={'completed_step': str(step), 'state_id': identity})
        (temporary / 'complete.json').write_text(json.dumps({'version': VERSION, 'step': step, 'id': identity}))
        target.mkdir(exist_ok=True)
        # After an explicit rollback, this step may belong to the old run.
        # Invalidate its marker only once the replacement is fully serialized.
        (target / 'complete.json').unlink(missing_ok=True)
        # Samples from an overwritten future checkpoint depict its old weights.
        # Keep samples-only directories when saving this step for the first time.
        samples = target / 'samples'
        if replacing and samples.exists() and samples.resolve().is_relative_to(root):
            shutil.rmtree(samples)
        for filename in ('model.safetensors', 'optimizer.bin', 'complete.json'):
            os.replace(temporary / filename, target / filename)
        trainer._last_saved_checkpoint_step = step
    finally:
        network.multiplier = multiplier
        if trainer.ema and ema_train:
            with isolated_rng():
                trainer.ema.train()
        if temporary.resolve().is_relative_to(root):
            shutil.rmtree(temporary)
    keep = trainer.save_config.max_step_saves_to_keep
    if keep > 0:
        for _, old in completed_checkpoints(root, trainer.name)[:-keep]:
            if old.resolve().is_relative_to(root):
                shutil.rmtree(old)
    return target


def restore_training_state(trainer, state):
    trainer._last_saved_checkpoint_step = None
    if state['optimizer_type'] != trainer.train_config.optimizer:
        raise ValueError('Checkpoint optimizer type differs from training configuration')
    if state['training_weights'] is not None:
        trainer.network.load_state_dict(state['training_weights'])
    trainer.optimizer.load_state_dict(state['optimizer'])
    trainer.lr_scheduler.load_state_dict(state['scheduler'])
    scaler = getattr(trainer.accelerator, 'scaler', None)
    if (state['scaler'] is None) != (scaler is None):
        raise ValueError('Checkpoint gradient scaler configuration differs')
    if scaler is not None:
        scaler.load_state_dict(state['scaler'])
    if (state['ema'] is None) != (trainer.ema is None):
        raise ValueError('Checkpoint EMA configuration differs')
    if trainer.ema:
        trainer.ema.load_state_dict(state['ema'])
    if (state['snr'] is None) != (trainer.snr_gos is None):
        raise ValueError('Checkpoint learnable SNR configuration differs')
    if trainer.snr_gos:
        with torch.no_grad():
            for name, value in state['snr']['parameters'].items():
                getattr(trainer.snr_gos, name).copy_(value)
        trainer.snr_gos.optimizer.load_state_dict(state['snr']['optimizer'])
        trainer.snr_gos.buffer = [value.to(trainer.snr_gos.device) for value in state['snr']['buffer']]
    trainer.completed_optimizer_steps = trainer.step_num = trainer.start_step = state['step']
    trainer.epoch_num = state['epoch']
