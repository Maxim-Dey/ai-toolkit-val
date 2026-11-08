"""Создать короткие GPU-конфиги отдельно от основного YAML; ничего не запускать."""
import argparse
import copy
from pathlib import Path, PurePosixPath

import yaml


MODELS = [
    ('qwen-image', 'Qwen/Qwen-Image', 'qwen_image'),
    ('qwen-image-2512', 'Qwen/Qwen-Image-2512', 'qwen_image'),
    ('qwen-image-edit-2511', 'Qwen/Qwen-Image-Edit-2511', 'qwen_image_edit_plus'),
    ('moss-sft', 'OpenMOSS-Team/MOSS-Image-SFT', 'moss_image'),
    ('moss-base', 'OpenMOSS-Team/MOSS-Image-Base', 'moss_image'),
    ('qwen-21-turbo', 'Qwen/Qwen-Image-2.1-Turbo', 'qwen_image_2'),
    ('qwen-21', 'Qwen/Qwen-Image-2.1', 'qwen_image_2'),
]


def make_configs(base, data_root, output_root, edit_sample_image):
    """data_root содержит отдельные train/val и необязательные masks/controls."""
    for name, repository, arch in MODELS:
        config = copy.deepcopy(base)
        config['config']['name'] = 'acceptance-' + name
        process = config['config']['process'][0]
        process['training_folder'] = str(output_root)
        process['model'].update(name_or_path=repository, extras_name_or_path=repository,
                                arch=arch, quantize=False, quantize_te=False, model_kwargs={})
        process['train'].update(steps=3, batch_size=1, gradient_accumulation=2)
        process['save'].update(save_every=2, max_step_saves_to_keep=2, dtype='float32')
        process['logging']['log_every'] = 100
        process['train_datasets'] = [{'train_path': str(PurePosixPath(data_root) / 'train'), 'resolution': [512],
                                      'caption_dropout_rate': .05, 'cache_latents_to_disk': True}]
        process['val_datasets'] = [{'val_path': str(PurePosixPath(data_root) / 'val'), 'resolution': [512],
                                    'val_every_n_steps': 2, 'val_seed_noise': 42,
                                    'val_level_noise_n': 8, 'val_seed_noise_n': 2}]
        process['sample'].update(sample_every=2, sample_start_step=0, sample_steps=4,
                                 width=512, height=512, samples=process['sample']['samples'][:1])
        if name == 'qwen-21-turbo':
            process['model']['model_kwargs']['turbo'] = True
            process['sample'].update(guidance_scale=1, sample_steps=8)
        if arch == 'qwen_image_edit_plus':
            process['train_datasets'][0]['control_path'] = str(PurePosixPath(data_root) / 'controls/train')
            process['val_datasets'][0]['control_path'] = str(PurePosixPath(data_root) / 'controls/val')
            process['sample']['samples'] = [{'prompt': 'Edit the reference image following the caption.',
                                             'ctrl_img': edit_sample_image}]
        yield name, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='config/qwen_image_21_kibm.yaml')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--edit-sample-image', required=True,
                        help='Путь на GPU-сервере к контрольному изображению для samples Edit-2511')
    parser.add_argument('--output-root', default='/ai-toolkit/output/validation-acceptance')
    parser.add_argument('--config-dir', default='testing/.validation-gpu-configs')
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.base).read_text(encoding='utf-8'))
    folder = Path(args.config_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for name, config in make_configs(base, args.data_root, args.output_root, args.edit_sample_image):
        path = folder / (name + '.yaml')
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
        print(path)


if __name__ == '__main__':
    main()
