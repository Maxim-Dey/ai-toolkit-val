import copy
from pathlib import Path
import unittest
from types import SimpleNamespace

import yaml

from toolkit.config_modules import (
    DatasetConfig, normalize_dataset_configs, preprocess_dataset_raw_config, validate_validation_model,
)


class ValidationConfigTest(unittest.TestCase):
    def test_gpu_configs_include_edit_controls_and_sampling_reference(self):
        from testing.make_validation_gpu_configs import make_configs
        path = Path(__file__).resolve().parents[1] / 'config/qwen_image_21_kibm.yaml'
        base = yaml.safe_load(path.read_text(encoding='utf-8'))
        original = copy.deepcopy(base)
        configs = dict(make_configs(base, '/data', '/output', '/data/reference.png'))
        self.assertEqual(len(configs), 7)
        process = configs['qwen-image-edit-2511']['config']['process'][0]
        self.assertEqual(process['model']['name_or_path'], 'Qwen/Qwen-Image-Edit-2511')
        self.assertFalse(process['model']['quantize'])
        self.assertFalse(process['model']['quantize_te'])
        train, val, _ = normalize_dataset_configs(process)
        train = [DatasetConfig(**raw) for raw in preprocess_dataset_raw_config(train)]
        validate_validation_model(SimpleNamespace(**process['model']), train + val.datasets)
        self.assertEqual(train[0].control_path, '/data/controls/train')
        self.assertEqual(val.datasets[0].control_path, '/data/controls/val')
        self.assertEqual(process['sample']['samples'][0]['ctrl_img'], '/data/reference.png')
        self.assertEqual(base, original)

    def test_qwen_edit_requires_controls_for_every_train_and_val_dataset(self):
        model = SimpleNamespace(arch='qwen_image_edit_plus', name_or_path='/local/edit-2511.safetensors', model_kwargs={})
        supported = [DatasetConfig(folder_path='train', control_path='train_controls'),
                     DatasetConfig(folder_path='val', control_path=['a', 'b'], mask_path='mask'),
                     DatasetConfig(folder_path='val', controls=['depth', 'mask'])]
        validate_validation_model(model, supported)
        for missing in (DatasetConfig(folder_path='train'), DatasetConfig(folder_path='val', mask_path='mask'),
                        DatasetConfig(folder_path='val', controls=['mask', 'sapiens2_mask'])):
            with self.subTest(dataset=missing.folder_path), self.assertRaisesRegex(ValueError, 'at least one control'):
                validate_validation_model(model, supported + [missing])

    def test_model_capabilities_do_not_depend_on_weight_path(self):
        plain = DatasetConfig(folder_path='val', mask_path='mask')
        controlled = DatasetConfig(folder_path='val', mask_path='mask', control_path=['a', 'b'])
        for arch, edit, supports in [('qwen_image', False, False), ('qwen_image_2', False, True),
                                      ('moss_image', False, False), ('moss_image', True, True)]:
            model = SimpleNamespace(arch=arch, name_or_path='/local/model.safetensors', model_kwargs={'edit': edit})
            validate_validation_model(model, [plain])
            if supports:
                validate_validation_model(model, [controlled])
            else:
                with self.assertRaisesRegex(ValueError, 'control inputs'):
                    validate_validation_model(model, [controlled])
            self.assertEqual(controlled.control_path, ['a', 'b'])
            self.assertEqual(controlled.mask_path, 'mask')
        with self.assertRaisesRegex(ValueError, 'arch'):
            validate_validation_model(SimpleNamespace(arch='flux'), [plain])

    def test_invalid_validation_parameters(self):
        for key, values in {
            'val_every_n_steps': [True, 1.5, 0, -1, '10'],
            'val_seed_noise': [False, 42.5, '42', None],
            'val_level_noise_n': [True, 2.5, 0, 1, 3, -2],
            'val_seed_noise_n': [False, 1.5, 0, -1],
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    normalize_dataset_configs({'val_datasets': [{'val_path': 'val', key: value}]})

    def test_multiple_validation_datasets_share_parameters(self):
        for key in ('val_every_n_steps', 'val_seed_noise', 'val_level_noise_n', 'val_seed_noise_n'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                normalize_dataset_configs({'val_datasets': [
                    {'val_path': 'a', key: 4}, {'val_path': 'b', key: 6},
                ]})
        _, val, _ = normalize_dataset_configs({'val_datasets': [
            {'val_path': 'a', 'val_seed_noise': -1}, {'val_path': 'b', 'val_seed_noise': -1},
        ]})
        self.assertEqual(val.val_seed_noise, -1)
        self.assertEqual(len(val.datasets), 2)

    def test_conflicts_and_empty_datasets(self):
        invalid = [
            ({'datasets': [], 'train_datasets': []}, 'datasets'),
            ({'val_datasets': [], 'train': {'validation_config': {}}}, 'validation_config'),
            ({'val_datasets': []}, 'val_datasets'),
            ({'val_datasets': None}, 'val_datasets'),
            ({'val_datasets': {}}, 'val_datasets'),
            ({'val_datasets': [None]}, 'val_datasets'),
            ({'val_datasets': [{'val_path': ''}]}, 'val_path'),
            ({'val_datasets': [{'val_path': []}]}, 'val_path'),
            ({'train_datasets': [{'train_path': 'a', 'folder_path': 'b'}]}, 'train_path'),
            ({'val_datasets': [{'val_path': 'a', 'dataset_path': 'b'}]}, 'val_path'),
        ]
        for config, message in invalid:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                normalize_dataset_configs(config)

    def test_random_validation_settings_are_rejected(self):
        for key, value in [('random_crop', True), ('flip_x', True), ('shuffle_tokens', True),
                           ('caption_dropout_rate', .1), ('augments', ['blur']),
                           ('control_from_same_folder', True), ('num_repeats', 2)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                normalize_dataset_configs({'val_datasets': [{'val_path': 'val', key: value}]})

    def test_agreed_yaml_is_normalized_without_mutation(self):
        path = Path(__file__).resolve().parents[1] / 'config/qwen_image_21_kibm.yaml'
        process = yaml.safe_load(path.read_text(encoding='utf-8'))['config']['process'][0]
        original = copy.deepcopy(process)
        train, val, new_mode = normalize_dataset_configs(process)
        train = [DatasetConfig(**item) for item in preprocess_dataset_raw_config(train)]
        self.assertTrue(new_mode)
        self.assertEqual(process, original)
        self.assertEqual(train[0].folder_path, original['train_datasets'][0]['train_path'])
        self.assertEqual(val.datasets[0].folder_path, original['val_datasets'][0]['val_path'])
        self.assertEqual(train[0].resolution, 1024)
        self.assertEqual(val.datasets[0].resolution, 1024)
        self.assertEqual(train[0].caption_dropout_rate, 0.05)
        self.assertEqual(val.datasets[0].caption_dropout_rate, 0)
        self.assertEqual(val.datasets[0].caption_ext, '.txt')
        self.assertEqual(val.datasets[0].default_caption, '')
        self.assertEqual((val.val_every_n_steps, val.val_seed_noise,
                          val.val_level_noise_n, val.val_seed_noise_n), (10, 42, 10, 2))

    def test_legacy_and_no_validation(self):
        datasets = [{'folder_path': 'train', 'resolution': [512, 1024], 'flip_x': True}]
        train, val, new_mode = normalize_dataset_configs({'datasets': datasets})
        self.assertFalse(new_mode)
        self.assertIsNone(val)
        self.assertEqual(train, datasets)
        self.assertIsNot(train, datasets)
        train, val, new_mode = normalize_dataset_configs({'train_datasets': [{'train_path': 'train'}]})
        self.assertTrue(new_mode)
        self.assertIsNone(val)
        self.assertEqual(train, [{'folder_path': 'train'}])

    def test_path_and_resolution_expansion(self):
        train, val, _ = normalize_dataset_configs({
            'train_datasets': [{'train_path': ['a', 'b'], 'resolution': [512, 1024]}],
            'val_datasets': [{'val_path': ['c', 'd'], 'resolution': [512, 1024]}],
        })
        self.assertEqual(len(preprocess_dataset_raw_config(train)), 4)
        self.assertEqual([(d.folder_path, d.resolution) for d in val.datasets],
                         [('c', 512), ('c', 1024), ('d', 512), ('d', 1024)])


if __name__ == '__main__':
    unittest.main()
