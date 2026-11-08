import copy
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image
from toolkit.config_modules import DatasetConfig
from toolkit.validation import input_manifest, verify_manifest


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder, color in [('val', 'red'), ('train', 'blue'), ('ctrl1', 'green'), ('ctrl2', 'white'), ('mask', 'gray')]:
            (self.root / folder).mkdir()
            Image.new('RGB', (32, 48), color).save(self.root / folder / 'a.png')
        (self.root / 'val/a.txt').write_text('caption', encoding='utf-8')
        self.config = DatasetConfig(folder_path=str(self.root / 'val'),
                                    control_path=[str(self.root / 'ctrl1'), str(self.root / 'ctrl2')],
                                    mask_path=str(self.root / 'mask'))

    def manifest(self, configs=None):
        return input_manifest(configs or [self.config], {'arch': 'qwen_image_2'})[0]

    def test_missing_required_input_and_empty_dataset(self):
        (self.root / 'ctrl2/a.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing control'):
            self.manifest()
        (self.root / 'empty').mkdir()
        with self.assertRaisesRegex(ValueError, 'Empty'):
            self.manifest([DatasetConfig(folder_path=str(self.root / 'empty'))])

    def test_overlap(self):
        shutil.copyfile(self.root / 'val/a.png', self.root / 'train/renamed.png')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            input_manifest([self.config], {}, [DatasetConfig(folder_path=str(self.root / 'train'))])

    def test_movement_order_deduplication_and_alternatives(self):
        original = self.manifest()
        self.assertEqual(original, self.manifest([self.config, copy.deepcopy(self.config)]))
        moved = self.root / 'moved'
        moved.mkdir()
        for folder in ('val', 'ctrl1', 'ctrl2', 'mask'):
            shutil.copytree(self.root / folder, moved / folder)
        config = copy.deepcopy(self.config)
        config.folder_path = str(moved / 'val')
        config.mask_path = str(moved / 'mask')
        config.control_path = [str(moved / 'ctrl1'), str(moved / 'ctrl2')]
        self.assertEqual(original, self.manifest([config]))
        alternative = copy.deepcopy(self.config)
        alternative.control_path.reverse()
        combined = self.manifest([self.config, alternative])
        self.assertEqual(combined, self.manifest([alternative, self.config]))
        self.assertEqual(len(combined['records']), 2)
        self.assertEqual(len({r['target'] for r in combined['records']}), 1)

    def test_all_inputs_and_preparation_are_checked(self):
        original = self.manifest()
        for folder in ('val', 'ctrl1', 'mask'):
            path = self.root / folder / 'a.png'
            data = path.read_bytes()
            Image.new('RGB', (32, 48), 'black').save(path)
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                verify_manifest(original, self.manifest())
            path.write_bytes(data)
        (self.root / 'val/a.txt').write_text('changed', encoding='utf-8')
        with self.assertRaises(ValueError):
            verify_manifest(original, self.manifest())
        (self.root / 'val/a.txt').write_text('caption', encoding='utf-8')
        self.config.invert_mask = True
        with self.assertRaises(ValueError):
            verify_manifest(original, self.manifest())


if __name__ == '__main__':
    unittest.main()
