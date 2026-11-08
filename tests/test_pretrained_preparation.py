"""权重准备中的完整性与原子发布检查；不访问网络。"""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_pretrained as preparation


class PreparationTests(unittest.TestCase):
    def test_resume_range_and_atomic_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'weight.pth.part').write_bytes(b'abc')
            response = io.BytesIO(b'def')
            response.status = 206
            response.headers = {'Content-Range': 'bytes 3-5/6'}
            record = {'path': 'weight.pth', 'bytes': 6, 'url': 'https://example.test/weight',
                      'sha256': hashlib.sha256(b'abcdef').hexdigest()}
            with patch.object(preparation, 'ROOT', root), patch.object(preparation.urllib.request, 'urlopen', return_value=response) as request:
                target = preparation.download(record)
                self.assertEqual(request.call_args.args[0].get_header('Range'), 'bytes=3-')
            self.assertEqual(target.read_bytes(), b'abcdef')
            self.assertFalse((root / 'weight.pth.part').exists())

    def test_existing_invalid_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / 'weight.pth'
            target.write_bytes(b'wrong!')
            record = {'path': target.name, 'bytes': 6, 'sha256': '0' * 64}
            with patch.object(preparation, 'ROOT', root), patch.object(preparation.urllib.request, 'urlopen', side_effect=AssertionError('network')):
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    preparation.download(record)
            self.assertEqual(target.read_bytes(), b'wrong!')
