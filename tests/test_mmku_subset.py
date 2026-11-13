import importlib.util
from pathlib import Path
import struct
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load('prepare_mmku_subset')
evaluate = load('evaluate_mmku_subset')


class MMKUSubsetTests(unittest.TestCase):
    def test_pair_by_entity_instead_of_misleading_numeric_id(self):
        train = [{'id': '9', 'image': 'images/Saiyaara_10.jpg'},
                 {'id': '7', 'image': 'images/Ballerina (2025 film)_7.jpg'}]
        test = [{'question_id': 9, 'image': 'images/Ballerina (2025 film)_2.jpg'},
                {'question_id': 7, 'image': 'images/Saiyaara_2.jpg'}]
        pairs = prepare.pair_unknown(train, test)
        for key, tr, te in pairs:
            self.assertEqual(prepare.entity(tr), prepare.entity(te))
            self.assertNotEqual(int(tr['id']), te['question_id'])
        self.assertEqual(pairs, prepare.pair_unknown(list(reversed(train)), list(reversed(test))))

    def test_duplicate_and_unmatched_entities_rejected(self):
        row = {'image': 'images/Saiyaara_1.jpg'}
        with self.assertRaises(ValueError):
            prepare.pair_unknown([row, row], [row])
        with self.assertRaises(ValueError):
            prepare.pair_unknown([row], [{'image': 'images/Other_2.jpg'}])

    def test_inference_never_receives_gold_or_entity(self):
        row = {'id': '7', 'entity': 'Saiyaara', 'answer': 'Mohit Suri',
               'messages': [{'role': 'user', 'content': '<image>\nWho directed this film?'}],
               'images': ['unknown/images/Saiyaara_2.jpg']}
        payload = evaluate.inference_payload(row, '/tmp/mmku-test')
        self.assertEqual(set(payload), {'messages', 'images'})
        self.assertNotIn(row['answer'], str(payload))
        self.assertEqual(row['images'][0], 'unknown/images/Saiyaara_2.jpg')
        with self.assertRaises(ValueError):
            evaluate.inference_payload(dict(row, messages=[{'role': 'assistant', 'content': row['answer']}]), '/tmp')

    def test_range_extracted_zip_entry_checks_crc(self):
        raw = b'test-image-bytes'
        comp = zlib.compress(raw)[2:-4]
        name = 'test.jpg'
        header = struct.pack('<4s5H3I2H', b'PK\x03\x04', 20, 0, 8, 0, 0,
                             zlib.crc32(raw), len(comp), len(raw), len(name), 0)
        entry = dict(name=name, compression=8, compressed_size=len(comp), size=len(raw), crc=zlib.crc32(raw))
        self.assertEqual(prepare.unpack_entry(entry, header + name.encode() + comp), raw)
        with self.assertRaises(ValueError):
            prepare.unpack_entry(dict(entry, crc=0), header + name.encode() + comp)


if __name__ == '__main__':
    unittest.main()
