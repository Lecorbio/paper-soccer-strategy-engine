"""Tiny packed/base-set parity fixtures; no campaign corpus or protected roots."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import weakref

from tools import compact_value_bfm_exclusion_index_v2 as packed

campaign = packed.campaign
STATE = campaign.legacy.STATE_FINGERPRINT_DOMAIN
FEATURE = campaign.legacy.FEATURE_FINGERPRINT_DOMAIN


def seal(path, body):
    campaign.seal(path, body)
    return campaign.record(path)


def rewrite(path, body):
    path.unlink()
    return seal(path, {key: value for key, value in body.items() if key != 'body_sha256'})


class PackedExclusionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.contract = self.root / 'campaign.json'; self.index = self.root / 'future/index.json'
        self.sources = []

    def source(self, role, domain, values):
        path = self.root / f'source-{len(self.sources)}.json'
        binding = seal(path, {'role': role, 'domain': domain, 'fingerprints': values,
            'sources': {'path': '/must-never-follow-historical-transcripts-or-metrics.json'},
            'contains_labels': False, 'contains_metrics': False, 'contains_transcripts': False})
        self.sources.append(binding)
        return binding

    def freeze(self):
        return seal(self.contract, {'exclusions': self.sources})

    def build(self):
        binding = packed.build_index(self.contract, self.index)
        return binding, packed.load_index(binding, contract_record=campaign.record(self.contract))

    def test_exact_full32_byte_union_order_and_readonly_lookup(self):
        values = [bytes(32).hex(), (bytes(31) + b'\x01').hex(), (b'\x01' + bytes(31)).hex(),
                  (b'\xff' * 31 + b'\x00').hex(), (bytes(31) + b'\xff').hex(),
                  (b'\x12\x00' * 16).hex(), (b'\xff' * 32).hex()]
        self.source('protected', STATE, [])
        self.source('prior-train', FEATURE, values + values[:3])
        self.source('live', STATE, values[::2])
        self.source('prior-train', FEATURE, list(reversed(values)))
        self.source('protected', STATE, values[1:3])
        self.source('empty-role', FEATURE, [])
        self.freeze()
        # Force multiple merge generations using only a few dozen keys.
        with patch.object(packed, 'CHUNK_RECORDS', 2), patch.object(packed, 'MERGE_FAN_IN', 2):
            binding, actual = self.build()
        expected = campaign.exclusion_sets(campaign.read(self.contract))
        self.assertEqual(list(actual), list(expected))
        queries = values + ['33' * 32, 'ff' * 31, 'FF' * 32, 'ff' * 33, '', 'invalid', None]
        for key in expected:
            self.assertEqual(len(actual[key]), len(expected[key]))
            self.assertFalse(actual[key]._lookup.values.flags.writeable)
            for value in queries:
                self.assertEqual(value in actual[key], value in expected[key], (key, value))
        index = campaign.read(campaign.verify(binding))
        for entry, key in zip(index['entries'], expected, strict=True):
            self.assertEqual(Path(entry['array']['path']).read_bytes(),
                             b''.join(sorted(bytes.fromhex(value) for value in expected[key])))
            self.assertEqual(entry['array']['bytes'], 32 * len(expected[key]))
        self.assertEqual(index['entries'][0]['source_ordinals'], [0, 4])
        self.assertEqual(index['entries'][1]['source_ordinals'], [1, 3])

    def test_membership_rejection_reason_and_whole_action_group_parity(self):
        initial = campaign.features.ReplayState()
        parent, _ = campaign.fresh_root(6, random.Random(51))
        child = campaign.successors(parent)[0][1]
        self.assertGreater(len(child.used_segments), 6)
        parent_fps, child_fps, initial_fps = map(campaign.fingerprints, (parent, child, initial))
        self.source('protected', STATE, [])
        self.source('prior-train', STATE, [initial_fps[STATE], parent_fps[STATE], child_fps[STATE]])
        self.source('live', FEATURE, [initial_fps[FEATURE], parent_fps[FEATURE]])
        self.source('prior-validation', FEATURE, [child_fps[FEATURE]])
        self.source('protected', STATE, [child_fps[STATE]])
        self.freeze(); _, actual = self.build()
        expected = campaign.exclusion_sets(campaign.read(self.contract))
        for state in (initial, parent, child):
            for split in ('train', 'validation'):
                self.assertEqual(campaign.rejection(state, split, actual), campaign.rejection(state, split, expected))
        self.assertIsNone(campaign.rejection(initial, 'train', actual))
        self.assertEqual(campaign.rejection(initial, 'validation', actual), 'prior-train')
        # The initially empty protected role must retain precedence after union.
        self.assertEqual(campaign.rejection(child, 'train', actual), 'protected')
        for state in (initial, parent):
            for split in ('train', 'validation'):
                self.assertEqual(campaign.preflight_group(state, split, actual),
                                 campaign.preflight_group(state, split, expected))
        self.assertTrue(campaign.preflight_group(initial, 'train', actual)['eligible'])
        self.assertFalse(campaign.preflight_group(parent, 'train', actual)['eligible'])

    def test_workers_and_existing_build_hash_sources_without_parsing_fingerprint_lists(self):
        self.source('prior-train', STATE, ['01' * 32]); self.freeze()
        binding, _ = self.build()
        real_read = campaign.read
        forbidden = {Path(row['path']) for row in self.sources}
        def read(path):
            self.assertNotIn(Path(path), forbidden, 'worker parsed a base exclusion JSON document')
            return real_read(path)
        with patch.object(campaign, 'read', side_effect=read), patch.object(packed, '_source_runs') as parse:
            loaded = packed.load_index(binding, contract_record=campaign.record(self.contract))
            self.assertIn('01' * 32, loaded['prior-train', STATE])
            self.assertEqual(packed.build_index(self.contract, self.index), binding)
        parse.assert_not_called()

    def test_builder_releases_each_source_document_before_reading_the_next(self):
        for ordinal in range(4):
            self.source('prior-train', STATE, [f'{ordinal:064x}', '01' * 32])
        self.freeze(); live_documents = []
        class Document(dict): pass
        real_read = campaign.read; source_paths = {Path(row['path']) for row in self.sources}
        def read(path):
            value = real_read(path)
            if Path(path) in source_paths:
                self.assertTrue(all(ref() is None for ref in live_documents))
                value = Document(value); live_documents.append(weakref.ref(value))
            return value
        with patch.object(campaign, 'read', side_effect=read):
            packed.build_index(self.contract, self.index)
        self.assertEqual(len(live_documents), 4)
        self.assertTrue(all(ref() is None for ref in live_documents))

    def test_empty_contract_and_empty_role_are_distinct_and_preserved(self):
        self.freeze(); _, loaded = self.build(); self.assertEqual(loaded, {})
        self.source('protected', STATE, [])
        other_contract = self.root / 'empty-role-contract.json'
        seal(other_contract, {'exclusions': self.sources})
        binding = packed.build_index(other_contract, self.root / 'empty-role/index.json')
        loaded = packed.load_index(binding, contract_record=campaign.record(other_contract))
        self.assertEqual(list(loaded), [('protected', STATE)])
        self.assertEqual(len(loaded['protected', STATE]), 0)

    def test_source_contract_loader_and_array_changes_fail_closed(self):
        self.source('protected', STATE, ['01' * 32, '02' * 32]); self.freeze()
        binding, _ = self.build()
        other_contract = self.root / 'other-contract.json'; seal(other_contract, {'exclusions': self.sources})
        with self.assertRaisesRegex(ValueError, 'source version, contract'):
            packed.load_index(binding, contract_record=campaign.record(other_contract))
        changed = packed._sources(); changed['membership_loader'] = {'changed': 'implementation'}
        with patch.object(packed, '_sources', return_value=changed), self.assertRaisesRegex(ValueError, 'source version, contract'):
            packed.load_index(binding, contract_record=campaign.record(self.contract))
        source = Path(self.sources[0]['path']); original = source.read_bytes(); source.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'changed artifact'):
            packed.load_index(binding, contract_record=campaign.record(self.contract))
        source.write_bytes(original)
        array = Path(campaign.read(self.index)['entries'][0]['array']['path'])
        array.write_bytes(b'\x01' * 32 + b'\x03' * 32)
        with self.assertRaisesRegex(ValueError, 'hash/count/size'):
            packed.load_index(binding, contract_record=campaign.record(self.contract))

    def test_resealed_unsorted_duplicate_and_partial_arrays_are_rejected(self):
        self.source('protected', STATE, ['01' * 32, '02' * 32]); self.freeze()
        binding, _ = self.build(); document = campaign.read(self.index)
        array = Path(document['entries'][0]['array']['path'])
        for payload in (b'\x02' * 32 + b'\x01' * 32, b'\x01' * 64, b'\x01' * 33):
            array.write_bytes(payload)
            changed = copy.deepcopy(document); changed['entries'][0]['array'] = campaign.record(array)
            changed_binding = rewrite(self.index, changed)
            with self.assertRaisesRegex(ValueError, 'strictly sorted|partial SHA256'):
                packed.validate_index(changed_binding, contract_record=campaign.record(self.contract))
            with self.assertRaisesRegex(ValueError, 'strictly sorted|partial SHA256'):
                packed.build_index(self.contract, self.index)

    def test_redirected_array_and_changed_role_order_are_rejected(self):
        self.source('protected', STATE, ['01' * 32]); self.source('live', FEATURE, ['02' * 32]); self.freeze()
        _, _ = self.build(); document = campaign.read(self.index)
        copied = self.root / 'same-bytes.bin'; copied.write_bytes(Path(document['entries'][0]['array']['path']).read_bytes())
        changed = copy.deepcopy(document); changed['entries'][0]['array'] = campaign.record(copied)
        binding = rewrite(self.index, changed)
        with self.assertRaisesRegex(ValueError, 'canonical path'):
            packed.load_index(binding, contract_record=campaign.record(self.contract))
        changed = copy.deepcopy(document); changed['entries'].reverse(); binding = rewrite(self.index, changed)
        with self.assertRaisesRegex(ValueError, 'union/order/count'):
            packed.load_index(binding, contract_record=campaign.record(self.contract))

    def test_noncanonical_source_hash_and_redirected_output_never_publish_index(self):
        self.source('prior-train', STATE, ['AB' * 32]); self.freeze()
        before = Path(self.sources[0]['path']).read_bytes()
        with self.assertRaisesRegex(ValueError, 'canonical lowercase SHA256'):
            packed.build_index(self.contract, self.index)
        self.assertFalse(self.index.exists())
        self.assertEqual(before, Path(self.sources[0]['path']).read_bytes())
        external = self.root / 'external'; external.mkdir()
        link = self.root / 'redirected'; link.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'not redirected'):
            packed.build_index(self.contract, link / 'index.json')
        self.assertEqual(list(external.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
