"""Independent trainer/export/native boundary checks for channel quantization."""
import hashlib
import importlib.util
import base64
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

for _name in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
              'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'

from tools import compact_value_bfm_train as trainer

ROOT = Path(__file__).resolve().parents[2]
BOT = ROOT / 'submissions/codingame/bots/compact_value_bfm'


def load_module(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class ChannelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exporter = load_module(BOT / 'export_model.py', 'channel_integration_export_model')
        cls.renderer = load_module(BOT / 'export_submission.py', 'channel_integration_export_submission')

    @staticmethod
    def channel_runtime(directory):
        architecture = trainer.ARCHITECTURES['capacity-12x8']
        codes = {key: ((trainer.np.arange(trainer.np.prod(shape), dtype=trainer.np.int32)
                       * 13 + offset) % 7 - 3).astype(trainer.np.int8).reshape(shape)
                 for offset, (key, shape) in enumerate(architecture.shapes.items())}
        scales = {'w1': trainer.np.asarray([(i + 1) / 2048 for i in range(12)], dtype=trainer.np.float32),
                  'w2': trainer.np.asarray([(i + 2) / 512 for i in range(8)], dtype=trainer.np.float32),
                  'w3': trainer.np.asarray([1 / 128], dtype=trainer.np.float32)}
        quantized = trainer.ChannelQuantizedWeights(codes, scales)
        runtime = trainer.write_runtime(Path(directory), architecture, quantized,
            arm=trainer.ARMS['search-target'], seed=20260907, float_epoch=1, qat_epoch=4,
            source_bundle_body_sha256='1' * 64,
            qat_profile=trainer.CHANNEL_PREDICTION_QAT_PROFILE, qat_evidence_sha256='2' * 64)
        return runtime, architecture, quantized

    @staticmethod
    def reseal(directory, document):
        body = {key: value for key, value in document.items() if key != 'body_sha256'}
        payload = trainer.canonical_json_bytes(trainer.body_hashed(body))
        path = Path(directory) / (hashlib.sha256(payload).hexdigest() + '.runtime.json')
        path.write_bytes(payload)
        return path

    @staticmethod
    def training_only_inputs():
        def pool(count, offset):
            return trainer.Dataset(
                indptr=trainer.np.arange(count + 1, dtype='<i8'),
                indices=trainer.np.arange(offset, offset + count, dtype='<u2'),
                targets=trainer.np.zeros(count, dtype='<f4'),
                weights=(1 + trainer.np.arange(count) % 3 / 4).astype('<f4'),
                group_ids=trainer.np.asarray([hashlib.sha256(str((offset, i)).encode()).digest()
                    for i in range(count)], dtype='V32'), split='train',
                source_manifest_sha256='3' * 64, source_npz_sha256='4' * 64,
                source_route='synthetic/train.json', teacher_predictions=None)
        class Inputs(SimpleNamespace):
            def __getattr__(self, name):
                raise AssertionError('calibration touched nontraining input: ' + name)
        return Inputs(new=pool(1057, 0), anchor=pool(3105, 2000))

    def test_training_fixture_is_exact_PCG64_and_never_reads_heldout(self):
        inputs = self.training_only_inputs()
        with tempfile.TemporaryDirectory() as temporary:
            fixture, document = trainer._channel_fixture(inputs, Path(temporary))
            repeated, duplicate = trainer._channel_fixture(inputs, Path(temporary))
            self.assertEqual(document, duplicate)
            generator = trainer.np.random.Generator(trainer.np.random.PCG64(20260908))
            expected_new = sorted(map(int, generator.choice(1057, 1024, replace=False)))
            expected_anchor = sorted(map(int, generator.choice(3105, 3072, replace=False)))
            self.assertEqual(document['identity']['sampled_indices'],
                {'new': expected_new, 'anchor': expected_anchor})
            self.assertEqual(len(fixture), 4096)
            self.assertEqual(fixture.indices.tolist(), expected_new + [2000 + i for i in expected_anchor])
            expected_weights = trainer.np.concatenate((inputs.new.weights[expected_new], inputs.anchor.weights[expected_anchor]))
            self.assertEqual(fixture.weights.tobytes(), expected_weights.tobytes())
            self.assertFalse(hasattr(fixture, 'targets'))
            self.assertFalse(hasattr(fixture, 'common_adjudicator'))
            with trainer.np.load(document['artifact']['path'], allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {'indptr', 'indices', 'weights', 'new_indices', 'anchor_indices'})
            for array in (fixture.indptr, fixture.indices, fixture.weights, repeated.indices):
                self.assertFalse(array.flags.writeable)

    def test_historical_profiles_and_attempts_one_through_four_stay_readable(self):
        from tools import compact_value_bfm_attribution_v2 as attribution
        from tools import compact_value_bfm_intervention_v2 as intervention
        from tests.codingame.test_compact_value_bfm_intervention_v2 import fourth_fixture
        expected = {
            'standard-v1': '9b7dd736fa296cf9869368656427d40dd6f2faa0412f1a2b23384c1b8d124f3c',
            'refined-adaptive-scales-v1': '7b4abbd1f5fbfb041c4578ce139fdea804e32017c6041fb0b1dbcb87da4af3bd',
            'retention-first-low-rate-v1': '3cdbd16892860b6db547ca510e97604ba7742ae07392df64fd817b315b975de3',
        }
        for name, digest in expected.items():
            with self.subTest(profile=name):
                self.assertEqual(trainer.qat_profile_contract(name)['body_sha256'], digest)
        self.assertEqual(set(attribution.profile_menu()['qat_and_scales']),
                         {'standard-v1', 'refined-adaptive-scales-v1'})
        self.assertEqual(set(attribution.profile_menu(after_attempts=3)['qat_and_scales']), set(expected))
        with tempfile.TemporaryDirectory() as temporary:
            _root, _parent, _report, previous, fourth = fourth_fixture(Path(temporary))
            for row in previous:
                expected_name = 'standard-v1' if row['attempt'] < 3 else 'refined-adaptive-scales-v1'
                self.assertEqual(intervention.expected_qat_profile(row['contract']), expected_name)
            self.assertEqual(intervention.expected_qat_profile(fourth), 'retention-first-low-rate-v1')
        with self.assertRaises(ValueError):
            intervention.expected_qat_profile({'attempt': 5, 'qat_profile': 'channel-prediction-qat-v1'})

    def test_calibration_keeps_frozen_targets_and_requantizes_current_masters(self):
        inputs = self.training_only_inputs(); architecture = trainer.ARCHITECTURES['capacity-12x8']
        parameters = trainer.initialize_parameters(architecture, 20260907)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture, fixture_document = trainer._channel_fixture(inputs, root / 'fixture')
            reference, document = trainer._channel_prediction_reference(parameters, architecture,
                fixture, fixture_document, root / 'reference')
            frozen_bytes = reference.tobytes(); frozen_parameters = document['pre_qat_parameters']
            self.assertFalse(reference.flags.writeable)
            current = {name: value.copy() for name, value in parameters.items()}
            current['w1'][0, 0] += trainer.np.float32(.125)
            current['w3'][0] += trainer.np.float32(.25)
            scales = {name: trainer.np.full(count, .125, dtype=trainer.np.float32)
                      for name, count in (('w1', 12), ('w2', 8), ('w3', 1))}
            expected = trainer.quantize_channels(current, architecture, scales)
            predictor = trainer.predict_dataset; observations = []
            def predict(values, arch, dataset, **kwargs):
                self.assertIs(dataset, fixture)
                self.assertEqual(trainer._parameter_identity(values, arch), trainer._parameter_identity(current, architecture))
                observations.append(1)
                return predictor(values, arch, dataset, **kwargs)
            # Restrict enumeration to its incumbent to isolate ownership and
            # continuity. Owner tests separately exercise the full search menu.
            with mock.patch.object(trainer, '_channel_coordinate_candidates', side_effect=lambda _, incumbent: (incumbent,)), \
                    mock.patch.object(trainer, 'predict_dataset', side_effect=predict), \
                    mock.patch.object(trainer, 'evaluate_validation_pair', side_effect=AssertionError('heldout calibration')):
                actual, report = trainer._channel_calibrate(current, architecture, fixture, reference, document,
                    scales, root / 'calibration', epoch=1)
            self.assertEqual(len(observations), 42)
            self.assertEqual(reference.tobytes(), frozen_bytes)
            self.assertEqual(report['training_reference'], document)
            self.assertNotEqual(report['master_parameters'], frozen_parameters)
            self.assertEqual(report['master_parameters'], trainer._parameter_identity(current, architecture))
            self.assertTrue(report['parameters_unchanged_by_calibration'])
            for name in ('w1', 'w2', 'w3'):
                self.assertEqual(actual.integer[name].tobytes(), expected.integer[name].tobytes())
                self.assertEqual(report['selected_code_sha256'][name], hashlib.sha256(expected.integer[name].tobytes()).hexdigest())
            matrix = trainer.np.load(report['prediction_matrix']['path'], mmap_mode='r', allow_pickle=False)
            try:
                self.assertEqual(matrix.shape, (672, 4096))
                self.assertEqual(report['prediction_rows'], 42)
                self.assertTrue(trainer.np.array_equal(matrix[0], matrix[41]))
                self.assertFalse(trainer.np.any(matrix[42:] != 0))
                weights = fixture.weights.astype(trainer.np.float64)
                delta = matrix[0].astype(trainer.np.float64) - reference.astype(trainer.np.float64)
                score = float(trainer.np.sum(weights * delta * delta) / trainer.np.sum(weights))
                self.assertEqual(score, report['selected_objective'])
            finally:
                matrix._mmap.close()

    def test_wrong_calibration_fixture_or_target_rejects_before_forward_or_output(self):
        inputs = self.training_only_inputs(); architecture = trainer.ARCHITECTURES['capacity-12x8']
        parameters = trainer.initialize_parameters(architecture, 20260907)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture, fixture_document = trainer._channel_fixture(inputs, root / 'fixture')
            reference, document = trainer._channel_prediction_reference(parameters, architecture,
                fixture, fixture_document, root / 'reference')
            wrong = (reference + trainer.np.float32(.25)).astype(trainer.np.float32); wrong.flags.writeable = False
            scales = {name: trainer.np.full(count, .125, dtype=trainer.np.float32)
                      for name, count in (('w1', 12), ('w2', 8), ('w3', 1))}
            with mock.patch.object(trainer, 'predict_dataset', side_effect=AssertionError('unbound forward')):
                with self.assertRaisesRegex(trainer.TrainingError, 'frozen reference'):
                    trainer._channel_calibrate(parameters, architecture, fixture, wrong, document,
                        scales, root / 'bad-target', epoch=1)
                saved = fixture.indices
                fixture.indices = saved.copy(); fixture.indices[0] += 1; fixture.indices.flags.writeable = False
                with self.assertRaisesRegex(trainer.TrainingError, 'audited identity'):
                    trainer._channel_prediction_reference(parameters, architecture, fixture, fixture_document, root / 'bad-freeze')
                with self.assertRaisesRegex(trainer.TrainingError, 'frozen reference'):
                    trainer._channel_calibrate(parameters, architecture, fixture, reference, document,
                        scales, root / 'bad-fixture', epoch=1)
            self.assertTrue(all(not (root / name).exists() for name in ('bad-target', 'bad-freeze', 'bad-fixture')))

    def test_seed_reader_rejects_foreign_channel_artifact_before_evidence_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root / 'seed'; output.mkdir()
            profile = trainer.qat_profile_contract(trainer.CHANNEL_PREDICTION_QAT_PROFILE)
            binding = trainer.body_hashed({'schema': 'papersoccer.compact-value-bfm-training-binding.v1',
                'settings': {'qat_profile': trainer.CHANNEL_PREDICTION_QAT_PROFILE,
                    'qat_profile_contract': profile, 'qat_learning_rate': .0000625}})
            forbidden = {'path': str(root / 'foreign' / ('a' * 64 + '.channel-reference.npy')),
                         'sha256': 'a' * 64, 'bytes': 1}
            with trainer.native_thread_execution_scope() as native:
                receipt = trainer.body_hashed({'schema': trainer.SEED_RECEIPT_SCHEMA, 'binding': binding,
                    'native_thread_execution': native, 'qat_profile': trainer.CHANNEL_PREDICTION_QAT_PROFILE,
                    'qat_profile_contract': profile, 'float_validation': {},
                    'quantized_training': {'training_prediction_reference': {'prediction': forbidden}}})
            payload = trainer.canonical_json_bytes(receipt); digest = hashlib.sha256(payload).hexdigest()
            receipt_path = output / 'seed-receipts' / (digest + '.seed-receipt.json')
            receipt_path.parent.mkdir(); receipt_path.write_bytes(payload)
            reference = output / 'reference.json'
            reference.write_bytes(trainer.canonical_json_bytes(trainer.body_hashed({
                'schema': trainer.SEED_REFERENCE_SCHEMA, 'receipt': 'seed-receipts/' + receipt_path.name,
                'receipt_sha256': digest, 'binding_body_sha256': binding['body_sha256']})))
            with mock.patch.object(trainer, 'validate_qat_execution_evidence', side_effect=AssertionError('artifact reader ran')) as read:
                with self.assertRaisesRegex(trainer.TrainingError, 'escaped its seed output'):
                    trainer._load_seed_receipt_from_reference(output, reference, binding)
            read.assert_not_called()

    def test_channel_outer_links_reject_transplanted_fixture_initialization_and_runtime(self):
        architecture = trainer.ARCHITECTURES['capacity-12x8']
        parameters = trainer.initialize_parameters(architecture, 20260907)
        inputs = self.training_only_inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _runtime, _architecture, quantized = self.channel_runtime(root / 'runtime')
            initial = trainer.write_float_checkpoint(root / 'initial', parameters, architecture)
            identity = trainer._parameter_identity(parameters, architecture)
            datasets = {name: trainer.dataset_identity(getattr(inputs, name)) for name in ('new', 'anchor')}
            validation = {name: {'sign_accuracy': .9, 'weighted_huber': .04}
                          for name in ('common_adjudicator', 'canonical_validation')}
            report = {'training_fixture': {'identity': {'datasets': datasets}},
                'training_prediction_reference': {'pre_qat_parameters': identity},
                'original_initialization_checkpoint': trainer._channel_artifact(initial),
                'selected_scales': trainer._channel_scale_document(quantized), 'selected_validation': validation,
                'selected_qat_epoch': 1,
                'history': [{'adaptive_scale_search': {'selected_code_sha256': {
                    name: hashlib.sha256(value.tobytes()).hexdigest()
                    for name, value in quantized.integer.items()}}}]}
            warmup = {'seed': 20260907, 'validation': validation,
                'optimizer': {'learning_rate': trainer.RANKING_FLOAT_LEARNING_RATE},
                'per_layer_update_evidence': trainer._parameter_update_evidence(parameters, parameters)}
            report['float_warmup_evidence_sha256'] = hashlib.sha256(trainer.canonical_json_bytes(warmup)).hexdigest()
            receipt = {'seed': 20260907, 'quantized_training': report, 'float_validation': validation,
                'float_training': warmup,
                'quantized_validation': copy.deepcopy(validation),
                'offline_gate': trainer.offline_advancement_gate(validation, validation)}
            binding = {'seed': 20260907, 'source_bundle_body_sha256': '1' * 64,
                'datasets': datasets, 'successor_ranking': {'initial_checkpoint': {'parameters': identity}}}
            runtime_selection = {'qat_profile': trainer.CHANNEL_PREDICTION_QAT_PROFILE,
                'qat_epoch': 1, 'float_epoch': 1, 'source_bundle_body_sha256': '1' * 64,
                'qat_evidence_sha256': hashlib.sha256(trainer.canonical_json_bytes(report)).hexdigest()}
            runtime_document = {'schema': trainer.CHANNEL_RUNTIME_SCHEMA}
            trainer._validate_channel_receipt_links(receipt, binding, root, parameters, quantized,
                runtime_selection, runtime_document)
            for fault in ('fixture', 'warmup', 'warmup-evidence', 'initialization', 'evidence', 'scales', 'metrics', 'gate', 'payload'):
                changed, expected, runtime = copy.deepcopy(receipt), copy.deepcopy(binding), copy.deepcopy(runtime_selection)
                decoded, runtime_body = quantized, runtime_document
                if fault == 'fixture': expected['datasets']['new']['sha256'] = 'f' * 64
                elif fault == 'warmup': changed['quantized_training']['training_prediction_reference']['pre_qat_parameters']['layers']['w1']['sha256'] = 'f' * 64
                elif fault == 'warmup-evidence': changed['float_training']['optimizer']['learning_rate'] = .00025
                elif fault == 'initialization': expected['successor_ranking']['initial_checkpoint']['parameters']['layers']['w1']['sha256'] = 'f' * 64
                elif fault == 'scales': changed['quantized_training']['selected_scales']['w1'][0] *= 2
                elif fault == 'metrics': changed['quantized_validation']['common_adjudicator']['weighted_huber'] += .01
                elif fault == 'gate': changed['offline_gate']['passed'] = False
                elif fault == 'payload':
                    codes = {name: value.copy() for name, value in quantized.integer.items()}
                    codes['w1'][0, 0] += 1  # Valid three-bit code, same scale/evidence binding.
                    alternate = trainer.ChannelQuantizedWeights(codes, quantized.scales)
                    alternate_path = trainer.write_runtime(root / 'alternate', architecture, alternate,
                        arm=trainer.ARMS['search-target'], seed=20260907, float_epoch=1, qat_epoch=1,
                        source_bundle_body_sha256='1' * 64,
                        qat_profile=trainer.CHANNEL_PREDICTION_QAT_PROFILE,
                        qat_evidence_sha256=runtime['qat_evidence_sha256'])
                    _arch, decoded, runtime, runtime_body = trainer.load_runtime(alternate_path)
                    self.assertEqual(runtime['qat_evidence_sha256'], runtime_selection['qat_evidence_sha256'])
                    self.assertNotEqual(hashlib.sha256(decoded.integer['w1'].tobytes()).hexdigest(),
                        report['history'][0]['adaptive_scale_search']['selected_code_sha256']['w1'])
                runtime['qat_evidence_sha256'] = ('f' * 64 if fault == 'evidence' else
                    hashlib.sha256(trainer.canonical_json_bytes(changed['quantized_training'])).hexdigest())
                with self.subTest(fault=fault), self.assertRaises(trainer.TrainingError):
                    trainer._validate_channel_receipt_links(changed, expected, root, parameters, decoded,
                        runtime, runtime_body)

    def test_legacy_v1_runtime_header_and_source_remain_exact_for_all_architectures(self):
        # Computed independently with immutable829c009 trainer/exporter/native
        # sources. These constants need no ignored snapshots or campaign inputs.
        expected = {
            'compact-8x8': (
                'bed1be3ea615d8658f1afa0c80e853b60b03dd197bb7343dc26b5670d0879714',
                '73ca68eea3d0f4d8fdd32a9602efc178aedaed44be0a70d54dc6e1e300240871',
                '8fbcdcae9f584a69c7a49cc02f96d6e8c487d3790cfabc2a8065ecf3152aa958', 79255),
            'source-neutral-8x16': (
                'edf8a70621524d0b2894a7b97522b122868a2c18f2b3b67d2e7491bf92bdf877',
                'ffa8477a6fbf31f58ad94667518ea0feb79a267202fe781bdfda214671be1e45',
                '08e509647da7dab59446ff07dacf4cdb6c3d4e88e670f402ff4bf0db6c8a4f33', 79302),
            'capacity-12x8': (
                'e96abb2e371ecd3f46b18ad95a9f6f744b9ad9701ddb0a567b47c9e377158dfb',
                '870711849a3228ffebe48149a67f2bea3360d38744ee78cc20e4a85d966bde4a',
                'c30b7459e6697c110d103b043f0f4362d79744b5bacafb9e0c6c2ce70afae684', 92142),
        }
        with tempfile.TemporaryDirectory() as temporary:
            for name, values in expected.items():
                architecture = trainer.ARCHITECTURES[name]
                quantized = trainer.QuantizedWeights(
                    {key: trainer.np.zeros(shape, dtype=trainer.np.int8)
                     for key, shape in architecture.shapes.items()},
                    {'w1': trainer.np.float32(.125), 'w2': trainer.np.float32(.25),
                     'w3': trainer.np.float32(.5)})
                runtime = trainer.write_runtime(Path(temporary) / name, architecture, quantized,
                    arm=trainer.ARMS['search-target'], seed=20260907, float_epoch=1, qat_epoch=4,
                    source_bundle_body_sha256='1' * 64)
                header, _metadata = self.exporter.render_header(runtime)
                _output, source = self.renderer.render(model_header=header)
                with self.subTest(architecture=name):
                    self.assertEqual((hashlib.sha256(runtime.read_bytes()).hexdigest(),
                        hashlib.sha256(header).hexdigest(), hashlib.sha256(source).hexdigest(),
                        len(source)), values)

    def test_channel_v2_roundtrip_and_export_use_exact_output_axes(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, architecture, expected = self.channel_runtime(temporary)
            loaded_architecture, actual, selection, document = trainer.load_runtime(runtime)
            exported, _payload, metadata = self.exporter.validate_runtime(runtime)
            self.assertEqual(document, exported)
            self.assertEqual(loaded_architecture, architecture)
            self.assertEqual(document['schema'], 'papersoccer.compact-value-bfm-runtime.v2')
            self.assertEqual(document['quantization']['scale_axis'], 'output')
            self.assertEqual(document['quantization']['scale_counts'], {'w1': 12, 'w2': 8, 'w3': 1})
            self.assertEqual(selection['qat_profile'], 'channel-prediction-qat-v1')
            self.assertEqual(selection['qat_evidence_sha256'], '2' * 64)
            for name in ('w1', 'w2', 'w3'):
                self.assertEqual(actual.integer[name].tobytes(), expected.integer[name].tobytes())
                self.assertEqual(actual.scales[name].tobytes(), expected.scales[name].tobytes())
            header, _metadata = self.exporter.render_header(runtime)
            _output, source = self.renderer.render(model_header=header)
            self.assertLessEqual(len(source), 93000)
            self.assertNotIn(b'#include "', source)

    def test_resealed_channel_contract_shape_values_and_version_confusion_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, _architecture, _quantized = self.channel_runtime(temporary)
            original = json.loads(runtime.read_bytes())
            mutations = {
                'short-w1': lambda d: d['quantization']['scales']['w1'].pop(),
                'nested-w2': lambda d: d['quantization']['scales'].update(w2=[[x] for x in d['quantization']['scales']['w2']]),
                'w3-is-one-output': lambda d: d['quantization']['scales'].update(w3=[.125] * 8),
                'v2-cannot-use-scalar': lambda d: d['quantization']['scales'].update(w1=.125),
                'axis': lambda d: d['quantization'].update(scale_axis='input'),
                'granularity': lambda d: d['quantization'].update(granularity='per-layer'),
                'counts': lambda d: d['quantization']['scale_counts'].update(w3=8),
                'boolean-count': lambda d: d['quantization']['scale_counts'].update(w3=True),
                'boolean-scale': lambda d: d['quantization']['scales']['w1'].__setitem__(0, True),
                'string-scale': lambda d: d['quantization']['scales']['w1'].__setitem__(0, '.125'),
                'zero-scale': lambda d: d['quantization']['scales']['w1'].__setitem__(0, 0.),
                'negative-scale': lambda d: d['quantization']['scales']['w1'].__setitem__(0, -.125),
                'noncanonical-float32': lambda d: d['quantization']['scales']['w1'].__setitem__(0, .1),
                'v1-schema-with-v2-contract': lambda d: d.update(schema='papersoccer.compact-value-bfm-runtime.v1'),
                'unknown-version': lambda d: d.update(schema='papersoccer.compact-value-bfm-runtime.v3'),
                'wrong-profile': lambda d: d['selection'].update(qat_profile='retention-first-low-rate-v1'),
                'float-seed': lambda d: d['selection'].update(seed=20260907.0),
                'missing-evidence': lambda d: d['selection'].pop('qat_evidence_sha256'),
                'wrong-evidence': lambda d: d['selection'].update(qat_evidence_sha256='A' * 64),
            }
            for name, mutation in mutations.items():
                document = copy.deepcopy(original); mutation(document)
                path = self.reseal(temporary, document)
                for label, loader in (('Python', trainer.load_runtime), ('exporter', self.exporter.validate_runtime)):
                    with self.subTest(case=name, loader=label), self.assertRaises((ValueError, trainer.TrainingError)):
                        loader(path)

    def test_resealed_forbidden_code_and_padding_rejected_by_both_loaders(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, _architecture, _quantized = self.channel_runtime(temporary)
            for fault in ('code-minus-four', 'padding'):
                document = json.loads(runtime.read_bytes()); quantization = document['quantization']
                payload = bytearray(base64.b64decode(quantization['payload_base64']))
                if fault == 'code-minus-four': payload[0] = (payload[0] & ~7) | 4
                else: payload[-1] |= 16  # 75,716 codes use only four bits of the last byte.
                quantization.update(payload_base64=base64.b64encode(payload).decode(),
                    payload_sha256=hashlib.sha256(payload).hexdigest())
                path = self.reseal(temporary, document)
                for label, loader in (('Python', trainer.load_runtime), ('exporter', self.exporter.validate_runtime)):
                    with self.subTest(fault=fault, loader=label), self.assertRaises((ValueError, trainer.TrainingError)):
                        loader(path)

    def test_native_json_loader_and_generated_v2_source_match_python_full_delta_and_mover(self):
        from tools import jacek_replay_features as features
        compiler = shlex.split(os.environ.get('CXX', 'c++'))
        if not compiler or shutil.which(compiler[0]) is None:
            self.skipTest('C++20 compiler unavailable for the isolated runtime fixture')
        body = r'''
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
namespace cv = compact_value_bfm;
int emit(const cv::QuantizedModel& model) {
  auto state = cv::initial_state();
  const auto prepared = model.prepare(cv::active_features(state));
  for (int step = 0; step < 24 && !state.terminal(); ++step) {
    const auto active = cv::active_features(state);
    const auto rotated = cv::active_features(cv::rotate_and_swap(state));
    if (!(active == rotated)) throw std::runtime_error("mover features changed");
    std::array<cv::Topology::Arc, 8> arcs{};
    const auto count = cv::legal_arcs(state, arcs);
    if (count == 0) throw std::runtime_error("fixture has no legal edge");
    const int direction = arcs[static_cast<std::size_t>(step) % count].direction;
    std::cout << static_cast<int>(state.to_move) << ' ' << std::hex
      << std::bit_cast<std::uint32_t>(model.evaluate(active)) << ' '
      << std::bit_cast<std::uint32_t>(model.evaluate_delta(prepared, active)) << ' '
      << std::bit_cast<std::uint32_t>(model.evaluate(rotated)) << ' ' << std::dec
      << direction << ' ';
    for (std::size_t index = 0; index < active.count; ++index) {
      if (index) std::cout << ',';
      std::cout << active.indices[index];
    }
    std::cout << '\n';
    if (!cv::apply_edge(state, direction)) throw std::runtime_error("fixture edge failed");
  }
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime, architecture, quantized = self.channel_runtime(root / 'runtime')
            header, _metadata = self.exporter.render_header(runtime)
            _output, source = self.renderer.render(model_header=header)
            (root / 'generated.cpp').write_bytes(source)
            (root / 'generated-probe.cpp').write_text(
                '#define COMPACT_VALUE_BFM_NO_MAIN\n#include "generated.cpp"\n' + body
                + '\nint main() { return emit(cv::deployment_model()); }\n')
            (root / 'loader-probe.cpp').write_text(
                '#include "tools/compact_value_bfm_runtime_loader.hpp"\n' + body + r'''
int main(int argc, char** argv) {
  try {
    if (argc < 2) return 3;
    auto runtime = papersoccer::compact_value_bfm_runtime::load(argv[1]);
    if (argc > 2) return 0;
    if (runtime.identity.runtime_schema != "papersoccer.compact-value-bfm-runtime.v2" ||
        runtime.identity.qat_profile != "channel-prediction-qat-v1" ||
        runtime.identity.qat_evidence_sha256 != std::string(64, '2')) return 4;
    return emit(*runtime.model);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 2;
  }
}
''')
            flags = ['-std=c++20', '-O1', '-ffp-contract=off', '-fno-fast-math', '-I', str(ROOT)]
            for kind, extra in (('generated', []), ('loader', [str(BOT / 'engine.cpp'),
                    str(ROOT / 'tools/compact_value_bfm_runtime_loader.cpp')])):
                compiled = subprocess.run([*compiler, *flags, str(root / (kind + '-probe.cpp')),
                    *extra, '-o', str(root / kind)], capture_output=True, text=True, timeout=120)
                self.assertEqual(compiled.returncode, 0, compiled.stderr[-4000:])
            generated = subprocess.run([str(root / 'generated')], capture_output=True, text=True, timeout=10)
            loaded = subprocess.run([str(root / 'loader'), str(runtime)], capture_output=True, text=True, timeout=10)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            self.assertEqual(loaded.returncode, 0, loaded.stderr)
            self.assertEqual(generated.stdout, loaded.stdout)
            state = features.ReplayState(); movers = set(); values = set()
            for line in generated.stdout.splitlines():
                mover, full, delta, rotated, direction, encoded = line.split()
                active = tuple(map(int, encoded.split(',')))
                self.assertEqual(int(mover), state.to_move)
                self.assertEqual(active, features.encode_active(state))
                predicted = trainer.scalar_quantized_forward(quantized, architecture, active)
                expected = struct.unpack('<I', predicted.tobytes())[0]
                self.assertEqual((int(full, 16), int(delta, 16), int(rotated, 16)), (expected,) * 3)
                movers.add(int(mover)); values.add(full)
                features.apply_primitive(state, int(direction))
            self.assertEqual(movers, {0, 1})
            self.assertGreater(len(values), 1)
            self.assertGreaterEqual(len(generated.stdout.splitlines()), 8)
            # Parser-only boundary cases: these arbitrary extreme descriptors
            # make no inference-quality or finite-activation claim.
            for scale in (float(trainer.np.nextafter(trainer.np.float32(0), trainer.np.float32(1))),
                          float(trainer.np.float32(2. ** 64))):
                extreme = json.loads(runtime.read_bytes())
                extreme['quantization']['scales'] = {name: [scale] * count
                    for name, count in (('w1', 12), ('w2', 8), ('w3', 1))}
                extreme_path = self.reseal(root, extreme)
                trainer.load_runtime(extreme_path)
                self.exporter.validate_runtime(extreme_path)
                checked = subprocess.run([str(root / 'loader'), str(extreme_path), 'validate-only'],
                    capture_output=True, text=True, timeout=10)
                with self.subTest(native_accepts_scale=scale):
                    self.assertEqual(checked.returncode, 0, checked.stderr)
            # Valid three-bit codes can still produce nonfinite effective
            # weights. Check each tensor's final output-channel orientation.
            for layer in ('w1', 'w2', 'w3'):
                for code in (-3, -2, 2, 3):
                    overflow = json.loads(runtime.read_bytes())
                    integers = {name: trainer.np.zeros(shape, dtype=trainer.np.int8)
                                for name, shape in architecture.shapes.items()}
                    if layer == 'w3': integers[layer][-1] = code
                    else: integers[layer][-1, -1] = code
                    overflow['quantization']['scales'][layer][-1] = float(trainer.np.finfo(trainer.np.float32).max)
                    flat = trainer.np.concatenate([integers[name].reshape(-1) for name in ('w1', 'w2', 'w3')])
                    packed = trainer.pack_signed_three_bit(flat)
                    overflow['quantization'].update(payload_base64=base64.b64encode(packed).decode(),
                        payload_sha256=hashlib.sha256(packed).hexdigest())
                    overflow_path = self.reseal(root, overflow)
                    with trainer.np.errstate(over='ignore', invalid='ignore'):
                        for label, reader in (('Python', trainer.load_runtime), ('exporter', self.exporter.validate_runtime)):
                            with self.subTest(overflow_layer=layer, code=code, reader=label), \
                                    self.assertRaises((ValueError, trainer.TrainingError)):
                                reader(overflow_path)
                    checked = subprocess.run([str(root / 'loader'), str(overflow_path), 'validate-only'],
                        capture_output=True, text=True, timeout=10)
                    with self.subTest(native_overflow_layer=layer, code=code):
                        self.assertEqual(checked.returncode, 2, checked.stderr)
            # A native loader must reject a fully rehashed mixed-version shape,
            # not just a mismatching outer checksum.
            original = json.loads(runtime.read_bytes())
            for field in ('axis', 'counts', 'scale', 'profile', 'forbidden-code'):
                changed = copy.deepcopy(original)
                if field == 'axis': changed['quantization']['scale_axis'] = 'input'
                elif field == 'counts': changed['quantization']['scale_counts']['w3'] = True
                elif field == 'scale': changed['quantization']['scales']['w1'][0] = .1
                elif field == 'profile': changed['selection']['qat_profile'] = 'retention-first-low-rate-v1'
                else:
                    data = bytearray(base64.b64decode(changed['quantization']['payload_base64']))
                    data[0] = (data[0] & ~7) | 4
                    changed['quantization'].update(payload_base64=base64.b64encode(data).decode(),
                        payload_sha256=hashlib.sha256(data).hexdigest())
                malformed = self.reseal(root, changed)
                checked = subprocess.run([str(root / 'loader'), str(malformed), 'validate-only'],
                    capture_output=True, text=True, timeout=10)
                with self.subTest(native_rejects=field): self.assertEqual(checked.returncode, 2)


if __name__ == '__main__':
    unittest.main()
