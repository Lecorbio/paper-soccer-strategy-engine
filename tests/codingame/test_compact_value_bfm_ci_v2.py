import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tools import compact_value_bfm_ci_v2 as ci


campaign = ci.campaign
ROOT = Path(__file__).resolve().parents[2]


def gh_run(publication):
    run_id = 12345
    return {
        'databaseId': run_id, 'workflowDatabaseId': ci.maintained.WORKFLOW_DATABASE_ID,
        'attempt': 1, 'name': ci.maintained.WORKFLOW_NAME,
        'workflowName': ci.maintained.WORKFLOW_NAME, 'event': 'workflow_dispatch',
        'headBranch': publication['branch'], 'headSha': publication['commit'],
        'status': 'completed', 'conclusion': 'success',
        'url': f'{ci.maintained.RUN_URL_PREFIX}{run_id}',
        'jobs': [{'name': name, 'status': 'completed', 'conclusion': 'success',
                  'databaseId': index + 10,
                  'url': f'{ci.maintained.RUN_URL_PREFIX}{run_id}/job/{index + 10}'}
                 for index, name in enumerate(ci.maintained.JOB_NAMES)]
                + [{'name': 'deploy', 'status': 'completed', 'conclusion': 'skipped'}],
    }


class CiSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.repository = self.root / 'repository'
        self.repository.mkdir()
        self.git('init', '-b', 'feature/trained-source')
        self.git('remote', 'add', 'origin', ci.maintained.REPOSITORY_URL + '.git')
        for relative in ci.COVERAGE_PATHS:
            campaign.once(self.repository / relative, (ROOT / relative).read_bytes())
        self.source = self.repository / ci.SUPPORTED_SOURCE
        campaign.once(self.source, b'int main() { return 0; }\n')
        self.git('add', '.')
        self.commit()
        self.selected_source = self.root / 'selected.cpp'
        campaign.once(self.selected_source, self.source.read_bytes())
        runtime_path = self.root / 'runtime.json'
        runtime = campaign.seal(runtime_path, {'quantization': {'payload_sha256': 'a' * 64}})
        self.selection = {
            'schema': campaign.ID + '.search-selection.v2', 'eligible_for_multi_opponent': True,
            'required_ablation_complete': True,
            'selected': {'source': campaign.record(self.selected_source),
                         'runtime': campaign.record(runtime_path),
                         'runtime_body_sha256': runtime['body_sha256'], 'payload_sha256': 'a' * 64},
        }
        self.selection_path = self.root / 'search-selection.json'
        campaign.seal(self.selection_path, self.selection)
        patch = mock.patch.object(ci, '_validated_selection', return_value=self.selection)
        self.validator = patch.start()
        self.addCleanup(patch.stop)
        self.publication_path = self.root / 'publication.json'

    def git(self, *arguments):
        return subprocess.run(['git', '-C', str(self.repository), *arguments],
                              capture_output=True, check=True).stdout

    def commit(self):
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 '-c', 'commit.gpgsign=false', 'commit', '-m', 'Freeze test source')

    def freeze(self, source=None):
        return ci.freeze_publication(self.publication_path, repository=self.repository,
                                     source=source or self.source, source_selection=self.selection_path,
                                     root=self.root, context=self.root / 'context', phase='attempt-001-full')

    def record(self, publication, payload=None):
        raw = self.root / 'raw.json'
        raw.write_text(json.dumps(payload or gh_run(publication), indent=2) + '\n')
        path = self.root / 'ci.json'
        ci.record_ci(path, publication_path=self.publication_path, raw_gh_path=raw,
                     fetched_at_utc='2026-09-05T04:00:00Z')
        return path, raw

    def test_exact_selected_tracked_source_and_raw_run_roundtrip(self):
        publication = self.freeze()
        self.assertEqual(publication['branch'], 'feature/trained-source')
        self.assertEqual(publication['commit'], self.git('rev-parse', 'HEAD').decode().strip())
        self.assertEqual(publication['source']['sha256'], campaign.sha(self.selected_source))
        self.assertEqual(len(publication['compile_coverage']), 4)
        path, raw = self.record(publication)
        evidence = ci.validate_ci(path)
        self.assertEqual(campaign.verify(evidence['raw_gh_payload']).read_bytes(), raw.read_bytes())
        self.assertTrue(evidence['source_specific_ci_passed'])
        self.assertFalse(evidence['qualification_passed'])
        self.assertFalse(evidence['campaign_success'])
        self.assertEqual(set(evidence['jobs']), set(ci.maintained.REQUIRED_JOB_IDS))
        self.assertGreaterEqual(self.validator.call_count, 3)
        with self.assertRaises(ci.maintained.UploadError):
            ci.maintained.validate_gh_run(gh_run(publication), expected_head=publication['commit'])

    def test_worktree_or_index_changes_cannot_bind_an_older_green_head(self):
        self.source.write_text('int main() { return 1; }\n')
        with self.assertRaisesRegex(ValueError, 'clean Git'):
            self.freeze()
        self.git('add', '.')
        with self.assertRaisesRegex(ValueError, 'clean Git'):
            self.freeze()

    def test_committed_uncompiled_source_route_is_rejected(self):
        other = self.repository / 'foo.cpp'
        other.write_bytes(self.source.read_bytes())
        self.git('add', '.')
        self.commit()
        with self.assertRaisesRegex(ValueError, 'explicitly compiled'):
            self.freeze(other)

    def test_exact_release_candidate_has_explicit_ci_coverage(self):
        for relative in ci.RELEASE_COVERAGE_PATHS:
            campaign.once(self.repository / relative, (ROOT / relative).read_bytes())
        release=self.repository/ci.RELEASE_SOURCE
        release.write_bytes(self.source.read_bytes())
        self.git('add','.')
        self.commit()
        publication=self.freeze(release)
        self.assertEqual(publication['source']['path'],ci.RELEASE_SOURCE)
        self.assertEqual(len(publication['compile_coverage']),7)
        path,_=self.record(publication)
        self.assertTrue(ci.validate_ci(path)['source_specific_ci_passed'])
        self.assertEqual((self.repository/ci.SUPPORTED_SOURCE).read_bytes(),self.selected_source.read_bytes())

    def test_release_candidate_requires_committed_harness_mapping(self):
        for relative in ci.RELEASE_COVERAGE_PATHS:
            campaign.once(self.repository / relative, (ROOT / relative).read_bytes())
        release=self.repository/ci.RELEASE_SOURCE
        release.write_bytes(self.source.read_bytes())
        (self.repository/ci.RELEASE_COVERAGE_PATHS[0]).write_text('# no source checks\n')
        self.git('add','.')
        self.commit()
        with self.assertRaisesRegex(ValueError,'release candidate compile coverage'):
            self.freeze(release)

    def test_selected_source_must_equal_committed_compiled_source(self):
        self.selected_source.write_text('int main() { return 1; }\n')
        self.selection['selected']['source'] = campaign.record(self.selected_source)
        with self.assertRaisesRegex(ValueError, 'exact selected source'):
            self.freeze()

    def test_runtime_and_actual_selection_are_required(self):
        self.selection['selected']['payload_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'runtime identity'):
            self.freeze()
        self.selection['selected']['payload_sha256'] = 'a' * 64
        self.validator.side_effect = ValueError('source selection not complete')
        with self.assertRaisesRegex(ValueError, 'not complete'):
            self.freeze()

    def test_changed_origin_and_compile_coverage_are_rejected(self):
        self.git('remote', 'set-url', 'origin', 'https://github.com/other/repository.git')
        with self.assertRaisesRegex(ValueError, 'fixed repository'):
            self.freeze()
        self.git('remote', 'set-url', 'origin', ci.maintained.REPOSITORY_URL + '.git')
        (self.repository / 'CMakeLists.txt').write_text('project(empty)\n')
        self.git('add', '.')
        self.commit()
        with self.assertRaisesRegex(ValueError, 'compile coverage'):
            self.freeze()

    def test_run_repository_workflow_event_branch_head_and_attempt_are_exact(self):
        publication = self.freeze()
        for key, value in (
            ('url', 'https://github.com/other/repository/actions/runs/12345'),
            ('workflowDatabaseId', ci.maintained.WORKFLOW_DATABASE_ID + 1),
            ('event', 'push'), ('headBranch', ci.maintained.BRANCH),
            ('headSha', 'f' * 40), ('attempt', 2), ('attempt', True),
        ):
            with self.subTest(key=key, value=value):
                payload = gh_run(publication)
                payload[key] = value
                with self.assertRaises(ValueError):
                    ci.validate_gh_run(payload, publication=publication)

    def test_five_distinct_successful_required_jobs(self):
        publication = self.freeze()
        mutations = [
            lambda jobs: jobs.pop(0),
            lambda jobs: jobs.append(copy.deepcopy(jobs[0])),
            lambda jobs: jobs[0].update(conclusion='skipped'),
            lambda jobs: jobs[0].update(conclusion='failure'),
            lambda jobs: jobs[1].update(databaseId=jobs[0]['databaseId'], url=jobs[0]['url']),
            lambda jobs: jobs[0].update(url='https://example.invalid/job/10'),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                payload = gh_run(publication)
                mutation(payload['jobs'])
                with self.assertRaises(ValueError):
                    ci.validate_gh_run(payload, publication=publication)

    def test_only_completed_skipped_deploy_may_be_omitted(self):
        publication = self.freeze()
        for changes in ({'name': 'unexpected'}, {'conclusion': None},
                        {'conclusion': 'success'}, {'status': 'queued'}):
            payload = gh_run(publication)
            payload['jobs'][-1].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ci.validate_gh_run(payload, publication=publication)

    def test_raw_payload_tampering_and_resealed_summary_tampering_fail(self):
        publication = self.freeze()
        path, _ = self.record(publication)
        evidence = campaign.read(path)
        bad = {key: value for key, value in evidence.items() if key != 'body_sha256'}
        bad['payload_sha256'] = 'e' * 64
        changed = self.root / 'changed.json'
        campaign.seal(changed, bad)
        with self.assertRaisesRegex(ValueError, 'differs from'):
            ci.validate_ci(changed)
        Path(evidence['raw_gh_payload']['path']).write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'changed artifact'):
            ci.validate_ci(path)

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'repeats a JSON key'):
            ci._json_payload(b'{"headSha":"a","headSha":"b"}')

    def test_historical_publication_remains_bound_when_local_head_advances(self):
        publication = self.freeze()
        other = self.repository / 'later.txt'
        other.write_text('unrelated next change\n')
        self.git('add', '.')
        self.commit()
        self.assertNotEqual(publication['commit'], self.git('rev-parse', 'HEAD').decode().strip())
        self.assertEqual(ci.validate_publication(self.publication_path), publication)


if __name__ == '__main__':
    unittest.main()
