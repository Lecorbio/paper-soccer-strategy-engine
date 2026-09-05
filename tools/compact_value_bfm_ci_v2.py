#!/usr/bin/env python3
"""Bind experimental-branch CI to an actually selected, committed source.

Only the maintained compact_value_bfm/submission.cpp route is supported: the
stock native build compiles it and submission_test.cpp includes it. Merely
committing another C++ file does not establish source-specific CI coverage.
These receipts establish CI evidence, never protected or live qualification.
No command publishes a source, dispatches a workflow, or edits a candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from collections.abc import Mapping

if __name__ == '__main__':
    for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_upload as maintained

PUBLICATION_SCHEMA = campaign.ID + '.ci-source-publication.v2'
CI_SCHEMA = campaign.ID + '.github-ci-evidence.v2'
SUPPORTED_SOURCE = 'submissions/codingame/bots/compact_value_bfm/submission.cpp'
COVERAGE_PATHS = ('.github/workflows/' + maintained.WORKFLOW_FILE, 'CMakeLists.txt',
                  'scripts/build-and-test.sh',
                  'submissions/codingame/bots/compact_value_bfm/submission_test.cpp')
COMMIT_RE = re.compile(r'[0-9a-f]{40}')
SHA_RE = re.compile(r'[0-9a-f]{64}')


def _git(repository, *arguments):
    result = subprocess.run(['git', '-C', str(repository), *arguments],
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError('Git source binding command failed: ' + arguments[0])
    return result.stdout


def _repository(repository):
    repository = Path(repository).resolve()
    if Path(_git(repository, 'rev-parse', '--show-toplevel').decode().strip()).resolve() != repository:
        raise ValueError('publication requires the actual Git repository root')
    origin = _git(repository, 'remote', 'get-url', 'origin').decode().strip()
    slug = maintained.REPOSITORY_SLUG
    if origin not in (f'https://github.com/{slug}', f'https://github.com/{slug}.git',
                      f'git@github.com:{slug}.git', f'ssh://git@github.com/{slug}.git'):
        raise ValueError('publication Git origin is not the fixed repository')
    return repository


def _clean_head(repository):
    if _git(repository, 'status', '--porcelain=v1', '--untracked-files=all'):
        raise ValueError('source publication requires a clean Git worktree and index')
    branch = _git(repository, 'symbolic-ref', '--quiet', '--short', 'HEAD').decode().strip()
    _git(repository, 'check-ref-format', '--branch', branch)
    commit = _git(repository, 'rev-parse', '--verify', 'HEAD^{commit}').decode().strip()
    if COMMIT_RE.fullmatch(commit) is None:
        raise ValueError('publication HEAD is not an exact commit')
    return branch, commit


def _committed_blob(repository, commit, relative):
    if COMMIT_RE.fullmatch(str(commit)) is None:
        raise ValueError('source publication commit is invalid')
    entry = _git(repository, 'ls-tree', '-z', commit, '--', relative).split(b'\0')
    if len(entry) != 2 or entry[1] or b'\t' not in entry[0]:
        raise ValueError('publication path is not a unique tracked file')
    descriptor, path = entry[0].split(b'\t', 1)
    mode, kind, object_id = descriptor.decode().split()
    if mode not in ('100644', '100755') or kind != 'blob' or path.decode() != relative:
        raise ValueError('publication path is not a committed regular file')
    data = _git(repository, 'cat-file', 'blob', object_id)
    return data, {'path': relative, 'git_blob': object_id, 'bytes': len(data),
                  'sha256': hashlib.sha256(data).hexdigest()}


def _validated_selection(root, context, phase, path):
    # Lazy import keeps the CI-only validator separate from source production.
    from tools import compact_value_bfm_search_v2 as search
    expected = search.directory(context, phase) / 'search-selection.json'
    if Path(path).resolve() != expected.resolve():
        raise ValueError('publication names a different source-selection artifact')
    return search.validate_selection(root, context, phase)


def _selection_identity(root, context, phase, path):
    selection = _validated_selection(root, context, phase, path)
    selected = selection.get('selected')
    if (selection.get('schema') != campaign.ID + '.search-selection.v2'
            or selection.get('eligible_for_multi_opponent') is not True
            or selection.get('required_ablation_complete') is not True
            or not isinstance(selected, Mapping)):
        raise ValueError('publication requires the validated trained source selection')
    source = campaign.verify(selected['source'])
    runtime_path = campaign.verify(selected['runtime'])
    runtime = campaign.read(runtime_path)
    body = selected.get('runtime_body_sha256')
    payload = selected.get('payload_sha256')
    if (SHA_RE.fullmatch(str(body)) is None or SHA_RE.fullmatch(str(payload)) is None
            or body != runtime['body_sha256']
            or payload != runtime['quantization']['payload_sha256']):
        raise ValueError('selected source runtime identity changed')
    return {'source_selection': campaign.record(path), 'selected_source': campaign.record(source),
            'runtime': campaign.record(runtime_path), 'runtime_body_sha256': body,
            'payload_sha256': payload}


def _coverage(repository, commit):
    records = []
    documents = {}
    for relative in COVERAGE_PATHS:
        data, record = _committed_blob(repository, commit, relative)
        documents[relative] = data.decode('utf-8')
        records.append(record)
    cmake = documents['CMakeLists.txt']
    script = documents['scripts/build-and-test.sh']
    workflow = documents[COVERAGE_PATHS[0]]
    required = ('set(PAPERSOCCER_CODINGAME_RESEARCH_BOTS\n  compact_value_bfm)',
                '${PAPERSOCCER_CODINGAME_RESEARCH_BOTS}',
                'foreach(PAPERSOCCER_CODINGAME_BOT IN LISTS PAPERSOCCER_CODINGAME_BUILD_BOTS)',
                'add_executable(${PAPERSOCCER_CODINGAME_PREFIX}_submission\n'
                '      "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/submission.cpp")')
    if (not all(item in cmake for item in required)
            or 'cmake "${build_args[@]}"' not in script
            or 'ctest "${ctest_args[@]}"' not in script
            or workflow.count('run: ./scripts/build-and-test.sh') != 3
            or '#include "submission.cpp"' not in documents[COVERAGE_PATHS[3]]):
        raise ValueError('committed workflow no longer has the supported source compile coverage')
    return records


def freeze_publication(output, *, repository, source, source_selection, root, context, phase):
    repository = _repository(repository)
    branch, commit = _clean_head(repository)
    source = Path(source)
    if source.is_symlink() or source.resolve() != repository / SUPPORTED_SOURCE:
        raise ValueError('CI publication supports only the compiled maintained submission.cpp route')
    identity = _selection_identity(root, context, phase, source_selection)
    committed, blob = _committed_blob(repository, commit, SUPPORTED_SOURCE)
    if (source.read_bytes() != committed or hashlib.sha256(committed).hexdigest()
            != identity['selected_source']['sha256']):
        raise ValueError('committed publication is not the exact selected source')
    committed.decode('ascii')
    coverage = _coverage(repository, commit)
    if _clean_head(repository) != (branch, commit):
        raise ValueError('Git publication changed while freezing source identity')
    return campaign.seal(output, {
        'schema': PUBLICATION_SCHEMA, 'repository': maintained.REPOSITORY_SLUG,
        'repository_path': str(repository), 'branch': branch, 'commit': commit,
        'source': blob, **identity, 'root': str(Path(root).resolve()),
        'context': str(Path(context).resolve()), 'phase': phase,
        'compile_coverage': coverage, 'clean_at_publication': True,
        'qualification_passed': False, 'campaign_success': False})


def validate_publication(path):
    publication = campaign.read(path)
    if (publication.get('schema') != PUBLICATION_SCHEMA
            or publication.get('repository') != maintained.REPOSITORY_SLUG
            or publication.get('clean_at_publication') is not True
            or publication.get('qualification_passed') is not False
            or publication.get('campaign_success') is not False
            or publication.get('source', {}).get('path') != SUPPORTED_SOURCE):
        raise ValueError('source publication contract changed')
    repository = _repository(publication['repository_path'])
    _git(repository, 'check-ref-format', '--branch', publication['branch'])
    identity = _selection_identity(publication['root'], publication['context'], publication['phase'],
                                   campaign.verify(publication['source_selection']))
    if any(publication.get(key) != value for key, value in identity.items()):
        raise ValueError('publication lost its selected source/runtime identity')
    data, blob = _committed_blob(repository, publication['commit'], SUPPORTED_SOURCE)
    if blob != publication['source'] or hashlib.sha256(data).hexdigest() != identity['selected_source']['sha256']:
        raise ValueError('publication source differs from the committed selected bytes')
    if _coverage(repository, publication['commit']) != publication['compile_coverage']:
        raise ValueError('publication compile coverage changed')
    return publication


def validate_gh_run(payload, *, publication):
    if not isinstance(payload, Mapping):
        raise ValueError('gh run payload must be an object')
    run_id = payload.get('databaseId')
    if (type(run_id) is not int or run_id <= 0
            or type(payload.get('workflowDatabaseId')) is not int
            or payload['workflowDatabaseId'] != maintained.WORKFLOW_DATABASE_ID
            or type(payload.get('attempt')) is not int or payload['attempt'] != 1
            or payload.get('name') != maintained.WORKFLOW_NAME
            or payload.get('workflowName') != maintained.WORKFLOW_NAME
            or payload.get('event') != 'workflow_dispatch'
            or payload.get('headBranch') != publication['branch']
            or payload.get('headSha') != publication['commit']
            or payload.get('status') != 'completed' or payload.get('conclusion') != 'success'
            or payload.get('url') != f'{maintained.RUN_URL_PREFIX}{run_id}'):
        raise ValueError('GitHub run is not the exact repository/workflow/branch/source commit')
    jobs = payload.get('jobs')
    if not isinstance(jobs, list):
        raise ValueError('GitHub run has no job evidence')
    normalized = {}
    deploy = False
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ValueError('GitHub job is malformed')
        name = job.get('name')
        if name == 'deploy':
            if deploy or job.get('status') != 'completed' or job.get('conclusion') != 'skipped':
                raise ValueError('only one completed skipped deploy job may be omitted')
            deploy = True
            continue
        if name not in maintained.JOB_NAMES:
            raise ValueError('GitHub run contains an unexpected job')
        key = maintained.JOB_NAMES[name]
        job_id = job.get('databaseId')
        if (key in normalized or type(job_id) is not int or job_id <= 0
                or job.get('status') != 'completed' or job.get('conclusion') != 'success'
                or job.get('url') != f'{maintained.RUN_URL_PREFIX}{run_id}/job/{job_id}'):
            raise ValueError('required GitHub job is duplicated, unsuccessful, or from another run')
        normalized[key] = {'name': name, 'database_id': job_id, 'status': 'completed',
                           'conclusion': 'success', 'url': job['url']}
    if (set(normalized) != set(maintained.REQUIRED_JOB_IDS)
            or len({job['database_id'] for job in normalized.values()}) != len(maintained.REQUIRED_JOB_IDS)):
        raise ValueError('CI requires all five distinct successful jobs')
    return {'repository': maintained.REPOSITORY_SLUG, 'run_id': run_id,
            'workflow_database_id': maintained.WORKFLOW_DATABASE_ID,
            'workflow_file': maintained.WORKFLOW_FILE, 'workflow_name': maintained.WORKFLOW_NAME,
            'attempt': 1, 'event': 'workflow_dispatch', 'branch': publication['branch'],
            'head_sha': publication['commit'], 'url': payload['url'],
            'jobs': {key: normalized[key] for key in maintained.REQUIRED_JOB_IDS}}


def _json_payload(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('raw GitHub payload repeats a JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def record_ci(output, *, publication_path, raw_gh_path, fetched_at_utc):
    publication = validate_publication(publication_path)
    maintained.parse_utc(fetched_at_utc, 'CI fetch time')
    raw = Path(raw_gh_path).read_bytes()
    normalized = validate_gh_run(_json_payload(raw), publication=publication)
    raw_path = Path(output).parent / ('gh-run-' + hashlib.sha256(raw).hexdigest() + '.json')
    campaign.once(raw_path, raw)
    return campaign.seal(output, {
        'schema': CI_SCHEMA, 'publication': campaign.record(publication_path),
        'raw_gh_payload': campaign.record(raw_path), 'fetched_at_utc': fetched_at_utc,
        'source': publication['source'], 'runtime_body_sha256': publication['runtime_body_sha256'],
        'payload_sha256': publication['payload_sha256'], **normalized,
        'source_specific_ci_passed': True, 'qualification_passed': False, 'campaign_success': False})


def validate_ci(path):
    evidence = campaign.read(path)
    if (evidence.get('schema') != CI_SCHEMA or evidence.get('source_specific_ci_passed') is not True
            or evidence.get('qualification_passed') is not False or evidence.get('campaign_success') is not False):
        raise ValueError('CI evidence improperly claims qualification or changes contract')
    publication = validate_publication(campaign.verify(evidence['publication']))
    raw = campaign.verify(evidence['raw_gh_payload']).read_bytes()
    normalized = validate_gh_run(_json_payload(raw), publication=publication)
    if (any(evidence.get(key) != value for key, value in normalized.items())
            or any(evidence.get(key) != publication[key]
                   for key in ('source', 'runtime_body_sha256', 'payload_sha256'))):
        raise ValueError('CI evidence differs from the raw run or selected source publication')
    maintained.parse_utc(evidence['fetched_at_utc'], 'CI fetch time')
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    for name in ('output', 'repository', 'source', 'source-selection', 'root', 'context'):
        freeze.add_argument('--' + name, type=Path, required=True)
    freeze.add_argument('--phase', required=True)
    record = commands.add_parser('record')
    for name in ('output', 'publication', 'raw-gh'):
        record.add_argument('--' + name, type=Path, required=True)
    record.add_argument('--fetched-at-utc', required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        result = freeze_publication(args.output, repository=args.repository, source=args.source,
                                    source_selection=args.source_selection, root=args.root,
                                    context=args.context, phase=args.phase)
    elif args.command == 'record':
        result = record_ci(args.output, publication_path=args.publication, raw_gh_path=args.raw_gh,
                           fetched_at_utc=args.fetched_at_utc)
    else:
        result = validate_ci(args.evidence)
    print(json.dumps({'schema': result['schema'], 'source_specific_ci_passed': result.get('source_specific_ci_passed', False),
                      'qualification_passed': False, 'campaign_success': False}))


if __name__ == '__main__':
    main()
