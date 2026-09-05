#!/usr/bin/env python3
"""Publish and locally verify the exact source that passed development.

``stage`` writes and stages only the supported candidate file, after validating
the completed development gate. Commit/push/CI dispatch remain separate steps.
``publication`` freezes that clean committed file through ci_v2. ``prepare``
requires its exact green CI, and ``run`` executes fresh, sequential local checks.
Only ``validate`` returning eligible_for_protected=True opens the next bridge.
Completed command receipts can resume; an interrupted or failed claimed command
is preserved and cannot silently be retried. No protected or live games run here.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sys
import tarfile

if __name__ == '__main__':
    for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_ci_v2 as ci
from tools import compact_value_bfm_development_v2 as development
from tools import compact_value_bfm_preflight as maintained

PREFIX = 'papersoccer_codingame_compact_value_bfm_release_'
TARGETS = tuple(PREFIX + name for name in ('submission', 'submission_test', 'feature_probe', 'inference_probe'))
TESTS = (PREFIX + 'submission_test', PREFIX + 'feature_parity')
REFEREE = 'papersoccer_codingame_referee'
RANK4 = 'papersoccer_codingame_rank_4_submission'
PANELS = ('gcc-release', 'clang-release', 'clang-sanitized')
POLICY = {'schema': campaign.ID + '.release-policy.v2', 'workers': 1,
          'fresh_builds': True, 'panels': list(PANELS), 'parity_states': 4096,
          'actual_model_full_delta_states': 4096,
          'internal_clocks_ms': [800, 155], 'external_deadlines_ms': [1000, 200],
          'empty_process_games': 2, 'colors': [0, 1], 'failures': 0,
          'source_reserve': 2000, 'source_limit_exclusive': 95000,
          'retry_interrupted_or_failed_command': False}
ENVIRONMENT = {**campaign.THREADS, 'PYTHONDONTWRITEBYTECODE': '1',
               'PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY': '1',
               'ASAN_OPTIONS': 'abort_on_error=1', 'UBSAN_OPTIONS': 'halt_on_error=1'}


def directory(context, phase):
    if not isinstance(phase, str) or re.fullmatch(r'attempt-[0-9]{3}-full', phase) is None:
        raise ValueError('release requires an exact full phase')
    return Path(context).resolve() / phase / 'release'


def prerequisites(root, context, phase):
    root, context = Path(root).resolve(), Path(context).resolve()
    contract = campaign.read(context / 'campaign.json')
    if campaign.verify(contract['parent_campaign']).parent != root:
        raise ValueError('release campaign parent changed')
    result = development.completed_development(context, phase)
    wins = result.get('candidate_wins_by_color')
    if (result.get('passed') is not True or result.get('games') != 1000 or result.get('pairs') != 500
            or type(result.get('candidate_wins')) is not int or result['candidate_wins'] < 550
            or not isinstance(wins, list) or len(wins) != 2
            or any(type(value) is not int or value < 265 for value in wins)
            or sum(wins) != result['candidate_wins'] or result.get('failures') != 0
            or type(result.get('paired_lower_95')) not in (float, int)
            or not math.isfinite(result['paired_lower_95']) or result['paired_lower_95'] <= .5
            or result.get('protected_gates_passed') is not False or result.get('campaign_success') is not False
            or result.get('policy') != development.POLICY
            or result.get('status') != 'development-passed-awaiting-exact-source-ci-and-protected-gates'):
        raise ValueError('release requires the completed 550/265/paired-CI/zero-failure development gate')
    selected = result['selected']
    identity = ci._selection_identity(root, context, phase, campaign.verify(selected['source_selection']))
    if (selected['source'] != identity['selected_source'] or selected['runtime'] != identity['runtime']
            or any(selected[key] != identity[key] for key in ('runtime_body_sha256', 'payload_sha256'))
            or selected.get('candidate_search_profile') not in development.gate.SEARCH_PROFILES):
        raise ValueError('development did not play the exact search-selected source/runtime')
    raw = campaign.verify(selected['source']).read_bytes(); raw.decode('ascii')
    if (95000 - len(raw) < 2000
            or b'constintbudget_ms=first_decision?800:155;' not in re.sub(rb'\s+', b'', raw)):
        raise ValueError('release source reserve or actual internal clocks changed')
    return {'root': str(root), 'context': str(context), 'phase': phase,
            'development': campaign.record(context / phase / 'development/assessment.json'),
            'selected': selected, 'source_selection': identity['source_selection']}


def _same(document, expected, message):
    if {key: value for key, value in document.items() if key != 'body_sha256'} != expected:
        raise ValueError(message)
    return document


def stage(root, context, phase, *, repository):
    """Write and git-add only the selected candidate; preserve any prior bytes."""
    with campaign.lease(Path(root).resolve()):
        ready = prerequisites(root, context, phase)
        repository = ci._repository(repository)
        target = repository / ci.RELEASE_SOURCE
        output = directory(context, phase)
        claim_path = output / 'publication-stage-claim.json'
        identity = {**ready, 'repository': str(repository), 'target': str(target)}
        if claim_path.exists():
            claim = campaign.read(claim_path)
            if any(claim.get(key) != value for key, value in identity.items()):
                raise ValueError('candidate publication stage claim changed')
            if claim['previous'] is not None:
                campaign.verify(claim['previous'])
        else:
            branch, commit = ci._clean_head(repository)
            if target.is_symlink():
                raise ValueError('candidate publication target is a symlink')
            previous = campaign.copy_checked(target, output / 'previous-candidate.cpp') if target.exists() else None
            claim = campaign.seal(claim_path, {'schema': campaign.ID + '.release-stage-claim.v2',
                **identity, 'previous': previous, 'base_branch': branch, 'base_commit': commit})
        raw = campaign.verify(ready['selected']['source']).read_bytes()
        if target.exists() and target.read_bytes() != raw:
            if claim['previous'] is None or target.read_bytes() != campaign.verify(claim['previous']).read_bytes():
                raise ValueError('candidate publication target contains an unrelated change')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.cpp.release-partial')
        if target.is_symlink() or temporary.exists():
            raise ValueError('candidate publication target or temporary is unsafe')
        if not target.exists() or target.read_bytes() != raw:
            with temporary.open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            temporary.replace(target)
        ci._git(repository, 'add', '--', ci.RELEASE_SOURCE)
        staged = ci._git(repository, 'show', ':' + ci.RELEASE_SOURCE)
        if staged != raw or target.read_bytes() != raw:
            raise ValueError('Git index did not preserve exact selected bytes')
        return campaign.seal(output / 'publication-stage.json', {
            'schema': campaign.ID + '.release-staged.v2', 'claim': campaign.record(claim_path),
            'source': campaign.record(target), 'selected': ready['selected'],
            'status': 'exact-source-staged-awaiting-commit-push-and-ci', 'campaign_success': False})


def publication(root, context, phase, *, repository):
    """Read-only with respect to Git: require already committed exact bytes."""
    with campaign.lease(Path(root).resolve()):
        ready = prerequisites(root, context, phase)
        path = directory(context, phase) / 'publication.json'
        if path.exists():
            result = ci.validate_publication(path)
            if result['selected_source'] != ready['selected']['source']:
                raise ValueError('publication belongs to another developed source')
            return result
        return ci.freeze_publication(path, repository=repository,
            source=Path(repository) / ci.RELEASE_SOURCE, source_selection=ready['source_selection']['path'],
            root=root, context=context, phase=phase)


def _ci_ready(root, context, phase, ci_path):
    ready = prerequisites(root, context, phase)
    evidence = ci.validate_ci(ci_path)
    expected_publication = directory(context, phase) / 'publication.json'
    if campaign.verify(evidence['publication']) != expected_publication:
        raise ValueError('release CI belongs to a different publication')
    published = ci.validate_publication(expected_publication)
    if (published['source']['path'] != ci.RELEASE_SOURCE
            or published['selected_source'] != ready['selected']['source']
            or published['source_selection'] != ready['source_selection']
            or any(published[key] != ready['selected'][key]
                   for key in ('runtime', 'runtime_body_sha256', 'payload_sha256'))):
        raise ValueError('release CI differs from the completed development source/runtime')
    return {**ready, 'publication': campaign.record(expected_publication), 'ci': campaign.record(ci_path)}, published


def _snapshot(output, published):
    archive = ci._git(published['repository_path'], 'archive', '--format=tar', published['commit'])
    archive_path = output / 'source.tar'
    campaign.once(archive_path, archive)
    repository = output / 'repository'
    files = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts or member.issym() or member.islnk():
                raise ValueError('committed source archive contains an unsafe path')
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError('committed source archive contains a nonregular file')
            target = repository / str(relative)
            campaign.once(target, tar.extractfile(member).read())
            target.chmod(member.mode & 0o777)
            files.append(campaign.record(target))
    path = output / 'snapshot.json'
    return campaign.seal(path, {'schema': campaign.ID + '.release-snapshot.v2',
        'commit': published['commit'], 'archive': campaign.record(archive_path),
        'repository': str(repository), 'files': files})


def _tool(path, family=None):
    path = Path(path).absolute()
    detail = maintained._executable(path, family) if family else maintained._versioned_tool(path, 'release tool')
    return {'file': campaign.record(path), 'command_path': str(path),
            'version_sha256': detail['version_sha256'], 'version_first_line': detail['version_first_line'],
            'family': family}


def identity_source(candidate):
    include = json.dumps(str(candidate))
    return ('#define COMPACT_VALUE_BFM_NO_MAIN\n#include ' + include + '\n#include <iostream>\n'
        'int main(){namespace cv=compact_value_bfm;std::cout<<cv::model::kRuntimeBodySha256<<"\\n"'
        '<<cv::model::kPayloadSha256<<"\\n"<<cv::deployment_model().payload_sha256()<<"\\n"'
        '<<cv::model::kInputs<<","<<cv::model::kHiddenOne<<","<<cv::model::kHiddenTwo<<","'
        '<<cv::model::kOutputs<<"\\n";}\n').encode('ascii')


def harness_sources(repository):
    candidate = repository / ci.RELEASE_SOURCE
    include = ('#include ' + json.dumps(str(candidate))).encode('ascii')
    sources = {'identity': identity_source(candidate)}
    for name, original_name in (('timing', 'timing_probe.cpp'), ('delta', 'inference_probe.cpp')):
        original = (repository / maintained.BOT_RELATIVE / original_name).read_bytes()
        if original.count(b'#include "submission.cpp"') != 1:
            raise ValueError('release harness has no unique source include')
        sources[name] = original.replace(b'#include "submission.cpp"', include)
    marker = b'const float value = cv::deployment_model().evaluate(features);'
    if sources['delta'].count(marker) != 1:
        raise ValueError('inference harness has no unique full evaluation point')
    sources['delta'] = sources['delta'].replace(marker, marker + b'''
    const auto &model = cv::deployment_model();
    static auto delta_previous = model.prepare(features);
    if (std::bit_cast<std::uint32_t>(value) !=
        std::bit_cast<std::uint32_t>(model.evaluate_delta(delta_previous, features))) return 2;
    delta_previous = model.prepare(features);
''')
    return sources


def commands(output, snapshot, toolchain, selected):
    repository = Path(snapshot['repository'])
    candidate = repository / ci.RELEASE_SOURCE
    executable = {name: item['command_path'] for name, item in toolchain.items()}
    steps = []
    def add(name, argv, markers=(), **extra):
        steps.append({'name': name, 'argv': list(map(str, argv)), 'markers': list(markers), **extra})
    for name in PANELS:
        sanitized = name == 'clang-sanitized'
        compiler = executable['gcc' if name == 'gcc-release' else 'clang']
        build = output / 'build' / name
        add(name + '-configure', [executable['cmake'], '-G', 'Unix Makefiles', '-S', repository, '-B', build,
            '-DCMAKE_BUILD_TYPE:STRING=' + ('Debug' if sanitized else 'Release'),
            '-DCMAKE_CXX_COMPILER:FILEPATH=' + compiler,
            '-DCMAKE_CXX_FLAGS:STRING=', '-DCMAKE_EXE_LINKER_FLAGS:STRING=',
            '-DPython3_EXECUTABLE:FILEPATH=' + executable['python'],
            '-DPAPERSOCCER_ENABLE_SANITIZERS:BOOL=' + ('ON' if sanitized else 'OFF'),
            '-DPAPERSOCCER_COMPACT_VALUE_BFM_RELEASE_SOURCE:FILEPATH=' + str(candidate),
            '-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON'])
        targets = (*TARGETS, REFEREE, RANK4) if name == 'clang-release' else TARGETS
        add(name + '-compile', [executable['cmake'], '--build', build, '--parallel', '1', '--target', *targets],
            panel=name, outputs=[str(build / target) for target in targets])
        add(name + '-ctest', [executable['ctest'], '--test-dir', build, '--output-on-failure',
            '--verbose', '--timeout', '300', '-j', '1', '-R', '^(' + '|'.join(TESTS) + ')$'],
            [*TESTS, '100% tests passed, 0 tests failed', 'compact_value_bfm submission tests passed',
             'compact feature parity passed states=4096'])
        add(name + '-inference', [executable['python'], repository / 'tools/compact_value_bfm_release_v2.py',
            '_inference', '--repository', repository, '--runtime', selected['runtime']['path'],
            '--probe', build / (PREFIX + 'inference_probe')])
    clang = output / 'build/clang-release'
    add('identity-compile', [executable['clang'], '-std=c++20', '-O3', output / 'identity.cpp',
        '-o', output / 'identity.bin'], outputs=[str(output / 'identity.bin')])
    add('identity', [output / 'identity.bin'])
    add('delta-compile', [executable['clang'], '-std=c++20', '-O3', output / 'delta.cpp',
        '-o', output / 'delta.bin'], outputs=[str(output / 'delta.bin')])
    add('delta-inference', [executable['python'], repository / 'tools/compact_value_bfm_release_v2.py',
        '_inference', '--repository', repository, '--runtime', selected['runtime']['path'],
        '--probe', output / 'delta.bin'])
    add('protocol', [executable['node'], repository / 'submissions/codingame/tools/protocol_smoke_test.mjs',
        clang / (PREFIX + 'submission')], ['Player 0 and Player 1 protocol smoke tests passed.'])
    add('timing-compile', [executable['clang'], '-std=c++20', '-O3', output / 'timing.cpp',
        '-o', output / 'timing.bin'], outputs=[str(output / 'timing.bin')])
    for color in (0, 1):
        add('timing-' + str(color), [output / 'timing.bin', str(color)])
        players = [clang / (PREFIX + 'submission'), clang / RANK4]
        if color == 1:
            players.reverse()
        add('empty-' + str(color), [clang / REFEREE, '--player-one', players[0], '--player-two', players[1],
            '--player-one-id', 'candidate' if color == 0 else 'rank4',
            '--player-two-id', 'rank4' if color == 0 else 'candidate',
            '--first-timeout-ms', '1000', '--later-timeout-ms', '200'])
    return steps


def prepare(root, context, phase, *, ci_path, python_path, gcc_path, clang_path,
            cmake_path=None, ctest_path=None, node_path=None):
    with campaign.lease(Path(root).resolve()):
        output = directory(context, phase)
        if (output / 'preflight-plan.json').exists():
            return validate_plan(root, context, phase)
        ready, published = _ci_ready(root, context, phase, ci_path)
        if (output / 'build').exists() or (output / 'steps').exists():
            raise ValueError('release preparation requires fresh build and execution directories')
        snapshot = _snapshot(output, published)
        repository = Path(snapshot['repository'])
        if campaign.sha(repository / 'tools/compact_value_bfm_release_v2.py') != campaign.sha(__file__):
            raise ValueError('release driver must equal the driver in the exact green source commit')
        toolchain = {'python': _tool(python_path), 'gcc': _tool(gcc_path, 'GNU'),
            'clang': _tool(clang_path, 'Clang'),
            **{name: _tool(value or shutil.which(name) or '') for name, value in
               (('cmake', cmake_path), ('ctest', ctest_path), ('node', node_path))}}
        for name, raw in harness_sources(repository).items():
            campaign.once(output / (name + '.cpp'), raw)
        return campaign.seal(output / 'preflight-plan.json', {
            'schema': campaign.ID + '.release-preflight-plan.v2', **ready, 'policy': POLICY,
            'snapshot': campaign.record(output / 'snapshot.json'), 'toolchain': toolchain,
            'producer': campaign.record(repository / 'tools/compact_value_bfm_release_v2.py'),
            'harnesses': {name: campaign.record(output / (name + '.cpp')) for name in ('identity', 'timing', 'delta')},
            'commands': commands(output, snapshot, toolchain, ready['selected']),
            'environment': ENVIRONMENT, 'fresh_builds_at_prepare': True,
            'eligible_for_protected': False, 'campaign_success': False})


def validate_plan(root, context, phase):
    output = directory(context, phase)
    plan = campaign.read(output / 'preflight-plan.json')
    ready, published = _ci_ready(root, context, phase, campaign.verify(plan['ci']))
    snapshot = campaign.read(campaign.verify(plan['snapshot']))
    expected_archive = ci._git(published['repository_path'], 'archive', '--format=tar', published['commit'])
    if (campaign.verify(snapshot['archive']).read_bytes() != expected_archive
            or snapshot['commit'] != published['commit']
            or Path(snapshot['repository']) != output / 'repository'):
        raise ValueError('release snapshot changed its exact committed source')
    with tarfile.open(fileobj=io.BytesIO(expected_archive)) as tar:
        expected = []
        for member in tar.getmembers():
            if member.isfile():
                path = Path(snapshot['repository']) / member.name
                if path.read_bytes() != tar.extractfile(member).read():
                    raise ValueError('release snapshot file differs from its committed blob')
                expected.append(campaign.record(path))
        if expected != snapshot['files']:
            raise ValueError('release snapshot file closure changed')
    observed = {str(path) for path in (output / 'repository').rglob('*') if path.is_file()}
    if observed != {item['path'] for item in snapshot['files']}:
        raise ValueError('release snapshot contains extra uncommitted source files')
    repository = Path(snapshot['repository'])
    if campaign.sha(repository / ci.RELEASE_SOURCE) != ready['selected']['source']['sha256']:
        raise ValueError('local candidate differs from published developed source')
    if campaign.sha(repository / 'submissions/codingame/bots/rank_4/submission.cpp') != development.gate.RANK4_SHA256:
        raise ValueError('release Rank4 reference changed')
    toolchain = plan['toolchain']
    if set(toolchain) != {'python', 'gcc', 'clang', 'cmake', 'ctest', 'node'}:
        raise ValueError('release toolchain roster changed')
    for name, item in toolchain.items():
        if _tool(item['command_path'], {'gcc': 'GNU', 'clang': 'Clang'}.get(name)) != item:
            raise ValueError('release compiler/interpreter/tool version changed')
    expected_harnesses = harness_sources(repository)
    for name, raw in expected_harnesses.items():
        if campaign.verify(plan['harnesses'][name]) != output / (name + '.cpp') or campaign.verify(plan['harnesses'][name]).read_bytes() != raw:
            raise ValueError('release harness changed its exact selected include')
    return _same(plan, {'schema': campaign.ID + '.release-preflight-plan.v2', **ready, 'policy': POLICY,
        'snapshot': campaign.record(output / 'snapshot.json'), 'toolchain': toolchain,
        'producer': campaign.record(repository / 'tools/compact_value_bfm_release_v2.py'),
        'harnesses': {name: campaign.record(output / (name + '.cpp')) for name in ('identity', 'timing', 'delta')},
        'commands': commands(output, snapshot, toolchain, ready['selected']), 'environment': ENVIRONMENT,
        'fresh_builds_at_prepare': True, 'eligible_for_protected': False, 'campaign_success': False},
        'release plan differs from developed source, exact CI, or execution policy')


def build_evidence(plan, step):
    """Reconcile actual candidate compile/link commands and source includes."""
    panel = step['panel']; sanitized = panel == 'clang-sanitized'
    output = directory(plan['context'], plan['phase'])
    repository = output / 'repository'; build = output / 'build' / panel
    compiler = plan['toolchain']['gcc' if panel == 'gcc-release' else 'clang']['command_path']
    cache = build / 'CMakeCache.txt'
    maintained.validate_cache_text(cache.read_text(), python_path=Path(plan['toolchain']['python']['command_path']), sanitized=sanitized)
    if ('CMAKE_CXX_COMPILER:FILEPATH=' + compiler) not in cache.read_text().splitlines():
        raise ValueError('candidate CMake cache changed compiler')
    candidate = repository / ci.RELEASE_SOURCE
    if ('PAPERSOCCER_COMPACT_VALUE_BFM_RELEASE_SOURCE:FILEPATH=' + str(candidate)) not in cache.read_text().splitlines():
        raise ValueError('candidate CMake cache changed exact source')
    compile_path = build / 'compile_commands.json'
    entries = json.loads(compile_path.read_bytes())
    artifacts = {'cache': campaign.record(cache), 'compile_commands': campaign.record(compile_path)}
    for target in TARGETS:
        source = candidate if target == PREFIX + 'submission' else build / 'compact-release-checks' / (target.removeprefix(PREFIX) + '.cpp')
        if source != candidate:
            original = repository / maintained.BOT_RELATIVE / source.name
            expected = original.read_bytes().replace(b'#include "submission.cpp"', ('#include ' + json.dumps(str(candidate))).encode('ascii'))
            if source.read_bytes() != expected:
                raise ValueError('CMake candidate harness does not include exact selected bytes')
        matches = [row for row in entries if Path(row['file']) == source]
        if len(matches) != 1:
            raise ValueError('candidate compile evidence missing or duplicated')
        command = matches[0].get('arguments') or shlex.split(matches[0]['command'])
        if Path(command[0]).resolve() != Path(compiler).resolve():
            raise ValueError('candidate compiled by another compiler')
        link = build / 'CMakeFiles' / (target + '.dir') / 'link.txt'
        link_text = link.read_text()
        if (('-fsanitize=address,undefined' in command) != sanitized
                or ('-fsanitize=address,undefined' in link_text) != sanitized
                or (sanitized and '-fno-sanitize-recover=all' not in command)):
            raise ValueError('candidate sanitizer instrumentation differs from panel')
        artifacts[target + '-source'] = campaign.record(source)
        artifacts[target + '-link'] = campaign.record(link)
    return artifacts


def validate_empty(raw, color):
    game = ci._json_payload(raw)
    ids = ['candidate', 'rank4'] if color == 0 else ['rank4', 'candidate']
    if (game.get('schema') != 'papersoccer.codingame-match.v1'
            or game.get('timeouts') != {'firstMillis': 1000, 'laterMillis': 200}
            or game.get('rules') != {'width': 8, 'height': 10, 'goalRule': 'OwnGoalsAllowed', 'blockedRule': 'MoverLoses'}
            or game.get('outcome', {}).get('forfeit') is not None):
        raise ValueError('empty process game failed or changed actual referee clocks/rules')
    for player, key in enumerate(('playerOne', 'playerTwo')):
        expected = {'id': ids[player], 'player': player,
                    'executable': PREFIX + 'submission' if ids[player] == 'candidate' else RANK4}
        if game['participants'][key] != expected:
            raise ValueError('empty process game candidate color or binary changed')
    state = campaign.features.ReplayState(); counts = [0, 0]; previous = '-'
    for index, row in enumerate(game['actions']):
        player = state.to_move
        deadline = 1000 if counts[player] == 0 else 200
        if (row['turn'] != index or row['player'] != player or row['botId'] != ids[player]
                or row['opponentAction'] != previous or row['accepted'] is not True
                or row['failureClassification'] is not None or row['deadlineMillis'] != deadline
                or type(row['durationMicros']) is not int or not 0 <= row['durationMicros'] < deadline * 1000):
            raise ValueError('empty process game has an illegal, failed, or late turn')
        campaign.features.apply_complete_turn(state, player, row['action'])
        previous = row['action']; counts[player] += 1
    if (state.winner not in (0, 1) or min(counts) == 0
            or game['outcome']['winnerId'] != ids[state.winner]
            or game['outcome']['loserId'] != ids[1 - state.winner]):
        raise ValueError('empty process game lacks a legal completed outcome')
    return {'color': color, 'turns': len(game['actions']), 'candidate_won': ids[state.winner] == 'candidate',
            'forfeits': 0, 'first_deadline_ms': 1000, 'later_deadline_ms': 200}


def _validate_output(plan, step, raw):
    name = step['name']
    if name.endswith('-inference'):
        receipt = ci._json_payload(raw); maintained.validate_parity_receipt(receipt)
        if (receipt['states'] != 4096 or receipt['feature_states'] != 4096
                or receipt['runtime_sha256'] != plan['selected']['runtime']['sha256']
                or receipt['probe_sha256'] != campaign.sha(step['argv'][-1])):
            raise ValueError('inference parity lost exact runtime/probe binding')
    elif name == 'identity':
        selected = plan['selected']
        if raw.decode('ascii').splitlines() != [selected['runtime_body_sha256'], selected['payload_sha256'],
                                               selected['payload_sha256'], '6301,12,8,1']:
            raise ValueError('compiled candidate runtime/payload/architecture identity differs')
    elif name in ('timing-0', 'timing-1'):
        color = int(name[-1]); matches = list(maintained.TIMING_LINE.finditer(raw.decode('ascii')))
        expected = [(f'player{color}_first', 800, 1000), (f'player{color}_later', 155, 200),
                    (f'player{color}_later_initial', 155, 200)]
        if len(matches) != 3:
            raise ValueError('one-worker timing coverage is incomplete')
        for match, (label, budget, deadline) in zip(matches, expected, strict=True):
            if match['label'] != label or int(match['budget']) != budget or int(match['elapsed']) >= deadline * 1000:
                raise ValueError('one-worker timing exceeds actual external deadline or changes internal budget')
    elif name in ('empty-0', 'empty-1'):
        validate_empty(raw, int(name[-1]))


def _step_claim(plan, plan_path, step):
    return {'schema': campaign.ID + '.release-command-claim.v2', 'plan': campaign.record(plan_path),
            'command': step, 'cwd': str(directory(plan['context'], plan['phase']) / 'repository'),
            'environment': ENVIRONMENT, 'nice': 0}


def validate_step(plan, plan_path, step):
    output = directory(plan['context'], plan['phase']) / 'steps' / step['name']
    _same(campaign.read(output / 'claim.json'), _step_claim(plan, plan_path, step), 'release command claim changed')
    result = campaign.read(output / 'execution.json')
    receipt = result['command_receipt']
    maintained.validate_command_receipt(receipt, label=step['name'], argv=step['argv'], required_markers=step['markers'])
    if receipt['cwd'] != str(directory(plan['context'], plan['phase']) / 'repository'):
        raise ValueError('release command cwd changed')
    for stream in ('stdout', 'stderr'):
        binding = result[stream]
        if campaign.verify(binding) != output / (stream + '.log'):
            raise ValueError('release command output path changed')
        if {key: binding[key] for key in ('bytes', 'sha256')} != receipt[stream]:
            raise ValueError('release command raw output digest changed')
    raw = campaign.verify(result['stdout']).read_bytes()
    combined = raw + b'\n' + campaign.verify(result['stderr']).read_bytes()
    if any(marker.encode() not in combined for marker in step['markers']):
        raise ValueError('release command raw success markers changed')
    _validate_output(plan, step, raw)
    artifacts = {path: campaign.record(path) for path in step.get('outputs', [])}
    if 'panel' in step:
        artifacts.update(build_evidence(plan, step))
    return _same(result, {'schema': campaign.ID + '.release-command.v2',
        'claim': campaign.record(output / 'claim.json'), 'command_receipt': receipt,
        'stdout': campaign.record(output / 'stdout.log'), 'stderr': campaign.record(output / 'stderr.log'),
        'artifacts': artifacts}, 'release command artifacts or source/compiler bindings changed')


def _run_step(plan, plan_path, step):
    output = directory(plan['context'], plan['phase']) / 'steps' / step['name']
    if (output / 'execution.json').exists():
        return validate_step(plan, plan_path, step)
    if output.exists():
        raise ValueError('release command already claimed without completed evidence; do not rerun it')
    if step['name'].endswith('-configure'):
        build = directory(plan['context'], plan['phase']) / 'build' / step['name'].removesuffix('-configure')
        if build.exists():
            raise ValueError('release panel is not a fresh build')
    campaign.seal(output / 'claim.json', _step_claim(plan, plan_path, step))
    receipt, out, err = maintained.run_command(step['name'], step['argv'],
        cwd=directory(plan['context'], plan['phase']) / 'repository', required_markers=step['markers'])
    campaign.once(output / 'stdout.log', out); campaign.once(output / 'stderr.log', err)
    artifacts = {}
    if receipt['passed']:
        artifacts = {path: campaign.record(path) for path in step.get('outputs', [])}
        if 'panel' in step:
            artifacts.update(build_evidence(plan, step))
    campaign.seal(output / 'execution.json', {'schema': campaign.ID + '.release-command.v2',
        'claim': campaign.record(output / 'claim.json'), 'command_receipt': receipt,
        'stdout': campaign.record(output / 'stdout.log'), 'stderr': campaign.record(output / 'stderr.log'),
        'artifacts': artifacts})
    return validate_step(plan, plan_path, step)


def _freeze_body(root, context, phase):
    output = directory(context, phase); plan_path = output / 'preflight-plan.json'
    plan = validate_plan(root, context, phase)
    executions = []
    for step in plan['commands']:
        validate_step(plan, plan_path, step)
        executions.append(campaign.record(output / 'steps' / step['name'] / 'execution.json'))
    preflight = {'schema': campaign.ID + '.release-preflight.v2', 'plan': campaign.record(plan_path),
        'executions': executions, 'policy': POLICY, 'source_specific_local_checks_passed': True,
        'actual_clock_empty_process_checks_passed': True, 'remaining_requirements': [],
        'protected_gates_passed': False, 'campaign_success': False}
    _same(campaign.read(output / 'preflight.json'), preflight, 'release preflight does not reproduce command evidence')
    return {'schema': campaign.ID + '.release-freeze.v2',
        **{key: plan[key] for key in ('root', 'context', 'phase', 'selected', 'development', 'source_selection', 'publication', 'ci')},
        'preflight': campaign.record(output / 'preflight.json'), 'policy': POLICY,
        'candidate_commit': ci.validate_publication(campaign.verify(plan['publication']))['commit'],
        'status': 'frozen-awaiting-two-independent-protected-gates', 'eligible_for_protected': True,
        'protected_gates_passed': False, 'qualification_passed': False, 'campaign_success': False}


def run(root, context, phase):
    with campaign.lease(Path(root).resolve()):
        plan = validate_plan(root, context, phase)
        if campaign.sha(__file__) != plan['producer']['sha256']:
            raise ValueError('release execution must use the driver frozen in the exact green commit')
        if os.getpriority(os.PRIO_PROCESS, 0) != 0:
            raise ValueError('uncontended release timing requires nice zero')
        output = directory(context, phase); plan_path = output / 'preflight-plan.json'
        previous = {key: os.environ.get(key) for key in ENVIRONMENT}
        os.environ.update(ENVIRONMENT)
        try:
            for step in plan['commands']:
                _run_step(plan, plan_path, step)
        finally:
            for key, value in previous.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value
        campaign.seal(output / 'preflight.json', {'schema': campaign.ID + '.release-preflight.v2',
            'plan': campaign.record(plan_path), 'executions': [campaign.record(output / 'steps' / step['name'] / 'execution.json')
                for step in plan['commands']], 'policy': POLICY, 'source_specific_local_checks_passed': True,
            'actual_clock_empty_process_checks_passed': True, 'remaining_requirements': [],
            'protected_gates_passed': False, 'campaign_success': False})
        return campaign.seal(output / 'freeze.json', _freeze_body(root, context, phase))


def validate(root, context, phase):
    """Independent read/verification API for the protected bridge (no lease)."""
    path = directory(context, phase) / 'freeze.json'
    frozen = campaign.read(path)
    return _same(frozen, _freeze_body(root, context, phase), 'release freeze differs from completed source-bound checks')


def preflight_boundaries(root, context, phase):
    """Yield fingerprint-only isolation evidence for both verified release games."""
    frozen = validate(root, context, phase)
    output = directory(context, phase)
    preflight = campaign.read(campaign.verify(frozen['preflight']))
    for color in (0, 1):
        execution_path = output / 'steps' / f'empty-{color}' / 'execution.json'
        if campaign.record(execution_path) not in preflight['executions']:
            raise ValueError('release empty game is absent from the frozen preflight')
        execution = campaign.read(execution_path)
        raw = campaign.verify(execution['stdout']).read_bytes()
        validate_empty(raw, color)
        game = ci._json_payload(raw)
        state = campaign.features.ReplayState()
        yield campaign.fingerprints(state)
        for row in game['actions']:
            campaign.features.apply_complete_turn(state, state.to_move, row['action'])
            yield campaign.fingerprints(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('stage', 'publication', 'prepare', 'run', 'validate', '_inference'))
    for name in ('root', 'context', 'repository', 'ci', 'python', 'gcc', 'clang', 'cmake', 'ctest', 'node', 'runtime', 'probe'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--phase')
    args = parser.parse_args()
    if args.command == '_inference':
        if not all((args.repository, args.runtime, args.probe)):
            parser.error('_inference requires repository/runtime/probe')
        print(json.dumps(maintained.run_inference_parity(repository=args.repository,
            runtime_path=args.runtime, probe_path=args.probe, states=4096)), flush=True)
        return
    if not all((args.root, args.context, args.phase)):
        parser.error('release requires root/context/phase')
    if args.command in ('stage', 'publication'):
        if args.repository is None: parser.error('publication requires repository')
        result = globals()[args.command](args.root, args.context, args.phase, repository=args.repository)
    elif args.command == 'prepare':
        if not all((args.ci, args.python, args.gcc, args.clang)):
            parser.error('prepare requires ci/python/gcc/clang')
        result = prepare(args.root, args.context, args.phase, ci_path=args.ci, python_path=args.python,
            gcc_path=args.gcc, clang_path=args.clang, cmake_path=args.cmake, ctest_path=args.ctest, node_path=args.node)
    elif args.command == 'run':
        result = run(args.root, args.context, args.phase)
    else:
        with campaign.lease(args.root.resolve()): result = validate(args.root, args.context, args.phase)
    print(json.dumps({'schema': result['schema'], 'status': result.get('status'),
        'eligible_for_protected': result.get('eligible_for_protected', False), 'campaign_success': False}), flush=True)


if __name__ == '__main__':
    main()
