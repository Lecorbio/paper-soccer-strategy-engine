"""Validate a completed failed pilot and carry its isolation evidence forward.

Completed full-stage failures use the separate full outcome validator. Missing
or interrupted stages never count as completed unsuccessful trained attempts.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path

from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_pilot_selection_v2 as selection
from tools import compact_value_bfm_ranking_store as storage
from tools import compact_value_bfm_stream_v2 as stream
from submissions.codingame.bots.compact_value_bfm import rank4_gate_support


def bound(record, path):
    if campaign.verify(record).resolve() != Path(path).resolve():
        raise ValueError('prior-attempt receipt points outside its expected stage')
    return campaign.read(path)


def validate_training(context, phase, contract):
    """Reload real seed completions and reproduce all-layer training evidence."""
    trainer = selection.trainer
    directory = context / phase
    training = campaign.read(directory / 'training.json')
    if training.get('smoke') is not False or training.get('mandatory_training_verified') is not True:
        raise ValueError('a completed nonsmoke trained attempt is required')
    roster = [(row['weight'], row['seed']) for row in training['results']]
    if len(roster) != 9 or set(roster) != {(w, s) for w in (0, .1, .25) for s in trainer.FIXED_SEEDS}:
        raise ValueError('a failed pilot must have all nine completed training seeds')
    campaign.verify(training['producer'])
    audit = bound(training['input_audit'], directory / 'training-input-audit.json')
    labels = bound(audit['labels'], directory / 'labels.json')
    positions = bound(audit['position_closure'], directory / 'positions.json')
    plan = bound(labels['plan'], directory / 'labels-plan.json')
    if (audit['bundle'] != contract['bundle'] or audit.get('protected_tests_opened') is not False
            or labels['teacher'] != contract['inputs']['teacher_runtime']
            or labels['positions'] != audit['position_closure']
            or labels.get('all_groups_exhaustive') is not True
            or labels.get('all_native_labels_validated') is not True
            or labels['groups'] != len(positions['rows'])
            or labels['deep_groups'] != math.ceil(labels['groups'] * .25)
            or plan['context'] != campaign.record(context / 'campaign.json')
            or plan['teacher'] != labels['teacher'] or plan['positions'] != labels['positions']
            or (plan['shallow_nodes'], plan['deep_nodes'], plan['deep_fraction']) != (64000, 500000, .25)):
        raise ValueError('prior training lost its accepted-teacher or isolation binding')
    for item in (labels['merged'], labels['teacher'], plan['producer'], audit['ranking_store'], audit['bundle']):
        campaign.verify(item)
    store = campaign.read(campaign.verify(audit['ranking_store']))
    if (store['sources'] != [labels['merged']] or store['teacher']['artifact_sha256'] != labels['teacher']['sha256']
            or len(store['groups']) != labels['groups'] or store.get('all_successors_preserved') is not True
            or store.get('all_rich_rows_validated') is not True or store.get('protected_tests_opened') is not False):
        raise ValueError('training ranking store differs from its completed teacher labels')
    for item in store['arrays'].values():
        campaign.verify(item)
    for shard in audit['shards'].values():
        for item in shard.values():
            campaign.verify(item)
    architecture = trainer.ARCHITECTURES['capacity-12x8']
    initial_record = contract['inputs']['attempt_one_initial_checkpoint']
    initial = trainer.load_float_checkpoint(campaign.verify(initial_record), architecture)
    for row in training['results']:
        receipt = row['seed_receipt']; binding = receipt['binding']; ranking = binding['successor_ranking']
        if (binding['input_audit'] != audit or binding['seed'] != row['seed']
                or binding['architecture']['name'] != architecture.name
                or binding['architecture']['dimensions'] != [6301, 12, 8, 1]
                or binding['architecture']['biases'] is not False
                or binding['settings']['qat_profile'] != 'standard-v1'
                or ranking['initial_checkpoint']['sha256'] != initial_record['sha256']
                or ranking['artifact_sha256'] != audit['ranking_store']['sha256']
                or ranking['body_sha256'] != store['body_sha256']
                or ranking['source_bundle_body_sha256'] != store['source_bundle_body_sha256']
                or ranking['teacher']['artifact_sha256'] != labels['teacher']['sha256']
                or binding['datasets']['new']['source_manifest_sha256'] != audit['shards']['train']['manifest']['sha256']
                or binding['datasets']['new']['source_npz_sha256'] != audit['shards']['train']['npz']['sha256']
                or ranking['loss_weight'] != row['weight']):
            raise ValueError('completed seed is not bound to this pilot training')
        arm_directory = directory / 'training' / f'lambda-{row["weight"]:.2f}'
        reference = trainer._seed_reference_path(arm_directory, architecture, trainer.ARMS['search-target'], row['seed'])
        loaded = trainer._load_seed_receipt_from_reference(arm_directory, reference, binding)
        if loaded != receipt:
            raise ValueError('embedded seed evidence differs from its completed reference')
        for key, receipt_key in (('runtime', 'quantized_runtime'), ('float_checkpoint', 'float_checkpoint')):
            if campaign.verify(row[key]).resolve() != (arm_directory / receipt[receipt_key]['path']).resolve():
                raise ValueError('trained artifact differs from its seed reference')
        source = campaign.verify(row['source']).read_bytes(); source.decode('ascii')
        if len(source) >= 95000 or row['source_reserve'] != 95000 - len(source):
            raise ValueError('trained source budget evidence changed')
        updated = trainer.load_float_checkpoint(campaign.verify(row['float_checkpoint']), architecture)
        changes = trainer._parameter_update_evidence(initial, updated)
        if (changes != row['master_updates'] or set(changes) != {'w1', 'w2', 'w3'}
                or any(not c['changed'] or not math.isfinite(c['l2_delta']) or c['l2_delta'] <= 0 for c in changes.values())):
            raise ValueError('prior attempt has no reproduced all-layer master updates')
        arch, quantized, _, _ = trainer.load_runtime(campaign.verify(row['runtime']))
        codes = trainer._quantized_update_evidence(trainer.quantize_fixed(initial, arch, quantized.scales), quantized)
        if codes != row['quantized_changes_vs_initialization'] or not any(c['changed_codes'] for c in codes.values()):
            raise ValueError('prior attempt retained the initialization payload')
    return training


def load_early_groups(context, phase, contract, training):
    audit = bound(training['input_audit'], context / phase / 'training-input-audit.json')
    index = campaign.verify(audit['ranking_store'])
    if index.resolve() != (context / phase / 'ranking-store/index.json').resolve() or audit['bundle'] != contract['bundle']:
        raise ValueError('prior early validation belongs to another training corpus')
    bundle = selection.trainer.FrozenBundle.load(campaign.verify(contract['bundle']))
    rankings = storage.RankingStore(index, bundle).labels()
    early = tuple(group for group in rankings.validation
        if sum(len(turn['action']) for turn in group.evidence['source_binding']['prefix']) <= 12)
    if not early:
        raise ValueError('prior pilot has no bound early held-out evidence')
    return early


def evaluate_early(row, early):
    # The pilot CLI establishes these environment limits before importing NumPy.
    # Programmatic callers must meet the same contract; setting it late is unsafe.
    trainer = selection.trainer
    with trainer.native_thread_execution_scope():
        architecture, quantized, _, _ = trainer.load_runtime(campaign.verify(row['runtime']))
        parameters = trainer.load_float_checkpoint(campaign.verify(row['float_checkpoint']), architecture)
        return trainer.successor_ranking_metrics(parameters, architecture, early, quantized=quantized)


def validate_selection(document, training, context, phase, contract):
    policy = campaign.read(campaign.verify(document['policy']))
    if {k: v for k, v in policy.items() if k != 'body_sha256'} != selection.SELECTION_POLICY:
        raise ValueError('prior pilot selection policy changed')
    if document.get('pilot_admitted') is not False or document.get('campaign_success') is not False:
        raise ValueError('offline selection cannot close an admitted or successful attempt')
    if [arm['lambda'] for arm in document['arms']] != [0, .1, .25]:
        raise ValueError('prior pilot selection arm roster changed')
    # Recompute every verdict before the maintained seed selector consumes it.
    for row in training['results']:
        receipt = row['seed_receipt']
        gate = selection.trainer.offline_advancement_gate(receipt['float_validation'], receipt['quantized_validation'])
        if receipt['offline_gate'] != gate:
            raise ValueError('prior seed canonical retention verdict does not reproduce validation')
    early = load_early_groups(context, phase, contract, training)
    for arm in document['arms']:
        records = [r for r in training['results'] if r['weight'] == arm['lambda']]
        receipt = selection.selection._selected_seed([r['seed_receipt'] for r in records])
        row = next(r for r in records if r['seed'] == receipt['seed'])
        if (arm['seed'] != row['seed'] or any(arm[k] != row[k] for k in ('source', 'runtime', 'float_checkpoint', 'source_reserve'))
                or arm['canonical_retention_passed'] != receipt['offline_gate']['passed']
                or arm['overall'] != receipt['quantized_validation']['successor_ranking']):
            raise ValueError('prior selection does not reproduce its completed seed evidence')
        if arm['early'] != evaluate_early(row, early):
            raise ValueError('prior selection early metrics do not reproduce the held-out trained model')
        for stratum in ('overall', 'early'):
            metrics = arm[stratum]
            if (not isinstance(metrics['groups'], int) or not isinstance(metrics['comparable_groups'], int)
                    or not 0 <= metrics['comparable_groups'] <= metrics['groups']
                    or any(not math.isfinite(metrics[k]) or metrics[k] < 0 for k in
                        ('mean_teacher_regret', 'float_vs_quantized_action_flip_rate'))):
                raise ValueError('invalid prior pilot selection metrics')
    comparisons = [selection.compare_candidate(document['arms'][0], arm) for arm in document['arms'][1:]]
    eligible = [arm for arm, result in zip(document['arms'][1:], comparisons) if result['eligible_for_rank4_screen']]
    winner = min(eligible, key=lambda arm: (arm['overall']['mean_teacher_regret'], -arm['overall']['top1_agreement'],
        arm['overall']['float_vs_quantized_action_flip_rate'], arm['lambda'])) if eligible else None
    status = 'model-selected-before-rank4-screen' if winner else 'offline-rejected-before-rank4-screen'
    if document['comparisons'] != comparisons or document['selected'] != winner or document['status'] != status:
        raise ValueError('prior rejection does not reproduce frozen model selection')
    return winner


def validate_screen(directory, outcome, selected, *, admitted=False):
    execution = bound(outcome['screen'], directory / 'rank4-screen/execution.json')
    claim = bound(execution['claim'], directory / 'rank4-screen/execution-claim.json')
    bank = bound(claim['bank'], directory / 'rank4-screen/bank.json')
    seed_claim = bound(bank['claim'], directory / 'rank4-screen/seed-claim.json')
    expected_inputs = {'selection': campaign.record(directory / 'model-selection.json'),
        'positions': campaign.record(directory / 'positions.json'), 'games': campaign.record(directory / 'games.json'),
        'context': campaign.record(directory.parent / 'campaign.json')}
    if seed_claim['inputs'] != expected_inputs or bank['pairs'] != 100 or len(bank['rows']) != 100:
        raise ValueError('screen bank lost its postselection frozen inputs')
    if claim['candidate'] != selected['source'] or claim['runtime'] != selected['runtime'] or claim.get('retry_allowed') is not False:
        raise ValueError('screen claim changed the selected trained source')
    for key in ('binary', 'gate_source', 'compiler'):
        campaign.verify(claim[key])
    contract = campaign.read(directory.parent / 'campaign.json')
    if claim['compiler'] != contract['compiler'] or execution['returncode'] not in (0, 2):
        raise ValueError('screen compiler or process completion changed')
    bank_rows = rank4_gate_support.validate_bank(campaign.verify(bank['tsv']))['openings']
    if [(r['opening_id'], r['transcript']) for r in bank_rows] != [(r['opening_id'], r['transcript']) for r in bank['rows']]:
        raise ValueError('screen bank rows differ from the executed TSV')
    raw = rank4_gate_support.validate_result(campaign.verify(execution['raw']), expected_bank_sha256=bank['tsv']['sha256'],
        expected_candidate_sha256=selected['source']['sha256'], expected_candidate_search_profile='standard-v1',
        require_trajectories=True, trajectory_bank=campaign.verify(bank['tsv']))
    runtime = campaign.read(campaign.verify(selected['runtime']))
    if (raw['config']['mode'] != 'actual-clock' or raw['config']['pair_offset'] != 0
            or raw['config']['pair_count'] != 100 or raw['config']['minimum_candidate_wins'] != 105
            or raw['bindings']['candidate_runtime_body_sha256'] != runtime['body_sha256']
            or raw['bindings']['candidate_payload_sha256'] != runtime['quantization']['payload_sha256']
            or raw['result'] != execution['result']
            or [(g['opening_id'], g['candidate_player']) for g in raw['games']] !=
                [(r['opening_id'], c) for r in bank['rows'] for c in (0, 1)]):
        raise ValueError('prior screen did not reproduce its source, clocks, or schedule')
    result = raw['result']
    if (outcome['selected'] != selected or outcome['games'] != 200 or result['games'] != 200
            or outcome['wins'] != result['candidate_wins'] or outcome['failures'] != result['failures']
            or (result['candidate_wins'] >= 105 and result['failures'] == 0) is not admitted):
        raise ValueError('prior pilot is not a completed ' + ('passing' if admitted else 'failed') + ' screen')
    validate_played_exclusions(directory, outcome, bank, raw)
    return bank, raw


def validate_played_exclusions(directory, outcome, bank, raw):
    if (raw['config'].get('trajectory_schema') != 'papersoccer.compact-value-bfm-rank4-trajectories.v1'
            or outcome.get('played_trajectory_closure_preserved') is not True):
        raise ValueError('completed screen requires every source-bound played trajectory')
    values = defaultdict(set)
    for game in raw['games']:
        prefix = game['root_transcript'].split('/')
        if game['transcript'].split('/')[:len(prefix)] != prefix:
            raise ValueError('screen continuation changed its root transcript')
        fingerprints, _ = trajectory_fingerprints(game['transcript'], len(prefix))
        for domain, members in fingerprints.items():
            values[domain].update(members)
    records = outcome['development_exclusions']; initial_count = len(bank['exclusions'])
    if records[:initial_count] != bank['exclusions'] or len(records) != initial_count + len(values):
        raise ValueError('screen exclusions omit bank or played-trajectory evidence')
    for item in bank['exclusions']:
        campaign.verify(item)
    for ordinal, (domain, fingerprints) in enumerate(sorted(values.items())):
        document = bound(records[initial_count + ordinal], directory / 'rank4-screen' / f'played-exclusion-{ordinal}.json')
        expected = {'schema': campaign.ID + '.pilot-screen-played-exclusions.v2',
            'role': 'mixed-development', 'domain': domain, 'fingerprints': sorted(fingerprints),
            'execution': campaign.record(directory / 'rank4-screen/execution.json'), 'bank_sha256': bank['tsv']['sha256'],
            'contains_transcripts': False, 'contains_labels': False, 'contains_metrics': False,
            'includes_all_played_postroot_boundaries': True, 'includes_terminal_features': True}
        if {key: value for key, value in document.items() if key != 'body_sha256'} != expected:
            raise ValueError('screen exclusions do not reproduce all played boundaries')


def failed_pilot(root, attempt):
    if isinstance(attempt, bool) or attempt not in (1, 2):
        raise ValueError('only the first two standard trained attempts are supported')
    phase = f'attempt-{attempt:03d}-pilot'; context = root / 'phases' / phase
    contract = campaign.read(context / 'campaign.json'); parent = bound(contract['parent_campaign'], root / 'campaign.json')
    if (contract['attempt'] != attempt or contract['phase'] != 'pilot' or contract['policy'] != campaign.POLICY
            or any(contract['inputs'][key] != parent['inputs'][key] for key in
                ('attempt_one_initial_checkpoint', 'teacher_runtime', 'attempt_zero_runtime'))):
        raise ValueError('prior pilot lost the frozen campaign lineage')
    if (root / 'phases' / f'attempt-{attempt:03d}-full').exists():
        raise ValueError('full-stage attempt outcomes require separate validated integration')
    directory = context / phase; path = directory / 'pilot-outcome.json'
    if not path.exists():
        raise ValueError('prior pilot is incomplete; any claimed screen remains spent')
    outcome = campaign.read(path)
    if (outcome.get('admitted') is not False or outcome.get('campaign_success') is not False
            or outcome.get('status') not in ('offline-rejected', 'rank4-screen-rejected')):
        raise ValueError('a verified completed unsuccessful pilot is required')
    document = bound(outcome['selection'], directory / 'model-selection.json')
    bound(document['training'], directory / 'training.json')
    with selection.trainer.native_thread_execution_scope():
        training = validate_training(context, phase, contract)
        selected = validate_selection(document, training, context, phase, contract)
    screen = None
    if outcome['status'] == 'offline-rejected':
        if selected is not None or (directory / 'rank4-screen').exists():
            raise ValueError('offline rejection cannot hide an eligible or spent screen')
    else:
        if selected is None:
            raise ValueError('a completed screen requires its frozen selected source')
        screen = validate_screen(directory, outcome, selected)
    return {'attempt': attempt, 'context': context, 'phase': phase, 'contract': contract,
        'outcome': campaign.record(path), 'selection': outcome['selection'], 'training': document['training'], 'screen': screen}


def trajectory_fingerprints(transcript, prefix_turns):
    """Exclude the root and every continuation boundary, including terminal features."""
    actions = transcript.split('/') if transcript else []
    if isinstance(prefix_turns, bool) or not isinstance(prefix_turns, int) or not 0 <= prefix_turns <= len(actions):
        raise ValueError('invalid continuation prefix')
    state = campaign.features.ReplayState(); result = defaultdict(set)
    for turn in range(len(actions) + 1):
        if turn >= prefix_turns:
            for domain, value in campaign.fingerprints(state).items():
                result[domain].add(value)
        if turn < len(actions):
            campaign.features.apply_complete_turn(state, state.to_move, actions[turn])
    return result, state


def collect_fingerprints(previous, *, expected_games=2000):
    context, phase = previous['context'], previous['phase']; directory = context / phase
    positions = campaign.read(directory / 'positions.json'); games = campaign.read(directory / 'games.json')
    plan = bound(positions['plan'], directory / 'positions-plan.json')
    bound(plan['context'], context / 'campaign.json'); bound(plan['games'], directory / 'games.json')
    chunks = [campaign.read(campaign.verify(item)) for item in positions['chunk_receipts']]
    if [row['output'] for row in chunks] != positions['census_files']:
        raise ValueError('prior census lost its validated position chunk binding')
    game_rows = {row['ordinal']: row for row in games['rows']}
    if len(game_rows) != games['games'] or games['games'] != expected_games or len(chunks) != len(game_rows):
        raise ValueError('prior pilot generation or census coverage is incomplete')
    if {chunk['inputs']['game']['ordinal'] for chunk in chunks} != set(game_rows):
        raise ValueError('prior pilot census omits or duplicates generated games')
    values = defaultdict(set)
    for chunk in chunks:
        item = chunk['inputs']['game']
        if game_rows.get(item['ordinal']) != item or chunk['inputs']['plan_body_sha256'] != plan['body_sha256']:
            raise ValueError('prior census references a different generated game')
        for row in stream.read_gzip(campaign.verify(chunk['output'])):
            if row['split'] != item['split'] or row['split'] not in ('train', 'validation'):
                raise ValueError('prior census split changed')
            role = 'prior-train' if row['split'] == 'train' else 'prior-validation'
            for member in row['closure']:
                for domain, value in member.items():
                    values[role, domain].add(value)
    schedule = campaign.read(directory / 'schedule.json')
    schedule_record = campaign.record(directory / 'schedule.json')
    bound(schedule['contract'], context / 'campaign.json')
    if schedule['games'] != expected_games or len(schedule['rows']) != expected_games:
        raise ValueError('prior pilot schedule is incomplete')
    for item in games['rows']:
        scheduled = schedule['rows'][item['ordinal']]
        if any(item[key] != value for key, value in scheduled.items()):
            raise ValueError('prior generated game differs from its frozen schedule')
        receipt = campaign.read(campaign.verify(item['receipt']))
        if receipt['returncode'] != 0 or receipt['inputs']['schedule'] != schedule_record:
            raise ValueError('prior generated game lacks a successful native receipt')
        for output in receipt['outputs']:
            campaign.verify(output)
        game = item['game']; transcript = game['transcript']; actions = transcript.split('/')
        if (hashlib.sha256(transcript.encode()).hexdigest() != game['transcript_sha256']
                or '/'.join(actions[:game['prefix_turns']]) != item['transcript']):
            raise ValueError('prior generated trajectory lost its source binding')
        fingerprints, terminal = trajectory_fingerprints(transcript, game['prefix_turns'])
        if terminal.winner != game['winner'] or terminal.winner is None:
            raise ValueError('prior generated trajectory is incomplete')
        role = 'prior-train' if item['split'] == 'train' else 'prior-validation'
        for domain, members in fingerprints.items():
            values[role, domain].update(members)
    coverage = 'not-opened'
    if previous['screen'] is not None:
        bank, raw = previous['screen']; roots = {row['opening_id']: row for row in bank['rows']}
        for row in roots.values():
            fps, state = trajectory_fingerprints(row['transcript'], len(row['transcript'].split('/')))
            if campaign.fingerprints(state) != row['fingerprints']:
                raise ValueError('screen root fingerprints differ from the played bank')
            for domain, members in fps.items():
                values['mixed-development', domain].update(members)
        for game in raw['games']:
            if 'transcript' not in game:
                raise ValueError('completed v2 screen has no source-bound played trajectory')
            root = roots[game['opening_id']]; prefix = root['transcript'].split('/')
            if game['transcript'].split('/')[:len(prefix)] != prefix:
                raise ValueError('screen trajectory differs from its frozen root')
            fps, _ = trajectory_fingerprints(game['transcript'], len(prefix))
            for domain, members in fps.items():
                values['mixed-development', domain].update(members)
        coverage = 'all-source-bound-trajectories'
    return values, coverage


def carry_failed_pilot(root, previous, destination):
    """Write only in the new context; never reopen the preceding phase receipts."""
    values, screen_coverage = collect_fingerprints(previous)
    directory = destination / 'exclusions' / f'failed-attempt-{previous["attempt"]:03d}'
    artifacts = []
    sources = {'outcome': previous['outcome'], 'training': previous['training'], 'selection': previous['selection'],
        'positions': campaign.record(previous['context'] / previous['phase'] / 'positions.json'),
        'games': campaign.record(previous['context'] / previous['phase'] / 'games.json')}
    for ordinal, ((role, domain), members) in enumerate(sorted(values.items())):
        path = directory / f'fingerprints-{ordinal}.json'
        campaign.seal(path, {'schema': campaign.ID + '.failed-pilot-exclusions.v2', 'role': role, 'domain': domain,
            'fingerprints': sorted(members), 'sources': sources, 'contains_labels': False, 'contains_metrics': False,
            'contains_transcripts': False, 'source_paths_followed_during_filtering': False})
        artifacts.append(campaign.record(path))
    path = directory / 'index.json'
    campaign.seal(path, {'schema': campaign.ID + '.failed-pilot-carry.v2', 'attempt': previous['attempt'],
        'sources': sources, 'artifacts': artifacts, 'screen_boundary_coverage': screen_coverage,
        'new_teacher_labels_required': True, 'early_training_exception_unchanged': True,
        'validation_never_exempt': True, 'terminal_feature_boundaries_included': True})
    return artifacts, campaign.record(path)


def failed_attempt(root, attempt):
    """Choose the actual terminal branch; never bypass an existing full stage."""
    root = Path(root).resolve()
    if isinstance(attempt, bool) or attempt not in (1, 2):
        raise ValueError('after two unsuccessful trained attempts an intervention binding is required')
    if (root / 'phases' / f'attempt-{attempt:03d}-full').exists():
        from tools import compact_value_bfm_full_outcome_v2 as full_outcome
        return full_outcome.failed_full(root, attempt)
    return failed_pilot(root, attempt)


def carry_failed_attempt(root, previous, destination):
    if previous.get('stage') == 'full':
        from tools import compact_value_bfm_full_outcome_v2 as full_outcome
        return full_outcome.carry_failed_full(root, previous, destination)
    return carry_failed_pilot(root, previous, destination)
