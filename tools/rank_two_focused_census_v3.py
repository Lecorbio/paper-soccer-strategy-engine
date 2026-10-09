"""Increment teacher ancestry with committed prefix reuse and bounded stages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_prior_v1 as validator
from tools import rank_two_focused_work_v1 as work
from tools import top_three_played_exclusions_v3 as census


def copy_state(state):
    return census.e.rules.ReplayState(ball=state.ball, to_move=state.to_move,
        winner=state.winner, used_segments=set(state.used_segments),
        visit_count=dict(state.visit_count))


class PrefixRows(census.RowCensus):
    """Only committed rows contribute to the shared caches or skipped prefix."""

    def __init__(self, root, db, budget, physical, anchor=None, teacher=False):
        super().__init__(root, db, budget)
        self.db, self.committed_physical = db, physical
        self.pending_physical, self.anchor, self.next_anchor = {}, anchor, None
        self.teacher, self.prefix_reuses, self.indexed_reuses = teacher, 0, 0

    def observe(self, state):
        self.observations += 1
        if self.observations % 256 == 0:
            self.guard()
        physical = (state.ball, state.to_move, state.winner,
                    sum(census.v2.EDGE_BITS[edge] for edge in state.used_segments))
        entry = self.pending_physical.get(physical, self.committed_physical.get(physical))
        canonical, feature = entry if entry is not None else (census.e.fingerprint(state), None)
        if feature is None:
            indexed = self.db.execute("SELECT 1 FROM keys WHERE category='states' AND value=?",
                                      (canonical,)).fetchone() is not None
            if indexed and not self.teacher:
                self.indexed_reuses += 1
            else:
                feature = census.banks.feature_key(state)
        if feature is not None:
            # Teacher classification includes repeated observations, even when
            # a plain trajectory previously populated the physical cache.
            self.states.add(canonical)
            self.features.add(feature)
            self.feature_by_state[canonical] = feature
        self.pending_physical[physical] = (canonical, feature)
        return canonical

    def trace(self, turns, reference, locator, kind, group=None):
        if group:
            self.groups.add(str(group))
        normalized = [[None, turn] if isinstance(turn, str) else
                      [turn.get('player_id', turn.get('player')), turn['action']]
                      for turn in turns]
        key = hashlib.sha256(census.e.campaign.canonical(normalized)).hexdigest()
        prefix = tuple((player, action) for player, action in normalized[:-1])
        if (kind == 'focused-teacher-successor' and self.anchor is not None
                and prefix == self.anchor['turns']):
            # The parent row has already committed all prefix states in both
            # the global and teacher namespaces. Only this new tail remains.
            state = copy_state(self.anchor['state'])
            primitive = self.anchor['result']['primitive_rows'] - 1
            boundaries = self.anchor['result']['boundary_rows']
            remaining = normalized[-1:]
            self.prefix_reuses = 1
        else:
            state = census.e.rules.ReplayState()
            self.observe(state)
            primitive, boundaries, remaining = 0, 1, normalized
        reason = None
        for player, action in remaining:
            if state.winner is not None:
                reason = 'turn-after-terminal'
                break
            mover = state.to_move
            if player is not None and player != mover:
                reason = 'wrong-player'
                break
            for direction in action:
                if direction not in '01234567' or state.winner is not None or state.to_move != mover:
                    reason = 'invalid-or-overlong-output'
                    break
                try:
                    census.e.rules.apply_primitive(state, direction)
                except ValueError:
                    reason = 'illegal-edge'
                    break
                self.observe(state)
                primitive += 1
            if reason:
                break
            if not action or state.winner is None and state.to_move == mover:
                reason = 'empty-or-incomplete-turn'
                break
            boundaries += 1
        result = dict(primitive_rows=primitive + 1, boundary_rows=boundaries,
                      rule_terminal=state.winner is not None, truncated_tail=reason)
        self.counts[kind] += 1
        self.counts['trajectory_instances'] += 1
        self.counts['observed_primitive_rows'] += result['primitive_rows']
        self.counts['observed_boundary_rows'] += result['boundary_rows']
        self.counts['terminal_trajectory_instances' if result['rule_terminal'] else
                    'partial_trajectory_instances'] += 1
        self.counts['truncated_tail_instances'] += reason is not None
        self.ledger.append(dict(input=reference, locator=locator, kind=kind, trace_sha256=key, **result))
        if kind == 'focused-teacher-parent' and reason is None and state.winner is None:
            self.next_anchor = dict(turns=tuple((player, action) for player, action in normalized),
                                    state=copy_state(state), result=result)

    def publish_committed(self, physical):
        physical.update(self.pending_physical)
        if len(physical) > 100000:
            physical.clear()
        return self.next_anchor if self.next_anchor is not None else self.anchor


def extend(plan_path):
    plan = campaign.read(plan_path)
    if plan['producer'] != campaign.record(__file__):
        raise ValueError('incremental prefix census producer changed')
    work.checked(plan, 'archive', launch=True)
    original = campaign.read(campaign.verify(plan['normalized_plan']))
    for name in ('records', 'scope', 'closure'):
        campaign.verify(original[name])
    for ref in original['producers'].values():
        campaign.verify(ref)
    for name in ('producer', 'teacher_validator', 'proposal_algorithm', 'prior_validator'):
        campaign.verify(original[name])
    closure = campaign.read(original['closure']['path'])
    for ref in closure['inputs']:
        campaign.verify(ref)
    if plan['committed_rows'] == 0:
        prior = campaign.read(campaign.verify(original['seed_prior']))
        if prior['inputs'] != original['seed_inputs']:
            raise ValueError('seed index chain changed')
        with validator.prior_census(prior):
            pass
        parent_database = prior['inputs']['database']
        parent_binding = prior['inputs']['binding']
    else:
        parent_database, parent_binding = plan['parent_database'], plan['parent_binding']
    campaign.verify(parent_database)
    campaign.verify(parent_binding)
    if plan.get('parent_wal'):
        campaign.verify(plan['parent_wal'])
    if not 1 <= plan['maximum_new_rows'] <= 50000:
        raise ValueError('bounded row count required')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'CLAIM.json').exists():
        raise RuntimeError('unknown spent census claim retained; classify its frontier before continuation')
    campaign.immutable(output / 'CLAIM.json', dict(plan=campaign.record(plan_path),
        committed_parent_rows=plan['committed_rows'], maximum_new_rows=plan['maximum_new_rows']))
    snapshot = output / 'parent-snapshot.sqlite3'
    shutil.copyfile(campaign.verify(parent_database), snapshot)
    if plan.get('parent_wal'):
        shutil.copyfile(campaign.verify(plan['parent_wal']), Path(str(snapshot) + '-wal'))
    directory = output / 'census'
    directory.mkdir(exist_ok=True)
    source = sqlite3.connect(snapshot.as_uri() + '?mode=ro', uri=True)
    database = directory / 'checkpoint.sqlite3'
    db = sqlite3.connect(database)
    budget = census.Budget(plan['cpu_seconds'], plan['maximum_rss_bytes'])
    scope, closure = campaign.read(original['scope']['path']), campaign.read(original['closure']['path'])
    binding = dict(schema=census.SCHEMA, scope=original['scope'], closure=original['closure'],
        records=original['records'], sources=original['producers'],
        inherited_inputs=[original['seed_inputs']['inventory']],
        indexed_seed=dict(database=original['seed_inputs']['database'],
                          receipt=original['seed_inputs']['receipt'], plan=plan['normalized_plan']),
        extension_producer=original['producer'], continuation_producer=plan['producer'],
        continuation_plan=campaign.record(plan_path), committed_parent_rows=plan['committed_rows'])
    physical, anchor, committed, reuses, indexed = {}, None, 0, 0, 0
    stop_reason = None
    try:
        for connection in (source, db):
            connection.execute('PRAGMA cache_size=-32768')
            connection.execute('PRAGMA mmap_size=0')
        source.backup(db, pages=256)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        previous = json.loads(db.execute("SELECT value FROM metadata WHERE name='binding'").fetchone()[0])
        if previous != campaign.read(parent_binding['path']):
            raise ValueError('copied parent database binding differs')
        if plan['committed_rows']:
            if census.checkpoint(db) != plan['committed_rows']:
                raise ValueError('classified frontier differs')
        with db:
            if not plan['committed_rows']:
                db.execute('DELETE FROM completed')
                db.execute('DELETE FROM metadata')
                bootstrap = dict(completed=0, historical_counts_unconfirmed={},
                    inherited_inventories=[dict(kind='exposure', input=original['seed_inputs']['inventory'])],
                    indexed_seed_receipt=original['seed_inputs']['receipt'])
                db.execute('INSERT INTO metadata VALUES (?,?)', ('bootstrap', json.dumps(bootstrap, sort_keys=True)))
                db.execute('INSERT INTO metadata VALUES (?,?)', ('binding', json.dumps(binding, sort_keys=True)))
            else:
                db.execute("UPDATE metadata SET value=? WHERE name='binding'", (json.dumps(binding, sort_keys=True),))
        campaign.immutable(directory / 'binding.json', binding)
        with campaign.verify(original['records']).open() as stream:
            for number, line in enumerate(stream, 1):
                digest = hashlib.sha256(line.encode()).hexdigest()
                if number <= plan['committed_rows']:
                    stored = db.execute('SELECT row_sha256 FROM completed WHERE number=?', (number,)).fetchone()
                    if stored is None or stored[0] != digest:
                        raise ValueError('consumed normalized row changed')
                    continue
                if committed >= plan['maximum_new_rows']:
                    stop_reason = 'declared row cap'
                    break
                try:
                    budget.guard()
                except (TimeoutError, MemoryError) as exc:
                    stop_reason = str(exc)
                    break
                if committed % 100 == 0:
                    work.checked(plan, 'archive')
                row = json.loads(line)
                if row['mode'] != 'trace':
                    raise ValueError('normalized trace row required')
                item = PrefixRows(scope['campaign_root'], db, budget, physical, anchor,
                                  row['kind'].startswith('focused-teacher'))
                item.trace(row['turns'], row['input'], row['locator'], row['kind'], row['group'])
                if any(item.counts[name] for name in census.FAILURES):
                    raise ValueError('normalized row failed rules')
                with db:
                    for category in ('states', 'features', 'groups'):
                        census.union(db, category, getattr(item, category))
                    if item.teacher:
                        census.union(db, 'teacher:focused_observed_states', item.states)
                        census.union(db, 'teacher:focused_observed_features', item.features)
                    db.execute('INSERT INTO completed VALUES (?,?,?,?)', (number, digest,
                        json.dumps(dict(item.counts), sort_keys=True), json.dumps(item.ledger, sort_keys=True)))
                # Cache publication follows, never precedes, the row commit.
                anchor = item.publish_committed(physical)
                committed += 1
                reuses += item.prefix_reuses
                indexed += item.indexed_reuses
                if committed % 100 == 0:
                    campaign.atomic(output / 'PROGRESS.json', dict(committed_rows=number,
                        total_rows=original['rows'], new_rows=committed, prefix_reuses=reuses))
        final = census.checkpoint(db)
        complete = final == original['rows']
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        if complete:
            receipt = census.export_inventory(db, directory / 'inventory-001.json', binding, scope,
                                              closure, 'complete', None, budget)
            if receipt['status'] != 'complete':
                raise ValueError('final ancestry receipt failed coverage')
            receipt.update(schema=census.SCHEMA, checkpoint=str(database.resolve()),
                historical_protected_reads=False, historical_ancestry_rescan=False, games_run=0,
                new_committed_rows=committed, prefix_reuses=reuses, indexed_state_reuses=indexed)
            campaign.immutable(directory / 'receipt-001.json', receipt)
    finally:
        db.close()
        source.close()
    campaign.verify(parent_database)
    if plan.get('parent_wal'):
        campaign.verify(plan['parent_wal'])
    for name in ('records', 'scope', 'closure'):
        campaign.verify(original[name])
    campaign.verify(plan['producer'])
    result = dict(passed=True, complete=complete, plan=campaign.record(plan_path),
        database=campaign.record(database), binding=campaign.record(directory / 'binding.json'),
        committed_rows=final, total_rows=original['rows'], new_rows=committed,
        prefix_reuses=reuses, indexed_state_reuses=indexed, stop_reason=stop_reason,
        normalized_plan=plan['normalized_plan'], historical_protected_reads=False,
        historical_ancestry_rescan=False, games_run=0)
    if complete:
        frozen = dict(schema='papersoccer.rank-two.live-prior-exposure.v1', historical_protected_reads=False,
            packet_id='focused-network-prefix-exposure', extension_plan=campaign.record(plan_path),
            inputs=dict(database=result['database'], binding=result['binding'], inventory=receipt['inventory'],
                        receipt=campaign.record(directory / 'receipt-001.json'),
                        closure=original['closure'], scope=original['scope']))
        with validator.prior_census(frozen):
            pass
        campaign.immutable(output / 'bank-prior.json', frozen)
        result.update(prior=campaign.record(output / 'bank-prior.json'), exposures=original['exposures'])
        campaign.immutable(output / 'ready.json', result)
    campaign.immutable(output / 'RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    result = extend(args.plan)
    print(json.dumps({k:result[k] for k in ('passed', 'complete', 'committed_rows', 'total_rows')}), flush=True)


if __name__ == '__main__':
    main()
