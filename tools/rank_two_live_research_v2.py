#!/usr/bin/env python3
"""Packet coverage, bounded causal-diagnostic selection, and explicit exposure export."""
from __future__ import annotations
import argparse,collections,hashlib,inspect,json,sqlite3,sys
from contextlib import contextmanager
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import top_three_experiments as e,rank_two_live_v2 as live

def canonical_transcript(text):
 reflected=''.join(str((8-int(ch))%8) if ch.isdigit() else ch for ch in text)
 return min(text,reflected)

def select(rows,seen=(),maximum=96):
 if not 0<=maximum<=96:raise ValueError('diagnostic position cap')
 seen=set(seen);pool=[r for r in rows if r['state_sha256'] not in seen];selected=[];games=collections.Counter();strata=collections.Counter();opponents=collections.Counter();phases=collections.Counter()
 while pool and len(selected)<maximum:
  eligible=[r for r in pool if games[r['game_id']]<3 and r['state_sha256'] not in seen]
  if not eligible:break
  def key(r):
   return (not r['priority_opponent'],strata[(r['result'],r['color'],r['phase'])],opponents[r['opponent_agent_id']],phases[(r['game_id'],r['phase'])],games[r['game_id']],r['game_id'],r['turn_index'])
  row=min(eligible,key=key);selected.append(row);seen.add(row['state_sha256']);games[row['game_id']]+=1;strata[(row['result'],row['color'],row['phase'])]+=1;opponents[row['opponent_agent_id']]+=1;phases[(row['game_id'],row['phase'])]+=1;pool.remove(row)
 return selected


def stream_record(path):
 path=Path(path).resolve();digest=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
 return dict(path=str(path),sha256=digest.hexdigest())

def verify_stream(ref):
 if stream_record(ref['path'])!=ref:raise ValueError('prior exposure identity changed')
 return Path(ref['path'])

def inventory_header(path):
 # The census exporter appends large arrays after the sorted metadata object.
 # Read only that metadata, and hash the full inventory separately in chunks.
 marker=b',"canonical_state_sha256":[';data=b''
 with Path(path).open('rb') as stream:
  while len(data)<=16*1024**2:
   block=stream.read(65536)
   if not block:break
   data+=block
   if marker in data:return json.loads(data.split(marker,1)[0]+b'}')
 raise ValueError('missing or oversized census inventory metadata')

@contextmanager
def prior_census(frozen):
 if frozen.get('schema')!='papersoccer.rank-two.live-prior-exposure.v1' or frozen.get('historical_protected_reads') is not False:raise ValueError('invalid frozen prior exposure')
 refs=frozen['inputs']
 for ref in refs.values():verify_stream(ref)
 receipt=e.read(refs['receipt']['path']);binding=e.read(refs['binding']['path'])
 closure=e.read(refs['closure']['path']);scope=e.read(refs['scope']['path'])
 header=inventory_header(refs['inventory']['path']);database=Path(refs['database']['path'])
 if (receipt.get('status')!='complete' or receipt.get('error') is not None
     or receipt.get('historical_protected_reads') is not False
     or receipt.get('inventory')!=refs['inventory']
     or Path(receipt['checkpoint']).resolve()!=database
     or receipt.get('completed_rows')!=receipt.get('total_rows')
     or binding.get('closure')!=refs['closure'] or binding.get('scope')!=refs['scope']
     or closure.get('scope')!=refs['scope'] or closure.get('rows')!=receipt['total_rows']
     or closure.get('all_payload_inputs_hash_verified_before_census') is not True
     or scope.get('historical_protected_reads') is not False
     or header.get('binding')!=binding or header.get('status')!='complete'
     or header.get('complete_coverage') is not True
     or header.get('historical_protected_reads') is not False
     or header.get('last_fully_processed_normalized_row')!=receipt['total_rows']
     or header.get('total_normalized_rows')!=receipt['total_rows']
     or header.get('state_function_sha256')!=hashlib.sha256(inspect.getsource(e.fingerprint).encode()).hexdigest()):
  raise ValueError('incomplete or inconsistent prior census binding')
 # A nonempty WAL is not covered by the database hash. Never checkpoint or write it.
 wal=Path(str(database)+'-wal')
 if wal.exists() and wal.stat().st_size:raise ValueError('prior census has unbound WAL changes')
 db=sqlite3.connect(database.as_uri()+'?mode=ro')
 try:
  db.execute('PRAGMA query_only=ON')
  metadata=dict(db.execute('SELECT name,value FROM metadata'))
  bootstrap=json.loads(metadata['bootstrap'])
  if json.loads(metadata['binding'])!=binding:raise ValueError('checkpoint binding mismatch')
  done=bootstrap['completed']
  for number, in db.execute('SELECT number FROM completed ORDER BY number'):
   if number!=done+1:raise ValueError('incomplete prior census checkpoint')
   done=number
  if (done!=receipt['total_rows'] or bootstrap['completed']!=receipt['bootstrap_completed_rows']
      or db.execute("SELECT count(*) FROM keys WHERE category='states'").fetchone()[0]!=receipt['states']):
   raise ValueError('incomplete prior census checkpoint')
  yield db
  verify_stream(refs['database'])
  if wal.exists() and wal.stat().st_size:raise ValueError('prior census changed during selection')
 finally:db.close()

def freeze_prior_exposure(root,packet_id):
 root=Path(root).resolve();output=root/'packets'/live.safe_id(packet_id)/'research-v2/prior-exposure.json'
 if output.exists():
  frozen=e.read(output)
  with prior_census(frozen):pass
  return e.raw_record(output)
 state_path=root.parent/'STATE.json';state=e.read(state_path);inventory=state['exposure_inventory']
 path=Path(inventory['path']);suffix=path.name.removeprefix('inventory-')
 if path.name==suffix:raise ValueError('expected completed census inventory')
 binding_ref=stream_record(path.parent/'binding.json');binding=e.read(binding_ref['path'])
 frozen=dict(schema='papersoccer.rank-two.live-prior-exposure.v1',frozen_at=live.now(),state_at_freeze=e.raw_record(state_path),historical_protected_reads=False,
  inputs=dict(inventory=inventory,database=stream_record(path.parent/'checkpoint.sqlite3'),receipt=stream_record(path.parent/('receipt-'+suffix)),binding=binding_ref,closure=binding['closure'],scope=binding['scope']))
 with prior_census(frozen):pass
 e.emit(output,frozen);return e.raw_record(output)

def novelty(rows,frozen):
 with prior_census(frozen) as db:
  matched={r['state_sha256'] for r in rows if db.execute("SELECT 1 FROM keys WHERE category='states' AND value=?",(r['state_sha256'],)).fetchone()}
 return [r for r in rows if r['state_sha256'] not in matched],dict(candidate_rows=len(rows),candidate_canonical_states=len({r['state_sha256'] for r in rows}),prior_exposure_matched_rows=sum(r['state_sha256'] in matched for r in rows),prior_exposure_matched_canonical_states=len(matched))

def strata_counts(rows):
 counts=collections.Counter((r['result'],r['color'],r['phase']) for r in rows);states=collections.defaultdict(set)
 for row in rows:states[(row['result'],row['color'],row['phase'])].add(row['state_sha256'])
 return [dict(result=k[0],color=k[1],phase=k[2],rows=v,canonical_states=len(states[k])) for k,v in sorted(counts.items())]

def report(root,packet_id,prior_exposure):
 frozen=e.read(e.verify(prior_exposure))
 # Verify the frozen cutoff before opening any new packet outcomes.
 with prior_census(frozen):pass
 root=Path(root);directory=root/'packets'/live.safe_id(packet_id);result=e.read(directory/'result.json');plan=e.read(e.verify(result['plan']))
 if plan['kind']=='qualification':raise ValueError('qualification windows are not research selection')
 inputs=[prior_exposure,*frozen['inputs'].values(),e.raw_record(directory/'result.json')];seen=set()
 original=root.parent/'diagnostics-v2/plan.json'
 if original.exists():
  inputs.append(e.raw_record(original));seen.update(r['state_sha256'] for r in e.read(original)['rows'])
 for prior in sorted(list((root/'packets').glob('*/research/selection.json'))+list((root/'packets').glob('*/research-v2/selection.json'))):
  if prior.parent.parent.name>=packet_id:continue
  inputs.append(e.raw_record(prior));seen.update(r['state_sha256'] for r in e.read(prior)['rows'])
 first=root/'attempts'/plan['slots'][0]['attempt_id']/'before/snapshot.json';before=e.read(first);inputs.append(e.raw_record(first))
 ranked=sorted([r for r in before['board']['users'] if r['codingamer']['userId']!=live.USER],key=lambda r:r['rank'])
 priority={r['agentId'] for r in ranked[:6]};priority.update(r['agentId'] for r in ranked if r['codingamer'].get('pseudo','').lower() in ('jacek','marchete'))
 game_ids=set();trajectory=set();opening=set();positions=set();rows=[];games=[];strata=collections.Counter();durations=[]
 for slot in plan['slots']:
  out=root/'attempts'/slot['attempt_id'];r=e.read(out/'result.json');inputs.append(e.raw_record(out/'result.json'));e.verify(r['manifest'])
  ref=e.raw_record(out/'research-games.json');inputs.append(ref)
  completed=e.read(e.verify(r['completed']));claim=e.read(out/'claim.json')
  durations.append((live.datetime.datetime.fromisoformat(completed['observed_at'])-live.datetime.datetime.fromisoformat(claim['claimed_at'])).total_seconds())
  for game in e.read(ref['path'])['games']:
   if game['game_id'] in game_ids:raise ValueError('duplicate game across attempts')
   game_ids.add(game['game_id']);games.append(game)
   turns=game['transcript'].split('/') if game['transcript'] else [];trajectory.add(canonical_transcript(game['transcript']));opening.add(canonical_transcript('/'.join(turns[:4])))
   op=game['opponent'];focus=game['focus'];clean=game['operational']['classification']=='clean'
   strata[(game['source_sha256'],op['agent_id'],op['submission_id'],focus['player_id'],focus['result'] if clean else 'operational')]+=1
   state=e.rules.ReplayState();prefix=[];edges=0
   for index,action in enumerate(turns):
    canonical=e.fingerprint(state);positions.add(canonical)
    if clean and state.to_move==focus['player_id'] and state.winner is None:
     rows.append(dict(game_id=game['game_id'],attempt_id=game['attempt_id'],source_sha256=game['source_sha256'],opponent=op['name'],opponent_agent_id=op['agent_id'],opponent_submission_id=op['submission_id'],priority_opponent=op['agent_id'] in priority,color=focus['player_id'],result=focus['result'],phase='early' if edges<=16 else 'middle' if edges<=60 else 'late',drawn_edges=edges,turn_index=index,prefix='/'.join(prefix),observed_action=action,state_sha256=canonical,opening_family=canonical_transcript('/'.join(turns[:4])),training_eligible=False,confirmation_eligible=False))
    e.rules.apply_complete_turn(state,state.to_move,action);prefix.append(action);edges+=len(action)
 fresh,coverage=novelty(rows,frozen)
 available=[r for r in fresh if r['state_sha256'] not in seen]
 selected=select(available,seen)
 coverage.update(available_rows=len(available),available_canonical_states=len({r['state_sha256'] for r in available}),available_strata=strata_counts(available),selected_strata=strata_counts(selected))
 output=directory/'research-v2';e.emit(output/'selection.json',dict(rows=selected,inputs=inputs,selection_rule='priority opponents; balance result/color/phase and opponent; max3pergame; canonical dedup incl frozen completed prior exposure and previous selections',independent_samples_claimed=False,producer=e.raw_record(__file__)))
 summary=dict(passed=True,prior_exposure=prior_exposure,novelty=coverage,packet=e.raw_record(directory/'plan.json'),games=len(games),unique_game_ids=len(game_ids),distinct_trajectories=len(trajectory),opening_families=len(opening),canonical_decision_positions=len(positions),selected_positions=len(selected),selected_games=len({r['game_id'] for r in selected}),priority_opponents=sorted(priority),observed_turnaround_seconds=durations,opponent_version_color_outcomes=[dict(source_sha256=k[0],opponent_agent_id=k[1],opponent_submission_id=k[2],color=k[3],outcome=k[4],games=v) for k,v in sorted(strata.items())],selection=e.raw_record(output/'selection.json'),claim='Descriptive unmatched schedules; games and repeated opening families are correlated. No causal strength estimate or optimal-move labels.')
 e.emit(output/'report.json',summary);return summary

def exposure(root,packet_ids,inherited,output):
 # Explicit packets only; never discover or open confirmation outcomes.
 root=Path(root);output=Path(output);records=[];inputs=[]
 for pid in packet_ids:
  directory=root/'packets'/live.safe_id(pid);plan=e.read(directory/'plan.json')
  qualification=plan['kind']=='qualification'
  e.read(directory/'result.json');inputs.extend([e.raw_record(directory/'plan.json'),e.raw_record(directory/'result.json')])
  for slot in plan['slots']:
   out=root/'attempts'/slot['attempt_id'];r=e.read(out/'result.json');ref=e.raw_record(out/('exposure-transcripts.json' if qualification else 'research-games.json'));inputs.extend([ref,e.raw_record(out/'result.json'),r['manifest']]);e.verify(r['manifest'])
   for i,game in enumerate(e.read(ref['path'])['games']):
    records.append(dict(mode='trace',kind='rank-two-live-research',input=ref,locator=f'/games/{i}/transcript',group=f"arena:{game['game_id']}",turns=game['transcript'].split('/') if game['transcript'] else [],training_eligible=False))
 cutoff=dict(cutoff_utc=live.now(),historical_protected_reads=False,excluded_current_live_details_directory=str(output/'protected-not-permitted'),excluded_current_agent_id=-1,resources=dict(workers=1,rss_bytes=4*1024**3,cpu_seconds=1200),purpose='Explicit research trajectories only; no training labels or protected outcomes.')
 e.emit(output/'CUTOFF.json',cutoff)
 scope=dict(schema='papersoccer.top-three.played-exclusions-scope.v2',campaign_root=str(e.ROOT/'results'),historical_protected_reads=False,cutoff=e.raw_record(output/'CUTOFF.json'),entries=[],inherit=[dict(kind='exposure',input=e.raw_record(inherited))])
 # Use the existing schema constant rather than a guessed compatibility spelling.
 from tools import top_three_played_exclusions_v2 as v2
 scope['schema']=v2.SCOPE_SCHEMA;e.emit(output/'scope.json',scope)
 e.campaign.immutable(output/'records.jsonl',b''.join(e.campaign.canonical(r) for r in records))
 closure=dict(schema='papersoccer.top-three.normalized-exposure-closure.v1',scope=e.raw_record(output/'scope.json'),cutoff=scope['cutoff'],records=e.raw_record(output/'records.jsonl'),rows=len(records),inputs=inputs,skipped_or_unreturned=[],all_payload_inputs_hash_verified_before_census=True,producer=e.raw_record(__file__))
 e.emit(output/'records.closure.json',closure);return closure

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--packet',required=True);p.add_argument('--prior-exposure',type=Path,required=True);p.add_argument('--prior-exposure-sha256',required=True);a=p.parse_args();print(json.dumps(report(a.root,a.packet,dict(path=str(a.prior_exposure.resolve()),sha256=a.prior_exposure_sha256)),indent=2))
