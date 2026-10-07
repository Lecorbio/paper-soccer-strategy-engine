#!/usr/bin/env python3
"""Hardened live research lifecycle; preserves the first rollout producer unchanged."""
from __future__ import annotations
import argparse, contextlib, datetime, fcntl, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import top_three_experiments as e, top_three_collect_replays as c
from tools import compact_value_bfm_live_v2 as identity, top_three_queue_v4 as queue
ROOT=Path(__file__).resolve().parents[1]
SCHEMA='papersoccer.rank-two.live-research.v2'
USER=5215477

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def safe_id(value):
 if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}',value): raise ValueError('invalid identifier')
 return value
@contextlib.contextmanager
def lock(root):
 # Shared by manual and scheduled live operations across worktrees.
 directory=queue.shared_root();directory.mkdir(parents=True,exist_ok=True)
 with (directory/'live-submission.lock').open('a') as f:
  try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError: raise RuntimeError('another live operation owns the submission lock')
  yield

def config(root):
 from tools import rank_two_campaign_v2 as policy
 root=Path(root).resolve();m=policy.load(root/'campaign.json');return m

def source_ref(path):
 r=e.raw_record(path);text=Path(path).read_text();text.encode('ascii')
 if not 0<len(text)<100000:raise ValueError('source size/encoding')
 return r

def admit(root,source,*,safety=None,pilot=None,correctness=None,development=None,panel=None):
 root=Path(root);m=config(root);src=source_ref(source);refs={}
 if src['sha256']==m['control']['source']['sha256']:
  release=ROOT/'submissions/codingame/releases/20260927-rank3/live-calibration.json'
  old=e.read(release)
  if old['coverage']['focus_operational_failures']!=0:raise ValueError('baseline operational evidence')
  refs['baseline_release']=e.raw_record(release);kind='verified-baseline'
 else:
  if not all([safety,pilot,correctness]):raise ValueError('correctness, safety and pilot required')
  for name,path in [('safety',safety),('correctness',correctness)]:
   item=e.read(path)
   if item.get('passed') is not True or item.get('candidate_source',item.get('source',{})).get('sha256')!=src['sha256']:raise ValueError(name+' source/pass differs')
   refs[name]=e.raw_record(path)
  p=e.read(pilot)
  if not p.get('complete') or p.get('stage')!='pilot':raise ValueError('complete pilot required')
  rows=[v for v in p['candidates'].values() if v['candidate_source']['sha256']==src['sha256']]
  if len(rows)!=1 or rows[0].get('passed') is not True or rows[0]['assessment'].get('passed') is not True or rows[0]['roots']!=64 or rows[0]['assessment']['failures']!=0 or rows[0]['assessment']['cluster_bootstrap']['upper_95']<0:raise ValueError('pilot admission failed')
  e.verify(p['plan']);e.verify(rows[0]['bank'])
  refs['pilot']=e.raw_record(pilot);kind='safe-pilot'
 if development or panel:
  if not development or not panel:raise ValueError('both development and panel required')
  d=e.read(development);rows=[v for v in d['candidates'].values() if v['candidate_source']['sha256']==src['sha256']]
  if not d.get('complete') or d.get('stage')!='development' or len(rows)!=1:raise ValueError('development identity')
  a=rows[0]['assessment']
  if rows[0]['roots']!=256 or a['failures'] or a['mean_uplift']<.03 or min(a['opponent_uplift'].values())<-.05:raise ValueError('development gate')
  p=e.read(panel)
  if not p.get('complete') or p.get('own_failures')!=0 or len(p.get('games',[]))!=12:raise ValueError('platform panel incomplete')
  selection=e.read(e.verify(p['selection']))
  if selection['source']['sha256']!=src['sha256']:raise ValueError('panel source differs')
  refs.update(development=e.raw_record(development),panel=e.raw_record(panel))
 value=dict(schema=SCHEMA,source=src,kind=kind,evidence=refs,promotion_eligible=bool(development and panel),manifest=e.raw_record(root/'campaign.json'))
 # Research and promotion admissions can coexist without changing earlier receipts.
 path=root/'admissions'/src['sha256']/('promotion.json' if development else 'research.json');e.emit(path,value);return e.raw_record(path)

def packets(root):return sorted((Path(root)/'packets').glob('*/plan.json'))
def guard_source(root,sha):
 from tools import rank_two_campaign_v2 as policy
 if (Path(root)/'STOP.json').exists() and sha!=policy.BASELINE_SHA:
  raise ValueError('operational stop: only incumbent restoration allowed')

def lease(root,aid,seconds=300):
 """Bounded resource reservation during editor interaction, not a supervisor."""
 out,a=attempt(root,aid)
 if not 1<=seconds<=300:raise ValueError('bounded UI lease maximum300s')
 with queue.lock(queue.shared_root()):
  owner=dict(pid=os.getpid(),birth=queue.identity(os.getpid()))
  atomic(out/'ui-lease.json',dict(owner=owner,expires=time.time()+seconds,attempt_id=aid))
  print('UI resource lease ready',flush=True)
  until=time.monotonic()+seconds
  while time.monotonic()<until and not (out/'submission.json').exists():time.sleep(.5)
  atomic(out/'ui-lease.json',dict(owner=owner,expires=0,attempt_id=aid))

def check_lease(out):
 p=out/'ui-lease.json'
 if not p.exists():raise ValueError('UI resource lease required before copyback/click')
 v=e.read(p)
 if v['expires']<=time.time() or not queue.live(v['owner']):raise ValueError('UI lease expired or stale')

def packet(root,kind,candidate=None):
 root=Path(root);m=config(root)
 with lock(root):
  for p in packets(root):
   if not (p.parent/'result.json').exists():raise ValueError('previous packet remains unresolved')
  baseline=admit(root,m['control']['source']['path']);control=e.read(e.verify(baseline))['source']
  if kind=='baseline_research':order=[control]*3;admissions=[baseline]*3
  else:
   if candidate is None:raise ValueError('candidate admission required')
   ref=e.raw_record(candidate);a=e.read(e.verify(ref));e.verify(a['source'])
   if a['manifest']!=e.raw_record(root/'campaign.json'):raise ValueError('admission manifest')
   for r in a['evidence'].values():e.verify(r)
   guard_source(root,a['source']['sha256'])
   if kind=='qualification':
    if not a['promotion_eligible']:raise ValueError('promotion gates required')
    if any(e.read(p).get('qualification_source')==a['source']['sha256'] for p in packets(root)):raise ValueError('source already has a qualification block')
    order=[a['source']]*3;admissions=[ref]*3
   elif kind=='candidate_research':order=[control,a['source'],a['source'],control,control,a['source']];admissions=[baseline,ref,ref,baseline,baseline,ref]
   else:raise ValueError('unsupported packet kind')
  number=len(packets(root))+1;pid=f'packet-{number:04d}'
  value=dict(schema=SCHEMA,packet_id=pid,kind=kind,manifest=e.raw_record(root/'campaign.json'),created_at=now(),slots=[dict(attempt_id=f'{pid}-slot-{i+1:02d}',source=s,admission=r) for i,(s,r) in enumerate(zip(order,admissions))],qualification_source=order[0]['sha256'] if kind=='qualification' else None)
  e.emit(root/'packets'/pid/'plan.json',value);return value

def next_slot(root):
 ps=packets(root)
 if not ps:return None
 p=ps[-1];plan=e.read(p)
 if (p.parent/'result.json').exists():return None
 for slot in plan['slots']:
  if not (Path(root)/'attempts'/slot['attempt_id']/'result.json').exists():return p,slot
 return None

def prepare(root,registry):
 root=Path(root);config(root)
 with lock(root):
  item=next_slot(root)
  if item is None:raise ValueError('no pending packet slot')
  p,slot=item;out=root/'attempts'/slot['attempt_id'];path=out/'plan.json'
  if path.exists():return e.read(path)
  guard_source(root,slot['source']['sha256'])
  admission=e.read(e.verify(slot['admission']));src=e.verify(slot['source']);c.arena.load_exclusion_registry(Path(registry),e.raw_record(registry)['sha256'])
  if admission['source']!=slot['source']:raise ValueError('admission source')
  for ref in admission['evidence'].values():e.verify(ref)
  e.campaign.immutable(out/'source.cpp',src.read_bytes())
  e.campaign.immutable(out/'producer.py',Path(__file__).read_bytes())
  value=dict(schema=SCHEMA,attempt_id=slot['attempt_id'],packet=e.raw_record(p),source=e.raw_record(out/'source.cpp'),admission=slot['admission'],registry=e.raw_record(registry),expected_games=90,user_id=USER,created_at=now(),producer=e.raw_record(out/'producer.py'))
  e.emit(path,value);return value

def attempt(root,aid):
 p=Path(root)/'attempts'/safe_id(aid);a=e.read(p/'plan.json')
 for k in ('packet','source','admission','registry'):e.verify(a[k])
 return p,a

def snapshot(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 a=c.Acquisition(out);board,br=a.fetch('leaderboard-v1',identity.LEADERBOARD_REQUEST);me=identity.user_row(board,USER)
 battles,rr=a.fetch('agent-battles-v1',[me['agentId'],None])
 value=dict(board=board,battles=battles,user=me,leaderboard_receipt=br,battles_receipt=rr,observed_at=now())
 e.emit(out/'snapshot.json',value);return value

def claim(root,aid,copyback):
 root=Path(root)
 with lock(root):
  out,a=attempt(root,aid)
  guard_source(root,a['source']['sha256']);check_lease(out)
  if (out/'claim.json').exists():raise ValueError('already claimed: observe identity; never click twice')
  # Refuse an unplanned attempt and an editor capture copied from the source path.
  pending=next_slot(root)
  if pending is None or pending[1]['attempt_id']!=aid:raise ValueError('not current serial slot')
  cp=Path(copyback).resolve()
  if cp==Path(a['source']['path']).resolve() or e.raw_record(cp)['sha256']!=a['source']['sha256']:raise ValueError('independent editor copyback mismatch')
  e.campaign.immutable(out/'editor-copyback.cpp',cp.read_bytes())
  before=snapshot(out/'before')
  unfinished=before['user'].get('percentage')!=100 or not before['battles'] or not all(g.get('done') is True for g in before['battles'])
  restoring=(root/'STOP.json').exists() and a['source']['sha256']==config(root)['control']['source']['sha256']
  if unfinished and not restoring:raise ValueError('existing calibration unfinished')
  value=dict(plan=e.raw_record(out/'plan.json'),copyback=e.raw_record(out/'editor-copyback.cpp'),before=e.raw_record(out/'before/snapshot.json'),claimed_at=now(),clicks_permitted=1,retry_click_allowed=False)
  e.emit(out/'claim.json',value);return value

def submission_identity(before,after):
 old=before['user'];new=after['user'];agent=new['agentId'];session=new.get('testSessionHandle')
 if not isinstance(agent,int) or agent<=0 or not isinstance(session,str) or not session:raise ValueError('invalid server identity')
 previous={p['submissionId'] for g in before['battles'] for p in g['players'] if p.get('playerAgentId')==old['agentId']}
 matching=[p for g in after['battles'] for p in g['players'] if p.get('playerAgentId')==agent]
 candidates={p['submissionId'] for p in matching}-previous
 if len(candidates)!=1:raise ValueError('new submission identity not singular: observe only')
 submission=candidates.pop()
 if any(p.get('userId')!=USER or p.get('testSessionHandle')!=session or p.get('submissionId')!=submission for p in matching):raise ValueError('mixed owner/session/submission')
 # Repeated submissions from one editor keep its testSessionHandle. The fresh
 # submission ID, matching owner and observed activation bind the new attempt.
 if new['codingamer']['userId']!=USER:raise ValueError('wrong owner')
 return dict(agent_id=agent,submission_id=submission,test_session_handle=session,user_id=USER)

def attest(root,aid):
 root=Path(root)
 with lock(root):
  out,a=attempt(root,aid)
  if (out/'submission.json').exists():return e.read(out/'submission.json')
  claim=e.read(out/'claim.json')
  if claim['plan']!=e.raw_record(out/'plan.json') or e.raw_record(e.verify(claim['copyback']))['sha256']!=a['source']['sha256']:raise ValueError('claim or copyback changed')
  before=e.read(e.verify(claim['before']))
  number=len(list((out/'identity').glob('*/snapshot.json')));after=snapshot(out/'identity'/f'{number:04d}')
  ids=submission_identity(before,after)
  protocol_path=out/'identity-producers'/hashlib.sha256(Path(__file__).read_bytes()).hexdigest();e.campaign.immutable(protocol_path,Path(__file__).read_bytes())
  value=dict(**ids,identity_protocol='fresh-submission-owner-session-v1',identity_producer=e.raw_record(protocol_path),source_sha256=a['source']['sha256'],attempt_id=aid,claim=e.raw_record(out/'claim.json'),attested_at=now(),observation=e.raw_record(out/'identity'/f'{number:04d}'/'snapshot.json'))
  e.emit(out/'submission.json',value);atomic(root/'active.json',value);return value

def validate_window(a,submission,snap):
 me=snap['user'];games=snap['battles'];ids=[]
 if me['agentId']!=submission['agent_id'] or me.get('testSessionHandle')!=submission['test_session_handle']:raise ValueError('active source/session changed')
 for g in games:
  focus=[p for p in g['players'] if p.get('playerAgentId')==submission['agent_id']]
  if len(focus)!=1 or focus[0]['submissionId']!=submission['submission_id'] or focus[0].get('testSessionHandle')!=submission['test_session_handle']:raise ValueError('mixed source window')
  ids.append(g['gameId'])
 if len(set(ids))!=len(ids):raise ValueError('duplicate game')
 complete=me.get('percentage')==100 and all(g.get('done') is True for g in games)
 if complete and len(games)!=a['expected_games']:raise ValueError('unexpected completed window size')
 return complete,sorted(ids)

def observe(root,aid):
 with lock(root):
  out,a=attempt(root,aid);sub=e.read(out/'submission.json')
  number=len(list((out/'observations').glob('*/snapshot.json')));directory=out/'observations'/f'{number:04d}';snap=snapshot(directory)
  complete,ids=validate_window(a,sub,snap)
  status=dict(attempt_id=aid,complete=complete,percentage=snap['user'].get('percentage'),done=sum(g.get('done') is True for g in snap['battles']),games=len(ids),observed_at=snap['observed_at'],snapshot=e.raw_record(directory/'snapshot.json'))
  atomic(out/'progress.json',status)
  if complete and not (out/'completed.json').exists():
   e.emit(out/'completed.json',dict(**status,game_ids=ids,completed_rank=snap['user']['rank'],score=snap['user']['score'],population=snap['board']['count'],definition='First observed completed window, never best observed rank.'))
  return status

def close(root,aid):
 root=Path(root)
 with lock(root):
  out,a=attempt(root,aid)
  if (out/'result.json').exists():return e.read(out/'result.json')
  completed=e.read(out/'completed.json');sub=e.read(out/'submission.json')
  collector=c.arena.ArenaBatchCollector(repository=ROOT,data_root=out/'archive',maximum_workers=1,exclusion_registry_path=Path(a['registry']['path']),exclusion_registry_sha256=a['registry']['sha256'],api=c.shared.PublicApi(maximum_attempts=4))
  binding=collector.bind_source(agent_id=sub['agent_id'],submission_id=sub['submission_id'],source_path=e.verify(a['source']),expected_source_sha256=a['source']['sha256'])
  receipt=collector.collect(run_id=aid,binding=binding,expected_games=a['expected_games'])
  path=Path(receipt['manifest_path']);path=path if path.is_absolute() else ROOT/path
  manifest=c.arena.validate_export_manifest(path,a['registry']['sha256'])
  if sorted(g['record']['game_id'] for g in manifest['games'])!=completed['game_ids']:raise ValueError('archive differs from first completed window')
  cov=manifest['coverage']
  if cov['focus_operational_failures']:
   atomic(root/'STOP.json',dict(reason='own-operational-failure',attempt_id=aid,manifest=e.raw_record(path),restore_required=True))
  if not cov['full_window_accounted'] or cov['accepted_games']!=a['expected_games']:raise ValueError('archive incomplete or protected outcomes unavailable')
  rows=[]
  for g in manifest['games']:
   r=g['record'];rows.append(dict(game_id=r['game_id'],source_sha256=a['source']['sha256'],attempt_id=aid,turns=r['replay']['valid_turns'],transcript=r['replay']['valid_transcript'],operational=r['operational'],focus=r['focus'],opponent=r['opponent'],training_eligible=False,confirmation_eligible=False))
  e.emit(out/'research-games.json',dict(manifest=e.raw_record(path),games=rows))
  e.emit(out/'exposure.json',dict(schema=SCHEMA,mode='live-research-trajectories',input=e.raw_record(out/'research-games.json'),training_eligible=False,confirmation_eligible=False))
  result=dict(schema=SCHEMA,attempt_id=aid,source_sha256=a['source']['sha256'],status='complete',percentage=100,completed_rank=completed['completed_rank'],coverage=cov,manifest=e.raw_record(path),submission=e.raw_record(out/'submission.json'),completed=e.raw_record(out/'completed.json'),closed_at=now(),use='research unless part of a predeclared qualification block')
  e.emit(out/'result.json',result)
  if cov['focus_operational_failures']:atomic(root/'STOP.json',dict(reason='own-operational-failure',attempt=e.raw_record(out/'result.json'),restore_required=True))
  return result

def abort_failed_packet(root):
 root=Path(root)
 with lock(root):
  stop=e.read(root/'STOP.json');failed=e.read(e.verify(stop.get('attempt',stop.get('manifest'))))
  if not failed['coverage']['focus_operational_failures']:raise ValueError('no verified own failure')
  p=packets(root)[-1];plan=e.read(p)
  value=dict(plan=e.raw_record(p),status='aborted-own-failure',failure=stop.get('attempt',stop.get('manifest')),uncompleted_attempts=[s['attempt_id'] for s in plan['slots'] if not (root/'attempts'/s['attempt_id']/'result.json').exists()],restore_required=True,closed_at=now())
  e.emit(p.parent/'result.json',value);return value

def finish_packet(root):
 from tools import rank_two_campaign_v2 as policy
 root=Path(root)
 with lock(root):
  p=packets(root)[-1];plan=e.read(p);windows=[e.read(root/'attempts'/s['attempt_id']/'result.json') for s in plan['slots']]
  assessment=policy.assess_live_windows(windows,source_sha256=plan['slots'][0]['source']['sha256'] if plan['kind']!='candidate_research' else plan['slots'][1]['source']['sha256'],attempt_ids=[s['attempt_id'] for s in plan['slots']],purpose=plan['kind'])
  value=dict(plan=e.raw_record(p),windows=[e.raw_record(root/'attempts'/s['attempt_id']/'result.json') for s in plan['slots']],assessment=assessment,restore_required=plan['kind']=='candidate_research',closed_at=now())
  e.emit(p.parent/'result.json',value);return value

def status(root):
 root=Path(root);p=packets(root);pending=next_slot(root)
 aid=pending[1]['attempt_id'] if pending else None
 out=root/'attempts'/aid if aid else None
 stage=('prepare' if not (out/'plan.json').exists() else 'claim' if not (out/'claim.json').exists() else 'attest' if not (out/'submission.json').exists() else 'observe' if not (out/'completed.json').exists() else 'close') if out else ('finish-packet' if p and not (p[-1].parent/'result.json').exists() else 'new-packet')
 return dict(next_stage=stage,attempt_id=aid,packet=str(p[-1]) if p else None,active=e.read(root/'active.json') if (root/'active.json').exists() else None,stopped=(root/'STOP.json').exists())

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);sub=p.add_subparsers(dest='cmd',required=True)
 sub.add_parser('status');sub.add_parser('finish-packet');sub.add_parser('abort-failed-packet')
 x=sub.add_parser('lease');x.add_argument('--attempt',required=True)
 x=sub.add_parser('packet');x.add_argument('--kind',choices=['baseline_research','candidate_research','qualification'],required=True);x.add_argument('--candidate',type=Path)
 x=sub.add_parser('prepare');x.add_argument('--registry',type=Path,required=True)
 for name in ['claim','attest','observe','close']:
  x=sub.add_parser(name);x.add_argument('--attempt',required=True)
  if name=='claim':x.add_argument('--copyback',type=Path,required=True)
 a=p.parse_args()
 if a.cmd=='packet':result=packet(a.root,a.kind,a.candidate)
 elif a.cmd=='prepare':result=prepare(a.root,a.registry)
 elif a.cmd=='claim':result=claim(a.root,a.attempt,a.copyback)
 elif a.cmd in ['attest','observe','close']:result=globals()[a.cmd](a.root,a.attempt)
 elif a.cmd=='finish-packet':result=finish_packet(a.root)
 elif a.cmd=='abort-failed-packet':result=abort_failed_packet(a.root)
 elif a.cmd=='lease':result=lease(a.root,a.attempt)
 else:result=status(a.root)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
