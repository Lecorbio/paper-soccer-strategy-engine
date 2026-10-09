#!/usr/bin/env python3
"""Cumulative, cutoff-bound exclusion census with explicit typed input adapters."""
from __future__ import annotations
import argparse
import collections
import copy
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import re
import resource
import sys
import time
try:
    from . import top_three_played_exclusions as legacy
except ImportError:
    import top_three_played_exclusions as legacy
e,banks=legacy.e,legacy.banks
SCOPE_SCHEMA='papersoccer.top-three.played-exclusion-scope.v2'
EDGE_BITS={edge:1<<index for edge,index in e.rules.EDGE_INDEX.items()}


def pointer(value, path=''):
    for part in path.split('/'):
        if part:value=value[int(part)] if isinstance(value,list) else value[part]
    return value


class Normalizer(legacy.Census):
    """Read/verify all payloads once, then freeze action-only records before hashing states."""
    def __init__(self, root, cutoff, output):
        super().__init__(root)
        self.cutoff=cutoff
        self.cutoff_time=datetime.fromisoformat(cutoff['cutoff_utc']).timestamp()
        self.output=output
        self.rows=0
        self.skipped=[]
        self.consumed=[]
        self.bound_only=False
        self.group_prefixes=collections.defaultdict(list)

    def read(self, reference, jsonl=False, text=False):
        path=Path(reference['path'])
        path=(e.ROOT/path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root):raise ValueError('input escapes explicit current campaign')
        denied=Path(self.cutoff['excluded_current_live_details_directory']).resolve()
        if path.is_relative_to(denied):raise ValueError('current live candidate details are excluded')
        if path.stat().st_mtime>self.cutoff_time+1e-6:
            raise ValueError('observation payload was modified after the frozen cutoff: '+str(path))
        raw=path.read_bytes()
        if len(raw)>512*1024**2 or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
            raise ValueError('input hash/size mismatch')
        item={'path':str(path),'sha256':reference['sha256']}
        if str(path) in self.inputs and self.inputs[str(path)]!=item:raise ValueError('conflicting input identities')
        self.inputs[str(path)]=item
        if len(self.inputs)>10000:raise ValueError('input count bound exceeded')
        if text:return raw.decode(),item
        return ([json.loads(line) for line in raw.splitlines() if line.strip()] if jsonl else json.loads(raw)),item

    def write(self,row):
        row['training_eligible']=False
        self.output.write(banks.canonical(row))
        self.rows+=1

    def iter_jsonl(self,reference):
        path=Path(reference['path'])
        path=(e.ROOT/path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root):raise ValueError('input escapes explicit current campaign')
        if path.is_relative_to(Path(self.cutoff['excluded_current_live_details_directory']).resolve()):
            raise ValueError('current live candidate details are excluded')
        if path.stat().st_mtime>self.cutoff_time+1e-6:raise ValueError('payload is newer than cutoff')
        if path.stat().st_size>2*1024**3:raise ValueError('streamed label file exceeds bound')
        sha=hashlib.sha256()
        with path.open('rb') as stream:
            while chunk:=stream.read(1024*1024):sha.update(chunk)
        if sha.hexdigest()!=reference['sha256']:raise ValueError('streamed input hash mismatch')
        item={'path':str(path),'sha256':reference['sha256']}
        self.inputs[str(path)]=item
        with path.open() as stream:
            for line in stream:
                if line.strip():yield json.loads(line),item

    def trace(self, turns, reference, locator, kind, group=None):
        self.write({'mode':'trace','turns':turns,'input':reference,'locator':locator,'kind':kind,'group':group})

    def branches(self,prefix,actions,reference,locator,kind):
        self.write({'mode':'branches','prefix':prefix,'actions':actions,'input':reference,'locator':locator,'kind':kind})

    def protocol(self,row,reference,locator,default_player=None):
        first=row.get('first',{})
        later=row.get('later',{})
        request=first.get('request',row.get('first_request'))
        player=row.get('player',default_player)
        if request is not None:
            fields=request.split()
            if len(fields)!=3 or fields[0] not in ('0','1') or int(fields[1])!=len(fields[2]):
                raise ValueError('malformed bound first protocol request')
            player=int(fields[0]);prefix='' if player==0 and fields[2]=='-' else fields[2]
        else:
            if player not in (0,1):raise ValueError('protocol lacks recoverable player')
            prefix='' if player==0 else row.get('first_opponent','0')
        action=first.get('action',row.get('first_action'))
        if not action:
            self.skipped.append({'input':reference,'locator':locator,'reason':'no first action returned'})
            self.trace(self.split(prefix),reference,locator+'/input','protocol-input')
            return
        turns=[*self.split(prefix),action]
        self.trace(turns,reference,locator+'/first','native-protocol-first')
        later_request=later.get('request',row.get('later_request'))
        if later_request is not None:
            fields=later_request.split()
            if len(fields)!=2 or int(fields[0])!=len(fields[1]):raise ValueError('malformed later request')
            incoming=fields[1]
        else:incoming=row.get('later_opponent')
        later_action=later.get('action',row.get('later_action'))
        if incoming is not None:
            turns.append(incoming)
            if later_action:turns.append(later_action)
            self.trace(turns,reference,locator+'/later','native-protocol-later')
        for field in ('warm_search_action','core_first_action','discarded_first_action'):
            if row.get(field):self.trace([*self.split(prefix),row[field]],reference,locator+'/'+field,'protocol-discarded-search')

    def entry(self,spec):
        kind=spec['kind']
        if kind=='teacher-jsonl':
            for i,(row,reference) in enumerate(self.iter_jsonl(spec['input'])):
                group=row['group'];prefix='/'.join(t['action'] for t in group['source_binding']['prefix'])
                if prefix not in self.group_prefixes[group['group_id']]:self.group_prefixes[group['group_id']].append(prefix)
                mover=e.state(prefix).to_move
                actions=[child['transcript'] for child in group['successors']]
                if mover==1:actions=[''.join(str((int(c)+4)%8) for c in action) for action in actions]
                for index in sorted({0,len(actions)-1}) if actions else []:
                    child=e.state(prefix)
                    e.rules.apply_complete_turn(child,child.to_move,actions[index])
                    if list(e.rules.encode_active(child))!=group['successors'][index]['active']:
                        raise ValueError('teacher action rotation does not match bound successor features')
                self.branches(prefix,actions,reference,str(i),spec.get('role','supervised-counterfactual'))
                if (i+1)%1000==0:
                    print(json.dumps({'stage':'normalizing-teacher','groups':i+1,'input':reference['path'],
                                      'cpu_seconds':time.process_time()}),flush=True)
            return
        if kind in {'experiment-result','pair-result','experiment-root','pair-root','native-game','native-bundle',
                    'native-panel','opening-game','transcripts','transcripts-jsonl','public-details','decisions','opening-probes-jsonl'}:
            return super().entry(spec)
        payload,reference=self.read(spec['input'],jsonl=kind.endswith('-jsonl'),text=kind in ('prefix-text','records-sequence'))
        if kind=='records-sequence':
            records=[];decoder=json.JSONDecoder();remaining=payload.lstrip()
            while remaining:
                record,end=decoder.raw_decode(remaining);records.append(record);remaining=remaining[end:].lstrip()
            payload=records;kind='records'
        node=pointer(payload,spec.get('pointer',''))
        if kind=='metadata':return
        if kind=='prefix-text':
            self.trace(self.split(node.strip()),reference,'','fixture-prefix');return
        if kind=='arena-manifest':
            if node.get('binding',{}).get('agent_id')==self.cutoff['excluded_current_agent_id']:
                raise ValueError('current candidate arena is excluded')
            for item in node['games']:
                child,ref=self.read({'path':item['record_path'],'sha256':item['record_sha256']})
                if child!=item['record']:raise ValueError('arena embedded/archived record mismatch')
                self.trace(child['replay']['observed_turns'],ref,'/replay/observed_turns','arena-game','arena:'+str(child['game_id']))
        elif kind=='clock-result':
            for ref in node['game_artifacts']:self.entry({'kind':'trace-game','input':ref})
        elif kind=='opening-result':
            for ref in node['artifacts']:self.entry({'kind':'trace-game','input':ref})
        elif kind=='trace-game':
            game=node.get('game')
            if game is None:
                self.skipped.append({'input':reference,'reason':'no parsed game returned','failure':node.get('stderr')})
                self.trace(self.split(node.get('entry',{}).get('prefix','')),reference,'/entry','failed-game-input')
            elif 'actions' in game:self.native(node,reference,'/game')
            else:self.trace(self.split(game['transcript']),reference,'/game','development-game',game.get('cluster_id'))
            for i,row in enumerate(node.get('persistent_telemetry',[])):
                self.decision(row,reference,f'/persistent_telemetry/{i}', '' if row['kind']=='unscored-startup' else None)
        elif kind=='platform-comparison':
            for i,row in enumerate(node['games']):
                record,ref=self.read(row['verified'])
                self.trace(self.split(record['transcript']),ref,'/transcript','private-platform-game')
                if row.get('warm_search_action'):
                    self.trace([row['warm_search_action']],reference,f'/games/{i}/warm_search_action','discarded-search')
        elif kind=='protocol':
            for i,row in enumerate(node if isinstance(node,list) else [node]):self.protocol(row,reference,str(i))
            if spec.get('diagnostic_prefix'):
                prefix,ref=self.read(spec['diagnostic_prefix'],text=True)
                for i,row in enumerate(node if isinstance(node,list) else [node]):
                    for j,action in enumerate(re.findall(r'R4T [^\n]* move=([0-7]+)',row.get('stderr',''))):
                        self.trace([*self.split(prefix.strip()),action],reference,f'{i}/stderr/{j}','virtual-fixed-prefix-probe')
        elif kind=='safety':
            for i,row in enumerate(node.get('native_protocol',[])):self.protocol(row,reference,f'/native_protocol/{i}')
            for i,row in enumerate(node.get('persistent_stress',[])):self.decision(row,reference,f'/persistent_stress/{i}')
            if node.get('worker_startup'):self.decision(node['worker_startup'],reference,'/worker_startup','')
            if isinstance(node.get('full_games'),list):
                for row in node['full_games']:self.entry({'kind':'native-game','input':row['record']})
        elif kind in ('records','records-jsonl'):
            values=node if isinstance(node,list) else [node]
            mapping={}
            if spec.get('fixtures'):
                fixture_data,_=self.read(spec['fixtures']['input'],text=spec['fixtures'].get('format')=='tsv')
                if spec['fixtures'].get('format')=='tsv':
                    for line in fixture_data.splitlines():
                        if not line or line.startswith('#'):continue
                        parts=line.split('\t');mapping[parts[0]]={'prefix':parts[spec['fixtures'].get('prefix_column',1)]}
                else:
                    for row in pointer(fixture_data,spec['fixtures'].get('pointer','')):mapping[row['id']]=row
            default=spec.get('default_prefix')
            if spec.get('prefix_input'):default=self.read(spec['prefix_input'],text=True)[0].strip()
            for i,row in enumerate(values):
                if spec.get('skip_without_action') and not row.get('action'):continue
                if mapping:
                    fixture=mapping[row.get(spec.get('id_field','id'))]
                    if fixture.get('mutation',0)!=0:
                        self.skipped.append({'input':reference,'locator':str(i),'reason':'explicit unreachable guard mutation'})
                        continue
                    prefix=fixture['prefix']
                else:prefix=row.get(spec.get('prefix_field','prefix'),default)
                if prefix is None:raise ValueError('record lacks bound prefix')
                fields=spec.get('action_fields',['action'])
                actions=[]
                for field in fields:
                    try:value=pointer(row,field)
                    except KeyError:continue
                    if value:actions.append(value)
                if actions:self.branches(prefix,actions,reference,str(i),spec.get('role','returned-counterfactual'))
                else:self.trace(self.split(prefix),reference,str(i),'fixture-prefix')
                if spec.get('reflected_prefix_field'):
                    self.trace(self.split(row[spec['reflected_prefix_field']]),reference,str(i)+'/reflected','reflected-fixture')
        elif kind=='summary-native-actions':
            fixtures={}
            for item in spec['fixtures']:
                values,_=self.read(item)
                fixtures.update({r['id']:r for r in values['fixtures']})
            for field in ('native_root_decisions','native_alternatives'):
                for i,row in enumerate(node[field]):
                    self.branches(fixtures[row['fixture']]['prefix'],[row['action']],reference,f'/{field}/{i}','bounded-counterfactual')
        elif kind=='teacher-summary':
            for i,row in enumerate(node['results']):
                for field in spec.get('bind_fields',[]):
                    if row.get(field):self.read(row[field])
                self.branches(row['fixture']['prefix'],[a['action'] for a in row['actions']],reference,str(i),'teacher-counterfactual')
        elif kind=='teacher-result':
            for i,row in enumerate(node['results']):
                if row.get('returncode')!=0 or row.get('timeout'):
                    self.skipped.append({'input':reference,'locator':str(i),'reason':'failed teacher attempt retained',
                                         'returncode':row.get('returncode'),'timeout':row.get('timeout')})
                self.entry({'kind':'teacher-jsonl','input':row['output']})
        elif kind=='one-ply-jsonl':
            try:from .top_three_proof_aware import physical_key
            except ImportError:from top_three_proof_aware import physical_key
            for i,row in enumerate(node):
                for name,probe in row['probes'].items():
                    prefixes=[p for p in self.group_prefixes[row['group_id']] if physical_key(e.state(p))==probe['root_key']]
                    if not prefixes:raise ValueError('native one-ply root does not match any bound teacher prefix')
                    self.branches(prefixes[0],[a['action'] for a in probe['actions']],reference,f'{i}/{name}','native-one-ply-aliases')
        elif kind=='profile-stdout':
            text,ref=self.read(spec['fixtures'],text=True)
            prefixes=[line.split('\t')[spec.get('prefix_column',2)] for line in text.splitlines() if line and not line.startswith('#')]
            actions=[line.split('\t')[0] for line in node['target_stdout'].splitlines()]
            if len(actions)!=len(prefixes)*spec['passes']:raise ValueError('profile stdout schedule mismatch')
            for i,action in enumerate(actions):self.branches(prefixes[i%len(prefixes)],[action],reference,str(i),'profile-returned-action')
        else:raise ValueError('unsupported typed adapter: '+kind)


class Census(legacy.Census):
    def __init__(self,root,limits):
        super().__init__(root)
        self.physical_seen={}
        self.feature_by_state={}
        self.pilot_states=set();self.pilot_features=set()
        self.teacher_classes=collections.defaultdict(set)
        self.limits=limits
        self.started=time.process_time()

    def guard(self):
        usage=resource.getrusage(resource.RUSAGE_SELF)
        rss=int(usage.ru_maxrss if sys.platform=='darwin' else usage.ru_maxrss*1024)
        if rss>self.limits['rss_bytes']:raise MemoryError('cumulative inventory exceeded RSS cap')
        if time.process_time()-self.started>self.limits['cpu_seconds']-40:raise TimeoutError('cumulative inventory CPU reserve reached')

    def observe(self,state):
        key=(state.ball,state.to_move,state.winner,sum(EDGE_BITS[edge] for edge in state.used_segments))
        # Visited vertices are fully determined by the used segments plus the
        # initial ball. Exact multiplicities do not affect rules/features.
        if key in self.physical_seen:return self.physical_seen[key]
        canonical=e.fingerprint(state)
        self.physical_seen[key]=canonical
        if canonical not in self.states:
            feature=banks.feature_key(state)
            self.states.add(canonical);self.features.add(feature);self.feature_by_state[canonical]=feature
        if len(self.physical_seen)%2048==0:self.guard()
        return canonical

    def branches(self,row):
        prefix=row['prefix'];self.trace(self.split(prefix),row['input'],row['locator']+'/parent',row['kind']+'-parent')
        state=e.state(prefix);mover=state.to_move
        trie={}
        for action in set(row['actions']):
            cursor=trie
            for char in action:cursor=cursor.setdefault(char,{})
            cursor['$']=True
        counts=collections.Counter()
        def walk(node,path):
            if '$' in node:
                counts['complete_actions' if state.winner is not None or state.to_move!=mover else 'incomplete_actions']+=1
            for direction,child in node.items():
                if direction=='$':continue
                if direction not in '01234567' or state.winner is not None or state.to_move!=mover:
                    counts['invalid_branch_tails']+=1;continue
                ball,player,winner=state.ball,state.to_move,state.winner
                try:e.rules.apply_primitive(state,direction)
                except ValueError:
                    counts['invalid_branch_tails']+=1;continue
                target=state.ball;edge=e.rules._segment(ball,target)
                canonical=self.observe(state);counts['primitive_trie_nodes']+=1
                if row['kind']=='pilot-supervised-counterfactual':
                    category='terminal' if state.winner is not None else 'handoff' if state.to_move!=mover else 'mandatory-rebound'
                    counts['pilot_'+category+'_observations']+=1
                    feature=self.feature_by_state.get(canonical)
                    if feature is None:
                        feature=banks.feature_key(state);self.feature_by_state[canonical]=feature
                    if canonical not in self.pilot_states:self.teacher_classes[category+'_state_gap'].add(canonical)
                    if feature not in self.pilot_features:self.teacher_classes[category+'_feature_gap'].add(feature)
                    if '$' in child and category!='mandatory-rebound' and canonical not in self.pilot_states:
                        counts['pilot_final_state_missing_from_original_inventory']+=1
                    if category=='mandatory-rebound' and not e.rules._boundary(state.ball):
                        degree=sum(state.ball in edge for edge in state.used_segments)
                        if degree<2:raise ValueError('unexpected rebound interior eligible as a fresh handoff')
                walk(child,path+direction)
                state.used_segments.remove(edge)
                state.visit_count[target]-=1
                if not state.visit_count[target]:del state.visit_count[target]
                state.ball,state.to_move,state.winner=ball,player,winner
        walk(trie,'')
        assert e.fingerprint(state)==e.fingerprint(e.state(prefix))
        self.counts['counterfactual_action_rows']+=len(row['actions'])
        self.counts['counterfactual_unique_actions']+=len(set(row['actions']))
        self.counts.update(counts)
        self.ledger.append({'input':row['input'],'locator':row['locator'],'kind':row['kind'],
                            'action_rows':len(row['actions']),**counts})
        self.guard()


def normalize(scope_path,expected,output):
    scope=e.read(e.verify({'path':str(Path(scope_path).resolve()),'sha256':expected}))
    if scope['schema']!=SCOPE_SCHEMA or scope['historical_protected_reads'] is not False:raise ValueError('explicit v2 scope required')
    cutoff=e.read(e.verify(scope['cutoff']))
    output=Path(output)
    with output.open('xb') as stream:
        normal=Normalizer(scope['campaign_root'],cutoff,stream)
        for spec in scope['entries']:normal.entry(spec)
    closure={'schema':'papersoccer.top-three.normalized-exposure-closure.v1','scope':e.raw_record(scope_path),
        'cutoff':scope['cutoff'],'records':e.raw_record(output),'rows':normal.rows,
        'inputs':sorted(normal.inputs.values(),key=lambda r:r['path']),'skipped_or_unreturned':normal.skipped,
        'all_payload_inputs_hash_verified_before_census':True,'producer':e.raw_record(__file__)}
    e.emit(output.with_suffix('.closure.json'),closure)
    return closure


def export(scope_path,expected,normalized_closure,output,cpu_budget=None):
    scope=e.read(e.verify({'path':str(Path(scope_path).resolve()),'sha256':expected}))
    closure=e.read(normalized_closure)
    if closure['scope']!=e.raw_record(scope_path):raise ValueError('normalized scope changed')
    e.verify(closure['producer']);records=e.verify(closure['records'])
    cutoff=e.read(e.verify(scope['cutoff']))
    limits=dict(cutoff['resources'])
    if cpu_budget is not None:
        if not 40<cpu_budget<=limits['cpu_seconds']:raise ValueError('invalid remaining census CPU budget')
        limits['cpu_seconds']=cpu_budget
    census=Census(scope['campaign_root'],limits)
    # Production normalization and export share one process; count its entire
    # CPU use instead of resetting the budget after parsing the label stream.
    census.started=0.0
    inherited=[]
    for spec in scope['inherit']:
        item=spec['input'];path=e.verify(item)
        fn=banks.learning_inventory if spec['kind']=='learning' else banks.exposure_inventory
        values,identity=fn(path,item['sha256'])
        if spec['kind']=='learning':
            census.pilot_states=values['states'];census.pilot_features=values['features']
        census.states.update(values['states']);census.features.update(values['features']);census.groups.update(values['clusters'])
        inherited.append({'kind':spec['kind'],'input':identity,'states':len(values['states']),'features':len(values['features'])})
    initial_states,initial_features=len(census.states),len(census.features)
    sources={'exporter':e.raw_record(__file__),'legacy_adapter':e.raw_record(legacy.__file__),
             'features':e.raw_record(e.rules.__file__),'experiments':e.raw_record(e.__file__),'banks':e.raw_record(banks.__file__)}
    status='complete';error=None
    processed=0
    try:
        with records.open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['mode']=='trace':census.trace(row['turns'],row['input'],row['locator'],row['kind'],row.get('group'))
                elif row['mode']=='branches':census.branches(row)
                else:raise ValueError('unknown normalized record mode')
                census.guard()
                processed+=1
                if processed%250==0:
                    print(json.dumps({'stage':'census','processed_records':processed,'total_records':closure['rows'],
                                      'states':len(census.states),'added_states':len(census.states)-initial_states,
                                      'cpu_seconds':time.process_time()}),flush=True)
    except (MemoryError,TimeoutError) as exc:
        status='resource-stopped';error=str(exc)
    if status=='complete' and any(census.counts[name] for name in (
            'pilot_final_state_missing_from_original_inventory','invalid_branch_tails','incomplete_actions')):
        status='validation-failed';error='a returned counterfactual branch failed exact-rule/endpoint validation'
    for item in sources.values():e.verify(item)
    output=Path(output)
    classes_path=output.with_suffix('.teacher-classes.json')
    e.emit(classes_path,{'schema':'papersoccer.top-three.teacher-intermediate-coverage.v1',
        'status':status,'baseline':'original pilot inventory only, before later played-state inheritance',
        'sets':{k:sorted(v) for k,v in census.teacher_classes.items()},
        'counts':{k:len(v) for k,v in census.teacher_classes.items()},
        'endpoint_validation_failures':census.counts['pilot_final_state_missing_from_original_inventory'],
        'interpretation':'A missing interior hash is a coverage gap, not proof of prior evaluation leakage. Exact mandatory-rebound interiors are not eligible complete-turn handoff roots; feature overlap is measured separately after freeze.'})
    ledger=output.with_suffix('.ledger.jsonl')
    e.campaign.immutable(ledger,b''.join(banks.canonical(row) for row in census.ledger))
    value={'schema':legacy.SCHEMA,'extension_schema':'papersoccer.top-three.cumulative-exclusions.v2',
        'campaign_root':str(census.root),'complete_coverage':status=='complete','status':status,'error':error,
        'scope_manifest':e.raw_record(scope_path),'cutoff':scope['cutoff'],'normalized_closure':e.raw_record(normalized_closure),
        'inputs':closure['inputs']+[spec['input'] for spec in scope['inherit']], 'inherited_inventories':inherited,
        'canonical_state_sha256':sorted(census.states),'canonical_feature_sha256':sorted(census.features),
        'root_group_ids':sorted(census.groups),'state_function_sha256':hashlib.sha256(inspect.getsource(e.fingerprint).encode()).hexdigest(),
        'sources':sources,'includes_primitive_states':True,'includes_turn_boundaries':True,
        'includes_terminal_and_valid_operational_prefixes':True,
        'rejected_action_policy':'Retain every legal primitive prefix and preserve invalid/unreturned suffix records; never fabricate moves or winners.',
        'counts':dict(census.counts),'unique_transcripts':len(census.cache),'trajectory_ledger':e.raw_record(ledger),
        'last_fully_processed_normalized_row':processed,'total_normalized_rows':closure['rows'],
        'teacher_intermediate_classification':e.raw_record(classes_path),
        'inherited_unique_states':initial_states,'inherited_unique_features':initial_features,
        'added_states':len(census.states)-initial_states,'added_features':len(census.features)-initial_features,
        'cpu_seconds':time.process_time()-census.started,'peak_rss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'training_eligible':False,'historical_protected_reads':False,'games_run':0,'confirmation_banks_generated':0}
    e.emit(output,value)
    if status!='complete':raise RuntimeError(error)
    banks.exposure_inventory(output,e.raw_record(output)['sha256'],census.root)
    return {'inventory':e.raw_record(output),'states':len(census.states),'features':len(census.features),
            'counts':dict(census.counts),'cpu_seconds':value['cpu_seconds']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('normalize','export'))
    parser.add_argument('--scope',type=Path,required=True);parser.add_argument('--scope-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--normalized-closure',type=Path)
    args=parser.parse_args()
    result=normalize(args.scope,args.scope_sha256,args.output) if args.command=='normalize' else export(
        args.scope,args.scope_sha256,args.normalized_closure,args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
