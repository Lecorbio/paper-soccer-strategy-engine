#!/usr/bin/env python3
"""Lossless, memory-mapped storage for exhaustive teacher ranking groups.

Every rich source row passes the maintained validator before being transcribed.
No successors, proofs, transcripts, values, or work budgets are discarded.
"""
from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import tempfile
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import jacek_replay_corpus as corpus
from tools import compact_value_bfm_train as trainer
import numpy as np

EXECUTING_SOURCE_BYTES=Path(__file__).read_bytes()
SCHEMA='papersoccer.compact-value-bfm-ranking-store.v2'
SUCCESSOR_DTYPE=np.dtype([
    ('identity','V32'),('value','<f8'),('mover','u1'),('solved','u1'),('winner','i1'),
    ('visits','<u8'),('selection_visits','<u8'),('active_begin','<u8'),('active_end','<u8'),
    ('transcript_begin','<u8'),('transcript_end','<u8'),
])


def build_store(sources: Sequence[Path], output: Path, bundle):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    index=output/'index.json';source_records=[campaign.record(p) for p in sources]
    if index.exists():
        document=campaign.read(index)
        if document['sources']!=source_records or document['source_bundle_body_sha256']!=bundle.body_sha256:
            raise ValueError('ranking-store resume source changed')
        for record in document['arrays'].values(): campaign.verify(record)
        return index
    groups=[];seen=set();first=None;successor_count=0;active_count=0;transcript_count=0
    import json
    with tempfile.TemporaryDirectory(prefix='ranking-store-',dir=output) as temporary:
        temporary=Path(temporary)
        with (temporary/'indices').open('wb') as indices, (temporary/'successors').open('wb') as successors, (temporary/'transcripts').open('wb') as transcripts:
            for source in sources:
                with Path(source).open('r',encoding='utf-8') as stream:
                    for line_number,line in enumerate(stream,1):
                        row=corpus.validate_complete_turn_action_group(json.loads(line))
                        group=row['group']
                        if not group['successors_exhaustive']: raise ValueError('nonexhaustive store group')
                        if group['group_id'] in seen: raise ValueError('duplicate/cross-split ranking group')
                        seen.add(group['group_id'])
                        if row['source_bundle_body_sha256']!=bundle.body_sha256: raise ValueError('ranking bundle changed')
                        immutable={key:row[key] for key in ('feature_schema','source_bundle_body_sha256','teacher','ranking')}
                        if first is None:
                            first=immutable
                            core={key:row['teacher'][key] for key in ('artifact_sha256','payload_sha256','feature_schema_sha256')}
                            if trainer._validate_teacher_identity(bundle,core)!=core: raise ValueError('teacher identity changed')
                        elif immutable!=first: raise ValueError('mixed ranking identities')
                        start=successor_count
                        for item in group['successors']:
                            active=np.asarray(item['active'],dtype='<u2');text=item['transcript'].encode('ascii')
                            metadata=np.zeros(1,dtype=SUCCESSOR_DTYPE)
                            metadata['identity'][0]=np.void(bytes.fromhex(item['successor_id']))
                            metadata['value'][0]=item['teacher_value'];metadata['mover'][0]=item['value_mover']
                            metadata['solved'][0]=int(item['proof']['solved'])
                            metadata['winner'][0]=-1 if item['proof']['proven_winner'] is None else item['proof']['proven_winner']
                            metadata['visits'][0]=item['visits'];metadata['selection_visits'][0]=item['selection_visits']
                            metadata['active_begin'][0]=active_count;active_count+=len(active);metadata['active_end'][0]=active_count
                            metadata['transcript_begin'][0]=transcript_count;transcript_count+=len(text);metadata['transcript_end'][0]=transcript_count
                            indices.write(active.tobytes());transcripts.write(text);successors.write(metadata.tobytes());successor_count+=1
                        groups.append({'split':row['split'],'begin':start,'end':successor_count,
                            'group':{key:value for key,value in group.items() if key!='successors'},
                            'source_ordinal':len(groups),'source_file':str(Path(source).resolve()),'source_line':line_number})
            for handle in (indices,successors,transcripts): handle.flush();os.fsync(handle.fileno())
        if first is None: raise ValueError('empty ranking store')
        arrays={}
        for name in ('indices','successors','transcripts'):
            file=temporary/name;destination=output/(campaign.sha(file)+f'.{name}.bin')
            arrays[name]=campaign.copy_checked(file,destination)
        builder=output/(campaign.hashlib.sha256(EXECUTING_SOURCE_BYTES).hexdigest()+'.builder.py')
        campaign.once(builder,EXECUTING_SOURCE_BYTES)
        campaign.seal(index,{'schema':SCHEMA,**first,'sources':source_records,'arrays':arrays,
            'groups':groups,'successor_count':successor_count,'active_count':active_count,
            'transcript_bytes':transcript_count,'successor_record_bytes':SUCCESSOR_DTYPE.itemsize,
            'all_rich_rows_validated':True,'all_successors_preserved':True,
            'builder':campaign.record(builder),'protected_tests_opened':False})
    return index


class MappedSuccessors(Sequence):
    def __init__(self,store,group):
        self.store=store;self.begin=group['begin'];self.end=group['end']
        self.root_termination=group['group']['termination_reason']

    def __len__(self): return self.end-self.begin

    def __getitem__(self,index):
        if isinstance(index,slice): return tuple(self[i] for i in range(*index.indices(len(self))))
        index=int(index)
        if index<0: index+=len(self)
        if index<0 or index>=len(self): raise IndexError(index)
        row=self.store.metadata[self.begin+index]
        begin,end=int(row['active_begin']),int(row['active_end'])
        tbegin,tend=int(row['transcript_begin']),int(row['transcript_end'])
        solved=bool(row['solved']);winner=int(row['winner'])
        return trainer.CompleteTurnSuccessor(
            successor_id=bytes(row['identity']).hex(),active=self.store.indices[begin:end],
            teacher_value=float(row['value']),value_mover=int(row['mover']),
            evidence={'transcript':self.store.transcripts[tbegin:tend].tobytes().decode('ascii'),
                'visits':int(row['visits']),'selection_visits':int(row['selection_visits']),
                'proof':{'solved':solved,'proven_winner':None if winner<0 else winner},
                'termination':{'reason':'subtree-solved' if solved else self.root_termination,
                    'value_status':'exact-sign' if solved else 'backed-up-at-root-termination'}})


class RankingStore:
    def __init__(self,index,bundle):
        self.index=Path(index);self.document=campaign.read(index);doc=self.document
        if (doc.get('schema')!=SCHEMA or doc.get('source_bundle_body_sha256')!=bundle.body_sha256
                or doc.get('all_rich_rows_validated') is not True or doc.get('all_successors_preserved') is not True
                or doc.get('protected_tests_opened') is not False or doc.get('successor_record_bytes')!=SUCCESSOR_DTYPE.itemsize):
            raise ValueError('ranking-store contract changed')
        core={key:doc['teacher'][key] for key in ('artifact_sha256','payload_sha256','feature_schema_sha256')}
        if trainer._validate_teacher_identity(bundle,core)!=core: raise ValueError('ranking-store teacher changed')
        for source in doc['sources']: campaign.verify(source)
        files={name:campaign.verify(record) for name,record in doc['arrays'].items()}
        if (files['indices'].stat().st_size!=doc['active_count']*2
                or files['successors'].stat().st_size!=doc['successor_count']*SUCCESSOR_DTYPE.itemsize
                or files['transcripts'].stat().st_size!=doc['transcript_bytes']):
            raise ValueError('ranking-store array length changed')
        self.indices=np.memmap(files['indices'],dtype='<u2',mode='r')
        self.metadata=np.memmap(files['successors'],dtype=SUCCESSOR_DTYPE,mode='r')
        self.transcripts=np.memmap(files['transcripts'],dtype='u1',mode='r')
        if (np.any(self.metadata['active_end']>len(self.indices))
                or np.any(self.metadata['active_begin']>=self.metadata['active_end'])
                or np.any(self.metadata['transcript_end']>len(self.transcripts))
                or np.any(self.metadata['transcript_begin']>=self.metadata['transcript_end'])):
            raise ValueError('ranking-store offsets invalid')
        self.groups={'train':[],'validation':[]};seen=set();end=0
        for row in doc['groups']:
            group=row['group']
            if row['begin']!=end or row['end']<=end or row['end']>len(self.metadata): raise ValueError('ranking-store group coverage changed')
            end=row['end']
            if group['group_id'] in seen or row['split'] not in self.groups: raise ValueError('ranking-store split/group collision')
            seen.add(group['group_id'])
            self.groups[row['split']].append(trainer.CompleteTurnGroup(
                group_id=group['group_id'],parent_mover=group['parent_mover'],
                successors=MappedSuccessors(self,row),successors_exhaustive=group['successors_exhaustive'],
                evidence={key:value for key,value in group.items() if key not in ('group_id','parent_mover','successors_exhaustive')}))
        if end!=len(self.metadata): raise ValueError('ranking-store unassigned successors')

    def labels(self):
        doc=self.document
        return trainer.SuccessorRankingLabels(
            train=tuple(sorted(self.groups['train'],key=lambda g:g.group_id)),
            validation=tuple(sorted(self.groups['validation'],key=lambda g:g.group_id)),
            teacher=doc['teacher'],source_bundle_body_sha256=doc['source_bundle_body_sha256'],
            artifact_sha256=campaign.sha(self.index),body_sha256=doc['body_sha256'],artifact_schema=SCHEMA)


def scalar_samples(group):
    """Derive the unchanged 75/25 scalar targets from a validated stored parent."""
    evidence=group.evidence;source=evidence['source_binding'];mover=group.parent_mover
    teacher=corpus._direct_teacher_target(float(evidence['root_value']),mover,
        bool(evidence['root_solved']),evidence['proven_winner'])
    target=.75*teacher+.25*(1.0 if source['winner']==mover else -1.0)
    state=corpus._prefix_state(source['prefix'])
    lineage=corpus.TeacherLineage(schema=corpus.COMPLETE_TURN_ACTION_GROUP_SCHEMA,
        position_id=source['position_id'],group_id=source['group_id'],
        root_group_id=source['root_group_id'],source=source['source'],
        split=source['split'],campaign_id=source['campaign_id'])
    return tuple(corpus.LabeledSample(corpus.features.encode_active(state,reflected=reflected),
        target,1.0,lineage.root_group_id,(lineage,)) for reflected in (False,True))
