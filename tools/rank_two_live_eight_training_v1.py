"""Warm-started live-data mixture training with bounded durable retention."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess

from tools import rank_two_focused_training_v9 as retained
from tools import rank_two_focused_checkpoint_v1 as storage
from tools import compact_representation_train as reader
from tools import rank_two_focused_work_v1 as work
from tools import rank_two_live_eight_v1 as campaign
from tools import rank_two_live_eight_model_v1 as model
from tools import rank_two_focused_native_v3 as native_probe

np=retained.np


def root_masses(train_ids,live_ids):
    train_ids,live_ids=set(train_ids),set(live_ids)
    old=train_ids-live_ids
    if not old or not live_ids or not live_ids<=train_ids or len(train_ids)>2048:
        raise ValueError('two disjoint eligible root pools required')
    # Mean over all root losses equals the declared half/half objective.
    total=len(train_ids)
    return {r:total*.5/(len(live_ids) if r in live_ids else len(old)) for r in train_ids}


def update_root(parameters,architecture,optimizer,groups,scales,bits,mass=1.):
    active,target,signs,spans=retained.root_rows(groups)
    if any(g['split']!='train' for g in groups):raise ValueError('TRAIN-only update required')
    q=None if scales is None else model.quantize(parameters,scales,bits)
    prediction,cache=retained.forward(parameters,architecture,active,q)
    huber,rank,derivative=retained.objective(groups,prediction,target,signs,spans)
    derivative*=np.float32(mass)
    gradients=retained.core._network_gradients(parameters,architecture,active,cache,derivative,parameters if q is None else q.effective())
    norm=math.sqrt(sum(float(np.sum(g.astype(np.float64)**2)) for g in gradients.values()))
    if not math.isfinite(norm):raise FloatingPointError('nonfinite training gradient')
    if norm>5:
        for g in gradients.values():g*=np.float32(5/norm)
    optimizer.update(parameters,gradients)
    if any(not np.all(np.isfinite(a)) for a in (*parameters.values(),*optimizer.first.values(),*optimizer.second.values())):
        raise FloatingPointError('nonfinite optimizer state')
    return dict(huber=huber*mass,ranking=rank*mass,objective=(huber+.25*rank)*mass)


def calibration(parameters,architecture,corpus,roots,bits,masses):
    maximum=(1 << (bits-1))-1
    candidates={k:[np.float32(max(float(np.percentile(np.abs(v),p))/maximum,1e-8)) for p in (95,99,100)] for k,v in parameters.items()}
    scales={k:v[-1] for k,v in candidates.items()};trials=[]
    for iteration in range(2):
        for layer in ('w1','w2','w3'):
            choices=[]
            for candidate in candidates[layer]:
                proposed={**scales,layer:candidate};q=model.quantize(parameters,proposed,bits);loss=0.
                for root in roots:
                    groups=corpus.root('train',root)
                    if any(g['split']!='train' for g in groups):raise ValueError('calibration must use TRAIN')
                    active,*_=retained.root_rows(groups)
                    target=retained.forward(parameters,architecture,active)[0]
                    prediction=retained.forward(parameters,architecture,active,q)[0]
                    loss+=masses[root]*float(np.mean((prediction-target)**2))/len(roots)
                trials.append(dict(pass_index=iteration,layer=layer,scale=float(candidate),weighted_float_mse=loss))
                choices.append((loss,float(candidate),proposed))
            scales=min(choices,key=lambda x:(x[0],x[1]))[2]
    return scales,dict(split='train',weight_bits=bits,passes=2,percentiles=[95,99,100],trials=trials,selected={k:float(v) for k,v in scales.items()})


def validation(corpus,parameters,architecture,bits,scales=None,native=None):
    rows=[];early=0
    for root in sorted(corpus.roots['validation']):
        groups=corpus.root('validation',root);active,target,signs,spans=retained.root_rows(groups)
        q=None if scales is None else model.quantize(parameters,scales,bits)
        prediction=retained.forward(parameters,architecture,active,q)[0]
        if native is not None:
            actual=np.asarray(native(root,active),dtype=np.float32)
            if actual.shape!=prediction.shape or not np.array_equal(actual,prediction):
                raise ValueError('native/reference float predictions differ')
            prediction=actual
        huber,ranking,_=retained.objective(groups,prediction,target,signs,spans)
        regrets=[];has_early=False
        for group,span in zip(groups,spans,strict=True):
            teacher=target[span]*signs[span]
            if group['exhaustive'] and len(teacher)>1 and float(np.max(teacher)-np.min(teacher))>1e-4:
                picked=int(np.argmax(prediction[span]*signs[span]));regrets.append(float(np.max(teacher)-teacher[picked]))
                has_early|=group['edges']<=16
        early+=has_early
        if regrets:rows.append(dict(root=root,regret=sum(regrets)/len(regrets),huber=huber,ranking=ranking))
    if len(rows)!=128 or early<100:raise ValueError('independent comparable128VALID/100early validation required')
    return dict(regret=sum(r['regret'] for r in rows)/len(rows),huber=sum(r['huber'] for r in rows)/len(rows),early_roots=early,roots=rows)


def float_probe(parameters,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    arrays='\n'.join('static const std::uint32_t '+key+'[]={'+','.join(str(int(v))+'U' for v in values.ravel().view(np.uint32))+'};'
                     for key,values in parameters.items())
    source='''#include <bit>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
'''+arrays+'''
float val(std::uint32_t x){return std::bit_cast<float>(x);}
float tanh_fast(float v){if(v<-4.95F)return -1.F;if(v>4.95F)return 1.F;
const float s=v*v;return v*(135135.F+s*(17325.F+s*(378.F+s)))/(135135.F+s*(62370.F+s*(3150.F+28.F*s)));}
int main(){std::string cmd;unsigned n;while(std::cin>>cmd>>n){
if(cmd!="eval"||!n||n>421)return 1;float first[8]={},second[8]={};
for(unsigned k=0;k<n;++k){unsigned index;if(!(std::cin>>index)||index>=6301)return 1;
for(unsigned h=0;h<8;++h)first[h]=first[h]+val(w1[index*8+h]);}
for(float&v:first)v=v<0?.01F*v:v*v;
for(unsigned i=0;i<8;++i)for(unsigned h=0;h<8;++h){volatile float term=first[i]*val(w2[i*8+h]);second[h]=second[h]+term;}
for(float&v:second)v=v<0?.01F*v:v;float out=0;
for(unsigned h=0;h<8;++h){volatile float term=second[h]*val(w3[h]);out=out+term;}
std::cout<<std::setprecision(9)<<tanh_fast(out)<<'\\n';}}
'''
    cpp=directory/'float.cpp';binary=directory/'float-probe';campaign.immutable(cpp,source.encode())
    result=subprocess.run(['/usr/bin/clang++','-std=c++20','-O3','-ffp-contract=off',str(cpp),'-o',str(binary)],text=True,capture_output=True,timeout=120)
    campaign.immutable(directory/'compile.json',dict(returncode=result.returncode,stderr=result.stderr,source=campaign.record(cpp)))
    if result.returncode:raise ValueError('native float probe compile failed')
    return binary


def resumable(directory,parameters,optimizer,cursor,binding):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    head=directory/'HEAD.json';previous=campaign.read(head) if head.exists() else None
    number=1 if previous is None else previous['sequence']+1
    generation=directory/'transient'/f'{number:08d}'
    receipt=storage.checkpoint(generation/'state.psc',parameters,optimizer,cursor,binding,parent=None)
    reference=campaign.record(generation/'state.json')
    journal=directory/'journal'/f'{number:08d}.json'
    campaign.immutable(journal,dict(sequence=number,cursor=cursor,parameter_sha256=receipt['parameter_sha256'],
                                  raw_sha256=receipt['storage']['raw_sha256'],optimizer_step=receipt['optimizer_step']))
    campaign.atomic(head,dict(sequence=number,receipt=reference,journal=campaign.record(journal)))
    # Only this writer's obsolete transient generation is disposable. Epoch
    # checkpoints are independent full states, never parents of transient data.
    if previous:
        old=Path(previous['receipt']['path']).parent
        if old.parent!=directory/'transient' or old==generation:raise ValueError('transient retention boundary differs')
        shutil.rmtree(old)
    return receipt


def validate_plan(plan,corpus,state):
    if (plan['producer']!=campaign.record(__file__) or plan['profile']!='dd8'
            or plan['weight_bits'] not in (4,6,7) or plan['teacher_nodes'] not in (256000,1000000)):
        raise ValueError('training recipe/producer differs')
    for ref in plan['inputs'].values():campaign.verify(ref)
    evidence=campaign.read(campaign.verify(plan['fresh_validation']))
    if (not evidence.get('passed') or evidence.get('roots')!=128 or evidence.get('early_roots',0)<100
            or evidence.get('ancestry')!=state.get('live_eight_ancestry_validated')):
        raise PermissionError('fresh validation ancestry receipt required')
    if len(corpus.roots['validation'])!=128 or len(corpus.roots['train'])>2048:
        raise ValueError('declared root bounds differ')
    roots=set(corpus.roots['train']);live_roots=set(plan['live_roots']);masses=root_masses(roots,live_roots)
    provenance=campaign.read(campaign.verify(plan['live_training_provenance']))
    if (not provenance.get('passed') or set(provenance['roots'])!=live_roots
            or not provenance.get('only_predesignated_new_exploration')
            or provenance.get('protected_games_read') is not False):
        raise PermissionError('live TRAIN provenance incomplete')
    campaign.resource_guard(plan['resource_budget'])
    return masses


def fit(plan_path):
    plan=campaign.read(plan_path);state=work.checked(plan,'train',launch=True)
    corpus=reader.Corpus(campaign.verify(plan['corpus']));masses=validate_plan(plan,corpus,state)
    out=Path(plan['output']);binding=campaign.record(plan_path)
    architecture,parameters=retained.initialize('dd8',plan['seed'])
    optimizer=retained.core.AdamW(parameters,learning_rate=.001,weight_decay=1e-5)
    parent=campaign.read(campaign.verify(plan['float_parent']))
    if parent['cursor']['phase']!='float':raise ValueError('common useful float parent required')
    storage.restore(parent,parameters,optimizer,parent['plan'])
    if (out/'resume/HEAD.json').exists():
        head=campaign.read(out/'resume/HEAD.json');latest=campaign.read(campaign.verify(head['receipt']))
        storage.restore(latest,parameters,optimizer,binding)
    roots=sorted(corpus.roots['train']);best_float=None;selected=None
    for phase,minimum,maximum,rate in [('float',10,plan.get('float_maximum',80),.001),('qat',8,plan.get('qat_maximum',32),.00025)]:
        if maximum>(80 if phase=='float' else 32) or maximum<minimum:raise ValueError('epoch cap differs')
        scales=None;history=[]
        if phase=='qat':
            storage.restore(best_float['checkpoint'],parameters,optimizer,binding)
            scales,report=calibration(parameters,architecture,corpus,roots,plan['weight_bits'],masses)
            campaign.immutable(out/'CALIBRATION.json',report)
        optimizer.learning_rate=np.float32(rate)
        receipts=[]
        for epoch in range(maximum):
            directory=out/phase/f'epoch-{epoch:02d}';receipt_path=directory/'receipt.json'
            if receipt_path.exists():
                receipt=campaign.read(receipt_path);storage.restore(receipt['checkpoint'],parameters,optimizer,binding)
            else:
                order=list(roots);order_seed=plan['seed']+epoch+(100000 if phase=='qat' else 0);random.Random(order_seed).shuffle(order)
                start=0
                if (out/'resume/HEAD.json').exists():
                    head=campaign.read(out/'resume/HEAD.json');current=campaign.read(campaign.verify(head['receipt']))
                    cursor=current['cursor']
                    if cursor['phase']==phase and cursor['epoch']==epoch:
                        if cursor['root_order']!=order:raise ValueError('resumed root order differs')
                        storage.restore(current,parameters,optimizer,binding);start=cursor['next_root']
                for index in range(start,len(order)):
                    work.checked(plan,'train')
                    update_root(parameters,architecture,optimizer,corpus.root('train',order[index]),scales,plan['weight_bits'],masses[order[index]])
                    resumable(out/'resume',parameters,optimizer,dict(phase=phase,epoch=epoch,next_root=index+1,root_order=order,rng_seed=order_seed),binding)
                checkpoint=storage.checkpoint(directory/'state.psc',parameters,optimizer,dict(phase=phase,epoch=epoch,next_root=len(order),root_order=order,rng_seed=order_seed),binding,parent=None)
                receipt=dict(phase=phase,epoch=epoch,checkpoint=checkpoint,plan=binding)
                if phase=='qat':
                    q=model.quantize(parameters,scales,plan['weight_bits'])
                    values=np.concatenate([q.integer[k].ravel() for k in ('w1','w2','w3')]).astype(int).tolist()
                    runtime=model.document('dd8',values,[float(scales[k]) for k in ('w1','w2','w3')],plan['weight_bits'],dict(plan=binding,phase=phase,epoch=epoch))
                    source,export=model.export(runtime,plan['clocks_ms'],verify_tokens=False)
                    receipt['runtime']=campaign.immutable(directory/'runtime.json',runtime)
                    receipt['source']=campaign.immutable(directory/'submission.cpp',source.encode())
                    receipt['export']=campaign.immutable(directory/'EXPORT.json',export)
                    binary=native_probe.build(source,export,directory/'native')
                else:
                    binary=float_probe(parameters,directory/'native')
                def native(root,active):
                    requests=['eval '+str(len(row))+' '+' '.join(map(str,row)) for row in active]
                    return [float(x) for x in native_probe.query(binary,requests)]
                receipt['metrics']=validation(corpus,parameters,architecture,plan['weight_bits'],scales,native)
                receipt['native']=campaign.record(binary)
                campaign.immutable(receipt_path,receipt)
            receipts.append(receipt);history.append(receipt['metrics']['regret'])
            chosen=min(receipts,key=lambda r:(r['metrics']['regret'],r['metrics']['huber'],r['epoch']))
            if phase=='float':best_float=chosen
            else:selected=chosen
            reason=retained.stopping(history,minimum,maximum)
            if reason:
                campaign.immutable(out/phase/'STOP.json',dict(reason=reason,budget_limited=reason=='budget-limited',epochs=len(receipts),selected=campaign.record(out/phase/f"epoch-{chosen['epoch']:02d}"/'receipt.json')))
                break
    result=dict(passed=True,playing_strength_qualified=False,selected=campaign.record(out/'qat'/f"epoch-{selected['epoch']:02d}"/'receipt.json'),best_float=campaign.record(out/'float'/f"epoch-{best_float['epoch']:02d}"/'receipt.json'),epochs_retained=True,root_update_journal= str(out/'resume/journal'),live_root_mass=.5,weight_bits=plan['weight_bits'])
    campaign.immutable(out/'RESULT.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();print(json.dumps(fit(a.plan)))
