"""Independent feature, inference and sanitizer checks on synthetic cases."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys

from tools import rank_two_focused_training_v1 as training
from tools import rank_two_focused_native_v2 as native
from tools import rank_two_focused_export_v1 as exporter
from tools import rank_two_focused_campaign_v1 as campaign
from tools import jacek_replay_features as rules

np=training.np


def cases(seed=2026100713,count=80):
    rng=random.Random(seed);result=[];proposals=[]
    while len(result)<count:
        state=rules.ReplayState();turns=[]
        for _ in range(160):
            if len(result)>=count:break
            result.append(dict(prefix="/".join(turns) or "-",mover=state.to_move,
                               terminal=state.winner is not None,
                               active=list(rules.encode_active(state)) if state.winner is None else None))
            if state.winner is not None:break
            mover=state.to_move;action=""
            while state.winner is None and state.to_move==mover:
                alternatives=[]
                for direction in range(8):
                    child=copy.deepcopy(state)
                    try:rules.apply_primitive(child,direction)
                    except ValueError:continue
                    alternatives.append((direction,child))
                if not alternatives:raise ValueError("nonterminal synthetic position has no move")
                proposals.append(dict(prefix="/".join(turns),partial=action,
                                      directions=[value[0] for value in alternatives]))
                direction,state=alternatives[rng.randrange(len(alternatives))];action+=str(direction)
            turns.append(action)
    return result,proposals


def tensors(runtime):
    values,_=exporter.validate(runtime);shape=runtime["architecture"]
    architecture=training.Architecture(runtime["profile"],shape[1],shape[2])
    flat=np.asarray(values,dtype=np.int8);first=shape[0]*shape[1];second=first+shape[1]*shape[2]
    integer=dict(w1=flat[:first].reshape(shape[0],shape[1]),w2=flat[first:second].reshape(shape[1],shape[2]),w3=flat[second:])
    quantized=training.core.QuantizedWeights(integer,{name:np.float32(scale) for name,scale in zip(("w1","w2","w3"),runtime["scales"],strict=True)})
    return architecture,quantized


def verify(directory,preflight):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    examples,proposals=cases()
    campaign.immutable(directory/"CASES.json",dict(seed=2026100713,synthetic=True,training_eligible=False,
                                                 cases=examples,proposals=proposals))
    comparisons={};nonterminal=[row for row in examples if not row["terminal"]]
    for profile in exporter.PROFILES:
        out=Path(preflight)/profile;runtime=campaign.read(out/"runtime.json");report=campaign.read(out/"EXPORT.json")
        source=(out/"submission.cpp").read_text();architecture,quantized=tensors(runtime)
        normal=native.build(source,report,directory/profile/"normal")
        active=[np.asarray(row["active"],dtype=np.intp) for row in nonterminal]
        expected,_=training.forward(quantized.effective(),architecture,active,quantized)
        answers=native.query(normal,["eval "+str(len(row))+" "+" ".join(map(str,row)) for row in active])
        actual=np.asarray([float(row) for row in answers],dtype=np.float32)
        if not np.allclose(expected,actual,rtol=0,atol=1e-6):raise ValueError("independent native inference differs: "+profile)
        features=native.query(normal,["features "+row["prefix"] for row in nonterminal])
        if [list(map(int,row.split())) for row in features]!=[row["active"] for row in nonterminal]:
            raise ValueError("independent mover-relative features differ: "+profile)
        requests=["state "+row["prefix"] for row in examples]
        normal_states=native.query(normal,requests)
        if any(row.startswith("ERROR") for row in normal_states):raise ValueError("native state/make-unmake failure")
        for case,answer in zip(examples,normal_states,strict=True):
            mover,terminal,count=map(int,answer.split())
            if mover!=case["mover"] or terminal!=int(case["terminal"]):raise ValueError("native rules/perspective differs")
        sanitized=native.build(source,report,directory/profile/"sanitized",sanitize=True)
        sanitized_states=native.query(sanitized,requests)
        if normal_states!=sanitized_states:raise ValueError("sanitized rule state differs")
        # Both transports exercise the same independent decoder, including tails.
        transport_requests=[];transport_expected=[]
        import base64
        for length in range(1,18):
            raw=bytes(range(length));text=base64.b85encode(raw,pad=True).decode()
            transport_requests.append("decode "+str(length)+" "+text);transport_expected.append(list(raw))
        decoded=native.query(sanitized,transport_requests)
        if [list(map(int,row.split())) for row in decoded]!=transport_expected:raise ValueError("native Base85 tail differs")
        errors=native.query(sanitized,["decode 4 ~~~~~","decode 4 0000\"","decode 1 VPa!s"])
        if any(not row.startswith("ERROR") for row in errors):raise ValueError("malformed Base85 accepted")
        comparisons[profile]=dict(inference_queries=len(active),feature_queries=len(nonterminal),
                                  rule_queries=len(examples),sanitizers=True,
                                  maximum_inference_error=float(np.max(np.abs(actual-expected))))
    result=dict(schema=campaign.SCHEMA+".independent-verification",passed=True,profiles=comparisons,
                both_orientations=sorted({row["mover"] for row in examples})==[0,1],
                cases=campaign.record(directory/"CASES.json"),new_games=0,
                whole_response_timing_certified=False,playing_strength_qualified=False)
    if not result["both_orientations"]:raise ValueError("missing orientation")
    campaign.immutable(directory/"RESULT.json",result)
    campaign.immutable(directory/"EXPOSURE_PENDING.json",dict(schema=campaign.SCHEMA+".synthetic-exposure",
                        cases=result["cases"],recipe=campaign.record(__file__),training_eligible=False,
                        before_next_fresh_bank=True))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--preflight",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(verify(args.output,args.preflight)))


if __name__=="__main__":main()
