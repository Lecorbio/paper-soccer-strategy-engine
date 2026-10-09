"""Isolated production-main protocol safety; mechanical games are not strength."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import select
import subprocess
import time

from tools import rank_two_focused_campaign_v1 as campaign
from tools import jacek_replay_features as rules


def compile_sources(preflight,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    builds={}
    for profile in ("dd8","dd12wide","control"):
        source=Path(preflight)/profile/"submission.cpp";directory=output/profile;directory.mkdir(exist_ok=True)
        binary=directory/"main"
        command=["/usr/bin/clang++","-std=c++20","-O3","-ffp-contract=off",str(source),"-o",str(binary)]
        process=subprocess.run(command,text=True,capture_output=True,timeout=120)
        campaign.immutable(directory/"compiler.stderr",process.stderr.encode())
        if process.returncode:raise ValueError("production main compilation: "+process.stderr[-1800:])
        builds[profile]=dict(source=campaign.record(source),binary=campaign.record(binary),flags=command[1:4])
    result=dict(compiled=True,builds=builds,compiler=campaign.record("/usr/bin/clang++"),new_games=0)
    campaign.immutable(output/"BUILD.json",result)
    return result


def random_turn(state,rng):
    mover=state.to_move;action="";proposals=[]
    while state.winner is None and state.to_move==mover:
        choices=[]
        for direction in range(8):
            child=copy.deepcopy(state)
            try:rules.apply_primitive(child,direction)
            except ValueError:continue
            choices.append((direction,child))
        if not choices:raise ValueError("nonterminal referee state has no legal direction")
        proposals.append(dict(partial=action,legal_directions=[item[0] for item in choices]))
        direction,state=choices[rng.randrange(len(choices))];action+=str(direction)
    return state,action,proposals


def game(build,color,seed,directory):
    directory=Path(directory);directory.mkdir(exist_ok=True)
    campaign.immutable(directory/"CLAIM.json",dict(build=build,color=color,seed=seed,kind="mechanical-production-main-safety"))
    state=rules.ReplayState();rng=random.Random(seed);transcript=[];answers=[];proposals=[]
    started=time.monotonic();process=subprocess.Popen([str(campaign.verify(build["binary"]))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
    process.stdin.write(str(color)+"\n");process.stdin.flush();first=True;last_opponent="-";failure=None
    try:
        for turn in range(317):
            if state.winner is not None:break
            if state.to_move!=color:
                state,action,choices=random_turn(state,rng);proposals.append(dict(prefix="/".join(transcript),choices=choices))
                transcript.append(action);last_opponent=action
                continue
            before="/".join(transcript);response_started=started if first else time.monotonic()
            process.stdin.write(str(0 if last_opponent=="-" else len(last_opponent))+"\n"+last_opponent+"\n");process.stdin.flush()
            ready,_,_=select.select([process.stdout],[],[],1.1 if first else .25)
            if not ready:raise TimeoutError("production response missing")
            action=process.stdout.readline().rstrip("\r\n");elapsed=1000*(time.monotonic()-response_started)
            if not action:raise ValueError("production returned empty/EOF action")
            limit=550 if first else 140
            answers.append(dict(prefix=before,action=action,elapsed_ms=elapsed,limit_ms=limit,first=first))
            if elapsed>limit:raise TimeoutError("whole-response envelope exceeded")
            rules.apply_complete_turn(state,color,action);transcript.append(action);first=False
            campaign.atomic(directory/"PROGRESS.json",dict(transcript=transcript,responses=answers,proposals=proposals,complete=False))
        if state.winner is None:raise RuntimeError("mechanical game exceeded finite edge-bound turn count")
        process.stdin.close();process.wait(timeout=3)
        if process.returncode:raise ValueError("production main returned nonzero exit")
    except BaseException as error:
        failure=type(error).__name__+": "+str(error)
        process.kill();process.wait()
    stderr=process.stderr.read();campaign.immutable(directory/"stderr.txt",stderr.encode())
    result=dict(source=build["source"],binary=build["binary"],color=color,seed=seed,
                transcript="/".join(transcript),responses=answers,proposals=proposals,
                complete=state.winner is not None and failure is None,failure=failure,
                first_max_ms=max((row["elapsed_ms"] for row in answers if row["first"]),default=0),
                later_max_ms=max((row["elapsed_ms"] for row in answers if not row["first"]),default=0),
                longest_response_edges=max((len(row["action"]) for row in answers),default=0),
                training_eligible=False,strength_eligible=False)
    campaign.immutable(directory/"RESULT.json",result)
    if failure:raise RuntimeError(failure)
    return result


def certify(build_path,output):
    build=campaign.read(build_path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results=[]
    for index,(profile,entry) in enumerate(build["builds"].items()):
        for color in (0,1):
            result=game(entry,color,2026100714+index*2+color,output/(profile+"-p"+str(color)))
            results.append(dict(profile=profile,receipt=campaign.record(output/(profile+"-p"+str(color))/"RESULT.json"),
                                first_max_ms=result["first_max_ms"],later_max_ms=result["later_max_ms"],longest_response_edges=result["longest_response_edges"]))
    report=dict(schema=campaign.SCHEMA+".mechanical-main-safety",passed=True,isolated=True,
                exact_builds=campaign.record(build_path),games=results,mechanical_games=6,
                own_failures=0,cold_warm_and_complete_protocol=True,
                long_turn_observed=all(max(row["longest_response_edges"] for row in results if row["profile"]==profile)>=8 for profile in build["builds"]),
                playing_strength_qualified=False,production_training_eligible=False)
    if not report["long_turn_observed"]:report["passed"]=False
    campaign.immutable(output/"RESULT.json",report)
    campaign.immutable(output/"EXPOSURE_PENDING.json",dict(schema=campaign.SCHEMA+".mechanical-game-exposure",
                        report=campaign.record(output/"RESULT.json"),game_receipts=[row["receipt"] for row in results],
                        recipe=campaign.record(__file__),training_eligible=False,before_next_fresh_bank=True))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    compile_parser=sub.add_parser("compile");compile_parser.add_argument("--preflight",type=Path,required=True);compile_parser.add_argument("--output",type=Path,required=True)
    certify_parser=sub.add_parser("certify");certify_parser.add_argument("--build",type=Path,required=True);certify_parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();report=compile_sources(args.preflight,args.output) if args.mode=="compile" else certify(args.build,args.output)
    print(json.dumps(report))


if __name__=="__main__":main()
