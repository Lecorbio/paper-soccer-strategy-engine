"""Fixed-work native parity probes; no strength or response-time certification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_export_v1 as exporter

PROBE = r'''
namespace ps=papersoccer;
namespace bt=papersoccer::turn_action_v2;
int main(){std::string line;while(std::getline(std::cin,line)){try{
std::istringstream in(line);std::string command;in>>command;
if(command=="weights"){for(auto value:bt::learned_eval::weights())std::cout<<int(value)<<' ';std::cout<<'\n';}
else if(command=="eval"){unsigned count;in>>count;if(!count||count>421)throw std::invalid_argument("count");
bt::learned_eval::Features features;features.count=count;
for(unsigned i=0;i<count;++i){unsigned value;in>>value;if(value>=6301)throw std::invalid_argument("index");features.indices[i]=value;}
std::cout<<std::setprecision(9)<<bt::learned_eval::evaluate(features)<<'\n';}
else if(command=="decode"){std::size_t expected;std::string encoded;in>>expected>>encoded;
for(auto value:bt::learned_eval::decode_base85(encoded,expected))std::cout<<unsigned(value)<<' ';std::cout<<'\n';}
else if(command=="search"||command=="features"||command=="state"){
std::uint64_t nodes=1000;if(command=="search")in>>nodes;
std::string prefix;in>>prefix;if(prefix=="-")prefix.clear();
ps::RulesConfig rules;rules.goal_rule=ps::GoalRule::OwnGoalsAllowed;rules.blocked_rule=ps::BlockedRule::MoverLoses;
auto state=ps::make_initial_state(rules);std::istringstream turns(prefix);std::string turn;
while(std::getline(turns,turn,'/'))bt::apply_encoded_turn(state,turn);
if(command=="state"){auto topology=std::make_shared<ps::detail::SearchTopology>(rules);
ps::detail::SearchPosition position(topology,state);auto before=position;
std::array<std::uint8_t,8> slots{};auto count=position.legal_slots(slots);
for(unsigned i=0;i<count;++i){position.make_move(slots[i]);position.unmake_move();if(!position.same_compact_state(before))throw std::logic_error("undo");}
std::cout<<int(state.to_move==ps::Player::Two)<<' '<<int(ps::is_terminal(state))<<' '<<count<<'\n';}
else{if(ps::is_terminal(state))throw std::invalid_argument("terminal");bt::SearchConfig config;config.max_nodes=nodes;
bt::CompleteTurnSearch search(state,config);
if(command=="features"){search.evaluation_snapshot();auto features=search.learned_features();
std::vector<unsigned> indices;for(unsigned i=0;i<features.count;++i)indices.push_back(features.indices[i]);std::sort(indices.begin(),indices.end());
for(auto value:indices)std::cout<<value<<' ';std::cout<<'\n';}
else{auto moves=search.run();auto next=state;std::string action;
for(auto move:moves){action.push_back(bt::encode_direction(next.ball,move.to));next=ps::apply_move(next,move);}
bt::apply_encoded_turn(state,action);auto stats=search.stats();
std::cout<<action<<' '<<stats.root_score<<' '<<stats.completed_turn_depth<<' '<<stats.attempted_turn_depth<<' '<<stats.nodes<<' '<<stats.leaf_evaluations<<' '<<stats.budget_exhausted<<'\n';}}}
else throw std::invalid_argument("command");
if(!in)throw std::invalid_argument("request");
}catch(const std::exception&error){std::cout<<"ERROR "<<error.what()<<'\n';}}}
'''


def probe_source(source, aliases):
    includes="\n".join(line for line in source.splitlines() if line.startswith("#include"))
    # Read system headers before the probe-only visibility macro. Production
    # bytes remain unchanged and the adapter is separately content addressed.
    return (includes+"\n#include <iomanip>\n#include <sstream>\n"
            "#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN\n#define private public\n"
            +source+exporter.alias_cleanup(aliases)+"#undef private\n"+PROBE)


def build(source, report, directory, compiler="/usr/bin/clang++", sanitize=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    cpp=directory/"probe.cpp";binary=directory/"probe"
    campaign.immutable(cpp,probe_source(source,report.get("aliases",{})).encode())
    flags=["-std=c++20","-O1" if sanitize else "-O3","-ffp-contract=off"]
    if sanitize:flags += ["-fsanitize=address,undefined","-fno-omit-frame-pointer"]
    command=[compiler,*flags,str(cpp),"-o",str(binary)]
    result=subprocess.run(command,text=True,capture_output=True,timeout=120)
    campaign.immutable(directory/"compiler.stderr",result.stderr.encode())
    if result.returncode:raise ValueError("native probe compilation failed: "+result.stderr[-2500:])
    campaign.immutable(directory/"BUILD.json",dict(source_sha256=report.get("source_sha256"),
                    adapter=campaign.record(cpp),binary=campaign.record(binary),
                    compiler=campaign.record(Path(compiler).resolve()),flags=flags,sanitizers=sanitize))
    return binary


def query(binary, requests):
    value=subprocess.run([str(binary)],input="\n".join(requests)+"\n",text=True,capture_output=True,timeout=120)
    if value.returncode:raise ValueError("native probe failed: "+value.stderr[-2000:])
    answers=value.stdout.splitlines()
    if len(answers)!=len(requests):raise ValueError("native response cardinality differs")
    return answers


def predict(source,report,active,directory):
    binary=build(source,report,directory)
    requests=["eval "+str(len(row))+" "+" ".join(map(str,row)) for row in active]
    answers=query(binary,requests)
    if any(answer.startswith("ERROR") for answer in answers):raise ValueError("native prediction query failed")
    return [float(answer) for answer in answers]


def preflight(directory):
    from tools import rank_two_network32_v2 as prior
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    forbidden=exporter.macro_names();_,control_weights,scales=prior.baseline()
    reports={};models={}
    for profile,shape in exporter.PROFILES.items():
        count=shape[0]*shape[1]+shape[1]*shape[2]+shape[2]
        values=control_weights if profile=="control" else [(index%15)-7 for index in range(count)]
        runtime=exporter.document(profile,values,scales,{"kind":"mechanical-preflight","production_eligible":False})
        source,report=exporter.export(runtime,forbidden,verify_tokens=True)
        out=directory/profile
        campaign.immutable(out/"submission.cpp",source.encode());campaign.immutable(out/"runtime.json",runtime)
        campaign.immutable(out/"EXPORT.json",report)
        binary=build(source,report,out/"normal")
        decoded=list(map(int,query(binary,["weights"])[0].split()))
        if decoded!=values:raise ValueError("native decoded tensors differ")
        models[profile]=(source,report,binary)
        reports[profile]={key:report[key] for key in ("characters","nonpayload_characters","payload_kind","below_stress_limit","nonpayload_target_passed")}
    # Compare native fixed-work search with identical incumbent tensors.
    original=prior.baseline()[0]
    legacy_report=dict(source_sha256=campaign.BASELINE,aliases={})
    original=original.replace("inline int base64_value(","inline int base64_value(")
    # The legacy evaluator has no Base85 entry point; remove only the optional
    # transport test branch from its diagnostic adapter.
    reference_cpp=probe_source(original,{})
    start=reference_cpp.index('else if(command=="decode")')
    end=reference_cpp.index('else if(command=="search"',start)
    reference_cpp=reference_cpp[:start]+reference_cpp[end:]
    reference_dir=directory/"reference";reference_dir.mkdir(exist_ok=True)
    campaign.immutable(reference_dir/"probe.cpp",reference_cpp.encode())
    process=subprocess.run(["/usr/bin/clang++","-std=c++20","-O3","-ffp-contract=off",str(reference_dir/"probe.cpp"),"-o",str(reference_dir/"probe")],text=True,capture_output=True,timeout=120)
    campaign.immutable(reference_dir/"compiler.stderr",process.stderr.encode())
    if process.returncode:raise ValueError("reference probe compilation failed: "+process.stderr[-2500:])
    requests=["search 4000 "+prefix for prefix in ("-","0","1","2","3","4","5","6","7")]
    expected=query(reference_dir/"probe",requests);actual=query(models["control"][2],requests)
    if expected!=actual:raise ValueError("fixed-work incumbent action/score/depth/work differs")
    result=dict(schema=campaign.SCHEMA+".native-preflight",passed=all(x["below_stress_limit"] and x["nonpayload_target_passed"] for x in reports.values()),
                profiles=reports,native_tensors_equal=True,fixed_work_queries=len(requests),
                fixed_work_incumbent_equal=True,compiler_tokens_equal=True,
                new_games=0,playing_strength_qualified=False,whole_response_timing_certified=False)
    campaign.immutable(directory/"RESULT.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(preflight(args.output)))


if __name__=="__main__":main()
