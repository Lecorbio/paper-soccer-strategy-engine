"""Classify a deadline-only H62 derivative without changing the frozen roster."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_runtime_v6 as runtime
from tools import rank_two_focused_work_v1 as work

SCHEMA=campaign.SCHEMA+'.h62-deadline-proposal.v1'

DRIVER=r'''
#define PAPER_SOCCER_JACEK_NATIVE_BFM_NO_MAIN
#include H62_SOURCE
#include <iomanip>
#include <sstream>
int main(){
  using namespace papersoccer;
  namespace bot=papersoccer::jacek_native_bfm;
  std::string line;
  while(std::getline(std::cin,line)){
    std::istringstream input(line);int expansions=0;std::string prefix;
    if(!(input>>expansions>>prefix))return 2;
    RulesConfig rules;rules.goal_rule=GoalRule::OwnGoalsAllowed;rules.blocked_rule=BlockedRule::MoverLoses;
    auto state=make_initial_state(rules);
    if(prefix!="-"){std::istringstream path(prefix);std::string action;while(std::getline(path,action,'/'))bot::apply_encoded_turn(state,action);}
    bot::SearchConfig config;config.max_expansions=expansions;config.max_tree_nodes=2048;
    const auto result=bot::choose_complete_turn(static_cast<const GameState&>(state),config);
    std::cout<<result.encoded<<'\t'<<std::setprecision(17)<<result.value<<'\t'<<result.solved<<'\t';
    if(result.solved_winner)std::cout<<static_cast<int>(*result.solved_winner);else std::cout<<-1;
    std::cout<<'\t'<<result.stats.tree_nodes<<'\t'<<result.stats.expansions<<'\t'<<result.stats.child_evaluations<<'\n'<<std::flush;
  }
}
'''


def materialize(source):
    start='const std::size_t first_child = nodes_.size();\nfor (CompleteTurnAction &action : generated.actions) {\n'
    replacement='const std::size_t first_child = nodes_.size();\nbool evaluation_truncated = false;\nfor (CompleteTurnAction &action : generated.actions) {\nif (nodes_.size() > first_child && deadline_expired()) {\nevaluation_truncated = true;\nbreak;\n}\n'
    finish='nodes_[index].exhaustive = generated.exhaustive;'
    if source.count(start)!=1 or source.count(finish)!=1:
        raise ValueError('frozen H62 expansion anchors differ')
    return source.replace(start,replacement).replace(finish,
        'nodes_[index].exhaustive = generated.exhaustive && !evaluation_truncated;')


def compile_driver(source,output,compiler,sanitize=False):
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    wrapper=directory/'driver.cpp';binary=directory/'driver'
    campaign.immutable(wrapper,DRIVER.encode())
    flags=['-O1','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if sanitize else ['-O3','-DNDEBUG']
    command=[compiler['executable']['path'],'-fintegrated-cc1','-std=c++20','-ffp-contract=off',*flags,
        '-DH62_SOURCE="'+str(source)+'"',str(wrapper),'-o',str(binary)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=120)
    campaign.immutable(directory/'COMPILE.json',dict(command=command,returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr,source=campaign.record(source),compiler=compiler))
    if result.returncode:raise RuntimeError('H62 proposal driver compilation failed: '+result.stderr[-1500:])
    return campaign.record(binary)


def run(plan_path):
    plan=campaign.read(plan_path)
    if plan['producer']!=campaign.record(__file__):raise ValueError('proposal producer differs')
    work.checked(plan,'classify',launch=True)
    for reference in plan['inputs'].values():campaign.verify(reference)
    original=campaign.verify(plan['original']);output=Path(plan['output'])
    proposed=output/'H62_DEADLINE_PROPOSAL.cpp'
    campaign.immutable(proposed,materialize(original.read_text()).encode())
    delta=subprocess.run(['/usr/bin/diff','-u',str(original),str(proposed)],capture_output=True,text=True)
    campaign.immutable(output/'SOURCE_DIFF.patch',delta.stdout.encode())
    examples=campaign.read(campaign.verify(plan['cases']))['cases']
    nonterminal=[x for x in examples if runtime.e.state(x['prefix']).winner is None]
    requests=[str(n)+' '+x['prefix'] for n in plan['fixed_expansions'] for x in nonterminal]
    answers={};builds={}
    for name,source in [('original',original),('proposal',proposed)]:
        work.checked(plan,'classify')
        binary=compile_driver(source,output/name,plan['compiler']);builds[name]=binary
        result=subprocess.run([binary['path']],input='\n'.join(requests)+'\n',capture_output=True,text=True,timeout=600)
        answers[name]=result.stdout.splitlines()
        campaign.immutable(output/name/'FIXED_WORK.json',dict(returncode=result.returncode,
            stderr=result.stderr,requests=requests,responses=answers[name],binary=binary))
        if result.returncode or len(answers[name])!=len(requests):raise RuntimeError('H62 fixed-work driver failed')
    if answers['original']!=answers['proposal']:raise ValueError('deadline-free fixed-work behavior differs')
    for request,answer in zip(requests,answers['proposal']):
        prefix=request.split(' ',1)[1];action=answer.split('\t')[0];state=runtime.e.state(prefix)
        runtime.e.rules.apply_complete_turn(state,state.to_move,action)
    sanitized=compile_driver(proposed,output/'sanitized',plan['compiler'],sanitize=True)
    checks=['1 '+x['prefix'] for x in nonterminal]
    result=subprocess.run([sanitized['path']],input='\n'.join(checks)+'\n',capture_output=True,text=True,timeout=600)
    campaign.immutable(output/'sanitized/RESULT.json',dict(returncode=result.returncode,stderr=result.stderr,
        requests=checks,responses=result.stdout.splitlines(),binary=sanitized))
    if result.returncode or len(result.stdout.splitlines())!=len(checks):raise RuntimeError('H62 proposal sanitizer check failed')
    value=dict(schema=SCHEMA,passed=True,original=plan['original'],proposed=campaign.record(proposed),
        diff=campaign.record(output/'SOURCE_DIFF.patch'),plan=campaign.record(plan_path),
        fixed_work_queries=len(requests),sanitized_queries=len(checks),fixed_work_bit_identical=True,
        exact_terminal_and_solved_labels_equal=True,omitted_children_marked_nonexhaustive=True,
        first_child_retained_for_legal_action=True,frozen_roster_modified=False,
        failed_panel_state_used=False,original_failure_cause='unknown',timing_certified=False,new_games=0)
    campaign.immutable(output/'RESULT.json',value)
    campaign.immutable(output/'EXPOSURE_PENDING.json',dict(schema=SCHEMA+'.exposure',cases=plan['cases'],
        inputs=[campaign.record(output/'original/FIXED_WORK.json'),campaign.record(output/'proposal/FIXED_WORK.json')],
        report=campaign.record(output/'RESULT.json'),training_eligible=False,before_next_fresh_bank=True))
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();result=run(args.plan);print(json.dumps(dict(passed=result['passed'],queries=result['fixed_work_queries'])))


if __name__=='__main__':main()
