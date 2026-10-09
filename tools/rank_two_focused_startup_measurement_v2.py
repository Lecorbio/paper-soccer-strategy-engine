"""Retained non-playing fixtures for linker and process-startup measurements."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_work_v1 as work

SCHEMA = campaign.SCHEMA + '.startup-measurement.v2'
FIXTURE = r'''
#ifdef FOCUSED_CONTROL_SOURCE
#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include FOCUSED_CONTROL_SOURCE
#endif
#include <chrono>
#include <iostream>
#include <string>
#include <sstream>
#include <time.h>
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif
static long long stamp() {
#ifdef __APPLE__
  mach_timebase_info_data_t ratio{};
  mach_timebase_info(&ratio);
  const auto ticks = mach_absolute_time();
  return static_cast<long long>((static_cast<__uint128_t>(ticks)*ratio.numer)/ratio.denom);
#else
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC, &value)) std::abort();
  return static_cast<long long>(value.tv_sec)*1000000000LL+value.tv_nsec;
#endif
}
int main() {
  const auto entered = stamp();
  std::cout << "READY\t" << entered << '\n' << std::flush;
  std::string line;
  while (std::getline(std::cin, line)) {
    const auto received = stamp();
    int mover = -1;
#ifdef FOCUSED_CONTROL_SOURCE
    papersoccer::RulesConfig rules;
    rules.goal_rule = papersoccer::GoalRule::OwnGoalsAllowed;
    rules.blocked_rule = papersoccer::BlockedRule::MoverLoses;
    auto state = papersoccer::make_initial_state(rules);
    if (line != "-") {
      std::istringstream input(line);
      std::string action;
      while (std::getline(input, action, '/'))
        papersoccer::turn_action_v2::apply_encoded_turn(state, action);
    }
    mover = state.to_move == papersoccer::Player::One ? 0 : 1;
#endif
    const auto reconstructed = stamp();
    std::cout << "ECHO\t" << received << '\t' << reconstructed << '\t'
              << mover << '\n' << std::flush;
  }
}
'''


def checked(plan_path):
    plan = campaign.read(plan_path)
    if plan['producer'] != campaign.record(__file__):
        raise ValueError('measurement producer differs')
    for reference in plan['inputs'].values():
        campaign.verify(reference)
    if plan['control']['sha256'] != campaign.BASELINE or plan['search_calls'] != 0:
        raise ValueError('measurement must use the incumbent without search')
    work.checked(plan, 'experiment', launch=True)
    return plan


def build(plan_path):
    plan = checked(plan_path)
    output = Path(plan['output']) / 'build'
    source = output / 'fixture.cpp'
    campaign.immutable(source, FIXTURE.encode())
    compiler = plan['compiler']
    if subprocess.check_output([compiler['executable']['path'], '--version'], text=True) != compiler['version']:
        raise ValueError('compiler version differs')
    campaign.verify(compiler['executable'])
    builds = {}
    for kind in ('minimal', 'control-reconstruction'):
        for linker in ('classic', 'default'):
            work.checked(plan, 'experiment')
            name = kind + '-' + linker
            directory = output / name
            directory.mkdir(parents=True, exist_ok=True)
            binary = directory / 'fixture'
            command = [compiler['executable']['path'], '-fintegrated-cc1', '-std=c++20',
                       '-O3', '-DNDEBUG', '-ffp-contract=off']
            if linker == 'classic':
                command += ['-Wl,-ld_classic']
            if kind == 'control-reconstruction':
                command += ['-DFOCUSED_CONTROL_SOURCE="' + str(campaign.verify(plan['control'])) + '"']
            command += [str(source), '-o', str(binary)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            compilation = campaign.immutable(directory / 'COMPILATION.json',
                dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
            if result.returncode:
                raise RuntimeError('measurement fixture compilation failed')
            metadata = subprocess.run(['/usr/bin/otool', '-hv', '-l', str(binary)],
                                      capture_output=True, text=True, timeout=30)
            builds[name] = dict(kind=kind, linker=linker, binary=campaign.record(binary),
                fixture=campaign.record(source), compilation=compilation,
                load_commands=campaign.immutable(directory / 'LOAD_COMMANDS.json',
                    dict(returncode=metadata.returncode, stdout=metadata.stdout, stderr=metadata.stderr)))
    receipt = dict(schema=SCHEMA+'.build', passed=True, plan=campaign.record(plan_path),
                   producer=campaign.record(__file__), builds=builds, compiler=compiler, new_games=0)
    campaign.immutable(Path(plan['output']) / 'BUILD.json', receipt)
    return receipt


def lines(child, count, timeout=3):
    data = bytearray()
    started = time.monotonic()
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ)
        while data.count(b'\n') < count:
            if time.monotonic() - started > timeout:
                raise RuntimeError('fixture response watchdog')
            if selector.select(.02):
                chunk = os.read(child.stdout.fileno(), 65536-len(data))
                if not chunk:
                    raise RuntimeError('fixture closed before response')
                data.extend(chunk)
                if len(data) >= 65536:
                    raise RuntimeError('oversized fixture response')
    if data.count(b'\n') != count or not data.endswith(b'\n'):
        raise RuntimeError('unexpected fixture output')
    return data.decode('ascii').rstrip('\n').split('\n')


def measure(plan_path):
    plan = checked(plan_path)
    output = Path(plan['output'])
    builds = campaign.read(campaign.verify(plan['build']))
    if builds['plan'] != plan['build_plan'] or builds['producer'] != plan['producer']:
        raise ValueError('measurement build binding differs')
    reports = []
    for index, name in enumerate(plan['order']):
        work.checked(plan, 'experiment')
        item = builds['builds'][name]
        directory = output / 'trials' / f'{index:03}'
        campaign.immutable(directory / 'CLAIM.json', dict(plan=campaign.record(plan_path), build=item))
        child = None
        response = []
        raw_timestamps = {}
        failure = None
        try:
            binary = campaign.verify(item['binary'])
            launched = time.monotonic_ns()
            child = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            spawned = time.monotonic_ns()
            child.stdin.write(b'-\n'); child.stdin.flush()
            response = lines(child, 2)
            completed = time.monotonic_ns()
            ready = response[0].split('\t'); echo = response[1].split('\t')
            if ready[0] != 'READY' or len(ready) != 2 or echo[0] != 'ECHO' or len(echo) != 4:
                raise ValueError('fixture response fields differ')
            entered, received, reconstructed = int(ready[1]), int(echo[1]), int(echo[2])
            raw_timestamps = dict(launched=launched, spawned=spawned, entered=entered, received=received, reconstructed=reconstructed, completed=completed)
            if not launched <= entered <= received <= reconstructed <= completed:
                raise ValueError('cross-process monotonic timestamp order differs')
            row = dict(launch_to_response_ms=(completed-launched)/1e6,
                popen_ms=(spawned-launched)/1e6, launch_to_main_ms=(entered-launched)/1e6,
                request_reconstruction_ms=(reconstructed-received)/1e6,
                main_to_response_ms=(completed-entered)/1e6, warm=[])
            for prefix in plan['warm_prefixes']:
                started = time.monotonic_ns()
                child.stdin.write((prefix+'\n').encode()); child.stdin.flush()
                text = lines(child, 1)[0]
                fields = text.split('\t')
                if fields[0] != 'ECHO' or len(fields) != 4 or int(fields[2]) < int(fields[1]):
                    raise ValueError('warm fixture response fields differ')
                row['warm'].append(dict(prefix=prefix, response=text,
                    elapsed_ms=(time.monotonic_ns()-started)/1e6))
            child.stdin.close()
            child.wait(timeout=3)
            if child.returncode != 0:
                raise RuntimeError('fixture nonzero exit')
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
            failure = type(error).__name__ + ': ' + str(error)
            row = {}
        finally:
            if child:
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=3)
                    except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=3)
                stderr = child.stderr.read().decode(errors='replace')
                child.stdout.close(); child.stderr.close()
                if not child.stdin.closed: child.stdin.close()
            else:
                stderr = ''
        result = dict(index=index, build=name, binary=item['binary'], metrics=row,
                      raw_first_response=response, raw_timestamps_ns=raw_timestamps, stderr=stderr, failure=failure)
        reports.append(campaign.immutable(directory / 'RESULT.json', result))
        if failure:
            break
    value = dict(schema=SCHEMA, passed=len(reports)==len(plan['order']) and failure is None,
        plan=campaign.record(plan_path), trials=reports, first_launch_of_each_binary_retained=True,
        isolated=True, search_calls=0, new_games=0, training_eligible=False, strength_eligible=False,
        original_failed_source_reexecuted=False, original_failure_cause='unknown')
    campaign.immutable(output / 'RESULT.json', value)
    campaign.immutable(output / 'EXPOSURE_PENDING.json', dict(schema=SCHEMA+'.exposure',
        recipe=campaign.record(plan_path), cases=[dict(prefix=p) for p in ['-',*plan['warm_prefixes']]],
        before_next_fresh_bank=True, training_eligible=False, no_gameplay=True))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('build','measure'))
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.plan) if args.stage == 'build' else measure(args.plan)
    print(json.dumps(dict(passed=result['passed'])))


if __name__ == '__main__':
    main()
