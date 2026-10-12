"""Account for an explicitly approved quarantine in one frozen validation bank."""
from tools import rank_two_live_eight_v1 as c


def root_accounting(frozen, evaluated, excluded, early):
    sets = []
    for values in (frozen, evaluated, excluded, early):
        values = list(values)
        if len(values) != len(set(values)) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError('unique explicit validation root identities required')
        sets.append(set(values))
    frozen, evaluated, excluded, early = sets
    if (len(frozen), len(evaluated), len(excluded)) != (128, 127, 1):
        raise PermissionError('approved128 frozen/127 evaluated/1 excluded accounting required')
    if evaluated & excluded or evaluated | excluded != frozen:
        raise PermissionError('evaluated and excluded roots must partition the frozen bank')
    if not early <= evaluated or len(early) < 100:
        raise PermissionError('at least100 clean comparable early roots required')
    return dict(frozen=128, evaluated=127, excluded=1, comparable_early=len(early))


def validate(evidence, actual_roots):
    approval = c.read(c.verify(evidence['gate_approval']))
    if (approval.get('approved_by') != 'human'
            or (approval['frozen_roots'], approval['evaluated_roots'], approval['excluded_roots']) != (128, 127, 1)
            or approval['minimum_comparable_early_roots'] != 100
            or not approval['no_seed_or_root_replacement']
            or not approval['no_qualification_rule_change']
            or evidence['generation_plan'] != approval['generation_plan']
            or evidence['plan'] != approval['validation_teacher_plan']
            or evidence['execution'] != approval['validation_teacher_execution']
            or evidence['excluded_root_ids'] != approval['excluded_root_ids']):
        raise PermissionError('exact human-approved bank and quarantine binding required')
    generated = c.read(c.verify(evidence['generation_plan']))
    frozen = sorted(row['root_group_id'] for row in generated['rows'])
    if generated['seed'] != approval['seed'] or frozen != evidence['frozen_root_ids']:
        raise PermissionError('frozen seed or root identities changed')
    if sorted(actual_roots) != evidence['evaluated_root_ids']:
        raise PermissionError('only the approved clean roots may be evaluated')
    accounting = root_accounting(frozen, evidence['evaluated_root_ids'],
        evidence['excluded_root_ids'], evidence['comparable_early_root_ids'])
    if (evidence['root_accounting'] != accounting or evidence['roots'] != 128
            or evidence['early_roots'] != accounting['comparable_early']):
        raise ValueError('reported frozen, evaluated, excluded or early counts differ')
    return accounting
