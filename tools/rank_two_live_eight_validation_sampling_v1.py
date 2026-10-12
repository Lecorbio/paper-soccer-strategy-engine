"""Bound validation parents by identity before observing teacher labels."""
from collections import defaultdict


def early_parents(positions, maximum=2):
    if type(maximum) is not int or not 1 <= maximum <= 2:
        raise ValueError('declared early validation parent cap is two')
    roots = set()
    grouped = defaultdict(dict)
    for row in positions:
        if row['split'] != 'validation':
            raise PermissionError('validation sampler cannot consume TRAIN parents')
        root = row['root_group_id']
        roots.add(root)
        if row['edges'] > 12:
            continue
        key = row['canonical_features']
        previous = grouped[root].get(key)
        if previous is None or (row['edges'], row['position_id']) < (previous['edges'], previous['position_id']):
            grouped[root][key] = row
    if set(grouped) != roots:
        raise ValueError('a frozen validation root has no early parent; retain it without replacement')
    selected = []
    for root in sorted(roots):
        ordered = sorted(grouped[root].values(), key=lambda row:
                         (row['edges'], row['canonical_features'], row['position_id']))
        selected.extend(ordered[:maximum])
    return selected
