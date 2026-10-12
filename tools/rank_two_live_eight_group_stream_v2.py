"""Read bound TRAIN or VALIDATION groups with an explicit partition expectation."""
from tools.rank_two_live_eight_group_stream_v1 import Reader


def successors(stream, expected_root, expected_group, expected_split):
    if expected_split not in ('train', 'validation'):
        raise PermissionError('explicit admitted partition required')
    reader = Reader(stream)
    reader.take('{')
    metadata, found = {}, False
    while reader.peek() != '}':
        key = reader.value()
        if not isinstance(key, str) or key in metadata or (key == 'successors' and found):
            raise ValueError('unique group object keys required')
        reader.take(':')
        if key == 'successors':
            if (metadata.get('split') != expected_split
                    or metadata.get('root_group_id') != expected_root
                    or metadata.get('group_id') != expected_group
                    or type(metadata.get('mover')) is not int
                    or type(metadata.get('exhaustive')) is not bool
                    or type(metadata.get('edges')) is not int):
                raise PermissionError('bound canonical partition metadata must precede successors')
            found = True
            reader.take('[')
            while reader.peek() != ']':
                child = reader.value()
                if not isinstance(child, dict):
                    raise ValueError('complete successor object required')
                yield metadata, child
                if reader.peek() == ',':
                    reader.take(',')
                    if reader.peek() == ']':
                        raise ValueError('trailing successor comma is invalid')
                elif reader.peek() != ']':
                    raise ValueError('successor array delimiter differs')
            reader.take(']')
        else:
            metadata[key] = reader.value()
        if reader.peek() == ',':
            reader.take(',')
            if reader.peek() == '}':
                raise ValueError('trailing group comma is invalid')
        elif reader.peek() != '}':
            raise ValueError('group object delimiter differs')
    reader.take('}')
    if not found or reader.peek():
        raise ValueError('one complete normalized group required')
