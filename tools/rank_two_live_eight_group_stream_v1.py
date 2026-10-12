"""Stream one immutable normalized group without loading its successor array."""
import json


class Reader:
    def __init__(self, stream):
        self.stream, self.buffer, self.position, self.eof = stream, '', 0, False
        self.decoder = json.JSONDecoder()

    def fill(self):
        if self.position:
            self.buffer = self.buffer[self.position:]
            self.position = 0
        chunk = self.stream.read(65536)
        self.buffer += chunk
        self.eof = not chunk
        if len(self.buffer) > 2 * 1024**2:
            raise ValueError('individual normalized group item exceeds stream bound')

    def space(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or self.eof:
                return
            self.fill()

    def peek(self):
        self.space()
        return self.buffer[self.position:self.position + 1]

    def take(self, expected):
        if self.peek() != expected:
            raise ValueError('normalized group JSON delimiter differs')
        self.position += 1

    def value(self):
        self.space()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                # A number can end at a chunk boundary before its remaining
                # digits arrive. Wait for its delimiter before accepting it.
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                if (end < len(self.buffer) and self.buffer[end] not in ' \t\r\n,:]}'
                        and not self.eof):
                    self.fill()
                    continue
                self.position = end
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.fill()


def successors(stream, expected_root, expected_group):
    reader = Reader(stream)
    reader.take('{')
    metadata, found = {}, False
    while reader.peek() != '}':
        key = reader.value()
        if not isinstance(key, str) or key in metadata or (key == 'successors' and found):
            raise ValueError('unique group object keys required')
        reader.take(':')
        if key == 'successors':
            if (metadata.get('split') != 'train' or metadata.get('root_group_id') != expected_root
                    or metadata.get('group_id') != expected_group
                    or type(metadata.get('mover')) is not int
                    or type(metadata.get('exhaustive')) is not bool):
                raise PermissionError('bound canonical TRAIN metadata must precede successors')
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
