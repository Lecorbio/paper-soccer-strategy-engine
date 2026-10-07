#!/usr/bin/env python3
"""Conservative C++ whitespace minification; literals and directives are opaque."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import re
import subprocess


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    start: int
    end: int


_LITERAL = re.compile(r'(?:u8|u|U|L)?(?:R"([^ ()\\\t\r\n]{0,16})\(|["\'])')
_NUMBER = re.compile(r'(?:\d|\.\d)(?:[\w.\']|[eEpP][+-])*')
_WORD = re.compile(r'[A-Za-z_\u0080-\uffff][\w\u0080-\uffff]*')
_OPERATORS = sorted(('##', '%:%:', '<=>', '<<=', '>>=', '->*', '...', '::', '.*', '->', '++', '--', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||', '*=', '/=', '%=', '+=', '-=', '&=', '^=', '|=', '<:', ':>', '<%', '%>', '%:'), key=len, reverse=True)
_SAFE = set('{};(),')


def _literal_end(source, start):
    match = _LITERAL.match(source, start)
    if not match:
        return None
    if match.group(1) is not None:
        end = source.find(')' + match.group(1) + '"', match.end())
        if end < 0:
            raise ValueError('unterminated raw literal')
        return end + len(match.group(1)) + 2
    quote = source[match.end() - 1]
    index = match.end()
    while index < len(source):
        if source[index] == '\\':
            index += 2
        elif source[index] == quote:
            return index + 1
        else:
            index += 1
    raise ValueError('unterminated literal')


def _directive_end(source, start):
    index = start
    while index < len(source):
        end = _literal_end(source, index)
        if end is not None:
            index = end
        elif source.startswith('/*', index):
            end = source.find('*/', index + 2)
            if end < 0:
                raise ValueError('unterminated directive comment')
            index = end + 2
        elif source.startswith('//', index):
            # Quotes in comments are not literal starts. Include continued lines.
            end = source.find('\n', index)
            while end >= 0 and source[:end].rstrip('\r').endswith('\\'):
                end = source.find('\n', end + 1)
            return len(source) if end < 0 else end + 1
        elif source[index] == '\n':
            if not source[:index].rstrip('\r').endswith('\\'):
                return index + 1
            index += 1
        else:
            # Skip pp-numbers, including digit separators, before quote parsing.
            number = _NUMBER.match(source, index)
            index = number.end() if number else index + 1
    return index


def lex(source):
    """Return opaque literals/directives and preprocessing-token-like spans."""
    index = 0
    line_clean = True
    while index < len(source):
        start = index
        if source[index].isspace():
            while index < len(source) and source[index].isspace():
                index += 1
            kind = 'space'
        elif line_clean and (source[index] == '#' or source.startswith('%:', index)):
            index = _directive_end(source, index)
            kind = 'directive'
        elif source.startswith('//', index):
            index = source.find('\n', index)
            if index < 0:
                index = len(source)
            kind = 'comment'
        elif source.startswith('/*', index):
            index = source.find('*/', index + 2)
            if index < 0:
                raise ValueError('unterminated comment')
            index += 2
            kind = 'comment'
        else:
            end = _literal_end(source, index)
            number = _NUMBER.match(source, index)
            word = _WORD.match(source, index)
            if end is not None:
                index, kind = end, 'literal'
            elif number:
                index, kind = number.end(), 'number'
            elif word:
                index, kind = word.end(), 'word'
            else:
                op = next((op for op in _OPERATORS if source.startswith(op, index)), source[index])
                index, kind = index + len(op), 'punct'
        value = source[start:index]
        yield Token(kind, value, start, index)
        if kind in ('space', 'comment'):
            if '\n' in value:
                line_clean = True
        else:
            line_clean = kind == 'directive' and value.endswith('\n')


def signature(source):
    return [(token.kind, token.text) for token in lex(source) if token.kind not in ('space', 'comment')]


def minify(source):
    # Phase-2 splices outside directives are intentionally unsupported. Handling
    # them after lexing could turn split comment delimiters into active syntax.
    tokens = list(lex(source))
    unprotected = ''.join('X' if t.kind in ('literal', 'directive') else t.text for t in tokens)
    if '\\\n' in unprotected or '\\\r\n' in unprotected:
        raise ValueError('line splice outside protected literal/directive')
    output = []
    previous = None
    pending = False
    for token in tokens:
        if token.kind in ('space', 'comment'):
            pending = True
            continue
        if previous is not None and pending:
            if token.kind == 'directive':
                if not output[-1].endswith('\n'):
                    output.append('\n')
            elif previous.kind == 'directive':
                if not previous.text.endswith('\n'):
                    output.append('\n')
            elif not ((previous.kind == 'punct' and previous.text in _SAFE) or
                      (token.kind == 'punct' and token.text in _SAFE)):
                output.append(' ')
        output.append(token.text)
        previous, pending = token, False
    result = ''.join(output) + '\n'
    if signature(result) != signature(source):
        raise ValueError('minification changed token/literal/directive sequence')
    return result


def verify_preprocessed(original, compact, compiler='/usr/bin/clang++'):
    """Compare compiler-preprocessed token streams using identical stdin identity."""
    command = [compiler, '-std=c++20', '-E', '-P', '-x', 'c++', '-']
    signatures = []
    for source in (original, compact):
        result = subprocess.run(command, input=source, text=True, capture_output=True, timeout=90)
        if result.returncode:
            raise ValueError('C++ preprocessing failed: ' + result.stderr)
        signatures.append(signature(result.stdout))
    if signatures[0] != signatures[1]:
        raise ValueError('compiler preprocessing changed token sequence')
    return {'passed': True, 'command': command, 'tokens': len(signatures[0]),
            'token_sha256': hashlib.sha256(repr(signatures[0]).encode()).hexdigest(),
            'compiler_version': subprocess.check_output([compiler, '--version'], text=True)}
