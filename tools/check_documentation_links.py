#!/usr/bin/env python3
"""Check repository Markdown links without network access or generated output."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\((<[^>]+>|[^)\n]+)\)")
REFERENCE = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s*(<[^>]+>|\S+)", re.M)


def prose(text: str) -> str:
    """Remove fenced code, preserving headings and ordinary link markup."""
    result = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            value = marker[1]
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
            continue
        if fence is None:
            result.append(line)
    return "\n".join(result)


def links(text: str) -> list[str]:
    text = prose(text)
    targets = [match[1] for match in INLINE_LINK.finditer(text)]
    targets.extend(match[2] for match in REFERENCE.finditer(text))
    return [value[1:value.index(">")] if value.startswith("<")
            else value.strip().split()[0] for value in targets]


def anchors(text: str) -> set[str]:
    text = prose(text)
    result = set(re.findall(r"\b(?:id|name)=[\"']([^\"']+)[\"']", text))
    used: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        heading = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", match[1])
        heading = html.unescape(re.sub(r"<[^>]+>", "", heading)).lower()
        slug = re.sub(r"[^\w\-\s]", "", heading, flags=re.UNICODE).replace(" ", "-")
        count = used.get(slug, 0)
        used[slug] = count + 1
        result.add(slug + (f"-{count}" if count else ""))
    return result


def check(root: Path, documents: list[Path], archived: list[dict]) -> tuple[list[str], int]:
    errors = []
    exceptions = {(row["source"], row["target"]): row for row in archived}
    seen = set()
    anchor_cache = {}
    for source in documents:
        text = source.read_text(encoding="utf-8")
        name = source.relative_to(root).as_posix()
        for link in links(text):
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            target = (source.parent / unquote(parsed.path)).resolve() if parsed.path else source
            if not target.exists():
                key = (name, link)
                exception = exceptions.get(key)
                if exception and hashlib.sha256(source.read_bytes()).hexdigest() == exception["source_sha256"]:
                    archive = urlsplit(exception["archive_url"])
                    if archive.scheme == "https" and archive.netloc == "github.com" and exception.get("reason"):
                        seen.add(key)
                        continue
                errors.append(f"{name}: missing target {link}")
            elif parsed.fragment and target.suffix.lower() in (".md", ".html"):
                if target not in anchor_cache:
                    anchor_cache[target] = anchors(target.read_text(encoding="utf-8"))
                if unquote(parsed.fragment) not in anchor_cache[target]:
                    errors.append(f"{name}: missing anchor {link}")
    for key in exceptions.keys() - seen:
        errors.append(f"unused or changed historical link exception: {key[0]} -> {key[1]}")
    return errors, len(seen)


def main() -> int:
    names = subprocess.check_output([
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md",
    ], cwd=ROOT).decode().split("\0")
    documents = sorted({ROOT / name for name in names if name and (ROOT / name).is_file()})
    archived = json.loads((ROOT / "docs/archived-links.json").read_text())
    errors, count = check(ROOT, documents, archived)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Checked {len(documents)} Markdown files; local links and anchors pass. "
          f"{count} hash-frozen references resolve through the documented archive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
