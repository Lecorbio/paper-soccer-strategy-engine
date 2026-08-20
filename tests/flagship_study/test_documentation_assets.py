from __future__ import annotations

import json
import pathlib
import re
import subprocess
import unittest
import urllib.parse

from benchmarks.flagship_study import charts


REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def tracked_files(pattern: str) -> list[pathlib.Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", pattern],
        cwd=REPOSITORY,
    )
    return [
        REPOSITORY / item.decode("utf-8")
        for item in output.split(b"\0")
        if item
    ]


def local_image_targets(markdown: pathlib.Path) -> list[pathlib.Path]:
    targets: list[pathlib.Path] = []
    text = markdown.read_text(encoding="utf-8")
    for match in MARKDOWN_IMAGE.finditer(text):
        raw = match.group(1).strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1]
        raw = raw.split(' "', 1)[0].split(" '", 1)[0]
        if raw.startswith(("http://", "https://", "data:")):
            continue
        path = urllib.parse.unquote(raw.split("#", 1)[0])
        if path:
            targets.append((markdown.parent / path).resolve())
    return targets


class DocumentationAssetTests(unittest.TestCase):
    def test_local_markdown_images_exist_and_docs_assets_are_referenced(self) -> None:
        referenced: set[pathlib.Path] = set()
        missing: list[str] = []
        for markdown in tracked_files("*.md"):
            for target in local_image_targets(markdown):
                referenced.add(target)
                if not target.is_file():
                    missing.append(
                        f"{markdown.relative_to(REPOSITORY)} -> "
                        f"{target.relative_to(REPOSITORY)}"
                    )

        self.assertEqual(missing, [], "missing local Markdown images:\n" + "\n".join(missing))
        assets = set(tracked_files("docs/assets/*"))
        self.assertEqual(
            assets - referenced,
            set(),
            "unreferenced docs assets: "
            + ", ".join(str(path.relative_to(REPOSITORY)) for path in assets - referenced),
        )

    def test_checked_in_flagship_charts_match_frozen_inputs(self) -> None:
        study = REPOSITORY / "benchmarks/flagship_study"
        manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
        selection = json.loads(
            (study / "selection_lock.json").read_text(encoding="utf-8")
        )
        analysis = json.loads(
            (study / "data/test.json").read_text(encoding="utf-8")
        )
        configurations = {
            value["id"]: value for value in manifest["configurations"]
        }
        labels = {
            identifier: value["public_label"]
            for identifier, value in configurations.items()
        }
        for identifier, configuration in configurations.items():
            if configuration["kind"] == "mcts":
                labels[identifier] = (
                    f"Tactical MctsBot ({configuration['settings']['iterations']} iter)"
                )
            elif configuration["kind"] in ("alpha-beta", "jacek-inspired"):
                labels[identifier] = (
                    f"{configuration['public_label']} "
                    f"({configuration['settings']['max_nodes'] // 1000}k nodes)"
                )

        expected = {
            "bradley_terry": charts.bradley_terry_svg(analysis, labels),
            "pareto": charts.pareto_svg(selection, labels),
            "calibration": charts.calibration_svg(analysis, labels),
        }
        for name, rendered in expected.items():
            relative = manifest["outputs"]["charts"][name]
            with self.subTest(chart=name):
                self.assertEqual(
                    (REPOSITORY / relative).read_text(encoding="utf-8"),
                    rendered,
                )


if __name__ == "__main__":
    unittest.main()
