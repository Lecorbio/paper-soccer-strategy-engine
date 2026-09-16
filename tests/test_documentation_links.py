import hashlib
from pathlib import Path
import tempfile
import unittest

from tools import check_documentation_links as docs


class DocumentationLinkTests(unittest.TestCase):
    def test_local_files_and_duplicate_heading_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, target = root / "index.md", root / "guide.md"
            source.write_text("[Read](guide.md#hello-world-1)\n")
            target.write_text("# Hello, world!\n## Hello, world!\n")
            self.assertEqual(docs.check(root, [source], []), ([], 0))
            source.write_text("[Read](guide.md#missing)\n")
            self.assertIn("missing anchor", docs.check(root, [source], [])[0][0])

    def test_fenced_examples_and_external_links_are_not_local_targets(self):
        self.assertEqual(docs.links("```md\n[example](absent.md)\n```\n[real](guide.md)"), ["guide.md"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "index.md"
            source.write_text("[external](https://example.com/docs#intro)\n")
            self.assertEqual(docs.check(root, [source], []), ([], 0))

    def test_historical_exception_requires_exact_source_bytes_and_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "frozen.md"
            source.write_text("[old](old.json)\n")
            exception = {"source": "frozen.md", "target": "old.json",
                         "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "archive_url": "https://github.com/example/repo/blob/frozen/old.json",
                         "reason": "frozen record"}
            self.assertEqual(docs.check(root, [source], [exception]), ([], 1))
            source.write_text(source.read_text() + "changed\n")
            self.assertTrue(docs.check(root, [source], [exception])[0])

    def test_reference_links_and_angle_bracket_paths(self):
        self.assertEqual(docs.links("[guide]: <a b.md>\n[read](<c d.md>)"), ["c d.md", "a b.md"])


if __name__ == "__main__":
    unittest.main()
