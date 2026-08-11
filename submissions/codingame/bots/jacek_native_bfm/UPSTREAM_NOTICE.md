# QtPaperSoccer provenance and modification notice

`jacek_native_bfm` is an independent, single-thread CodinGame adaptation of
the publicly documented Paper Soccer architecture in Jacek Dermont's
[QtPaperSoccer](https://github.com/jdermont/QtPaperSoccer) project. The pinned
reference revision is
[`366d5304c09c2c820bd3ef4ea94624c034b8d955`](https://github.com/jdermont/QtPaperSoccer/commit/366d5304c09c2c820bd3ef4ea94624c034b8d955)
from 2026-03-08.

The upstream project is licensed under Apache License 2.0. A verbatim copy is
provided in [APACHE-2.0.txt](APACHE-2.0.txt). Apache-licensed rights and
notices remain applicable to any upstream-derived portion; the repository's
MIT license applies only to independently authored portions.

This adaptation is materially changed for the CodinGame environment:

- Qt, the GUI, desktop threading, and upstream pool allocation are removed;
- the repository's neutral rules and compact-position APIs replace the
  upstream application model;
- time handling, diagnostics, deterministic tests, and the text protocol are
  new;
- the checked-in evaluator is trained independently for this repository; and
- no upstream network checkpoint, private CodinGame source, replay book,
  incumbent evaluator, or incumbent search is included.

The pinned public code is evidence for architectural choices, not evidence
that this candidate reproduces Jacek's unpublished CodinGame bot or its
strength.
