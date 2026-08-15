# Arena derivation disposition

Content-addressed derivations under `rejected-window/` are retained only to
audit the collector's initial per-game classification. They are not training
inputs. A source-attributable operational failure rejects the complete window,
even when the preliminary derivation labels individual clean games as
`eligible` candidates.

`fresh_corpus.py` independently enforces this rule: any derivation whose
summary reports a nonzero `focus_operational_failures` count is rejected before
game bindings are loaded. The content-addressed window-rejection report is the
authoritative usage disposition.
