# Superseded preregistration

`manifest-eab2728f.json` is retained as an audit record of the first frozen
manifest (SHA-256 `eab2728f4f5915926639ab67f20ab94137afe0275543f8eafd34b8047ab4ecf3`).
It was superseded after four development-only pilot units exposed enough
runtime to begin review, but before any validation unit or test outcome was
accessed.

The independent review found that duplicate processes could race while
publishing the same raw shard and that validation recorded only its pre-run
power/thermal snapshot. The replacement preregistration closes both gaps and
pins the Python analysis runtime. The twelve already committed opening banks
are reused byte-for-byte with unchanged paths, seeds, and SHA-256 hashes; no
opening was regenerated in response to an outcome. Raw results from the
superseded manifest remain isolated under its ignored manifest-hash namespace
and are not inputs to the replacement study.
