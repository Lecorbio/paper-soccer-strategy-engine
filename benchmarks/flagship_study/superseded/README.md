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

`manifest-0031a81c.json` records an intermediate corrected preregistration
(SHA-256 `0031a81c193a63fe503d153e9fd4922e2cc8f4e6d64448752502805f0e4d8f41`).
It was superseded before executing any unit because the final fail-closed
review found that unavailable or malformed `pmset` output was not yet rejected
as an unknown Low Power Mode state. No raw-result namespace exists for that
manifest. The v3 runner accepts only an explicit disabled value and rejects
missing, malformed, unavailable, enabled, or conflicting mode records.

The completed v3 manifest was archived byte-for-byte as
`manifest-b7553a24.json` before the clean v4 source boundary was committed. V3
completed all 1,800 development and 3,600 validation games with zero
truncations, then stopped before test when
its calibration fitter's scale-dependent coefficient cap rejected a finite
alpha-beta fit. [`../V3_VALIDATION_FAILURE.md`](../V3_VALIDATION_FAILURE.md)
records the exact data hashes and numerical diagnosis. V4 repairs that
implementation defect prospectively, changes all validation-specific seeds,
and generates four validation banks disjoint from every v3 opening. The
development banks and never-evaluated test banks remain byte-for-byte fixed.
