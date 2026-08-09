# First live replay model iteration: rejected at equal clock

The resumed append-only round crossed the frozen 50-game floor, produced the
first shared action-conditioned policy candidate, and exercised every allowed
gate. The selected candidate passed preflight, passed the 7-5 default floor,
and passed fixed-work development 51-45. It then lost equal-clock development
29-67, worse than the maintained 39-57 baseline. The maintained model was
restored and **no performance submission was made**.

## Collection identity and boundary

CodinGame history version 55 was the single explicit collection upload. The
authenticated editor source was the unchanged maintained 99,697-character
source, SHA-256
`4d77f9e20e756b2e11b0f4dc04ff9ed2fed37e85158f358dda104d8281b5b243`.
It became agent `6604698`, submission `41114191`, in session
`77331573cfcf07bc257c0a715fe9675351097800`. Public processing completed at
100% with 90 finished battles, rank 24, and score 35.13. The exact binding is
in `active_submissions/20260809T200708Z-agent-6604698.json`. No second
collection upload was attempted.

Poll 19 froze 68 independent new accepted games at
`2026-08-09T20:06:16Z`. The snapshot is
`corpora/687468e84c475107eee840f4d731fbc51182e8cfc20d2bf2cd7039d344f48f97.json`.
It binds the exclusion registry, final poll, 68 discoveries and accepted
records, and collector SHA-256
`5c73139e9607ea97477aef212b6c015207cf66fb462fda8759003f98a4e897ef`.
The offline store audit reports 68 accepted records, 68 discoveries, 139 raw
and 139 normalized response objects, 520 receipts, and zero payload conflicts.

No protected evaluation, rank-one lock, arena diagnostic, exposed
development, prospective, or sealed-final action entered the snapshot. The
root `matches.json` was not read or changed. All states and reflections from a
game share one global split group; canonical train/validation/test overlap was
purged in that order.

## Labels and deterministic training

The snapshot contains 5,482 directly supervised strong-opponent primitives:
one game from the frozen rank 6-10 tier and 67 from rank 11-20. Direct live
final outcomes have zero value weight. The bot's 4,204 played primitives were
never copied as expert actions. Instead,
`e406c9ad5d3796524e29ba2e052a84f778c43ab8f1eb7c5817f6ad1da6ff2e98.relabel.jsonl`
records deeper neural-only PUCT distributions and mover-relative root values.
It performed 41,830,000 simulations and 30,485,580 neural evaluations at a
10,000-simulation request and 100,000-node hard cap. There were 981
visit-max disagreements with the played move; low played probability,
entropy, disagreement, and tactical-punishment flags determine the frozen
priority weights.

Every split was normalized per player and game and then rebalanced to exactly
75% anchored public-expert mass and 25% live direct/relabelled mass for both
policy and value. Three 40-epoch action-head fits were trained before selection:

| Seed | Int4 validation value | Int4 validation policy | Frozen sum |
|---:|---:|---:|---:|
| 20260810 | 0.645121 | 1.094832 | 1.739953 |
| 20260811 | 0.628448 | 1.052427 | **1.680875** |
| 20260812 | 0.627000 | 1.077978 | 1.704978 |

Seed `20260811` was selected solely by the predeclared quantized validation
sum; test outcomes were not used. The three content-addressed artifacts are in
`candidates/`. The selected model hash is
`549e443d6aa37532f9569116b675e5c3cd6f484cdd38cb1e15ea60ee846f83f2`.
Its generated header hash is
`87c0c0db23e2586fd501dbf8590df53ccab70620694f8195e50cb98fa5b345dd`,
and its 98,062-character candidate submission hash is
`1af09752e5c1ff5b1fc475d5006fc0999172bdb9ae546aeb0634ea19d4b6659d`.

The shared eight-unit scorer combines the 32-value state projection with 16
per-direction consequences: canonical direction, rebound/handoff, immediate
win/loss, both goal-distance changes, remaining degree, safe/dead frontier,
continuation size, layer fill/closure, escape routes, and opponent mobility.
Weights are shared across all eight directions. The mover-relative value head
and neural-only decision path are unchanged.

## Frozen gates

Release compilation, generated-source checks, the source-size cap, exact
Python/C++ feature parity (5,656/5,656 float32 values), rotation/reflection and
int4 golden tests, legality, protocol, construction-inclusive timing, node-cap,
and ASan/UBSan checks all passed. The complete CodinGame Python suite passed
31 tests. Apple ASan does not support leak detection; the same sanitizer
binaries passed without the unsupported `detect_leaks` option.

| Gate | Candidate | Rank 5 | Colors | Operational result | Decision |
|---|---:|---:|---|---|---|
| Six paired defaults, 2,000 vs 5,000 | 7 | 5 | 5-1 / 2-4 | clean | pass at floor |
| 48 exposed openings, fixed work | 51 | 45 | 25-23 / 26-22 | clean | pass |
| 48 exposed openings, equal 20 ms | 29 | 67 | 16-32 / 13-35 | clean | **reject** |

The content-addressed reports are
`evaluations/e03b137968ceefe791307e79e7da2a4785aab5e768903c65f54f9fe7e8221b72.json`,
`evaluations/4b623f096f1e3c7951cb12174123f1775421ee039b5a63300b016f75d49480da.json`,
and
`evaluations/94a8934c2f5f5c17c2e95dc25fa58706fddf780fe020a4f5b7d858fd30112139.json`.
No prospective, validation, test, or sealed-final opening bank was inspected
or run for this exploratory candidate.

## Decision

The action-conditioned representation transfers at fixed work but is too slow
and/or too weak per neural evaluation at equal time. Its 29 wins are ten below
the maintained equal-clock baseline, so it is not upload-worthy. The checked-in
model remains the maintained seed-20260809 artifact
`d7e5255fd2c3f7a203796829c21c8a7580b384de13942e466f4069081e2126b2`.
There is no performance agent to ingest for a next generation.
