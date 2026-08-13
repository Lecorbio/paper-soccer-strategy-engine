# Campaign evidence report (working snapshot)

This report freezes the collection decision, immediate safety action, and
rollback audit for the five-hour `jacek_arena_bfm` campaign. It remains a
working snapshot until the final repository commit, CI run, and campaign-end
live rank are recorded.

## Fresh construction and scratch evidence

- Campaign boundary: `2026-08-13T10:12:52Z`.
- Initial random-bootstrap corpus: 2,000 games, 30,659 value rows, manifest
  `72eb091180fc0191e679b97d1b789a05a13420bc8f661123f11069728344b60c`.
- Interim scratch continuation: 2,000 games, 76,562 value rows, manifest
  `3b77cdb7adf5936fe7358f92555475fc4a5fb624886f8a3ad177c275ce9fcab4`.
- Cumulative valid fresh scratch corpus: 4,000 games and 107,221 value rows,
  balanced at 1,000 games for each opening depth `0,4,8,12`.
- Historical model weights, corpora, actions, labels, and replay content used:
  none.  Arena games used by the scratch continuation: none.

## First live candidate

| Field | Exact value |
| --- | --- |
| Model | `fresh-32x32-s101-7f23a40ba6ca` |
| Source SHA-256 | `3883f4c3f29a32c039492adc6151e94b5dfd84653ce0dfb2383356e7f5e3c9f8` |
| Source size | 88,664 ASCII bytes |
| Commit | `dbce4dec8ca2f31ef7790992dcddda7948eef132` |
| Agent | `6615613` |
| Submission | `41130787` |
| Upload | `2026-08-13T10:55:59Z` |
| Exact-window manifest | `8b8b3e5e59cda6817e54d17da7ef531a75bfd49ae5d0d7cefb1b650fb5795de0` |

The exact matching-submission window completed at 90 games: 74 were clean,
unambiguous rule terminals; three were timeouts attributable to this bot; 13
contained opponent operational failures.  The focus timeout game IDs are
`898882047`, `898882199`, and `898882273`.

At exact completion the candidate ranked 24. Its 74 clean games were 41-33
overall: 20-14 as player 0 and 21-19 as player 1. It had no clean top-five or
Jacek match before the rollback. These results cannot override the three
operational failures.

Any focus operational failure rejects the whole window.  Therefore all 90
games, including the 74 otherwise clean terminals, are excluded from training.
The exact fresh-arena usage report is:

| Use | Games |
| --- | ---: |
| State value | 0 |
| Opponent action ranking | 0 |
| Arena validation | 0 |
| Final live holdout | 0 |

The immutable rejection record is
`results/jacek_arena_bfm/reports/48224d76f1e42ea453bf5afe1ab96abfbcd9cf8081a017f15f936b2f573fe9a6.json`.

## Safety rollback

The three own timeouts disqualified the fresh candidate and immediately
triggered the authorized safe rollback:

| Field | Exact value |
| --- | --- |
| Source SHA-256 | `d9d96f83197f13b7212e7b652851097053ee7f1662845e06dd722d1c0bc24f71` |
| Source size | 99,810 ASCII bytes |
| Runtime | `C=0.95`, 80,000 nodes, 800/155 ms |
| Agent | `6615714` |
| Submission | `41130866` |
| Upload | `2026-08-13T11:13:22Z` |

The editor paste/copy-back matched exactly and Play My Code produced legal
stdout with expected H62 telemetry before submission.  The H62 artifact is an
evaluation/rollback source only; it is not imported into the new engine,
model, scratch games, actions, labels, or training lineage.

The exact rollback accounting window completed at 90 games under manifest
`44530074995e48754100ace955d4d75c16021947194b7e2c3e87a802b7de7cb9`:
76 clean rule terminals, 14 opponent operational failures, and zero H62
operational failures. Its rollback-only derivation is
`bc7336d949ec947740856c953e8f58660a90e43073891fe43cb3a19a723a82c9`.
The two-window campaign sequence validates as sequential and complete under
report `595e759d72eb6ee9ca75a4dc17a54a94cff3f1ab3affd03bad34ed55cb7edaa0`.
At exact completion the rollback ranked 9. Its 76 clean games were 49-27
overall, 4-17 against the frozen top five, and 0-5 against Jacek; the color
split was 20-14 as player 0 and 29-13 as player 1. The content-addressed
outcome summary is
`539905832f487bb8f23ef0e90b46640038eeada1be9ee31e8b54d3e4a7e1ffe1`.

CodinGame upload bytes are editor-attested by exact copy-back equality.  The
public CodinGame API does not expose the editor source bytes, so the attestation
cannot be replaced by an API readback.

## Mixture and selection disposition

The 25%, 40%, and 55% arena-exposure mixtures and their required two seeds
were not run.  No eligible fresh arena training window exists after the
whole-window rejection, so those exposure levels cannot be realized.
Scratch-only retrains may be evaluated separately, but they must not be named
or selected as arena mixtures.

Two otherwise identical one-epoch, random-initialized scratch-only retrains
were run on the frozen cumulative 4,000-game corpus. Seed 101
(`fresh-32x32-s101-ad87209e4c6c`) was selected for the offline namespace
artifact over seed 1701 on frozen validation MSE (`0.883150` versus
`0.891634`) and sign accuracy (`0.627034` versus `0.618694`). The
content-addressed decision is
`4617ca6c044709110be852ae02e09aa76ee40091ebc0f6682cf439c9e40c9d5a`.
It remains offline and does not supersede the live H62 rollback.

Postmortem replay of the three exact pre-timeout states isolated unbounded
complete-turn generation across the deadline. The exact uploaded source
reproduced local maxima of `159.912`, `161.361`, and `162.114` ms against its
155 ms later-turn budget. The offline fix checks deadlines during generation,
reserves 6 ms for finalization, retains a deterministic legal emergency turn,
and uses a 128 ms later budget; its corresponding maxima were `124.844`,
`126.253`, and `126.734` ms. This diagnostic evidence is explicitly
nontraining and is bound by report SHA
`acb532751a214ffcdfaa02f55bb57df7d38be29d8615e8d6bd00d909f447c78b`.

The offline-hardened source then ran the exact 1,000-game fast screen against
frozen H62, balanced 500/500 colors. All 1,000 games were operationally clean,
but the source lost `57-943`. It therefore fails the mandatory strength gate;
the 212-game actual-clock qualification was not run. The isolated aggregate
report has SHA
`ea8882f89199a09859c951d4da240e8b4f65d768909d42a606866470951587df`.

The fresh source did not qualify as a finalist because it had three own
operational failures.  Rank 4 was not achieved by that candidate.  The live
campaign conclusion remains the exact safe rollback unless later accounting
uncovers a fact that must be added to this report.
