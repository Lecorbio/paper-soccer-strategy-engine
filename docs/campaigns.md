# Campaign catalog

This is the current navigation layer over historical evidence. A completed
experiment's protocol remains useful documentation; its example commands do
not allocate another run. Local qualification and public platform results are
reported separately.

| Family | Recorded status and supported result | Entry point / evidence |
| --- | --- | --- |
| Canonical contest submissions | Historical Rank-4 snapshot: version 56, 66–24, score 44.29750553418035. Rank-5 is an immutable predecessor. | [Submission archive](../submissions/codingame/README.md); per-bot generators and ledgers. |
| Flagship demo study | Published 4,800-game test study under demo rules; predecessor v3 stopped at validation. | `benchmarks/flagship_study/`; [report](../benchmarks/flagship_study/REPORT.md), [history](history.md). |
| Game Review | Accepted `deep-400k` demo analysis profile with frozen native/Wasm and accuracy gates. | `benchmarks/game_review_gate/gate.py`; [report](../benchmarks/game_review_gate/REPORT.md). |
| Local contest leaderboard | Frozen 22-bot, 990-game local comparison, not a public ladder reproduction. | `benchmarks/codingame_leaderboard/leaderboard.py`; [workflow](../benchmarks/codingame_leaderboard/README.md). |
| Replay BFM and self-search v5/v6 | Historical training/reference families used by subsequent rebuild and teacher work. Outcome belongs to each run's own final summary. | `tools/jacek_selfsearch_workflow.py`; [replay model](jacek-replay-bfm.md), [self-search](jacek-selfsearch-campaign.md); local `results/jacek_replay_bfm/`. |
| Replay rebuild, 20260826 v1 | Closed with `no-development-qualified-candidate`; all four candidate families exhausted their route. The resume-at-login job is retired. | `tools/jacek_replay_rebuild.py`; [protocol](jacek-replay-rebuild.md); local `replay-rebuild-20260826-v1/run/final-summary.json`. |
| Exploratory self-search, 20260827 v1 | Pilot rejected. The separate large-teacher continuation has its own evidence and authorization. | `tools/jacek_selfsearch_exploratory.py`; [protocol](jacek-selfsearch-exploratory.md); local `selfsearch-exploratory-20260827-v1/final-summary.json`. |
| Large-teacher campaign | Accepted local teacher after 10,000 generated games, four strength panels, retention, and uncontended timing. | [Public report](../benchmarks/large_teacher_campaign/REPORT.md); [self-search protocol](jacek-selfsearch-campaign.md). |
| Compact discrete-v3 / Rank-4 challenger | Recovered incumbent passed two protected gates and completed one 90-game live diagnostic. Attempt zero did not perform new distillation training. | [Outcome](compact-value-bfm-rank4-campaign-outcome.md); [challenger](compact-value-bfm-rank4-teacher-challenger.md), [release](compact-value-bfm-rank4-teacher-release.md). |
| Compact mandatory-training v2 | Closed without a qualifying newly trained model. Four pilots had 0/36 retention passes; the final pair failed its declared criterion. Six of eight short jobs were used. | `tools/compact_value_bfm_campaign_v2.py`; [outcome](../benchmarks/compact_value_bfm/TRAINED_V2_OUTCOME.md), [contract](compact-value-bfm-trained-v2.md). |
| Rank-4/Jacek hybrid research | Historical experiment with a separate archive of unfinished local work. Consolidation implies no new promotion. | [Bot record](../submissions/codingame/bots/rank_4_jacek_hybrid/README.md); tracked `results/rank_4_jacek_hybrid/` and retained local evidence. |

## Terms used in results

- **Position accuracy / Huber:** evaluator agreement on labelled positions;
  neither is a game win rate.
- **Retention:** absolute evaluator floors and allowed deterioration relative
  to specified frozen references. Both named pools must pass where required.
- **Teacher regret:** loss in teacher value from the student's chosen action.
  Lower is better; use the declared float or quantized control.
- **Development gate:** an unprotected experiment used to choose or reject a candidate.
- **Protected gate:** a fresh, source-bound final test whose data is excluded
  from training and model selection. A spent gate is not freely retried.
- **Local qualification:** passing a particular local contract, opponent set,
  source identity, and clock. It does not confer a public rank.
- **Live result:** observed platform games tied to the uploaded source,
  calibration state, and collection window.

## Before continuing a campaign

Read its terminal receipt and stopping rule first. New work needs a new plan,
budget, source identity, and output namespace where the prior plan stopped.
Retained absolute paths, models, datasets, and exclusions remain part of the
evidence even after old Git branch names disappear.

The generic [compact evidence publisher](../benchmarks/compact_value_bfm/README.md)
has a separate incomplete publication contract. Its placeholder report is not
the status of the completed challenger or closed training-v2 experiment.
