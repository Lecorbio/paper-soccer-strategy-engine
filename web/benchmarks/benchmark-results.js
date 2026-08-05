globalThis.PaperSoccerBenchmarkResults = {
  "calibration": [
    {
      "brierScore": 0.20360308758391332,
      "decisionCount": 153427,
      "id": "alpha-beta-50k",
      "label": "Hand-evaluated AlphaBetaBot",
      "logLoss": 0.5817811076050585,
      "samples": 153385
    },
    {
      "brierScore": 0.19191784952514035,
      "decisionCount": 158999,
      "id": "jacek-20k",
      "label": "Neural alpha-beta (JacekInspiredBot)",
      "logLoss": 0.5511572884639822,
      "samples": 158831
    },
    {
      "brierScore": 0.14791759324012183,
      "decisionCount": 60157,
      "id": "mcts-1000",
      "label": "Tactical MctsBot",
      "logLoss": 0.4609805780177983,
      "samples": 60157
    },
    {
      "brierScore": 0.21568840956991409,
      "decisionCount": 136088,
      "id": "rank5-fixed-50k",
      "label": "Rank5DerivedBot — fixed 50k demo profile",
      "logLoss": 0.614574507512456,
      "samples": 39098
    }
  ],
  "caveats": {
    "latency": "Latency is native, single-threaded validation p95 on the study gate machine and will vary across hardware.",
    "rank5": "Rank5DerivedBot adapts search code from the rank 5/206 CodinGame submission to different demo rules and a fixed-work profile. These measurements do not evaluate the authentic ranked submission.",
    "relativeStrength": "Bradley-Terry strengths are relative within these four entrants; their zero point is not an absolute playing-strength scale.",
    "validationReference": "Rank5Derived's 50% validation strength is a defined common-opponent reference level, not an independently observed score."
  },
  "entrants": [
    {
      "bradleyTerry": {
        "estimate": 0.5512377382724543,
        "lower": 0.48955786032099274,
        "upper": 0.6137660396931038
      },
      "family": "jacek_inspired",
      "id": "jacek-20k",
      "label": "Neural alpha-beta (JacekInspiredBot)",
      "shortLabel": "Neural alpha-beta",
      "validation": {
        "fixed": false,
        "gateEligible": true,
        "latencyDecisions": 33161,
        "p95LatencyMs": 35.718084,
        "pairs": 200,
        "paretoOptimal": true,
        "selected": true,
        "strength": 0.555,
        "strengthIsReference": false,
        "strengthLower": 0.51,
        "strengthUpper": 0.5975
      }
    },
    {
      "bradleyTerry": {
        "estimate": 0.4372816040653935,
        "lower": 0.376394885334156,
        "upper": 0.49852754849499525
      },
      "family": "rank5_derived",
      "id": "rank5-fixed-50k",
      "label": "Rank5DerivedBot — fixed 50k demo profile",
      "shortLabel": "Rank5Derived",
      "validation": {
        "fixed": true,
        "gateEligible": true,
        "latencyDecisions": 65605,
        "p95LatencyMs": 31.383417,
        "pairs": null,
        "paretoOptimal": true,
        "selected": true,
        "strength": 0.5,
        "strengthIsReference": true,
        "strengthLower": null,
        "strengthUpper": null
      }
    },
    {
      "bradleyTerry": {
        "estimate": 0.1761225176283836,
        "lower": 0.12014704228054846,
        "upper": 0.23335979957754632
      },
      "family": "alpha_beta",
      "id": "alpha-beta-50k",
      "label": "Hand-evaluated AlphaBetaBot",
      "shortLabel": "Hand alpha-beta",
      "validation": {
        "fixed": false,
        "gateEligible": true,
        "latencyDecisions": 31120,
        "p95LatencyMs": 24.273083,
        "pairs": 200,
        "paretoOptimal": true,
        "selected": true,
        "strength": 0.43,
        "strengthIsReference": false,
        "strengthLower": 0.385,
        "strengthUpper": 0.475
      }
    },
    {
      "bradleyTerry": {
        "estimate": -1.1646418599662314,
        "lower": -1.2424453717022215,
        "upper": -1.0920148185402883
      },
      "family": "mcts",
      "id": "mcts-1000",
      "label": "Tactical MctsBot",
      "shortLabel": "Tactical MCTS",
      "validation": {
        "fixed": false,
        "gateEligible": true,
        "latencyDecisions": 11089,
        "p95LatencyMs": 35.683625,
        "pairs": 200,
        "paretoOptimal": false,
        "selected": true,
        "strength": 0.175,
        "strengthIsReference": false,
        "strengthLower": 0.1425,
        "strengthUpper": 0.2125
      }
    }
  ],
  "links": {
    "report": "https://github.com/Lecorbio/paper-soccer-strategy-engine/blob/main/benchmarks/flagship_study/REPORT.md"
  },
  "matchups": [
    {
      "classification": "stronger",
      "games": 800,
      "id": "test-alpha-beta-vs-jacek",
      "leftId": "alpha-beta-50k",
      "leftScore": 0.39625,
      "leftScoreLower": 0.365,
      "leftScoreUpper": 0.42625,
      "pairs": 400,
      "rightId": "jacek-20k",
      "strongerId": "jacek-20k"
    },
    {
      "classification": "stronger",
      "games": 800,
      "id": "test-alpha-beta-vs-rank5",
      "leftId": "alpha-beta-50k",
      "leftScore": 0.42125,
      "leftScoreLower": 0.3925,
      "leftScoreUpper": 0.45,
      "pairs": 400,
      "rightId": "rank5-fixed-50k",
      "strongerId": "rank5-fixed-50k"
    },
    {
      "classification": "statistically_unresolved",
      "games": 800,
      "id": "test-jacek-vs-rank5",
      "leftId": "jacek-20k",
      "leftScore": 0.51375,
      "leftScoreLower": 0.4825,
      "leftScoreUpper": 0.545,
      "pairs": 400,
      "rightId": "rank5-fixed-50k",
      "strongerId": null
    },
    {
      "classification": "stronger",
      "games": 800,
      "id": "test-mcts-vs-alpha-beta",
      "leftId": "mcts-1000",
      "leftScore": 0.1825,
      "leftScoreLower": 0.1575,
      "leftScoreUpper": 0.2075,
      "pairs": 400,
      "rightId": "alpha-beta-50k",
      "strongerId": "alpha-beta-50k"
    },
    {
      "classification": "stronger",
      "games": 800,
      "id": "test-mcts-vs-jacek",
      "leftId": "mcts-1000",
      "leftScore": 0.14875,
      "leftScoreLower": 0.125,
      "leftScoreUpper": 0.17375,
      "pairs": 400,
      "rightId": "jacek-20k",
      "strongerId": "jacek-20k"
    },
    {
      "classification": "stronger",
      "games": 800,
      "id": "test-mcts-vs-rank5",
      "leftId": "mcts-1000",
      "leftScore": 0.19625,
      "leftScoreLower": 0.17,
      "leftScoreUpper": 0.2225,
      "pairs": 400,
      "rightId": "rank5-fixed-50k",
      "strongerId": "rank5-fixed-50k"
    }
  ],
  "schema": "papersoccer.benchmark-summary.v1",
  "study": {
    "entrantCount": 4,
    "games": 4800,
    "headline": "Neural alpha-beta has the highest strength estimate, while its matchup with Rank5Derived remains statistically unresolved.",
    "id": "competitive-demo-bots-flagship-2026-v4",
    "latencyGateMs": 50,
    "openingDepths": [
      4,
      8,
      12,
      20
    ],
    "pairs": 2400,
    "title": "Competitive demo-rule Paper Soccer bot study"
  },
  "validationCandidates": [
    {
      "budget": 100000,
      "family": "alpha_beta",
      "fixed": false,
      "gateEligible": true,
      "id": "alpha-beta-100k",
      "label": "Hand-evaluated AlphaBetaBot",
      "latencyDecisions": 31290,
      "p95LatencyMs": 25.011875,
      "pairs": 200,
      "paretoOptimal": true,
      "selected": false,
      "strength": 0.44,
      "strengthIsReference": false,
      "strengthLower": 0.3975,
      "strengthUpper": 0.4825
    },
    {
      "budget": 20000,
      "family": "alpha_beta",
      "fixed": false,
      "gateEligible": true,
      "id": "alpha-beta-20k",
      "label": "Hand-evaluated AlphaBetaBot",
      "latencyDecisions": 31733,
      "p95LatencyMs": 13.281208,
      "pairs": 200,
      "paretoOptimal": true,
      "selected": false,
      "strength": 0.39,
      "strengthIsReference": false,
      "strengthLower": 0.3475,
      "strengthUpper": 0.4325
    },
    {
      "budget": 50000,
      "family": "alpha_beta",
      "fixed": false,
      "gateEligible": true,
      "id": "alpha-beta-50k",
      "label": "Hand-evaluated AlphaBetaBot",
      "latencyDecisions": 31120,
      "p95LatencyMs": 24.273083,
      "pairs": 200,
      "paretoOptimal": true,
      "selected": true,
      "strength": 0.43,
      "strengthIsReference": false,
      "strengthLower": 0.385,
      "strengthUpper": 0.475
    },
    {
      "budget": 100000,
      "family": "jacek_inspired",
      "fixed": false,
      "gateEligible": false,
      "id": "jacek-100k",
      "label": "Neural alpha-beta (JacekInspiredBot)",
      "latencyDecisions": 33224,
      "p95LatencyMs": 58.719958,
      "pairs": 200,
      "paretoOptimal": false,
      "selected": false,
      "strength": 0.51,
      "strengthIsReference": false,
      "strengthLower": 0.4625,
      "strengthUpper": 0.5575
    },
    {
      "budget": 20000,
      "family": "jacek_inspired",
      "fixed": false,
      "gateEligible": true,
      "id": "jacek-20k",
      "label": "Neural alpha-beta (JacekInspiredBot)",
      "latencyDecisions": 33161,
      "p95LatencyMs": 35.718084,
      "pairs": 200,
      "paretoOptimal": true,
      "selected": true,
      "strength": 0.555,
      "strengthIsReference": false,
      "strengthLower": 0.51,
      "strengthUpper": 0.5975
    },
    {
      "budget": 50000,
      "family": "jacek_inspired",
      "fixed": false,
      "gateEligible": false,
      "id": "jacek-50k",
      "label": "Neural alpha-beta (JacekInspiredBot)",
      "latencyDecisions": 33369,
      "p95LatencyMs": 58.231083,
      "pairs": 200,
      "paretoOptimal": false,
      "selected": false,
      "strength": 0.5275,
      "strengthIsReference": false,
      "strengthLower": 0.4775,
      "strengthUpper": 0.5775
    },
    {
      "budget": 1000,
      "family": "mcts",
      "fixed": false,
      "gateEligible": true,
      "id": "mcts-1000",
      "label": "Tactical MctsBot",
      "latencyDecisions": 11089,
      "p95LatencyMs": 35.683625,
      "pairs": 200,
      "paretoOptimal": false,
      "selected": true,
      "strength": 0.175,
      "strengthIsReference": false,
      "strengthLower": 0.1425,
      "strengthUpper": 0.2125
    },
    {
      "budget": 2000,
      "family": "mcts",
      "fixed": false,
      "gateEligible": false,
      "id": "mcts-2000",
      "label": "Tactical MctsBot",
      "latencyDecisions": 14554,
      "p95LatencyMs": 69.956542,
      "pairs": 200,
      "paretoOptimal": false,
      "selected": false,
      "strength": 0.275,
      "strengthIsReference": false,
      "strengthLower": 0.235,
      "strengthUpper": 0.3175
    },
    {
      "budget": 4000,
      "family": "mcts",
      "fixed": false,
      "gateEligible": false,
      "id": "mcts-4000",
      "label": "Tactical MctsBot",
      "latencyDecisions": 16964,
      "p95LatencyMs": 136.38475,
      "pairs": 200,
      "paretoOptimal": false,
      "selected": false,
      "strength": 0.385,
      "strengthIsReference": false,
      "strengthLower": 0.34,
      "strengthUpper": 0.4275
    },
    {
      "budget": 50000,
      "family": "rank5_derived",
      "fixed": true,
      "gateEligible": true,
      "id": "rank5-fixed-50k",
      "label": "Rank5DerivedBot — fixed 50k demo profile",
      "latencyDecisions": 65605,
      "p95LatencyMs": 31.383417,
      "pairs": null,
      "paretoOptimal": true,
      "selected": true,
      "strength": 0.5,
      "strengthIsReference": true,
      "strengthLower": null,
      "strengthUpper": null
    }
  ]
};
