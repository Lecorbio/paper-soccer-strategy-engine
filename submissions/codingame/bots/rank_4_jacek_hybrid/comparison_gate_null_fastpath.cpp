// DEVELOPMENT-only driver. The generic gate's Rank-4 reference slot is
// deliberately rebound to the archived pre-fastpath hybrid engine.
#define PAPER_SOCCER_GATE_RANK4_SLOT_HAS_EXACT_PROOF
#define choose_rank4 choose_prefastpath
#include "comparison_gate.cpp"
