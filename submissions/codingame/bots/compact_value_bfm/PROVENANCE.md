# Compact value BFM provenance boundary

Runtime code is a neutral adaptation of the repository's compact arena
state/topology and publicly described complete-turn best-first minimax ideas.
It deliberately imports no older BFM checkpoint, corpus, policy target,
strength claim, replay book, action lookup, or deployment receipt.

The only deployable model route is the strict content-addressed runtime
validated by `export_model.py`. The generated `submission.cpp` is derived only
from `model.hpp`, `engine.hpp`, `engine.cpp`, and `bot.cpp`. Raw campaign data,
labels, opening banks, and game transcripts are not submission inputs.
