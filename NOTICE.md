# Notices and license scope

The MIT license in `LICENSE` applies to original repository content authored
by Mikołaj Trębacz. It does not purport to relicense third-party material or
the underlying data identified below.

## CodinGame-derived records

Some historical experiment artifacts were built from public CodinGame arena
games involving other participants. The affected material includes:

- `submissions/codingame/bots/*/replay_book.json` and generated replay headers;
- replay-trained model JSON/headers under `submissions/codingame/bots/`;
- `arena_batch_*.json`, arena-loss regressions, and replay-screen reports; and
- generated `submission.cpp` files to the extent they embed that data.

These artifacts are retained for result verification and provenance.

No standalone redistribution license for those game records, participant
names, or other CodinGame site content was identified. The repository's MIT
license covers only the owner's original portions of combined artifacts and
grants no rights in the underlying third-party records. CodinGame's current
contest and site terms may impose additional conditions:
[CodinGame contest rules](https://www.codingame.com/rules).

The independently implemented Jacek-inspired feature representation credits
Jacek Dermont's article. The repository does not include the article text,
unpublished weights, or copied submission code, and the article itself is not
licensed by this repository:
[Inputs for Neural Networks for the Board Games: Paper Soccer](https://www.codingame.com/playgrounds/157341/inputs-for-neural-networks-for-the-board-games/paper-soccer).

## QtPaperSoccer-derived architecture

The `jacek_native_bfm` and `compact_value_bfm` CodinGame research tracks and
the offline `JacekReplayBfmBot` architecture are independent adaptations
of public architectural behavior in Jacek Dermont's
[QtPaperSoccer](https://github.com/jdermont/QtPaperSoccer), pinned at commit
`366d5304c09c2c820bd3ef4ea94624c034b8d955` (2026-03-08). The upstream project
is Apache-2.0 licensed. Its full license and the adaptation's modification and
provenance notice are retained in
`submissions/codingame/bots/jacek_native_bfm/`.

The repository's MIT license does not relicense Apache-licensed upstream
portions. No upstream network checkpoint or unpublished CodinGame source is
included. The replay BFM bootstrap weights are generated independently and
are explicitly untrained.

## Emscripten-generated browser modules

`web/papersoccer-wasm.js` and `web/papersoccer-analysis-wasm.js` are generated
with Emscripten 6.0.2 and include Emscripten runtime code and linked
system-library code. Their upstream license notices remain applicable.
Emscripten is offered under the MIT and University of Illinois/NCSA licenses;
the MIT notice is reproduced here:

Copyright (c) 2010-2014 Emscripten authors, see AUTHORS file.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The version-pinned upstream license bundle, including the Node.js notice, is
available in the
[Emscripten 6.0.2 license](https://github.com/emscripten-core/emscripten/blob/6.0.2/LICENSE).
The single-file C++/WebAssembly outputs also link selected system-library
objects. The exact retained objects are linker-dependent; the conservative
notice set for these builds is:

- [musl libc copyright and license](https://github.com/emscripten-core/emscripten/blob/6.0.2/system/lib/libc/musl/COPYRIGHT);
- [LLVM compiler-rt license](https://github.com/emscripten-core/emscripten/blob/6.0.2/system/lib/compiler-rt/LICENSE.TXT);
- [LLVM libc++ license](https://github.com/emscripten-core/emscripten/blob/6.0.2/system/lib/libcxx/LICENSE.TXT);
- [LLVM libc++abi license](https://github.com/emscripten-core/emscripten/blob/6.0.2/system/lib/libcxxabi/LICENSE.TXT); and
- [LLVM libunwind license](https://github.com/emscripten-core/emscripten/blob/6.0.2/system/lib/libunwind/LICENSE.TXT).

Those runtime portions are not relicensed under the repository's MIT license.

## Development dependency

NumPy is installed separately for research and training and is not vendored.
It remains subject to its own license expression (BSD-3-Clause, 0BSD, MIT,
Zlib, and CC0-1.0): [NumPy 2.5.1](https://pypi.org/project/numpy/2.5.1/).
