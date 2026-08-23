# TODO

Open development tasks, grouped by what they block. Nothing here is required
for the engine to run correctly today.

## Verification

- ~~Run `run.py test --model PATH` against the published `LiquidAI/LFM2.5-2.6B`
  weights.~~ Done: `info` and `generate` run against the published checkpoint
  (2.69B, 30 layers: 8 attention + 22 convolution); prefill and decode both
  produce coherent text.
- ~~Confirm the tokenizer flavour of the published checkpoint.~~ LFM2.5 ships
  byte level BPE (128000 pieces, 124 added tokens, llama3 split), which the
  engine reads.
- Confirm the chat template. Prompt shaping detects `<|im_start|>` in the
  vocabulary rather than evaluating the Jinja template, and `--raw` bypasses it.
  The checkpoint's chat_template.jinja is present; the detector matched the
  chatml flavour.
- ~~Build and run the test suite on Windows with MSVC.~~ Compiles clean with
  MSVC 19.42 after two fixes in app_core.c: the `near` local (Win32 macro)
  renamed to `round`, and an MSVC atomic shim (Interlocked-based) so the Win32
  thread pool engages instead of falling back to single threaded. 73 checks pass.
- Build and run the test suite on arm64. The NEON instantiation of the vector
  vocabulary has not been compiled on hardware.
- Add a perplexity command so accuracy loss from `--quant q8` can be reported as
  a number rather than as a correlation.

## Performance

- Runtime SIMD dispatch. The vector width is chosen at compile time, so a binary
  built with `-march=native` will not run on an older host. Function multi-
  versioning would let one binary pick its width at startup.
- An AVX-512 VNNI path for the q8 dot product, and an AMX path where available.
- Memoise activation quantisation. In q8 mode the same activation block is
  packed once per plane that consumes it — three times in attention, twice in
  the feed forward. One pack per block would remove roughly a third of the
  quantisation work.
- Pack weight panels for prefill. `dense` streams rows in their stored layout;
  a blocked panel layout would improve cache reuse at large batch widths.
- Fuse the attention score row. Scores are materialised per head before the
  softmax; a flash-style tiling would keep them in registers and cut the scratch
  that grows with context.
- Store the key/value cache at half width. It is f32 today, which is the
  dominant term in state memory at long context.
- Prefetch and NUMA placement for the weight mapping on large hosts.

## Formats

- q4 weights, for hosts where 3 GiB is still too much.
- Per-channel rather than per-block scales as an alternative q8 layout, to be
  measured against the current one.
- Write a packed engine file, so a q8 repack can be done once rather than at
  every load. Reading Hugging Face folders directly stays the default.

## Architecture coverage

- The mixture-of-experts variant (`model_type: "lfm2_moe"`).
- The vision variant (`model_type: "lfm2_vl"`).
- Rotary scaling types beyond `default` and `linear` — `yarn`, `llama3`,
  `dynamic`. The loader warns and falls back to plain rotary today.
- Sliding window attention, if a future member of the family uses it. The layer
  plan already carries a per-layer kind.
- Batched sequences: several states advanced in one forward pass, which would
  turn many single-token decodes into one wide matrix multiply.

## Tokenizer

- Unigram and WordPiece models, and the Metaspace pre-tokenizer. Currently
  reported as `ILL_VOCAB` rather than approximated.
- Replace the character-class range table with generated Unicode property
  tables. Runes below `0x80` follow the exact ASCII rule; above it, a rune is a
  letter unless it falls in a listed range of spaces, punctuation, or symbols.
  The ranges cover ordinary prose but not every script.
- Evaluate the Jinja `chat_template` for the subset that chat templates actually
  use, so prompt shaping comes from the checkpoint rather than from detection.
- Speed the added-token scan. It is linear in the number of added tokens at
  every input position; an Aho-Corasick automaton would make it linear in the
  input.

## Engine

- Implement a device backend against the existing `IllBackend` seam. The seam
  is in place and the CPU backend is the reference implementation of it, but
  nothing has been written on the other side of it yet.
- Save and restore a prompt cache to disk, so a long shared prefix is paid for
  once.
- Speculative decoding with a smaller draft model.
- Replay-based `ill_state_crop` for models with convolution layers. A partial
  rewind currently reports `ILL_STATE` because the convolution window is
  recurrent.
- Streaming decode of partial UTF-8. A token that ends mid-sequence is written
  as-is; buffering the tail would avoid a transient replacement character in
  terminal output.
