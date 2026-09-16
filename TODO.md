# TODO (Open)

Open work, most consequential first. Nothing here is required for the engine to
run correctly today; everything here either makes it faster, makes it honest
about what it costs, or lets it read a checkpoint it currently refuses.

The order comes from where the time goes, and the cost split in `CHANGES.md`'s
standing results is the argument for it. Decode reads every weight once a token
and is within about a tenth of what the memory on the measured host can
deliver, so there is almost nothing left in the decode kernel: what remains is
to read fewer bytes, or to get more than one token out of a read. `--draft`
now does the second for greedy decoding; the items at the top do the first, and
extend the second to the sampling most callers actually use. Prefill is
arithmetic bound and still has room. Everything below that is coverage,
reliability and reach.

## Engine

1. **q4 weights.** Halving the bytes a token reads is the largest decode win
   left, and it is the item that puts the 2.6B checkpoint
   under 1.5 GiB, where a laptop with 4 GiB of usable memory can hold it. Keep
   the same per-block shape as q8 so the loader, the repack and the plane
   dispatch all widen rather than fork. `perplexity` is the instrument that
   says whether it is landable, and the row it has to beat is what q8 measured
   — see the standing results.

2. **Speculative sampling above temperature zero.** `--draft` refuses to
   engage unless sampling is greedy, because accepting a candidate on the
   ground that it equals an argmax is not a test a distribution can pass, and
   a repetition penalty rewrites the row it is applied to. Most callers do not
   decode greedily, so most callers get none of what drafting is worth. The
   fix is the standard one — accept a candidate with probability
   `min(1, p_target/p_draft)` and resample from the residual when it is
   rejected — and it needs the draft to carry a distribution rather than a
   bare token, which an n-gram scan does not have. Decide what a match is
   worth as a probability before writing any of it; that choice is the whole
   design.

3. **Fuse the attention score row.** Scores are materialised per head before
   the softmax, so the scratch grows with context and the row is written and
   read again for no reason. A flash-style tiling with a running maximum and a
   running sum keeps the row in registers, and it is the difference between
   attention that fits in cache at long context and attention that does not.
   **Decision**: the running softmax sums in a different order, so the output
   moves; re-take the parity run and say so.

4. **Pack the weight panel for prefill.** `dense` streams weight rows in their
   stored layout, so a prefill wide enough to reuse a panel still re-reads it
   from wherever it fell out to. A blocked panel layout would keep the reused
   half in cache. The companion move — fusing more activation rows against one
   weight row — is spent: eight is where the registers run out on a machine
   with thirty-two of them, and sixteen measured level with four.

5. **An AMX path for the q8 dot where the host has one.** The VNNI path folds
   four byte products into a lane; AMX does a tile at a time and is the next
   step up on the hosts that carry it. It is worth less than it looks on a
   single sequence — AMX wants many activation rows to fill a tile, so this is
   a prefill item and a batching item, not a decode item.

6. **Store the key/value cache at half width.** It is f32 today, which is the
   dominant term in state memory once the context is long — a 4096 token state
   is 0.15 GiB and the window the checkpoint advertises is 131072. bf16 halves
   it for a rounding error the keys and values already carry, since they were
   bf16 in the checkpoint. **Decision**: the scores change in the last bits, so
   the output moves.

7. **Runtime SIMD dispatch.** The vector width is chosen at compile time, so a
   binary built with `-march=native` faults on an older host and a binary built
   to be portable leaves half the machine unused. This matters more now than it
   did: the fastest path is gated on VNNI, so the gap between the portable
   build and the native one is wider than it was. Compiling the kernel set once
   per width with a target attribute and choosing at startup gives one binary
   that runs everywhere at the width the host actually has. It is a reliability
   item before it is a speed item — the failure it removes is a crash with no
   diagnosis.
8. **Rewind into the middle of a pass.** A mark can only be returned to at
   the point it was taken, so a round that rejects a proposal drops the
   confirmed tokens back into the next round's pass and carries them there.
   That costs nothing in weight reads — the pass was going to happen — but it
   widens every pass while the carry lasts, and a carry that reaches the cap
   spends a whole pass committing. Keeping the convolution signal for each
   token of a batch, rather than only the window at its end, would let the
   state stop exactly where the acceptance did. It is `conv_count * dim`
   floats a token, so it is only affordable while a batch is small, which a
   drafted batch is.


9. **Prefetch the next panel while the current one is in flight.** Decode
   streams gigabytes a token along an entirely predictable stride, and cores
   waiting on that stride are cores doing nothing. Issuing a prefetch for the
   rows a chore will reach next costs one instruction per cache line. Pair it
   with NUMA-aware placement of the weight mapping on hosts with more than one
   node, where the wrong node doubles the latency of every one of those reads.
   This is the one item that could still find something in the decode kernel,
   and the margin it is chasing is small — see the cost split.

10. **Save and restore a prompt cache to disk.** A long shared prefix — a
    system message, a document, a code file — is paid for at every start
    today. Writing the state after the prefix and reading it back turns the
    second run's prefill into a file read. The identity that says a cache
    belongs to this prompt must be two independent mixes over the same bytes
    with the shape compared beside them, as the keep file already is.

11. **Grammar-constrained sampling.** A mask over the logits that admits only
    tokens keeping the output valid against a grammar makes a malformed answer
    impossible rather than unlikely, which is worth more than any retry loop.
    Tool calling and JSON replies are the cases; the sampler layer is where the
    mask belongs, and the tokenizer already has the piece table the mask needs.
    Keep it to a grammar the engine can compile itself — no dependency.

12. **Per-channel rather than per-block q8 scales**, as an alternative layout to
    be measured against the current one. It trades a scale read per row for a
    scale read per block and may quantise better on rows with a flat range.
    There is a second reason to look now: the paired dot spends three of its
    eight operations building the two block scales into one vector, and a
    layout with fewer scales in play would not need them.

13. **Write a packed engine file**, so a q8 or q4 repack is done once rather
    than at every load. Reading Hugging Face folders directly stays the
    default; this is a second path, not a replacement, and it must carry enough
    identity that a stale pack is detected rather than used.

14. **Rotary scaling types beyond `default` and `linear`** — `yarn`, `llama3`,
    `dynamic`. The loader warns and falls back to plain rotary, which silently
    produces wrong positions past the training window on a checkpoint that uses
    one of these. This is a correctness gap wearing a coverage item's clothes.

15. **Sliding window attention**, if a member of the family uses it. The layer
    plan already carries a per-layer kind, so it is a third case rather than a
    change of shape.

16. **The mixture-of-experts variant** (`model_type: "lfm2_moe"`). A router and
    a per-token expert selection, which also makes the weight read per token
    depend on the routing — the one place in this engine where decode stops
    being a fixed stride.

17. **Batched sequences: several states advanced in one forward pass**, which
    turns many single-token decodes into one wide matrix multiply. This is the
    serving item: it does nothing for one user and most of what a server needs.
    `ill_state_mark` did the harder half of it, and item 5 wants it.

18. **Speed the added-token scan.** It is linear in the number of added tokens
    at every input position, and the published checkpoint has 124 of them. An
    Aho-Corasick automaton makes it linear in the input instead.

19. **Evaluate the Jinja `chat_template`** for the subset chat templates
    actually use, so prompt shaping comes from the checkpoint rather than from
    detecting `<|im_start|>` in the vocabulary. Detection is a guess that
    happens to be right on this family; a checkpoint that shapes turns
    differently would be shaped wrongly and produce plausible nonsense.

20. **Replace the character-class range table with generated Unicode property
    tables.** Runes below `0x80` follow the exact ASCII rule; above it a rune is
    a letter unless it falls in a listed range, and the ranges cover ordinary
    prose rather than every script. A checkpoint tokenised in a script outside
    them splits differently to the reference.

21. **Unigram and WordPiece tokenizer models, and the Metaspace pre-tokenizer.**
    Reported as `ILL_VOCAB` rather than approximated, which is the right
    refusal and still a refusal.

22. **Implement a device backend against the existing `IllBackend` seam.** The
    seam is in place and the CPU backend is the reference implementation of it;
    nothing has been written on the other side. Six operations is the whole
    surface.

## Verification

23. **Build and run the test suite on arm64.** The NEON instantiation of the
    vector vocabulary has not been compiled on hardware, and it now carries a
    paired dot that nothing has exercised there. On MSVC/arm64 the engine also
    still takes the single threaded fallback: the atomic shim's loads are plain
    volatile reads, which are acquire on x86 but not under `/volatile:iso` on
    arm64, and `ill_cpu_pause` has no MSVC arm64 arm. Both need real barriers
    before that target can enable the pool. **Blocked** on an arm64 host.

24. **Run the caveman tune against the published weights.** The pipeline is
    verified end to end on a synthetic six layer checkpoint carrying the real
    tokenizer and chat template; what is missing is the number for what the
    register costs in accuracy. **Blocked**: it needs a card, not a change.

25. **Run the abliteration against the published weights.** The drive is
    verified end to end on a synthetic four layer checkpoint over the real
    128000 entry vocabulary; what is missing is the pair of numbers that says
    what it bought and what it cost — refusals on the held out harmful prompts
    beside the base's, and a `--check` afterwards to say whether the register
    and the answers survived the edit. The search costs a hundred generations
    and a hundred forward passes per trial over two hundred trials.
    **Blocked**: it needs a card, not a change.

26. **Grow `data/tune.jsonl` past its 202 hand written rows** with `--make-data`
    against a published reasoning corpus. The press is deletion only and takes
    about 29% off the prose it is given, which is the floor rather than the
    ceiling: a hand written caveman answer restructures and reaches 2.4x. The
    rows that press well are the verbose ones, so a corpus of terse answers is
    the wrong source.

## Research

Each item carries what kind of claim it is, what it does to the output, the
risk that would sink it, and the stop rule that ends the experiment. None of
these is scheduled; they are here so the next person does not have to find them
again.

27. **Draft with the model's own q4 weights, verify with its q8 weights.**
    *Adaptation* (QuantSpec and ML-SpecQD do this on cards),
    *model-preserving* — the verifier decides every token, so the text is the
    q8 text. One checkpoint, two planes over the same rows, and the draft costs
    half the bytes of the verifier. It is a better fit here than a draft model
    because there is no second checkpoint to ship and the draft agrees with the
    verifier by construction rather than by training. **Risk**: the extra pass
    is only worth it if runs of accepted tokens are long, and q4 drift on a
    2.6B model may be enough to break them. **Experiment**: `--draft` already
    marks, verifies and rewinds, so this is a second weight plane and a draft
    pass where the n-gram scan sits; draft four tokens and record the mean
    accepted run over the tuning corpus. **Stop rule**: abandon if the mean
    accepted run is below 1.6 tokens at k=4, which is where the second weight
    read stops paying for itself.

28. **A shortlist for the vocabulary head.** *Hypothesis*, *model-preserving if
    a bound is carried, approximate otherwise*. The head is 128000 rows of
    2048, which at q8 is 262 MB of the 2.83 GiB a token reads — near a tenth
    of decode, spent to rank a vocabulary from which one token is taken.
    Cluster the rows once at load, score the cluster centroids, and expand only
    the clusters whose bound can still contain the maximum. **Risk**: the
    bound is loose enough that most clusters expand anyway, and the clustering
    costs more at load than it returns. **Experiment**: build the centroids,
    measure the fraction of rows actually touched per step at greedy and at
    top-p 0.95. **Stop rule**: abandon if more than 40% of rows are touched, at
    which point the scattered reads cost more than the sequential ones saved.

29. **Training-free activation sparsity.** *Adaptation* (TEAL), *approximate*.
    Magnitude-thresholding the hidden state before each projection lets a row
    whose activation is zero skip its weight read entirely, and the published
    result is 40-50% sparsity for a small accuracy cost on Llama-class models.
    **Risk**: the win needs a gather, and a gather over a memory bound stream
    can cost more than the sequential read it replaces — this is the failure
    mode that makes CPU sparsity papers rarer than GPU ones. **Experiment**:
    apply it to the feed forward only, where the SwiGLU gate already says which
    rows are small. **Stop rule**: abandon if decode gains less than 1.15x at
    the sparsity where `perplexity` rises by less than 0.1.

30. **Lookup-table mixed-precision matrix multiply.** *Adaptation* (T-MAC),
    *model-preserving* — exact for the format it implements. Below q8, a dot
    product can be a table lookup rather than a multiply: precompute every
    product of an activation block against the 16 possible q4 nibbles, then
    index. **Risk**: the table has to stay in the fastest cache or the random
    access costs more than the multiply saved, which is the whole difficulty.
    **Experiment**: only after item 1 exists, against its dot product.
    **Stop rule**: abandon if it does not beat the direct q4 dot by 1.2x on
    decode.

31. **Keys quantised per channel, values per token.** *Established* (KIVI),
    *approximate*. Keys carry a few channels with very large magnitudes that
    dominate a per-token range; values do not. Quantising each along the axis
    that suits it is what makes a 4-bit key/value cache hold accuracy where a
    naive one does not. **Risk**: a per-channel key scale is read across the
    grain of the score loop, which may cost more than the narrower cache saves.
    **Experiment**: after item 6, extend it downward. **Stop rule**: abandon
    below 8 bits if perplexity rises by more than 0.05.

32. **Skip attention layers to make a self-draft.** *Hypothesis*, approximate
    as a draft and *model-preserving* in what it emits, since the pass verifies.
    Twenty-two of the thirty layers are convolution, whose cost does not grow
    with context; the eight attention layers are the ones that do. A draft that
    runs the convolution layers and skips some of the attention ones is cheap
    in exactly the place a long context is expensive. **Risk**: attention is
    where this architecture does its recall, so a draft without it may agree
    only on function words. **Experiment**: drop the last four attention
    layers from the draft pass and record the mean accepted run. **Stop rule**:
    the same as item 27 — below 1.6 tokens at k=4, it does not pay.

## Non-text media

The model this engine was written for has no vision tower and no audio tower,
so nothing here is reachable from the published checkpoints. These sit last for
that reason, not because the work is small.

33. **The vision variant** (`model_type: "lfm2_vl"`). A patch embedding and an
    image tower ahead of the same stack, and a second token stream to splice
    into the prompt.

34. **An audio front end**, if a member of the family grows one. The same shape
    of problem as the vision tower: a separate encoder whose output joins the
    text stream as embeddings rather than as tokens.
