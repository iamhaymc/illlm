# TODO (Open)

Open work, most consequential first, the text engine then the picture engine.
Nothing here is required for either engine to run correctly today; everything
here either makes one faster, makes it honest about what it costs, or lets it
read a checkpoint it currently refuses.

The order comes from where the time goes, and the cost split in `CHANGES.md`'s
standing results is the argument for it. Decode reads every weight once a
token, so its rate is bytes over bandwidth and there are only two moves: read
fewer bytes, or get more than one token out of a read. Both have now been made
once — q4 for the first, `--draft` for the second — and both left something
behind. q4 came off the memory ceiling and is now held by its own kernel, and
it gives up on confident text what it saves in bytes; `--draft` only engages
for greedy sampling, which is not how most callers decode. The items at the
top are those two debts. Prefill is arithmetic bound and still has room.
Everything below that is coverage, reliability and reach.

## Engine

1. **A q4 that survives text the model is sure about.** q4 reads 0.55 of what
   q8 reads and decodes 1.23x faster, and on ordinary prose it costs nothing a
   perplexity can see — but on text the model should find easy it nearly
   doubles perplexity, 0.68 nats a token to 1.26. That is where four bits over
   a 32 value block goes: not into the model's uncertainty, into its
   confidence. Narrowing the block, or spending a second scale on the rows that
   matter, or an importance-weighted pack that fits where the activations
   actually are, would each buy some of it back. Sparing the output head does
   not — it was measured and it is worse; see the refusal register before
   trying it.

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

3. **Take q4 off the kernel ceiling.** q4 decode is the first thing in this
   engine that is not waiting on memory: 11.9 tok/s over 1.57 GiB is 20.0 GB/s
   against a 36.6 GB/s sweep, so a third of the machine's bandwidth is idle
   while the unpack and the dot catch up. The byte count says 1.8x and the
   measurement says 1.23x, and the difference is all arithmetic. It is also why
   q4 prefill is 0.93x of q8's, since prefill has no arithmetic to spare. Fewer
   instructions per unpacked block is the whole of it.

4. **Fuse the attention score row.** Scores are materialised per head before
   the softmax, so the scratch grows with context and the row is written and
   read again for no reason. A flash-style tiling with a running maximum and a
   running sum keeps the row in registers, and it is the difference between
   attention that fits in cache at long context and attention that does not.
   **Decision**: the running softmax sums in a different order, so the output
   moves; re-take the parity run and say so.

5. **Pack the weight panel for prefill.** `dense` streams weight rows in their
   stored layout, so a prefill wide enough to reuse a panel still re-reads it
   from wherever it fell out to. A blocked panel layout would keep the reused
   half in cache. The companion move — fusing more activation rows against one
   weight row — is spent: eight is where the registers run out on a machine
   with thirty-two of them, and sixteen measured level with four.

6. **An AMX path for the q8 dot where the host has one.** The VNNI path folds
   four byte products into a lane; AMX does a tile at a time and is the next
   step up on the hosts that carry it. It is worth less than it looks on a
   single sequence — AMX wants many activation rows to fill a tile, so this is
   a prefill item and a batching item, not a decode item.

7. **Store the key/value cache at half width.** It is f32 today, which is the
   dominant term in state memory once the context is long — a 4096 token state
   is 0.15 GiB and the window the checkpoint advertises is 131072. bf16 halves
   it for a rounding error the keys and values already carry, since they were
   bf16 in the checkpoint. **Decision**: the scores change in the last bits, so
   the output moves.

8. **Runtime SIMD dispatch.** The vector width is chosen at compile time, so a
   binary built with `-march=native` faults on an older host and a binary built
   to be portable leaves half the machine unused. This matters more now than it
   did: the fastest path is gated on VNNI, so the gap between the portable
   build and the native one is wider than it was. Compiling the kernel set once
   per width with a target attribute and choosing at startup gives one binary
   that runs everywhere at the width the host actually has. It is a reliability
   item before it is a speed item — the failure it removes is a crash with no
   diagnosis.
9. **Rewind into the middle of a pass.** A mark can only be returned to at
   the point it was taken, so a round that rejects a proposal drops the
   confirmed tokens back into the next round's pass and carries them there.
   That costs nothing in weight reads — the pass was going to happen — but it
   widens every pass while the carry lasts, and a carry that reaches the cap
   spends a whole pass committing. Keeping the convolution signal for each
   token of a batch, rather than only the window at its end, would let the
   state stop exactly where the acceptance did. It is `conv_count * dim`
   floats a token, so it is only affordable while a batch is small, which a
   drafted batch is.


10. **Prefetch the next panel while the current one is in flight.** Decode
    streams gigabytes a token along an entirely predictable stride, and cores
    waiting on that stride are cores doing nothing. Issuing a prefetch for the
    rows a chore will reach next costs one instruction per cache line. Pair it
    with NUMA-aware placement of the weight mapping on hosts with more than one
    node, where the wrong node doubles the latency of every one of those reads.
    This is the one item that could still find something in the decode kernel,
    and the margin it is chasing is small — see the cost split.

11. **Save and restore a prompt cache to disk.** A long shared prefix — a
    system message, a document, a code file — is paid for at every start
    today. Writing the state after the prefix and reading it back turns the
    second run's prefill into a file read. The identity that says a cache
    belongs to this prompt must be two independent mixes over the same bytes
    with the shape compared beside them, as the keep file already is.

12. **Grammar-constrained sampling.** A mask over the logits that admits only
    tokens keeping the output valid against a grammar makes a malformed answer
    impossible rather than unlikely, which is worth more than any retry loop.
    Tool calling and JSON replies are the cases; the sampler layer is where the
    mask belongs, and the tokenizer already has the piece table the mask needs.
    Keep it to a grammar the engine can compile itself — no dependency.

13. **Per-channel rather than per-block q8 scales**, as an alternative layout to
    be measured against the current one. It trades a scale read per row for a
    scale read per block and may quantise better on rows with a flat range.
    There is a second reason to look now: the paired dot spends three of its
    eight operations building the two block scales into one vector, and a
    layout with fewer scales in play would not need them.

14. **Write a packed engine file**, so a q8 or q4 repack is done once rather
    than at every load. Reading Hugging Face folders directly stays the
    default; this is a second path, not a replacement, and it must carry enough
    identity that a stale pack is detected rather than used.

15. **Rotary scaling types beyond `default` and `linear`** — `yarn`, `llama3`,
    `dynamic`. The loader warns and falls back to plain rotary, which silently
    produces wrong positions past the training window on a checkpoint that uses
    one of these. This is a correctness gap wearing a coverage item's clothes.

16. **Sliding window attention**, if a member of the family uses it. The layer
    plan already carries a per-layer kind, so it is a third case rather than a
    change of shape.

17. **The mixture-of-experts variant** (`model_type: "lfm2_moe"`). A router and
    a per-token expert selection, which also makes the weight read per token
    depend on the routing — the one place in this engine where decode stops
    being a fixed stride.

18. **Batched sequences: several states advanced in one forward pass**, which
    turns many single-token decodes into one wide matrix multiply. This is the
    serving item: it does nothing for one user and most of what a server needs.
    `ill_state_mark` did the harder half of it, and item 6 wants it.

19. **Speed the added-token scan.** It is linear in the number of added tokens
    at every input position, and the published checkpoint has 124 of them. An
    Aho-Corasick automaton makes it linear in the input instead.

20. **Evaluate the Jinja `chat_template`** for the subset chat templates
    actually use, so prompt shaping comes from the checkpoint rather than from
    detecting `<|im_start|>` in the vocabulary. Detection is a guess that
    happens to be right on this family; a checkpoint that shapes turns
    differently would be shaped wrongly and produce plausible nonsense.

21. **Replace the character-class range table with generated Unicode property
    tables.** Runes below `0x80` follow the exact ASCII rule; above it a rune is
    a letter unless it falls in a listed range, and the ranges cover ordinary
    prose rather than every script. A checkpoint tokenised in a script outside
    them splits differently to the reference.

22. **Unigram and WordPiece tokenizer models, and the Metaspace pre-tokenizer.**
    Reported as `ILL_VOCAB` rather than approximated, which is the right
    refusal and still a refusal.

23. **Implement a device backend against the existing `IllBackend` seam.** The
    seam is in place and the CPU backend is the reference implementation of it;
    nothing has been written on the other side. Six operations is the whole
    surface.

## The picture engine

`app/yolo.c` runs every published yolo26 checkpoint and matches the reference
on all five tasks. What is below is speed, reach and the parts of the store it
refuses.

These are numbered from 36 rather than from 24, because the research items
below keep their original numbering so that old citations to them resolve, and
a new item taking a used number would break that. The numbers say what an item
is, not where it sits.

The order comes from where the time goes: a 640 picture spends 98% of
its run in the forward pass, on one thread, with no intrinsics, so the first
two items are worth more than everything after them together.

36. **Run the forward pass on more than one thread.** 405 ms of a 405 ms
    detection is one core's arithmetic, and a convolution over a feature map is
    the most parallel thing in either engine — every output row is independent
    of every other. `YoloOptions.thread_count` is already in the interface and
    is ignored. The seam to split at is `conv_run`, over the output rows of one
    group, because that keeps the weight panel shared and the writes disjoint.
    `app/core.c` has a pool with the platform arms already written; copying it
    rather than sharing it is the right move here, since the file list turning
    on a shared header is what 1.7.0 refused. On a four-core host this is the
    difference between 405 ms and something near 110 ms, and it is worth more
    than every other item in this section.

37. **Write the convolution kernel in intrinsics, per width.** The tile kernel
    is a fixed-size array with constant bounds, which a compiler turns into
    vector registers and which measured 3.0x over the row-add form — but it is
    at the compiler's discretion, which is why `-O3` is 70% slower than `-O2`
    here and why the build has to say so. An `ILL_SIMD_NAME`-style guarded
    block per width, as `app/core.c` carries, makes the register allocation the
    file's decision rather than the optimiser's, and takes the flag sensitivity
    out with it. Measure against the current 376 ms before writing any of it;
    if the gap is under 1.3x it is not worth the instruction sets.

38. **Quantise the weights.** yolo26x is 142 MB of fp16, widened to 284 MB of
    f32 at load because every kernel reads floats. An f16 plane read and widened
    in the dot, as `app/core.c` does for bf16, halves the memory and is exact
    for the format; q8 halves it again for a cost that has to be measured on
    mAP rather than asserted. This matters most at the large end, where the
    weights stop fitting in cache between layers.

39. **Reuse the lowered patch matrix across a group.** `yolo_cpu_lower` builds
    one for every group of every convolution, and for a 3x3 that is nine reads
    of the input for one pass over it. A 1x1 convolution already skips it
    entirely. The two cheap moves are lowering once for all groups where the
    group count is small, and skipping the lowering for a 3x3 at stride one by
    running three row-shifted 1x1 multiplies over the input in place — which is
    the shape most of the backbone is.

40. **Batch several pictures through one forward pass.** A session runs one
    picture at a time and `YoloPlane` already carries a batch size that nothing
    sets above one. A convolution over eight pictures is one matrix multiply
    with eight times the columns, which is where the blocked kernel is at its
    best. This is the serving item, and it does nothing for one picture.

41. **A device backend against the `YoloBackend` seam.** Nine operations is the
    whole surface, the CPU table is the reference implementation of it, and
    `yolo_model_backend_set` is the switch. Nothing has been written on the
    other side. Note that `conv_room` has to answer for the backend that will
    run the convolution, not for the CPU one.

42. **Read a deflated archive.** `torch.save` writes its zip stored, so every
    published checkpoint loads; an archive that has been repacked — by a
    release pipeline, by a user unzipping and rezipping — is refused with a
    message saying so. An inflate is about 250 lines and would also let the
    engine read a `.tar.gz`. It is a reach item, not a correctness one: the
    refusal is honest and names the cause.

43. **Read a zip64 archive.** Refused by name today. No yolo26 reaches four
    gigabytes, so this is only reachable on a checkpoint nobody has published,
    and it is two extra fields in the directory walk.

44. **Widen the pickle reader to protocols 0 and 1.** The text opcodes —
    `INT`, `STRING`, `UNICODE`, `PUT`, `GET`, `OBJ`, `INST` — are refused by
    name. Torch has written protocol 2 since it started writing zips, so this
    is unreachable through `torch.save`, and it is here so that the refusal is
    a decision rather than an oversight.

45. **Verify the oriented head on a picture it was trained for.** yolo26n-obb
    matches the reference exactly on bus.png — one box, its centre, size and
    angle — but a street photograph is nothing a DOTA checkpoint was trained
    for, so that says the decode is right and not that the model is. The
    rotated path needs an aerial picture with real objects in it: the angle
    decode, `dist2rbox`, and the probabilistic overlap all go untested on
    anything but noise today. **Blocked** on a picture, not a change.

46. **Take the mask lift off a full-size plane a box.** A segmentation result
    allocates one byte a source pixel for every box kept. At the default limit
    of 300 boxes on a 4K picture that is 2.5 GiB, which nothing asks for today
    and nothing prevents either. A run-length row list, or a plane cropped to
    the box with its offset carried beside it, costs a caller one indirection
    and removes the cliff.

47. **Match Pillow's resize for the non-bilinear filters.** The classifier path
    reproduces Pillow's bilinear exactly, which is what `torchvision`'s
    `Resize` uses by default. A checkpoint whose `transforms` ask for bicubic
    or Lanczos would be shaped with the wrong filter and silently scored a few
    points off. The tap computation is already general; only the filter
    function and its support are bilinear-specific.

48. **Evaluate mAP against a dataset rather than against two pictures.** Every
    claim in `CHANGES.md` 1.7.0 is agreement with the reference on bus and
    zidane. That is the right check for an engine — it is the reference's
    answer or it is not — but it says nothing about a checkpoint over a
    distribution, and it would not catch a decode that is wrong only at a shape
    neither picture has. COCO val is 5000 pictures and the engine is fast
    enough to run it. **Blocked** on the dataset, not a change.

49. **The remaining yolo families.** `A2C2f` and `ABlock` (yolo12), `RepConv`
    and `RepVGGDW`, `WorldDetect` and the `YOLOE` heads, and `RTDETRDecoder`
    are reported as `YOLO_ERR_MODULE` naming the module, which is the right
    refusal and still a refusal. Each is a few dozen lines against the graph
    builder and the forward pass, and the pickle already hands over everything
    they need. Take them in the order someone asks for them.

50. **Read a `.pt` that holds a bare `state_dict`.** The loader wants the
    module tree, because that is what carries the graph. A checkpoint saved as
    weights alone — which is what a training script that calls
    `torch.save(model.state_dict())` produces — has no graph in it at all, and
    would need the yaml path this engine deliberately does not have. Decide
    whether that is worth a second loader before writing one; the refusal names
    the cause today.

## Verification

24. **Build and run the test suite on arm64.** The NEON instantiation of the
    vector vocabulary has not been compiled on hardware, and it now carries a
    paired dot that nothing has exercised there. On MSVC/arm64 the engine also
    still takes the single threaded fallback: the atomic shim's loads are plain
    volatile reads, which are acquire on x86 but not under `/volatile:iso` on
    arm64, and `ill_cpu_pause` has no MSVC arm64 arm. Both need real barriers
    before that target can enable the pool. **Blocked** on an arm64 host.

25. **Run the caveman tune against the published weights.** The pipeline is
    verified end to end on a synthetic six layer checkpoint carrying the real
    tokenizer and chat template; what is missing is the number for what the
    register costs in accuracy. **Blocked**: it needs a card, not a change.

26. **Run the abliteration against the published weights.** The drive is
    verified end to end on a synthetic four layer checkpoint over the real
    128000 entry vocabulary; what is missing is the pair of numbers that says
    what it bought and what it cost — refusals on the held out harmful prompts
    beside the base's, and a `--check` afterwards to say whether the register
    and the answers survived the edit. The search costs a hundred generations
    and a hundred forward passes per trial over two hundred trials.
    **Blocked**: it needs a card, not a change.

27. **Grow `data/tune.jsonl` past its 202 hand written rows** with `--make-data`
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

28. **Draft with the model's own q4 weights, verify with its q8 weights.**
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

29. **A shortlist for the vocabulary head.** *Hypothesis*, *model-preserving if
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

30. **Training-free activation sparsity.** *Adaptation* (TEAL), *approximate*.
    Magnitude-thresholding the hidden state before each projection lets a row
    whose activation is zero skip its weight read entirely, and the published
    result is 40-50% sparsity for a small accuracy cost on Llama-class models.
    **Risk**: the win needs a gather, and a gather over a memory bound stream
    can cost more than the sequential read it replaces — this is the failure
    mode that makes CPU sparsity papers rarer than GPU ones. **Experiment**:
    apply it to the feed forward only, where the SwiGLU gate already says which
    rows are small. **Stop rule**: abandon if decode gains less than 1.15x at
    the sparsity where `perplexity` rises by less than 0.1.

31. **Lookup-table mixed-precision matrix multiply.** *Adaptation* (T-MAC),
    *model-preserving* — exact for the format it implements. Below q8, a dot
    product can be a table lookup rather than a multiply: precompute every
    product of an activation block against the 16 possible q4 nibbles, then
    index. **Risk**: the table has to stay in the fastest cache or the random
    access costs more than the multiply saved, which is the whole difficulty.
    **Experiment**: against `ill_dense_nib`'s dot, which q4 now provides.
    **Stop rule**: abandon if it does not beat the direct q4 dot by 1.2x on
    decode.

32. **Keys quantised per channel, values per token.** *Established* (KIVI),
    *approximate*. Keys carry a few channels with very large magnitudes that
    dominate a per-token range; values do not. Quantising each along the axis
    that suits it is what makes a 4-bit key/value cache hold accuracy where a
    naive one does not. **Risk**: a per-channel key scale is read across the
    grain of the score loop, which may cost more than the narrower cache saves.
    **Experiment**: after item 7, extend it downward. **Stop rule**: abandon
    below 8 bits if perplexity rises by more than 0.05.

33. **Skip attention layers to make a self-draft.** *Hypothesis*, approximate
    as a draft and *model-preserving* in what it emits, since the pass verifies.
    Twenty-two of the thirty layers are convolution, whose cost does not grow
    with context; the eight attention layers are the ones that do. A draft that
    runs the convolution layers and skips some of the attention ones is cheap
    in exactly the place a long context is expensive. **Risk**: attention is
    where this architecture does its recall, so a draft without it may agree
    only on function words. **Experiment**: drop the last four attention
    layers from the draft pass and record the mean accepted run. **Stop rule**:
    the same as item 28 — below 1.6 tokens at k=4, it does not pay.

## Non-text media

The model this engine was written for has no vision tower and no audio tower,
so nothing here is reachable from the published checkpoints. These sit last for
that reason, not because the work is small.

34. **The vision variant** (`model_type: "lfm2_vl"`). A patch embedding and an
    image tower ahead of the same stack, and a second token stream to splice
    into the prompt.

35. **An audio front end**, if a member of the family grows one. The same shape
    of problem as the vision tower: a separate encoder whose output joins the
    text stream as embeddings rather than as tokens.
