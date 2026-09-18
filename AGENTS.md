# AGENTS

How to work in this repository. It is written for an agent picking the project
up cold, and it describes conventions that are already load-bearing rather than
preferences — the documents cite each other, the tests are the parity argument,
and the version log is an archive other files index into.

Read `GUIDE.md` before changing `app/core.c`. Read `CHANGES.md`'s **standing
results** before quoting a number or reopening an idea.

---

## 1. The files, and what each is for

The source is arranged in three directories: `app/` holds the engines and the
command line, `test/` holds the two test suites, and `util/` holds the Python
workflows and the tune. Checkpoints live in `ckpt/`, the tuning corpus in
`data/`.

There are **two engines**, and they share nothing but the conventions in this
file. `app/core.c` runs the Liquid text architecture from a Hugging Face
folder; `app/yolo.c` runs yolo26 from an Ultralytics `.pt`. They are separate
translation units with separate prefixes — `ill_` and `yolo_` — because a
picture model and a language model have no kernel, no store and no vocabulary
in common, and folding them together would give one file with two halves that
never call each other.

There is also **one file that is neither engine and serves both**.
`app/vulk.c` fills `IllBackend`'s six operations and `YoloBackend`'s nine over
Vulkan, and is compiled into whichever translation unit asks for it. It is not
a third engine and does not run a model; it is the other side of two seams.

| file              | what belongs in it                                                        | what must never go in it                            |
| ----------------- | ------------------------------------------------------------------------- | --------------------------------------------------- |
| `app/core.c`      | the text engine, one translation unit, fifteen layers bottom-up           | anything a second file would have to be created for |
| `app/main.c`      | the command line front end; includes `core.c`                             | engine logic                                        |
| `app/yolo.c`      | the picture engine, one translation unit, eleven layers bottom-up, with its own command line under `YOLO_MAIN` | anything the text engine also needs — copy it or leave it |
| `app/vulk.c`      | the Vulkan backend for both seams: the ABI, the SPIR-V assembler, fourteen kernels, both tables | anything an engine still needs on a host with no device |
| `test/test.c`     | the unit tests for both engines; includes `../app/core.c` and `../app/yolo.c` | anything that needs a checkpoint it cannot make     |
| `test/test.py`    | the reference comparison against `transformers`                           | anything the C unit tests already cover             |
| `util/make.py`    | install, build, test, run, bench, clean                                   | a second build system                               |
| `util/tune.py`    | the tune, the caveman rule engine, and the abliteration pass              | anything the engine does at inference               |
| `data/tune.jsonl` | the tuning corpus, one JSON row per line                                  | anything a generated corpus should hold             |
| `CHANGES.md`      | **the archive** — every version, its reasoning, its numbers, its refusals | anything still open                                 |
| `TODO.md`         | **open items only**, most consequential first                             | progress, rationale, history, closed work           |
| `README.md`       | what the engine is, how to run it, what the flags do, where it stands     | version archaeology, pass-by-pass narrative         |
| `GUIDE.md`        | a tour of the implementation, layer by layer                              | performance history                                 |
| `AGENTS.md`       | this file                                                                 | anything specific to one change                     |

The `.py` files are the reference comparison, the build workflow and the tune,
not part of the engine. `util/make.py` is the only build system there is.

**The file list is closed.** Do not add a source file, a header, or a build
tool. It has been opened twice, and both arguments are on the record because
they are the ones to make again.

At 1.7.0, for `app/yolo.c`: a picture engine shares no kernel, no store and no
vocabulary with a text one, so the alternative was one file with two halves
that never call each other. Nothing smaller than a second architecture reopens
the list on that argument.

At 1.8.0, for `app/vulk.c`, on a different one: it is a single file that is
neither engine and fills the seam of both. Folding it into `app/core.c` would
put a Vulkan loader and a SPIR-V assembler inside the text engine and leave the
picture engine unable to reach them; writing it twice would be the same two
thousand lines of device, memory, pipeline and dispatch in two places. Nothing
smaller than a second implementation of *both* seams reopens the list on that
one.

`RESEARCH.md` used to exist and was folded into the end of `TODO.md`;
`CHANGES.md` says so, and old citations to it by idea number still resolve.

**`app/vulk.c` depends on nothing at all**, and that is deliberate rather than
incidental: there is no `vulkan.h`, no SDK and no `-lvulkan`. It declares the
two dozen structures and fifty entry points it uses against the published ABI
and opens the loader with `dlopen` at run time, so a build with `--vulkan`
still compiles and still runs on a host with no Vulkan — the backend reports
itself unavailable and the caller keeps the CPU one. Its shaders are assembled into SPIR-V at run
time by the assembler in the same file, so the build gains no step and the tree
gains no opaque blob. Keep it that way: a GPU must never become a build
requirement for a project whose claim is that a C compiler is enough.

**`app/yolo.c` depends on `app/libc11/stb_image.h` and
`app/libc11/stb_image_write.h`**, which this repository already vendors, and on
nothing else. That is not a dependency being added — the headers are in the
tree and no package manager is involved — and `-DYOLO_NO_STB` builds without
them, reading and writing binary PNM only, which is what `test/test.c` does.

## 2. The versioning pattern in `CHANGES.md` — preserve it exactly

This is the rule most easily broken by accident and the most expensive to
repair, because every other document indexes into it.

- **`CHANGES.md` is append-only and ordered oldest first.** New entries go at
  the **end**. Never rewrite, reorder, renumber, or "tidy" an entry already
  there, even when a later version proves it wrong — a correction is a new
  entry that says what it corrects, which is how 0.8.11's ranking, 0.8.14's
  correction of it and 0.9.2's correction of the mlp ratio all read today.
- **One version number per change set**, as `## X.Y.Z — a short lowercase
phrase`, separated from the entry before it by a blank line, `---`, and a
  blank line. Bump the patch digit; the minor digit moves when a run of work
  closes a theme.
- **A version number marks an engine change.** Documentation-only work does not
  take one — commit it without a version, as this repository already does for
  doc commits.
- **Entry shape** for a new entry, in this order, omitting what does not apply:
  a paragraph saying what was wrong or missing; `### What it took`;
  `### What it is worth`; `### That it is the same answer`; `### Code`;
  `### Tests`. End the Tests section with `**N pass**, M before.` Early entries
  use an older set — `### Scope`, `### Why`, `### Known gaps`, `### Testing` —
  and are left as they are; match the recent shape in anything new.
- **`## Standing results` sits above the log** and holds only what spans
  versions: the hosts, the cost splits, the refusal register, the llama.cpp
  prior. Add to it when a fact outlives the version that found it. It is the
  one section of that file that may be edited in place.

## 3. `TODO.md` is a flat list of open items, by impact

- Numbered, most consequential first, engine items then research items.
- Each item is **what to do and why it is worth doing**. Nothing else — no
  progress, no measurements taken along the way, no closed sub-tasks, no
  history. Those go to `CHANGES.md` when the item closes.
- Mark **blocked** where the work is scoped but the hardware is not here, and
  **decision** where a measured win is deliberately untaken because it would
  change what the engine outputs.
- When an item closes, delete it from `TODO.md` and write the version entry.
  Do not leave a struck-through line or a "closed by" note; `CHANGES.md` is the
  ledger and searching it is the intended way to find out when something moved.
- Research items keep the original `RESEARCH.md` numbering so old citations
  resolve. Each carries its label (**established** / **adaptation** /
  **hypothesis**, and model-preserving / approximate / training required), its
  main risk, and an experiment with a **stop rule**.

## 4. Measure, then claim

The project's habit, and the reason several entries read as corrections of
earlier ones.

- **Never quote a rate without the host that produced it**, and quote that
  host's bare memory sweep beside any ratio. A ceiling is a host's, not the
  engine's. The hosts are tabulated in the standing results.
- **`GiB/s` is not comparable across bit widths.** Count multiply-adds. Two
  versions of one entry were wrong for ignoring this.
- Take a figure as the **minimum a phase reaches** over runs alternating
  between the two builds, on a quiet machine with the checkpoint in the page
  cache. Watch for a host that drifts over a series — reverse the order of the
  builds and see whether the first run of _whichever_ went first is ahead.
- **A refusal is a result and is written up with its numbers.** Say what was
  tried, what it measured, and why it does not pay. Add it to the refusal
  register so the next person does not have the same idea twice — several
  entries exist only because someone did.
- When an entry's premise turns out wrong, **close it by the correction** and
  say so plainly rather than quietly deleting it.
- Do not claim novelty. Do not present a cache hit as acceleration of the thing
  it skipped.

## 5. Output must not move, and that is checked rather than argued

- A change that is not meant to alter results must be held **bit for bit**, not
  to a tolerance. The comparison is against a binary built from the previous
  commit (`git archive HEAD app/core.c app/main.c`, build it aside, compare).
- Use greedy `chat` text for anything touching the cache path. **`logits` does
  not read `--keep`** — only `main_serve` primes from a keep file — so a
  comparison taken through `logits` tests nothing there.
- Where a change _does_ move output, it is a **decision**: say so in
  `CHANGES.md`, re-take the affected parity run, and expect the test that holds
  the old kernel bit-for-bit to fail deliberately.
- A cache or store must never hand back rows for input the caller did not
  supply. Identities are two independent mixes over the same bytes, never one,
  and the shape is compared exactly beside them.

## 6. Tests

- Every change adds tests, and the entry says how many pass and how many passed
  before.
- **Check that a new test bites.** Mutate the implementation it covers and
  confirm the test fails. Entries record this — dropping a flag from a recall
  failing two tests, removing the samples from an identity failing three.
- Tests state _what the caller may rely on_, not what the code happens to do.
  Name them as sentences.
- The suite must pass on the default and `--tuned` builds; `--wide` must at
  least compile. `python util/make.py test --tuned`.
- `--vulkan` adds the backend's checks, and they are **only** run on a host
  that has a Vulkan device; without one the suite says it skipped, which is
  right — there is nothing to compare against. A change to `app/vulk.c` is not
  tested until it has been run somewhere with a device, and
  `python util/make.py test --vulkan` is how. Its checks are a tolerance rather
  than bit for bit, because a device adds in a different order; that is a
  property of the backend and not a licence to loosen anything else.
- It must also pass on `--portable` and `--no-simd`, and the picture engine
  must report the **same detections** under all three: its output is compared
  against the reference to the digit, so a build flavour that moves it is a
  bug rather than a tolerance.

## 7. Prose style

The documents are written to be read start to finish, and the voice is
consistent across all of them. Match it.

- **Plain words over jargon**, and concrete nouns: _a picture's rows_, _a clip_,
  _the sweep_, _a run_, _what a token costs_. Not _artifacts_, _leveraging_,
  _performant_.
- **Say the number.** "18.96 ms of a 36.9 ms step — 52%" beats "a significant
  share". If there is no number, say there is none.
- **Bold the claim, not the topic.** A bolded sentence should be the thing a
  reader must not miss.
- Explain **why**, especially why something was _not_ done. A comment that says
  what the code does is worth less than one saying what was tried instead.
- Em dashes and semicolons are used freely; sentences may be long. British
  spelling. No emoji, no exclamation marks, no headings that promise more than
  the section delivers.
- Do not write "simply", "just", "obviously", or "of course" about anything that
  took a measurement to establish.

## 8. Code style

`GUIDE.md` §2 is authoritative; the short form:

- Every name is `noun_verb` or `noun_noun`, module first: `model_load`,
  `session_open`, `plane_row`, `keep_note_share`.
- Field suffixes carry meaning: `_count` a quantity, `_size` a dimension,
  `_limit` a cap, `_list` an array, `_room` scratch, `_flag` a boolean,
  `_sheet` a weight matrix, `_span` a view the engine does not own.
- C11, declarations at the top of a block, no VLAs, no compiler extensions
  outside an intrinsics block guarded by its `ILL_SIMD_NAME` level.
- The public interface lives inside `ILL_CORE_INCLUDED`, the include guard at
  the top of `app/core.c`; everything below it in the same file is private.
- A layer may depend only on the layers beneath it.
- Comments above a function explain the decision. Long comments are normal here
  and are the reason the file is readable at fourteen thousand lines.

## 9. Working on this repository

- Prepend the compiler to `PATH`, then `python util/make.py test --tuned`. There is no
  other build step. `--vulkan` compiles `app/vulk.c` in as well; it needs no
  SDK, and on a host with no device the build still succeeds and the checks
  skip.
- Do not add dependencies. Not one, not for tests, not for tooling.
- Generate test media rather than downloading it: both engines read binary PNM,
  and Python's `wave` module writes the RIFF the audio tower wants.
- The picture engine's parity argument is `ultralytics` itself, run out of the
  `ckpt/yolo26/py` submodule. It is not installed by `make.py install` and is
  not required to build or run anything; it is how a claim about matching the
  reference is checked. **Compare on the same decoded pixels** — stb and
  OpenCV do not agree on a JPEG to the last unit, so a comparison through
  `.jpg` measures the two decoders as much as the engine. Write the picture out
  as PNG or PNM first.
- Commit messages follow the entry they describe — a title naming the change,
  then prose with the numbers in it. Doc-only commits say what moved and why.
