# AGENTS.md

How to work in this repository. It is written for an agent picking the project
up cold, and it describes conventions that are already load-bearing rather than
preferences — the documents cite each other, the tests are the parity argument,
and the version log is an archive other files index into.

Read `GUIDE.md` before changing `app_core.c`. Read `CHANGES.md`'s **standing
results** before quoting a number or reopening an idea.

---

## 1. The files, and what each is for

| file         | what belongs in it                                                        | what must never go in it                            |
| ------------ | ------------------------------------------------------------------------- | --------------------------------------------------- |
| `app_core.c` | the whole engine, one translation unit, eleven layers bottom-up           | anything a second file would have to be created for |
| `app_main.c` | the command line front end                                                | engine logic                                        |
| `app_test.c` | the unit tests; includes `app_core.c`                                     | anything that needs a checkpoint it cannot make     |
| `CHANGES.md` | **the archive** — every version, its reasoning, its numbers, its refusals | anything still open                                 |
| `TODO.md`    | **open items only**, most consequential first                             | progress, rationale, history, closed work           |
| `README.md`  | what the engine is, how to run it, what the flags do, where it stands     | version archaeology, pass-by-pass narrative         |
| `GUIDE.md`   | a tour of the implementation, layer by layer                              | performance history                                 |
| `AGENTS.md`  | this file                                                                 | anything specific to one change                     |

The four `.py` files are the reference comparison and the build workflow, not
part of the engine. `run.py` is the only build system there is.

**The file list is closed.** Do not add a source file, a header, or a build
tool. `RESEARCH.md` used to exist and was folded into the end of `TODO.md`;
`CHANGES.md` says so, and old citations to it by idea number still resolve.

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
  commit (`git archive HEAD app_core.c app_main.c`, build it aside, compare).
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
  least compile. `python run.py test --tuned`.

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
  outside an intrinsics block guarded by its `APP_SIMD_*` level.
- The public interface lives inside `APP_CORE_INCLUDED`; everything else is
  inside `APP_CORE_IMPLEMENTED` and is private.
- A layer may depend only on the layers beneath it.
- Comments above a function explain the decision. Long comments are normal here
  and are the reason the file is readable at fourteen thousand lines.

## 9. Working on this repository

- Prepend the compiler to `PATH`, then `python run.py test --tuned`. There is no
  other build step.
- Do not add dependencies. Not one, not for tests, not for tooling.
- Generate test media rather than downloading it: the engine reads binary PNM,
  and Python's `wave` module writes the RIFF the audio tower wants.
- Commit messages follow the entry they describe — a title naming the change,
  then prose with the numbers in it. Doc-only commits say what moved and why.
