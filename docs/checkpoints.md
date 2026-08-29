# Checkpoints

Design settled 2026-08-23 (grilling rounds 1–6), **built the same day**. The problem: 28 checkpoints in
one flat, unsortable list at the bottom of the Model sheet, ordered by mtime and nothing else. Picking
one to serve meant reading every row.

## Facts that shaped it

* A checkpoint is a run: `runs/<name>/last.pt`. Splitting the run name on the first `_` gives `ab` (14),
  `overnight` (5), `sweep` (4), `pilot` (4), `smoke` (1) — families worth filtering by, derived from the
  data rather than hardcoded.
* **The "63 MB" the old rows showed was `bytes_seen`, not file size.** It is `step × batch × seq ×
  grad_accum` (`mote/infer/engine.py`), while the files on disk are 135–406 MB and appeared nowhere in the
  API. Both are worth sorting by, so both are now labelled wherever they appear.
* 2 of 28 have `val_bpb: null` (never evaluated) and many share `step: 2000`, so null handling and a
  tie-break are load-bearing, not edge cases.
* `discover_checkpoints` globs `runs/*/*.pt`, so a run that saved intermediates contributes several rows.
  Naming them all after the run puts identical-looking rows in the list that load different models —
  `displayName` keeps the file stem for anything that is not `last.pt`.
* Describing one checkpoint costs a `torch.load` and a full read of the run log, and the sheet is now
  opened far more often than the old buried section was.

## Surfaces

### 1. The composer pill — the fast path

`CheckpointPicker.svelte` takes the left slot of the composer's tools row, where Sampling used to sit;
Sampling keeps its place beside it and drops to a bare icon below 34rem. The pill reads the run name,
truncated. Opens a **popover on a laptop, a bottom sheet on a phone** — the split Claude's own apps
make, which is why Sampling also became a bottom sheet on a phone rather than leaving two adjacent
buttons behaving differently.

* Up to 5 **recently loaded** checkpoints (`model.recents`, in `localStorage`, written only by a load
  that succeeded). Before that history exists it falls back to the **5 best `val_bpb`**, so the picker
  is never empty and never useless.
* Rows are name + `step 2,000 · 1.761 bits/byte`, with a check on the loaded one. Rows do one thing —
  load — so the whole row stays a target.
* A **challenger line** below them: `Challenger <name>` with a Clear, or `Set a challenger →`. Where
  Claude's picker keeps Effort.
* `All N checkpoints →` opens the sheet.

The composer disables entirely during a swap, as before; the `Loading <id> —` line above it is the
feedback and the pill is unreachable for those seconds.

### 2. The Checkpoints sheet — the full list

`CheckpointsSheet.svelte`, a fourth `SheetView`. Reached from the picker and from the Model sheet, not
from the header: checkpoints belong to the composer, where you are when you want one.

* **Sort**: a native `<select>` over six keys — saved, bits/byte, step, name, file size, bytes seen —
  plus a direction toggle. Native so a phone gets its own picker, restyled so it does not look like a
  foreign object.
* **Nulls sink in both directions**, so ascending by bits/byte gives best-first with the un-evaluated
  ones last (verified: 0.954 first, both nulls last; descending puts 3.020 first and the nulls still
  last). Ties break by newest first.
* **Search** matches the display name, not the `runs/…/last.pt` wrapper. No autofocus — opening the
  sheet on a phone must not summon the keyboard over the list.
* **Chips**: `Evaluated` · `Beats loaded` · one per family with 2+ members. *Beats loaded* compares
  against the served checkpoint only and disables itself, with a reason, when that checkpoint has no
  `val_bpb` of its own.
* **Rows**: name, step, bits/byte, saved. Sorting by a size promotes it into the line as `file 405 MB`
  or `seen 369 MB` — never a bare figure. Load and Challenger become 40px icon buttons below 34rem.
* **Sort and filters persist** (`ckptview.svelte.ts`). Because a saved filter can hide rows on a later
  visit, `N of M shown · Clear` sits above the list whenever anything is filtering.

### 3. The Model sheet — what is running

Lost the list. Keeps its loaded-checkpoint detail block and gains a Serving / Challenger summary with a
Clear, plus `All N checkpoints →`.

## Header

One overflow menu at every viewport, holding Model, Diagnostics and Training. `Alt+1/2/3` are gone;
`Cmd/Ctrl+K` stays. Two controls in the nav rather than four, so 320px is no longer tight.

## Backend

`GET /api/checkpoints` gained `file_size_bytes` from `stat()`. The file-derived half of each row is
memoised in `_CKPT_ROWS`, keyed on the checkpoint's mtime and size **and the run log's mtime** — the log
is what gains `val_bpb` while a run is still going, and keying on the `.pt` alone would freeze a row at
null. Nothing is cached while no engine exists to describe with, for the same reason.
`tests/test_checkpoints_api.py` covers all four cases. The mock backend serves the same field; its runs
moved under `mock/` so the run names carry realistic families while the path still says what it is.

## Verified

`svelte-check` and `vite build` clean, the Python suite passing, no console errors. In the browser at
1280px and at 320/375px: no horizontal overflow, both themes, the sort rules above, chips composing to
`5 of 14 shown`, the mobile bottom sheet, the Panels menu, and a real swap through the picker updating
the pill and reordering the shortlist off `recents`.
