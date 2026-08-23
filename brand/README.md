# Mote — logo candidates

**Adopted (2026-08-23): `boundary-ring`**, with two changes from the candidate below — the gap is
turned to half past one (at twelve, with something in it, it is the power glyph), and the stroke,
dot and gap grow as the tile shrinks (`ring_params` in `build.py`: 7/9/37 design units at ≥64 px,
8/9.5/38 at 32 px, 11/11/42 at 16 px) instead of scaling the 512 down to a hairline. The mark is
defined once, in `build.py`; `web/icons/make_icons.py` imports it for the app icons and
`favicon.svg`, and the header (`Wordmark.svelte`) and pairing page carry the 32-px SVG body.
The serif `m` below is the icon the studio shipped before — a letter for *Morpheme*, the
package, not a mark for *Mote*, the model.

Open `contact-sheet.html` to compare them: every mark, in both schemes, at 96 / 32 / 16 px.

## The candidates

| Mark | What it claims | Against it |
| --- | --- | --- |
| `serif-m` | The status quo, included so the others have something to beat. | A letterform, not a symbol; it says "Morpheme", not "Mote". |
| `mote` | The name, literally: one speck. The only mark in the category that isn't trying. | At a glance it's a bullet point. Needs the wordmark to carry it. |
| `boundary-ring` | A continuous stream with one break in it, and the mote sitting in the break — dynamic chunking finding its own boundary. Centred and round, so it sits in the same family as the sunburst / flower / four-pointed star without copying any of them. | The hairline thins out below 32 px; a real icon build should thicken the stroke at small sizes rather than scale it. |
| `break` | The same idea unrolled along a byte stream. | Reads as a diagram, not a badge; the only asymmetric one, which hurts it as an avatar. |
| `byte-0x6d` | The byte `01101101` — which is lowercase `m` — as eight cells. Byte-level, and the letter, encoded rather than typeset. | "Grid of squares" is the most crowded space in tech logos, and the joke needs explaining. |

## Layout

* `marks/*.svg` — the source. Transparent, `viewBox="0 0 100 100"`, drawn in `currentColor` so the
  embedding context picks the colour. This is what would go into the Svelte header.
* `tiles/*-{light,dark}.svg` — the same marks with an app-icon tile baked in, for viewing bare.
* `png/<mark>-<px>-{light,dark}.png` — 512 / 180 / 64 / 32 / 16, supersampled ×4 and downsampled,
  so the small sizes are honest about what survives.
* `contact-sheet.html` — all of it on one page.

Colours are the studio's `--accent` on `--bg`, from `web/src/app.css`: rust `#a34a1f` on cream
`#faf9f7`, peach `#e0a070` on `#131211`. Preview tiles are rounded at 22%; real iOS and Android
tiles ship square and let the OS mask them, which is what `make_icons.py` already does.

## Regenerating

```bash
python brand/build.py
```

Pillow only — no rasteriser dependency. Each mark is authored twice in `build.py`, once as an SVG
body and once as a Pillow draw function, both in the same 0–100 design units; if you edit one, edit
the other.

## When one is picked

1. Replace the glyph draw in `web/icons/make_icons.py` with that mark's draw function (or import it
   from here) and regenerate `web/public/icons/`.
2. Add `web/public/favicon.svg` from `marks/<name>.svg` and link it in `web/index.html`.
3. Drop the mark into `web/src/components/Header.svelte` beside the name.
