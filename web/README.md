# Morpheme — web client

The front end for Morpheme Studio: a single page for talking to a byte-level H-Net and, when
you want it, looking inside the stream it produces.

## Commands

```
npm install     # once (Node 24, npm 11)
npm run dev     # http://localhost:5173 — standalone, backed by the dev mock
npm run build   # → web/dist, which the Python backend serves at /
npm run check   # svelte-check, TypeScript strict; must report 0 errors
npm run preview # serve the built bundle (no mock — needs the real backend)
```

`npm install` may report that `esbuild` has an unapproved install script. Run
`npm install-scripts approve esbuild` once; Vite needs its platform binary.

## How it talks to the backend

Same origin, always. In production the FastAPI app (`morpheme.serve.app`, port 7860) serves
`web/dist` at `/` and answers `/api`, `/v1` and `/ws/generate` itself, so no proxy or base URL
is configured anywhere.

In `npm run dev` there is no Python process, so `mock/` registers a Vite plugin that answers
exactly the same paths from the dev server. It is `apply: 'serve'` only — nothing under `mock/`
is imported by `src/`, and nothing from it can reach a production build.

**The mock's data is fabricated and says so.** `status_note` begins `DEV MOCK — no model is
loaded…` and is printed verbatim in the Diagnostics sheet; the device is named
`Mock device (development server, no GPU)`; checkpoints and runs are prefixed `mock_`. The app
itself contains no placeholder values: a field the backend has not sent renders as `—`, never
as a zero.

The mock streams at 26–57 bytes/s, opens chunks on word-like units, accepts runs of bytes from
the multi-byte head — cutting a draft short sometimes, so the `fix` source and the
`spec_rounds` / `spec_fixes` / `spec_replays` counters are exercised the way the engine reports
them — includes a multi-byte UTF-8 character in every reply so the `pending` path is exercised,
and emits `stats` every 16 bytes and `diagnostics` at each chunk boundary. One
training run reports `running: true` and its log genuinely grows while the dev server is up, so
the polling path is real. One checkpoint reports `val_bpb: null`, which the backend does for any
run whose log has no eval record yet, so the "not measured" rendering is visible in dev too.

## Layout

```
index.html                  # single entry; inline SVG favicon, no external requests
vite.config.ts              # svelte plugin + dev-only mock plugin
src/
  main.ts                   # mount
  app.css                   # design tokens (light + dark), base type, shared controls
  App.svelte                # shell: header, conversation, composer, sheets, shortcuts
  lib/
    types.ts                # the docs/api.md contract, typed
    api.ts                  # HTTP client (/api/*), typed errors
    ws.ts                   # /ws/generate: reconnect with backoff, queue, cancel
    trace.svelte.ts         # compact per-byte model for one reply (typed arrays)
    format.ts               # byte/count/percent/date formatting, absolute and relative
    persist.ts              # failure-tolerant localStorage
    download.ts             # hand a generated file to the browser
    brand.ts                # the model's name, mirroring serve/identity.py
    clock.svelte.ts         # one 30 s tick shared by every relative timestamp
    actions.ts              # dismissable, autosize, tip (tooltips)
    chart.ts, views.ts      # small shared types
    stores/
      chat.svelte.ts        # transcript, conversations, streaming pipeline, export
      model.svelte.ts       # /api/model + checkpoints + hot swap
      settings.svelte.ts    # sampling params as overrides on the model's defaults
      diagnostics.svelte.ts # live values for the reply in flight
      notice.svelte.ts      # the undo bar's one transient message
      ui.svelte.ts          # view preferences, edit target, switcher state
  components/
    Header.svelte           # wordmark, conversation switcher (filter/rename/export)
    UndoBar.svelte          # "deleted — undo", above the composer
    Conversation.svelte     # transcript, opening state, follow-the-stream scrolling
    Message.svelte          # one turn, its actions and its measured footer
    ChunkedText.svelte      # plain reply, or the same reply with structure marked
    Composer.svelte         # textarea, sampling popover, context line, send/stop
    SamplingControls.svelte
    Sheet.svelte            # right-hand panel (bottom sheet under 40rem), focus-trapped
    ModelSheet.svelte       # checkpoint, architecture, device, checkpoint list + swap
    DiagnosticsSheet.svelte # throughput, multi-byte head, context, retention, exchange mass
    TrainingSheet.svelte    # run picker, curves, last evaluation
    ByteInspector.svelte    # every byte of one reply, windowed
    Curve.svelte, Sparkline.svelte, Bars.svelte, Icon.svelte, Wordmark.svelte
mock/
  index.ts                  # the Vite plugin: HTTP routes + websocket upgrade
  data.ts                   # fabricated model, checkpoints, training log
  generate.ts               # byte-stream simulator
```

## Notes on how it is built

**Streaming.** Socket events are pushed into a queue and applied once per animation frame
(with a `setTimeout` backstop, because `requestAnimationFrame` stalls in a hidden tab). Per-byte
values live in growable typed arrays inside `ByteTrace`; the only fine-grained reactive signal
is a version counter bumped on flush. No byte creates a DOM node unless you open a detail view —
a 272-byte reply keeps the whole page at ~86 elements. Structure view adds one span per chunk
(not per byte) and caps at 1500 chunks; the byte inspector is windowed, so 4096 bytes cost ~33
rows.

**Chunk spans.** `chunk` events are used only as "a chunk closed" signals. Their `start`/`end`
come from `morpheme/serve/engine.py` in whole-context coordinates (prompt included) and its
`end` is one short of its own `bytes` count, while `byte.i` is reply-local — so chunk spans are
rebuilt from the `chunk` index that every byte already carries, which is unambiguous and in the
same frame as everything else. The dev mock emits the same awkward values on purpose, so this
does not silently regress.

**Conversations.** The backend stores nothing, so the full message list is sent every turn.
The transcript is kept in `localStorage` (up to 30 conversations, plus the byte traces of the 8
most recent replies in each, so Structure and Bytes survive a reload for those). The header
switcher lists them with a title filter, inline rename, per-item delete and Markdown/JSON
export. Nothing disappears in silence: deleting shows an undo bar for eight seconds, and so does
anything that replaces turns; the conversation evicted at the 30 cap is announced even though
its storage has already gone.

**Destructive edits.** Editing a prompt or retrying it discards the replies that followed,
because they were conditioned on a continuation that no longer exists — the undo bar holds the
whole transcript for eight seconds. The one exception is retrying the *newest* prompt: there is
no tail to lose, so the old reply is kept as a sample and the `1/n` control flips between them.

**Provenance.** Each reply records the sampling parameters and checkpoint step it was drawn at,
captured at send time rather than read back later. The footer stays quiet when a reply used the
checkpoint's own defaults and names the difference when it did not; the Bytes sheet and the JSON
export carry the full values. A checkpoint hot-swapped mid-conversation draws a rule across the
transcript, because two replies from two different models otherwise look identical.

**Sampling.** `/api/model.defaults` is the baseline; your changes are stored as overrides, so
Reset restores what the checkpoint recommends and a checkpoint that ships different defaults is
honoured for anything you have not touched.

**Keyboard.** Enter sends, Shift+Enter opens a line, Up on an empty composer edits the last
prompt. Esc cancels an edit, then closes a layer, then stops a reply, then returns focus to the
composer — in that order. Ctrl/⌘+K opens the conversation switcher; Alt/⌥+1/2/3 open the three
panels. In an edit box Enter makes a line and Ctrl/⌘+Enter saves, because a prompt being
rewritten is usually several paragraphs. Modifiers only, never a bare letter: the composer is a
prose field. Alt rather than Ctrl for the digits, because the browser reserves Ctrl+1..8 for its
own tabs. Sheets trap Tab and restore focus on close.

**Theme and motion.** Both schemes are authored as tokens and chosen by `prefers-color-scheme`;
all text pairs clear WCAG AA (the faintest is 5.07:1 dark, 5.16:1 light). Animation is limited to
short entrances and the streaming caret, and `prefers-reduced-motion` disables it.

**No network at runtime.** No webfont, no CDN, no analytics. The display face is a
system serif stack (Iowan Old Style / Palatino / Charter / Cambria / Georgia); UI text is
`system-ui`; the byte inspector is `ui-monospace`.
