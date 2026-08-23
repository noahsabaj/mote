<script lang="ts">
  // What /help opens. It is a popover rather than a message, because the transcript is a
  // record of what Mote actually produced — it feeds the exports and the byte traces — and
  // a fabricated turn in it would make every one of those slightly untrue.

  import { COMMANDS } from '../lib/commands';
</script>

<div class="help">
  <h2>Commands</h2>
  <dl>
    {#each COMMANDS as c (c.name)}
      <dt>/{c.name}</dt>
      <dd>{c.hint}</dd>
    {/each}
  </dl>

  <h2>Typing</h2>
  <p>
    A command has to be the whole message. <code>/clear the table</code> is a message for Mote,
    and <code>//clear</code> sends the literal text.
  </p>
  <p>
    Anything you type while a reply is running waits its turn below it. Drag a waiting item by
    its handle to reorder it, or hold Alt and use the arrow keys; click it to edit it.
    <em>Interrupt</em> ends the reply so the queue starts now.
  </p>
</div>

<style>
  .help {
    font-size: 0.8125rem;
    line-height: 1.6;
    color: var(--ink-2);
  }

  h2 {
    margin: 0 0 0.4rem;
    font-family: var(--font-read);
    font-size: 0.8125rem;
    font-weight: 600;
    color: var(--ink);
  }
  h2 + p,
  dl + h2 {
    margin-top: 0;
  }
  dl + h2 {
    margin-top: 0.9rem;
  }

  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.15rem 0.7rem;
    margin: 0;
  }
  dt {
    color: var(--accent-ink);
  }
  dd {
    margin: 0;
    color: var(--ink-3);
  }

  p {
    margin: 0.5rem 0 0;
  }

  code {
    font-family: var(--font-mono, ui-monospace, monospace);
    font-size: 0.9em;
    color: var(--ink);
  }

  em {
    font-style: normal;
    color: var(--ink);
  }
</style>
