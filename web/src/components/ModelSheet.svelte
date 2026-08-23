<script lang="ts">
  import { model } from '../lib/stores/model.svelte';
  import { prefs } from '../lib/stores/prefs.svelte';
  import { ui } from '../lib/stores/ui.svelte';
  import { displayName } from '../lib/checkpoints';
  import Icon from './Icon.svelte';
  import { bytes, count, minutes, num, pct, when } from '../lib/format';

  $effect(() => {
    void model.refreshCheckpoints();
    void prefs.refresh();
  });

  const info = $derived(model.info);
  const loadedCkpt = $derived(model.checkpoints.find((c) => c.loaded));
</script>

{#if model.error}
  <p class="fail">
    <Icon name="alert" size={14} />
    {model.error}
  </p>
{:else if info}
  <section>
    <p class="note"><span class="tag">{info.status}</span> {info.status_note}</p>
  </section>

  <section>
    <h3>Checkpoint</h3>
    <dl class="rows">
      <dt>Path</dt>
      <dd>{info.checkpoint.path}</dd>
      <dt>Step</dt>
      <dd>{info.checkpoint.step.toLocaleString()}</dd>
      <dt>Bytes seen</dt>
      <dd>{bytes(info.checkpoint.bytes_seen)}</dd>
      <dt>Val bits/byte</dt>
      <dd>{num(info.checkpoint.val_bpb, 3)}</dd>
      <dt>Trained</dt>
      <dd>{minutes(info.checkpoint.trained_minutes)}</dd>
      <dt>Saved</dt>
      <dd>{when(info.checkpoint.created_at)}</dd>
    </dl>
  </section>

  <section>
    <h3>Architecture</h3>
    <dl class="rows">
      <dt>Parameters</dt>
      <dd>{count(info.params)}</dd>
      <dt>Outer width</dt>
      <dd>{info.architecture.outer_width}</dd>
      <dt>Encoder</dt>
      <dd>{info.architecture.encoder_layers} layers</dd>
      <dt>Main</dt>
      <dd>{info.architecture.main}</dd>
      <dt>Decoder</dt>
      <dd>{info.architecture.decoder_layers} layers</dd>
      <dt>Multi-byte head</dt>
      <dd>{info.architecture.mbp_layers} layers</dd>
      <dt>Context</dt>
      <dd>{info.context_limit_bytes.toLocaleString()} bytes</dd>
    </dl>
  </section>

  <section>
    <h3>Identity and pushback</h3>
    <p class="meta lead">
      Greedy answers to held-out prompts — phrasings, facts and pushback wordings absent from the
      identity training data — scored on this checkpoint: whether it knows what it is, and what it
      does when it is contradicted.
    </p>
    {#if info.probe}
      <dl class="rows">
        <dt>Knows what it is</dt>
        <dd>{pct(info.probe.identity_acc, 0)} of {info.probe.n_identity} identity prompts</dd>
        <dt>Holds a right answer</dt>
        <dd>{pct(info.probe.hold_rate, 0)}</dd>
        <dt>Accepts a true correction</dt>
        <dd>{pct(info.probe.concede_rate, 0)}</dd>
        <dt>Facts probed</dt>
        <dd>{info.probe.n_facts}</dd>
        {#if info.probe.identity_acc_seen !== undefined && info.probe.hold_rate_seen !== undefined && info.probe.concede_rate_seen !== undefined}
          <dt>On training-style prompts</dt>
          <dd>{pct(info.probe.identity_acc_seen, 0)} identity · {pct(info.probe.hold_rate_seen, 0)} holds · {pct(info.probe.concede_rate_seen, 0)} concedes</dd>
        {/if}
      </dl>
    {:else}
      <p class="none">Not measured for this checkpoint.</p>
    {/if}
  </section>

  <section>
    <h3>Device</h3>
    <dl class="rows">
      <dt>Name</dt>
      <dd>{info.device.name}</dd>
      <dt>VRAM</dt>
      <dd>
        {info.device.vram_used_mb.toLocaleString()} of
        {info.device.vram_total_mb.toLocaleString()} MB in use
      </dd>
      <dt>Kernels</dt>
      <dd>
        Mamba-3 {info.kernels.mamba3 ? 'compiled' : 'PyTorch fallback'} · SSD
        {info.kernels.ssd ? 'compiled' : 'PyTorch fallback'}
      </dd>
    </dl>
  </section>

  <!-- The list itself lives in its own sheet, where it has room to be sorted and filtered
       (docs/checkpoints.md). What stays here is the answer to "what is running right now". -->
  <section>
    <h3>Checkpoints</h3>
    {#if model.checkpointError}
      <p class="fail small">
        <Icon name="alert" size={13} />
        {model.checkpointError}
      </p>
    {/if}
    <dl class="rows">
      <dt>Serving</dt>
      <dd>{loadedCkpt ? displayName(loadedCkpt.id) : displayName(info.checkpoint.path)}</dd>
      <dt>Challenger</dt>
      <dd>
        {#if info.challenger}
          {displayName(info.challenger.id)} · {num(info.challenger.val_bpb, 3)} bits/byte
          <button class="quiet inline" onclick={() => model.clearChallenger()}>Clear</button>
        {:else}
          None. Compare and arena mode need one to draw a blind second reply.
        {/if}
      </dd>
    </dl>
    <button class="quiet browse" onclick={() => (ui.sheet = 'checkpoints')}>
      All {model.checkpoints.length} checkpoints
      <Icon name="chevron" size={13} />
    </button>
  </section>

  <section>
    <h3>Preferences</h3>
    {#if prefs.error}
      <p class="fail small"><Icon name="alert" size={13} /> {prefs.error}</p>
    {:else if !prefs.summary || prefs.summary.pairs === 0}
      <p class="meta">No votes yet. Retry, Compare, or arena mode put two replies up for a vote.</p>
    {:else}
      <p class="meta">
        {prefs.summary.pairs} pairs · {prefs.summary.votes.user} of your votes · {prefs.summary.votes.claude} rater
        votes{#if prefs.summary.agreement.n > 0}
          · agreement {pct(prefs.summary.agreement.rate ?? 0)} on {prefs.summary.agreement.n}{/if}
      </p>
      {#if prefs.summary.table.length > 0}
        <ul class="ckpts">
          {#each prefs.summary.table as row (row.a + row.b)}
            <li>
              <div class="who">
                <span class="id">{row.a} vs {row.b}</span>
                <span class="meta">
                  {row.a_wins} – {row.b_wins} · {row.ties} ties · {row.both_bad} both bad · {row.n} votes
                </span>
              </div>
            </li>
          {/each}
        </ul>
      {/if}
    {/if}
  </section>
{:else}
  <p class="meta">Reading the model…</p>
{/if}

<style>
  .lead {
    margin: 0 0 0.6rem;
    line-height: 1.5;
  }

  .none {
    margin: 0;
    font-size: 0.8125rem;
    color: var(--ink-3);
  }
  section + section {
    margin-top: 1.7rem;
  }

  h3 {
    font-size: 0.9375rem;
    font-weight: 600;
    margin-bottom: 0.45rem;
  }

  .note {
    margin: 0;
    font-size: 0.875rem;
    line-height: 1.6;
    color: var(--ink-2);
  }

  .tag {
    display: inline-block;
    margin-right: 0.55em;
    font-size: 0.6875rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--accent-ink);
  }

  .fail {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin: 0 0 0.8rem;
    font-size: 0.875rem;
    color: var(--accent-ink);
  }
  .fail.small {
    font-size: 0.8125rem;
  }

  .ckpts {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .ckpts li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.7rem 0;
    border-top: 1px solid var(--rule);
  }
  .ckpts li:first-child {
    border-top: 0;
  }

  .who {
    min-width: 0;
  }

  .id {
    display: block;
    font-size: 0.875rem;
    font-family: var(--font-mono);
    overflow-wrap: anywhere;
  }
  .who :global(.meta) {
    display: block;
    margin-top: 0.1rem;
    font-size: 0.75rem;
  }

  .browse {
    width: 100%;
    justify-content: space-between;
    margin-top: 0.5rem;
    padding: 0 0.55rem;
    border: 1px solid var(--rule);
    border-radius: var(--radius-sm);
    font-size: 0.875rem;
  }
  .browse :global(svg) {
    color: var(--ink-3);
  }

  .inline {
    min-height: 0;
    padding: 0 0.2em;
    font-size: 0.75rem;
    text-decoration: underline;
    text-underline-offset: 2px;
  }
</style>
