<script lang="ts">
  // Live values only. A field that the backend has not sent stays empty rather than showing
  // a zero, because "not measured" and "measured as zero" are different facts.
  import { diagnostics } from '../lib/stores/diagnostics.svelte';
  import { model } from '../lib/stores/model.svelte';
  import { auth } from '../lib/stores/auth.svelte';
  import { statusNote } from '../lib/status';
  import Sparkline from './Sparkline.svelte';
  import Bars from './Bars.svelte';
  import { num, pct } from '../lib/format';

  const s = $derived(diagnostics.stats);
  const d = $derived(diagnostics.latest);
  const contextFrac = $derived(s ? Math.min(1, s.context_bytes / s.context_limit) : 0);
  const status = $derived(
    auth.required ? 'locked' : model.error ? 'offline' : (model.info?.status ?? 'loading')
  );
</script>

<section class="honesty">
  <h3>Status</h3>
  <p class="tag" class:off={auth.required || !!model.error || !model.info}>{status}</p>
  <p class="meta">{statusNote()}</p>
</section>

{#if !s && !d}
  <p class="empty">
    Diagnostics are read off the running model. Send a message and they fill in as the reply
    streams.
  </p>
{:else}
  <p class="state">
    {#if diagnostics.live}
      Streaming now.
    {:else}
      From the last reply.
    {/if}
  </p>

  <section>
    <h3>Throughput</h3>
    <p class="meta">
      Decoding speed of the loaded model on this machine. Every byte is one model step.
    </p>
    <dl class="rows">
      <dt>Bytes/s</dt>
      <dd>{s ? num(s.bytes_per_sec, 1) : '—'}</dd>
      <dt>Bytes</dt>
      <dd>{s ? `${s.bytes} in ${num(s.elapsed_ms / 1000, 1)} s` : '—'}</dd>
      <dt>Bytes per chunk</dt>
      <dd>{s ? num(s.bytes_per_chunk, 2) : '—'}{s ? ` · ${s.chunks} chunks` : ''}</dd>
    </dl>
    {#if diagnostics.rate.length > 1}
      <div class="plot">
        <p class="meta">Bytes/s at each stats update, from the start of the reply.</p>
        <Sparkline
          values={diagnostics.rate}
          min={0}
          max={Math.max(...diagnostics.rate) * 1.15}
          label="Bytes per second over the reply"
          height={38}
        />
      </div>
    {/if}
  </section>

  <section>
    <h3>Context</h3>
    <p class="meta">
      Prompt plus reply so far, in bytes. When the next turn would not fit, the oldest turns are
      dropped and the reply is marked truncated.
    </p>
    <dl class="rows">
      <dt>Used</dt>
      <dd>{s ? `${s.context_bytes} of ${s.context_limit} bytes` : '—'}</dd>
    </dl>
    {#if s}
      <div class="gauge" aria-hidden="true">
        <div class="gauge-fill" style="width: {contextFrac * 100}%"></div>
      </div>
      <p class="meta">{pct(contextFrac)} of the window.</p>
    {/if}
  </section>

  <section>
    <h3>Boundary probability</h3>
    <p class="meta">
      The router's p(boundary) for the last bytes: how different each byte's encoder state is
      from the previous one. Above 0.5 a new chunk starts and the main network runs.
    </p>
    {#if d && d.boundary_probs.length}
      <div class="plot">
        <Sparkline
          values={d.boundary_probs}
          label="Boundary probability of the most recent bytes"
          height={44}
        />
      </div>
    {:else}
      <p class="none">Not reported yet.</p>
    {/if}
  </section>

  <section>
    <h3>Mamba-3 retention</h3>
    <p class="meta">
      Per head, the share of its state carried to the next byte (exp(A·Δt)) at the newest chunk
      boundary: 1.0 remembers everything so far, 0 forgets instantly. Read live from the tensors.
    </p>
    <h4>Encoder</h4>
    <Bars values={d?.mamba3.encoder_retention ?? []} prefix="h" />
    <h4>Decoder</h4>
    <Bars values={d?.mamba3.decoder_retention ?? []} prefix="h" />
  </section>

  <section>
    <h3>Relation exchange mass</h3>
    <p class="meta">
      For the newest chunk, how much of each Relation layer's output is drawn from earlier chunks
      (g) rather than from the chunk itself (1 − g). Low means "ask self", high means "ask others".
    </p>
    <Bars values={d?.relation.exchange_mass ?? []} prefix="L" />
  </section>
{/if}

<style>
  .honesty {
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--rule);
  }
  .tag {
    margin: 0 0 0.35rem;
    font-size: 0.6875rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--accent-ink);
  }
  .tag.off {
    color: var(--ink-3);
  }
  .honesty .meta {
    margin: 0;
    line-height: 1.5;
  }

  .empty,
  .state {
    margin: 0 0 1.2rem;
    font-size: 0.875rem;
    line-height: 1.6;
    color: var(--ink-2);
  }
  .state {
    margin-bottom: 1.4rem;
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

  h4 {
    margin: 0.9rem 0 0;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--ink-3);
  }

  section :global(p.meta) {
    margin: 0 0 0.5rem;
    line-height: 1.5;
  }

  .plot {
    margin-top: 0.7rem;
  }

  .gauge {
    height: 5px;
    margin-top: 0.6rem;
    border-radius: 3px;
    background: var(--surface-2);
    overflow: hidden;
  }
  .gauge-fill {
    height: 100%;
    background: var(--accent);
    transition: width 200ms ease;
  }

  .none {
    margin: 0;
    font-size: 0.8125rem;
    color: var(--ink-3);
  }
</style>
