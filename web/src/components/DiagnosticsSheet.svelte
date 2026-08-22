<script lang="ts">
  // Live values only. A field that the backend has not sent stays empty rather than showing
  // a zero, because "not measured" and "measured as zero" are different facts.
  import { diagnostics } from '../lib/stores/diagnostics.svelte';
  import { settings } from '../lib/stores/settings.svelte';
  import Sparkline from './Sparkline.svelte';
  import Bars from './Bars.svelte';
  import { num, pct } from '../lib/format';

  const s = $derived(diagnostics.stats);
  const d = $derived(diagnostics.latest);
  const contextFrac = $derived(s ? Math.min(1, s.context_bytes / s.context_limit) : 0);
</script>

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
    <h3>Multi-byte head</h3>
    <dl class="rows">
      <dt>Accept rate</dt>
      <dd>{s ? pct(s.mbp_accept_rate, 1) : '—'}</dd>
      <dt>Proposed</dt>
      <dd>{s ? `${s.mbp_accepted} accepted of ${s.mbp_proposed}` : '—'}</dd>
      <dt>Threshold τ</dt>
      <dd>{settings.params.accept_threshold.toFixed(2)} · {settings.params.n_candidates} candidates</dd>
    </dl>
  </section>

  <section>
    <h3>Context</h3>
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
    <p class="meta">The router's p(boundary) over the last bytes it emitted.</p>
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
    <p class="meta">Mean exp(A·Δt) per head — how much state each head carries forward.</p>
    <h4>Encoder</h4>
    <Bars values={d?.mamba3.encoder_retention ?? []} prefix="h" />
    <h4>Decoder</h4>
    <Bars values={d?.mamba3.decoder_retention ?? []} prefix="h" />
  </section>

  <section>
    <h3>Relation exchange mass</h3>
    <p class="meta">g per layer for the newest chunk — how much it draws from earlier chunks.</p>
    <Bars values={d?.relation.exchange_mass ?? []} prefix="L" />
  </section>
{/if}

<style>
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
