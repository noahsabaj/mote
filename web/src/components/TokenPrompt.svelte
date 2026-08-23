<script lang="ts">
  // Shown when the backend answers 401 / closes the socket with 4401. One field, one button.
  import Sheet from './Sheet.svelte';
  import { api, ApiError } from '../lib/api';
  import { auth } from '../lib/stores/auth.svelte';
  import { chat } from '../lib/stores/chat.svelte';
  import { model } from '../lib/stores/model.svelte';

  let code = $state('');
  let value = $state('');
  let busy = $state(false);
  let error = $state<string | null>(null);

  function unlocked(token: string) {
    auth.set(token);
    code = '';
    value = '';
    error = null;
    void model.refresh();
    chat.reconnect();
  }

  async function submitCode(e: SubmitEvent) {
    e.preventDefault();
    const c = code.replace(/\D/g, '');
    if (c.length !== 6 || busy) return;
    busy = true;
    error = null;
    try {
      const res = await api.pair(c);
      unlocked(res.token);
    } catch (err) {
      error = err instanceof ApiError ? err.message : String(err);
    } finally {
      busy = false;
    }
  }

  function submitToken(e: SubmitEvent) {
    e.preventDefault();
    if (!value.trim()) return;
    unlocked(value);
  }
</script>

<Sheet title="Pair this device" subtitle="This studio is protected" onclose={() => auth.dismiss()}>
  <div class="prompt">
    <p>
      {#if auth.rejected}
        The token stored here was refused — the server's token has probably changed. Pair again.
      {:else}
        On the PC, open <code>http://127.0.0.1:7861/pair</code>: scan its QR with this phone's camera, or type
        the six-digit code it shows.
      {/if}
    </p>
    <form class="row" onsubmit={submitCode}>
      <label>
        <span>Pairing code</span>
        <input
          class="code"
          bind:value={code}
          inputmode="numeric"
          pattern="[0-9]*"
          maxlength="6"
          autocomplete="one-time-code"
          placeholder="000000"
          data-autofocus
        />
      </label>
      <button type="submit" disabled={busy || code.replace(/\D/g, '').length !== 6}>Pair</button>
    </form>
    {#if error}<p class="error">{error}</p>{/if}
    <details>
      <summary>Paste the full token instead</summary>
      <form class="row" onsubmit={submitToken}>
        <label>
          <span>Token</span>
          <input type="password" bind:value autocomplete="off" spellcheck="false" />
        </label>
        <button type="submit" disabled={!value.trim()}>Unlock</button>
      </form>
    </details>
  </div>
</Sheet>

<style>
  .prompt {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    max-width: 34rem;
  }
  .row {
    display: flex;
    gap: 0.6rem;
    align-items: flex-end;
    flex-wrap: wrap;
  }
  .row label {
    flex: 1 1 12rem;
  }
  input.code {
    font-size: 1.6rem;
    letter-spacing: 0.25em;
    text-align: center;
    max-width: 11rem;
  }
  .error {
    color: var(--accent-ink);
  }
  details {
    font-size: 0.85rem;
    color: var(--ink-3);
  }
  details summary {
    cursor: pointer;
  }
  details form {
    margin-top: 0.6rem;
  }
  p {
    margin: 0;
    color: var(--ink-2);
    font-size: 0.9rem;
    line-height: 1.5;
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    font-size: 0.8125rem;
    color: var(--ink-3);
  }
  input {
    font: inherit;
    font-family: var(--font-mono, ui-monospace, monospace);
    padding: 0.55rem 0.7rem;
    border: 1px solid var(--rule-strong, var(--rule));
    border-radius: 8px;
    background: var(--bg);
    color: var(--ink);
  }
  input:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  button {
    font: inherit;
    font-size: 0.875rem;
    padding: 0.5rem 1rem;
    border: 1px solid var(--accent);
    border-radius: 8px;
    background: var(--accent);
    color: var(--bg);
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.45;
    cursor: default;
  }
</style>
