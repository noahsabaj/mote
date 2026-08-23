<script lang="ts">
  // Shown when the backend answers 401 / closes the socket with 4401. One field, one button.
  import Sheet from './Sheet.svelte';
  import { auth } from '../lib/stores/auth.svelte';
  import { chat } from '../lib/stores/chat.svelte';
  import { model } from '../lib/stores/model.svelte';

  let value = $state('');

  function submit(e: SubmitEvent) {
    e.preventDefault();
    if (!value.trim()) return;
    auth.set(value);
    value = '';
    void model.refresh();
    chat.reconnect();
  }
</script>

<Sheet title="Access token" subtitle="This studio is protected" onclose={() => auth.dismiss()}>
  <form class="prompt" onsubmit={submit}>
    <p>
      {#if auth.rejected}
        The token stored in this browser was refused — the server's token has probably changed.
      {:else}
        Paste the token the server printed when it started (<code>--token</code> or
        <code>MORPHEME_TOKEN</code>). It is kept in this browser only.
      {/if}
    </p>
    <label>
      <span>Token</span>
      <input type="password" bind:value autocomplete="off" spellcheck="false" required />
    </label>
    <button type="submit" disabled={!value.trim()}>Unlock</button>
  </form>
</Sheet>

<style>
  .prompt {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    max-width: 34rem;
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
    align-self: flex-start;
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
