<script lang="ts">
  // The rule inside the wordmark is the same mark the training log uses to print a learned
  // chunk boundary: Morph|eme.
</script>

<span class="wordmark" aria-label="Morpheme">
  <span aria-hidden="true">Morph</span><span class="bar" aria-hidden="true"></span><span
    aria-hidden="true">eme</span
  >
</span>

<style>
  .wordmark {
    font-family: var(--font-read);
    font-size: 1.0625rem;
    font-weight: 600;
    letter-spacing: 0.005em;
    color: var(--ink);
    display: inline-flex;
    align-items: baseline;
    white-space: nowrap;
    user-select: none;
  }

  .bar {
    display: inline-block;
    width: 1px;
    height: 0.92em;
    margin: 0 0.13em;
    background: var(--accent);
    transform: translateY(0.08em);
  }
</style>
