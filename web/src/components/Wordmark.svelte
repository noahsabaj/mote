<script lang="ts">
  // The mark beside the name. The mark is the boundary ring from brand/build.py (a continuous
  // stream with one break, and the mote sitting in the break), at the 32-px optical weights since
  // it renders at about 20 px here; currentColor carries the accent in both themes.
</script>

<span class="wordmark">
  <svg class="mark" viewBox="0 0 100 100" aria-hidden="true" focusable="false">
    <circle
      cx="50"
      cy="50"
      r="27"
      fill="none"
      stroke="currentColor"
      stroke-width="8"
      stroke-dasharray="131.65 38"
      stroke-dashoffset="150.65"
      transform="rotate(-45 50 50)"
    />
    <circle cx="69.09" cy="30.91" r="9.5" fill="currentColor" />
  </svg>
  Mote
</span>

<style>
  .wordmark {
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
    font-family: var(--font-read);
    font-size: 1.0625rem;
    font-weight: 600;
    letter-spacing: 0.005em;
    color: var(--ink);
    white-space: nowrap;
    user-select: none;
  }
  .mark {
    width: 1.2em;
    height: 1.2em;
    flex: none;
    color: var(--accent);
  }
</style>
