<script lang="ts">
  // Inline SVG only — no icon font, no emoji. 1.5px strokes on a 16 grid.
  type Name =
    | 'send'
    | 'stop'
    | 'sliders'
    | 'close'
    | 'copy'
    | 'redo'
    | 'plus'
    | 'trash'
    | 'alert'
    | 'check'
    | 'chevron'
    | 'structure'
    | 'sun'
    | 'moon'
    | 'auto'
    | 'model'
    | 'diagnostics'
    | 'training';

  let { name, size = 16 }: { name: Name; size?: number } = $props();

  const PATHS: Record<Name, string> = {
    send: 'M8 13.5V3.2M8 3.2 3.8 7.4M8 3.2l4.2 4.2',
    stop: '',
    sliders: 'M2.5 4.5h11M2.5 11.5h11',
    close: 'M4 4l8 8M12 4l-8 8',
    copy: 'M5.5 5.5V3.2A.7.7 0 0 1 6.2 2.5h6.1a.7.7 0 0 1 .7.7v6.1a.7.7 0 0 1-.7.7h-2.3M3.2 5.5h6.1a.7.7 0 0 1 .7.7v6.6a.7.7 0 0 1-.7.7H3.2a.7.7 0 0 1-.7-.7V6.2a.7.7 0 0 1 .7-.7Z',
    redo: 'M13.2 6.6A5.4 5.4 0 1 0 13.4 9M13.2 2.9v3.7h-3.7',
    plus: 'M8 3.2v9.6M3.2 8h9.6',
    trash: 'M3 4.4h10M6.4 4.4V3.1a.6.6 0 0 1 .6-.6h2a.6.6 0 0 1 .6.6v1.3M4.4 4.4l.5 8.2a.6.6 0 0 0 .6.6h5a.6.6 0 0 0 .6-.6l.5-8.2',
    alert: 'M8 5.6v3.2M8 11.4h.01M6.9 2.6 1.7 11.6a1.2 1.2 0 0 0 1.1 1.9h10.4a1.2 1.2 0 0 0 1.1-1.9L9.1 2.6a1.2 1.2 0 0 0-2.2 0Z',
    check: 'M3 8.4l3.4 3.4L13 4.6',
    chevron: 'M5.5 3.5 10 8l-4.5 4.5',
    structure: 'M2.5 4h4M9.5 4h4M2.5 12h4M9.5 12h4M8 2.2v11.6',
    sun: 'M8 1.6v1.7M8 12.7v1.7M1.6 8h1.7M12.7 8h1.7M3.5 3.5l1.2 1.2M11.3 11.3l1.2 1.2M3.5 12.5l1.2-1.2M11.3 4.7l1.2-1.2',
    moon: 'M13.2 9.8A5.6 5.6 0 0 1 6.2 2.8a5.6 5.6 0 1 0 7 7Z',
    auto: '',
    model: 'M8 1.9 13.7 5v6L8 14.1 2.3 11V5ZM2.5 5.1 8 8.1l5.5-3M8 8.1V14',
    diagnostics: 'M1.7 8.6h2.6l1.7-4.9 2.7 7.6 1.5-2.7h4.1',
    training: 'M2.6 2.2v11.2h11.2M5 10.9l2.9-3.4 2.3 1.9 3.2-4.6'
  };
</script>

<svg
  width={size}
  height={size}
  viewBox="0 0 16 16"
  fill="none"
  stroke="currentColor"
  stroke-width="1.5"
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden="true"
  focusable="false"
>
  {#if name === 'stop'}
    <rect x="4.4" y="4.4" width="7.2" height="7.2" rx="1.4" fill="currentColor" stroke="none" />
  {:else if name === 'sliders'}
    <path d={PATHS.sliders} />
    <circle cx="6" cy="4.5" r="1.9" fill="var(--bg)" />
    <circle cx="10.5" cy="11.5" r="1.9" fill="var(--bg)" />
  {:else if name === 'sun'}
    <circle cx="8" cy="8" r="2.9" />
    <path d={PATHS.sun} />
  {:else if name === 'auto'}
    <circle cx="8" cy="8" r="5.4" />
    <path d="M8 2.6a5.4 5.4 0 0 1 0 10.8Z" fill="currentColor" stroke="none" />
  {:else}
    <path d={PATHS[name]} />
  {/if}
</svg>

<style>
  svg {
    display: block;
    flex: none;
  }
</style>
