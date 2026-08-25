<script lang="ts">
  import type { Snippet } from 'svelte';
  import Icon from './Icon.svelte';
  import { ui } from '../lib/stores/ui.svelte';

  let {
    title,
    subtitle,
    onclose,
    children
  }: {
    title: string;
    subtitle?: string;
    onclose: () => void;
    children: Snippet;
  } = $props();

  let panel = $state<HTMLElement | null>(null);
  const titleId = `sheet-${Math.random().toString(36).slice(2, 8)}`;

  function focusables(): HTMLElement[] {
    if (!panel) return [];
    return Array.from(
      panel.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )
    ).filter((el) => el.offsetParent !== null || el === document.activeElement);
  }

  $effect(() => {
    const previous = document.activeElement as HTMLElement | null;
    // A sheet with something to fill in says so; the rest open on Close, which is the first
    // control in the header.
    queueMicrotask(() =>
      (panel?.querySelector<HTMLElement>('[data-autofocus]') ?? focusables()[0])?.focus()
    );
    // The opener is often a menu item that has since unmounted (the Panels menu closes as the sheet
    // opens): focus then fell to <body> and the next Tab started from the top of the page (QA
    // 2026-08-24). The composer is where the keyboard belongs when nothing is open.
    return () => {
      if (previous?.isConnected) previous.focus?.();
      else ui.focusComposer += 1;
    };
  });

  function onkeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.stopPropagation();
      onclose();
      return;
    }
    if (e.key !== 'Tab') return;
    const items = focusables();
    if (items.length === 0) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }
</script>

<div class="scrim" role="presentation" onclick={onclose}></div>
<div
  class="sheet"
  bind:this={panel}
  role="dialog"
  aria-modal="true"
  aria-labelledby={titleId}
  onkeydown={onkeydown}
  tabindex="-1"
>
  <header>
    <div class="titles">
      <h2 id={titleId}>{title}</h2>
      {#if subtitle}<p class="meta">{subtitle}</p>{/if}
    </div>
    <button class="quiet close" onclick={onclose} aria-label="Close {title}">
      <Icon name="close" />
    </button>
  </header>
  <div class="body">
    {@render children()}
  </div>
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    background: rgb(0 0 0 / 0.28);
    z-index: 40;
    animation: fade 140ms ease;
  }

  .sheet {
    position: fixed;
    z-index: 41;
    inset: 0 0 0 auto;
    width: min(30rem, 100vw);
    display: flex;
    flex-direction: column;
    background: var(--bg);
    border-left: 1px solid var(--rule);
    box-shadow: var(--shadow);
    animation: slide 180ms cubic-bezier(0.22, 0.8, 0.3, 1);
  }

  header {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1.15rem var(--gutter) 0.9rem;
    border-bottom: 1px solid var(--rule);
  }

  .titles {
    flex: 1;
    min-width: 0;
  }

  h2 {
    font-family: var(--font-read);
    font-size: 1.2rem;
    font-weight: 600;
  }

  .titles :global(p) {
    margin: 0.2rem 0 0;
  }

  .close {
    margin: -0.25rem -0.35rem 0 0;
  }

  .body {
    flex: 1;
    overflow-y: auto;
    overscroll-behavior: contain;
    padding: 1.1rem var(--gutter) 2.5rem;
  }

  @keyframes fade {
    from {
      opacity: 0;
    }
  }
  @keyframes slide {
    from {
      transform: translateX(1.5rem);
      opacity: 0;
    }
  }

  @media (max-width: 40rem) {
    .sheet {
      inset: auto 0 0 0;
      width: 100%;
      /* dvh: with vh the bottom of the drawer sat under a phone's toolbar */
      height: min(88dvh, 100%);
      border-left: 0;
      border-top: 1px solid var(--rule);
      border-radius: 16px 16px 0 0;
      animation-name: slideUp;
    }
    /* A real thumb target: the header's close control was 25×28 on a phone (QA 2026-08-24). */
    .close {
      width: 40px;
      min-height: 40px;
      justify-content: center;
      margin: -0.45rem -0.6rem 0 0;
    }
    @keyframes slideUp {
      from {
        transform: translateY(2rem);
        opacity: 0;
      }
    }
  }
</style>
