// Small DOM actions shared by the popovers and the composer.

export interface DismissOptions {
  onDismiss: () => void;
  /** element that opened the layer — clicks on it are ignored so it can toggle */
  trigger?: HTMLElement | null;
}

export function dismissable(node: HTMLElement, options: DismissOptions) {
  let opts = options;

  function onPointerDown(e: PointerEvent) {
    const target = e.target as Node;
    if (node.contains(target)) return;
    if (opts.trigger && opts.trigger.contains(target)) return;
    opts.onDismiss();
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.stopPropagation();
      opts.onDismiss();
    }
  }

  function onFocusIn(e: FocusEvent) {
    const target = e.target as Node;
    if (node.contains(target)) return;
    if (opts.trigger && opts.trigger.contains(target)) return;
    opts.onDismiss();
  }

  document.addEventListener('pointerdown', onPointerDown, true);
  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('focusin', onFocusIn);

  return {
    update(next: DismissOptions) {
      opts = next;
    },
    destroy() {
      document.removeEventListener('pointerdown', onPointerDown, true);
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('focusin', onFocusIn);
    }
  };
}

/** Grow a textarea with its content, up to a cap, without layout thrash on every keypress. */
export function autosize(node: HTMLTextAreaElement, maxPx: number) {
  let max = maxPx;
  let frame = 0;

  const measure = () => {
    frame = 0;
    node.style.height = 'auto';
    node.style.height = `${Math.min(node.scrollHeight, max)}px`;
    node.style.overflowY = node.scrollHeight > max ? 'auto' : 'hidden';
  };
  const schedule = () => {
    if (frame === 0) frame = requestAnimationFrame(measure);
  };

  measure();
  node.addEventListener('input', schedule);

  return {
    update(next: number) {
      max = next;
      schedule();
    },
    destroy() {
      node.removeEventListener('input', schedule);
      if (frame) cancelAnimationFrame(frame);
    }
  };
}

/* -------------------------------------------------------------------- tooltips */

// One tooltip exists at a time, parented to <body> so the transcript's own scroll box can
// never clip it. Opening is delayed just enough that sweeping across a row of icons does
// not flash three of them; while one is already up, its neighbours open instantly.

const TIP_DELAY = 140;
const WARM_MS = 400;

let tipNode: HTMLElement | null = null;
let tipOwner: HTMLElement | null = null;
let warmUntil = 0;
let tipSeq = 0;

function placeTip(el: HTMLElement, anchor: HTMLElement): void {
  const M = 8;
  const a = anchor.getBoundingClientRect();
  const t = el.getBoundingClientRect();
  let top = a.top - t.height - 6;
  if (top < M) top = a.bottom + 6; // no room above — sit under the control instead
  const half = a.left + a.width / 2 - t.width / 2;
  const left = Math.min(Math.max(half, M), window.innerWidth - t.width - M);
  el.style.top = `${Math.round(top)}px`;
  el.style.left = `${Math.round(left)}px`;
}

function reposition(): void {
  if (tipNode && tipOwner) placeTip(tipNode, tipOwner);
}

function hideTip(): void {
  if (!tipNode) return;
  window.removeEventListener('scroll', hideTip, true);
  window.removeEventListener('resize', reposition);
  tipNode.remove();
  tipNode = null;
  tipOwner?.removeAttribute('aria-describedby');
  tipOwner = null;
  warmUntil = Date.now() + WARM_MS;
}

function showTip(anchor: HTMLElement, label: string): void {
  hideTip();
  const el = document.createElement('div');
  el.className = 'tip';
  el.setAttribute('role', 'tooltip');
  el.id = `tip-${(tipSeq += 1)}`;
  el.textContent = label;
  document.body.appendChild(el);
  placeTip(el, anchor);
  anchor.setAttribute('aria-describedby', el.id);
  tipNode = el;
  tipOwner = anchor;
  // The anchor moves with the transcript, and a tooltip left behind is worse than none.
  window.addEventListener('scroll', hideTip, true);
  window.addEventListener('resize', reposition);
}

/** `use:tip={'Edit'}` — a hover/focus label. Pass '' to disable it for now. */
export function tip(node: HTMLElement, label: string) {
  let text = label;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const open = () => {
    if (!text) return;
    if (timer !== null) clearTimeout(timer);
    const delay = Date.now() < warmUntil ? 0 : TIP_DELAY;
    timer = setTimeout(() => {
      timer = null;
      showTip(node, text);
    }, delay);
  };

  const close = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    if (tipOwner === node) hideTip();
  };

  // A touch device has no hover, so a pointer tooltip would only appear on tap — after the
  // button has already done its job. Keyboard focus still gets one.
  const onEnter = () => {
    if (matchMedia('(hover: hover)').matches) open();
  };
  const onFocus = () => {
    if (node.matches(':focus-visible')) open();
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') close();
  };

  node.addEventListener('pointerenter', onEnter);
  node.addEventListener('pointerleave', close);
  node.addEventListener('pointerdown', close);
  node.addEventListener('focus', onFocus);
  node.addEventListener('blur', close);
  node.addEventListener('keydown', onKey);

  return {
    update(next: string) {
      text = next;
      if (tipOwner === node && tipNode) {
        tipNode.textContent = next;
        placeTip(tipNode, node);
      }
      if (!next) close();
    },
    destroy() {
      close();
      node.removeEventListener('pointerenter', onEnter);
      node.removeEventListener('pointerleave', close);
      node.removeEventListener('pointerdown', close);
      node.removeEventListener('focus', onFocus);
      node.removeEventListener('blur', close);
      node.removeEventListener('keydown', onKey);
    }
  };
}
