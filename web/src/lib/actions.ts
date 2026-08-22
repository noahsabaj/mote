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
