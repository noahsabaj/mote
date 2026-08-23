// Whether the viewport is a phone, as a reactive value rather than a CSS-only fact.
//
// Most of the app answers this in a media query, which is the right tool when only the styling
// changes. The checkpoint picker changes *what it renders* — a popover over the composer on a
// laptop, a bottom sheet on a phone, the way Claude's own apps split it — and that decision has
// to be made in script. The breakpoint matches Sheet's, since a phone is exactly the width at
// which a sheet stops being a side drawer.

const PHONE = '(max-width: 40rem)';

class Layout {
  phone = $state(false);

  constructor() {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mq = window.matchMedia(PHONE);
    this.phone = mq.matches;
    mq.addEventListener('change', (e) => (this.phone = e.matches));
  }
}

export const layout = new Layout();
