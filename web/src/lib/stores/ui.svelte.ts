// View preferences that outlive a reload but carry no data.

import * as persist from '../persist';

export type Theme = 'system' | 'light' | 'dark';
const THEMES: Theme[] = ['system', 'light', 'dark'];

class Ui {
  /** Show learned chunk boundaries and parallel acceptances inside replies. */
  structure = $state<boolean>(persist.read('ui.structure', false));
  /** 'system' follows the OS; 'light'/'dark' force a scheme via data-theme on <html>. */
  theme = $state<Theme>(persist.read('ui.theme', 'system'));
  /**
   * Id of the user turn open for editing, or null. It lives here rather than inside the
   * message so that only one box can be open at a time and so the composer's Up-arrow can
   * reach back into the transcript to open the last prompt.
   */
  editing = $state<string | null>(null);
  /** The header's conversation menu. Here so a keyboard shortcut can reach it. */
  switcher = $state(false);
  /**
   * The /help popover. It lives here because /help can arrive either straight from the
   * composer or from an item that was edited into a command while sitting in the queue.
   */
  help = $state(false);
  /** Bumped to ask the composer for focus; it is a signal, not a value. */
  focusComposer = $state(0);

  toggleStructure(): void {
    this.structure = !this.structure;
    persist.write('ui.structure', this.structure);
  }

  cycleTheme(): void {
    this.theme = THEMES[(THEMES.indexOf(this.theme) + 1) % THEMES.length];
    persist.write('ui.theme', this.theme);
    this.applyTheme();
  }

  applyTheme(): void {
    const el = document.documentElement;
    if (this.theme === 'system') delete el.dataset.theme;
    else el.dataset.theme = this.theme;
  }
}

export const ui = new Ui();
