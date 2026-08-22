// View preferences that outlive a reload but carry no data.

import * as persist from '../persist';

export type Theme = 'system' | 'light' | 'dark';
const THEMES: Theme[] = ['system', 'light', 'dark'];

class Ui {
  /** Show learned chunk boundaries and parallel acceptances inside replies. */
  structure = $state<boolean>(persist.read('ui.structure', false));
  /** 'system' follows the OS; 'light'/'dark' force a scheme via data-theme on <html>. */
  theme = $state<Theme>(persist.read('ui.theme', 'system'));

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
