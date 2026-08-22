// View preferences that outlive a reload but carry no data.

import * as persist from '../persist';

class Ui {
  /** Show learned chunk boundaries and parallel acceptances inside replies. */
  structure = $state<boolean>(persist.read('ui.structure', false));

  toggleStructure(): void {
    this.structure = !this.structure;
    persist.write('ui.structure', this.structure);
  }
}

export const ui = new Ui();
