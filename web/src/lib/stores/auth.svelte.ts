// The studio's one secret: an access token the server may require (`--token`). Stored in
// localStorage so a phone pastes it once; revoked by changing it on the server.

import * as persist from '../persist';

const KEY = 'token';

class AuthStore {
  token = $state<string | null>(persist.read<string | null>(KEY, null));
  /** the backend refused a request: show the prompt */
  required = $state(false);
  /** a stored token was refused (as opposed to no token at all) */
  rejected = $state(false);

  require(): void {
    this.rejected = this.token !== null;
    this.required = true;
  }

  set(token: string): void {
    const t = token.trim();
    if (!t) return;
    this.token = t;
    persist.write(KEY, t);
    this.required = false;
    this.rejected = false;
  }

  dismiss(): void {
    this.required = false;
  }

  clear(): void {
    this.token = null;
    persist.drop(KEY);
  }
}

export const auth = new AuthStore();
