// One source for the honesty sentence, shown in Diagnostics and as the status badge's
// tooltip. `status_note` is printed exactly as the backend sends it.
import { model } from './stores/model.svelte';
import { auth } from './stores/auth.svelte';

export function statusNote(): string {
  if (auth.required) return 'This studio needs its access token before anything here is measured.';
  if (model.error) return `Backend unreachable — ${model.error} Nothing on this screen is measured.`;
  if (model.info) return model.info.status_note;
  return 'Reading the checkpoint the backend has open.';
}
