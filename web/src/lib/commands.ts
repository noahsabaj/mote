// The few things you can type into the composer that are instructions to the studio rather
// than messages for Mote.
//
// Parsing is deliberately exact: the whole trimmed message must be the command and nothing
// else. Mote reads raw bytes, so "/clear is a shell builtin, right?" is a question you may
// genuinely want to put to it, and a parser that swallowed it would be worse than no parser
// at all. `//clear` sends the literal six bytes, and an unknown `/celar` is simply a message.

export type CommandName = 'clear' | 'help';

export interface Command {
  name: CommandName;
  /** One line, used by both the menu and /help. */
  hint: string;
}

export const COMMANDS: Command[] = [
  { name: 'clear', hint: 'Delete this conversation and start a fresh one' },
  { name: 'help', hint: 'What you can type here' }
];

const NAMES = new Set<string>(COMMANDS.map((c) => c.name));

/** The command this message *is*, or null if it is an ordinary message. */
export function parseCommand(text: string): CommandName | null {
  const m = /^\/([a-z]+)$/.exec(text.trim());
  return m && NAMES.has(m[1]) ? (m[1] as CommandName) : null;
}

/**
 * `//clear` is you asking for the literal text, so one slash comes off. Anything else
 * beginning with `//` is left exactly as typed — a message that is only `// TODO` is a
 * message, not an escape, and quietly rewriting it would be its own small lie.
 */
export function unescapeCommand(text: string): string {
  const t = text.trim();
  const m = /^\/\/([a-z]+)$/.exec(t);
  return m && NAMES.has(m[1]) ? t.slice(1) : text;
}

/** True while the field still reads as someone part-way through typing a command. */
export function looksLikeCommand(text: string): boolean {
  return /^\/[a-z]*$/.test(text);
}

/** The menu's contents for what has been typed so far. */
export function matching(text: string): Command[] {
  const q = text.slice(1).toLowerCase();
  return COMMANDS.filter((c) => c.name.startsWith(q));
}
