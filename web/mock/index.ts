// Dev-only Vite plugin that answers the docs/api.md contract so `npm run dev` runs
// standalone. `apply: 'serve'` means none of this is reachable from a production build;
// in production the same paths are served by the Python backend on the same origin.

import type { IncomingMessage, ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { Plugin, ViteDevServer } from 'vite';
import { WebSocketServer, type WebSocket } from 'ws';
import { CHECKPOINTS, checkpointList, modelPayload, runLog, runs, state } from './data';
import { previewContext, runGeneration, type Session } from './generate';

const SWAP_MS = 1400;

function json(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Mote-Mock', '1');
  res.end(payload);
}

async function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  if (!chunks.length) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8')) as Record<string, unknown>;
  } catch {
    return {};
  }
}

// Training jobs (docs/shape.md): an in-memory queue so the Training tab's controls work offline.
type MockJob = { id: string; argv: string[]; state: string; created_at: number; started_at: number | null; ended_at: number | null; error: string | null; resumed: boolean };
const MOCK_JOBS: MockJob[] = [];
let jobSeq = 0;

function jobsStatus() {
  return {
    current: MOCK_JOBS.find((j) => j.state === 'running') ?? null,
    queued: MOCK_JOBS.filter((j) => j.state === 'queued'),
    recent: MOCK_JOBS.filter((j) => !['queued', 'running'].includes(j.state)).slice(-10).reverse()
  };
}

// Preference votes (docs/prefs.md): kept in memory for the session, enough for the Model sheet's table.
const PREF_VOTES: { vote: string | null; a: string; b: string }[] = [];

function prefsSummary() {
  const table = new Map<string, { a: string; b: string; a_wins: number; b_wins: number; ties: number; both_bad: number; n: number }>();
  for (const v of PREF_VOTES) {
    if (!v.vote) continue;
    const [x, y] = v.a <= v.b ? [v.a, v.b] : [v.b, v.a];
    const flipped = x !== v.a;
    const row = table.get(`${x}|${y}`) ?? { a: x, b: y, a_wins: 0, b_wins: 0, ties: 0, both_bad: 0, n: 0 };
    row.n += 1;
    if (v.vote === 'tie') row.ties += 1;
    else if (v.vote === 'both_bad') row.both_bad += 1;
    else if ((v.vote === 'a') !== flipped) row.a_wins += 1;
    else row.b_wins += 1;
    table.set(`${x}|${y}`, row);
  }
  const user = PREF_VOTES.filter((v) => v.vote).length;
  return {
    pairs: PREF_VOTES.length,
    votes: { user, claude: 0 },
    unrated_by_claude: PREF_VOTES.length,
    table: [...table.values()],
    agreement: { n: 0, agree: 0, rate: null },
    rubric: 'mock00000000'
  };
}

function rubricText(): string {
  try {
    return readFileSync(resolve(__dirname, '../../docs/rubric.md'), 'utf-8');
  } catch {
    return '# Rubric\n\n(docs/rubric.md could not be read by the mock)';
  }
}

export function mockBackend(): Plugin {
  return {
    name: 'mote:mock-backend',
    apply: 'serve',
    configureServer(server: ViteDevServer) {
      server.middlewares.use((req, res, next) => {
        void handle(req, res, next);
      });

      const wss = new WebSocketServer({ noServer: true });
      server.httpServer?.on('upgrade', (req, socket, head) => {
        const path = (req.url ?? '').split('?')[0];
        if (path !== '/ws/generate') return; // leave Vite's own HMR socket alone
        wss.handleUpgrade(req, socket as never, head, (ws) => {
          wss.emit('connection', ws, req);
        });
      });
      wss.on('connection', (ws: WebSocket) => {
        let session: Session | null = null;
        ws.on('message', (raw) => {
          let msg: { type?: string } & Record<string, unknown>;
          try {
            msg = JSON.parse(String(raw)) as { type?: string };
          } catch {
            return;
          }
          if (msg.type === 'generate') {
            session?.cancel();
            session = runGeneration(ws, msg as never);
          } else if (msg.type === 'stop') {
            session?.cancel();
          }
        });
        ws.on('close', () => session?.cancel());
      });

      server.config.logger.info(
        '\n  [33mmote mock backend[0m  /api, /v1 and /ws/generate answer with fabricated data\n'
      );
    }
  };
}

async function handle(
  req: IncomingMessage,
  res: ServerResponse,
  next: (err?: unknown) => void
): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  const path = url.pathname;
  if (!path.startsWith('/api/') && !path.startsWith('/v1/')) return next();

  if (path === '/api/health') {
    return json(res, 200, { ok: true, model_loaded: !state.swapping });
  }

  if (path === '/api/model') {
    return json(res, 200, modelPayload());
  }

  if (path === '/api/checkpoints' && req.method === 'GET') {
    return json(res, 200, checkpointList());
  }

  if (path === '/api/training/queue' && req.method === 'GET') {
    return json(res, 200, jobsStatus());
  }

  if (path === '/api/training/start' && req.method === 'POST') {
    const body = await readJson(req);
    const argv = (body.args ?? []) as string[];
    const rec = { id: `j${(jobSeq += 1)}`, argv, state: 'queued', created_at: Date.now() / 1000, started_at: null as number | null, ended_at: null as number | null, error: null, resumed: false };
    MOCK_JOBS.push(rec);
    setTimeout(() => {
      if (rec.state === 'queued' && !MOCK_JOBS.some((j) => j.state === 'running')) {
        rec.state = 'running';
        rec.started_at = Date.now() / 1000;
        setTimeout(() => {
          if (rec.state === 'running') {
            rec.state = 'done';
            rec.ended_at = Date.now() / 1000;
          }
        }, 30000);
      }
    }, 1500);
    return json(res, 200, { submitted: rec.id, ...jobsStatus() });
  }

  if (path === '/api/training/stop' && req.method === 'POST') {
    const body = await readJson(req);
    const id = (body.id ?? null) as string | null;
    const rec = id ? MOCK_JOBS.find((j) => j.id === id) : MOCK_JOBS.find((j) => j.state === 'running');
    if (!rec) return json(res, 404, { detail: 'no such job (or nothing running)' });
    rec.state = 'cancelled';
    rec.ended_at = Date.now() / 1000;
    return json(res, 200, jobsStatus());
  }

  if (path === '/api/challenger/load' && req.method === 'POST') {
    const body = await readJson(req);
    const id = String(body.id ?? '');
    if (!CHECKPOINTS.some((c) => c.id === id)) return json(res, 404, { detail: `No checkpoint ${id}` });
    await new Promise((r) => setTimeout(r, 300));
    state.challengerId = id;
    return json(res, 200, modelPayload());
  }

  if (path === '/api/challenger' && req.method === 'DELETE') {
    state.challengerId = null;
    return json(res, 200, modelPayload());
  }

  if (path === '/api/prefs/vote' && req.method === 'POST') {
    const body = await readJson(req);
    const vote = body.vote as string | null;
    const pair = body.pair as { a_source?: { checkpoint?: string; step?: number }; b_source?: { checkpoint?: string; step?: number } };
    PREF_VOTES.push({ vote, a: `${pair.a_source?.checkpoint}@${pair.a_source?.step}`, b: `${pair.b_source?.checkpoint}@${pair.b_source?.step}` });
    return json(res, 200, { pair: `p${PREF_VOTES.length}`, ...prefsSummary() });
  }

  if (path === '/api/prefs/summary' && req.method === 'GET') {
    return json(res, 200, prefsSummary());
  }

  if (path === '/api/prefs/rubric' && req.method === 'GET') {
    return json(res, 200, { text: rubricText(), hash: 'mock00000000' });
  }

  if (path === '/api/checkpoints/load' && req.method === 'POST') {
    const body = await readJson(req);
    const id = String(body.id ?? '');
    if (!CHECKPOINTS.some((c) => c.id === id)) {
      return json(res, 404, { detail: `No checkpoint ${id}` });
    }
    if (state.swapping) return json(res, 503, { detail: 'A checkpoint swap is already running.' });
    state.swapping = true;
    await new Promise((r) => setTimeout(r, SWAP_MS));
    state.loadedId = id;
    state.swapping = false;
    return json(res, 200, modelPayload());
  }

  if (path === '/api/context' && req.method === 'POST') {
    const body = await readJson(req);
    const msgs = (body.messages ?? []) as { role: string; content: string }[];
    const preview = previewContext(msgs, Number(body.max_bytes ?? 512), String(body.fold ?? 'auto'), (body.card ?? null) as string | null);
    // the engine's prefix cache already holds everything before the newest user turn (see mock/generate.ts)
    const older = msgs.slice(0, -1).map((m) => `${m.role}: ${m.content}`).join('\n');
    const reusable = msgs.length > 1 ? Math.min(new TextEncoder().encode(older).length, preview.used) : 0;
    return json(res, 200, { ...preview, reusable });
  }

  if (path === '/api/training/runs' && req.method === 'GET') {
    return json(res, 200, runs());
  }

  const logMatch = /^\/api\/training\/runs\/([^/]+)\/log$/.exec(path);
  if (logMatch) {
    const id = decodeURIComponent(logMatch[1]);
    if (!runs().some((r) => r.id === id)) return json(res, 404, { detail: `No run ${id}` });
    const since = Number(url.searchParams.get('since') ?? '0');
    return json(res, 200, runLog(id, Number.isFinite(since) ? since : 0));
  }

  // Provided only so the contract is complete; the studio itself uses the websocket.
  if (path === '/v1/chat/completions' && req.method === 'POST') {
    const body = await readJson(req);
    const text = 'Dynamic chunking is a routing decision, not a lookup.';
    if (body.stream) {
      res.statusCode = 200;
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-store');
      res.setHeader('X-Mote-Mock', '1');
      for (const piece of text.match(/.{1,8}/g) ?? []) {
        res.write(
          `data: ${JSON.stringify({
            object: 'chat.completion.chunk',
            choices: [{ index: 0, delta: { content: piece }, finish_reason: null }]
          })}\n\n`
        );
        await new Promise((r) => setTimeout(r, 40));
      }
      res.write('data: [DONE]\n\n');
      res.end();
      return;
    }
    return json(res, 200, {
      object: 'chat.completion',
      choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop' }]
    });
  }

  return json(res, 404, { detail: `No mock route for ${path}` });
}
