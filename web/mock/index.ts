// Dev-only Vite plugin that answers the docs/api.md contract so `npm run dev` runs
// standalone. `apply: 'serve'` means none of this is reachable from a production build;
// in production the same paths are served by the Python backend on the same origin.

import type { IncomingMessage, ServerResponse } from 'node:http';
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
  res.setHeader('X-Morpheme-Mock', '1');
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

export function mockBackend(): Plugin {
  return {
    name: 'morpheme:mock-backend',
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
        '\n  [33mmorpheme mock backend[0m  /api, /v1 and /ws/generate answer with fabricated data\n'
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
    return json(
      res,
      200,
      previewContext(
        (body.messages ?? []) as { role: string; content: string }[],
        Number(body.max_bytes ?? 512),
        String(body.fold ?? 'auto'),
        (body.card ?? null) as string | null
      )
    );
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
      res.setHeader('X-Morpheme-Mock', '1');
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
