"""Offline search index: English Wikipedia intros -> SQLite FTS5 with BM25 (docs/search.md).

    python -m morpheme.data.build_wiki_index --out data/wiki_intros.sqlite          # build (resumable)
    python -m morpheme.data.build_wiki_index --out data/wiki_intros.sqlite --query "capital of burkina faso"

Downloads FineWiki's 15 enwiki parquet files one at a time (each ~2.5 GB, deleted after indexing
unless --keep), keeps each article's intro - the text between the title heading and the first
section heading - capped at --max-bytes at a UTF-8 boundary, and writes it to an FTS5 table
(porter + unicode61 tokenizer, title weighted above body). A `done` table records finished files,
so a killed build resumes where it stopped. About 6.3M articles, ~6 GB.

`search(conn, query, k)` is the retriever the dialogue builder, the eval and the studio fallback use.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import List

REPO = "HuggingFaceFW/finewiki"
HEADING = re.compile(r"^#{1,2} ")
TOKEN = re.compile(r"\w+", re.UNICODE)


def intro_of(text: str, max_bytes: int) -> str:
    lines = text.split("\n")
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    out: List[str] = []
    for ln in lines:
        if HEADING.match(ln):
            break
        out.append(ln)
    body = " ".join(" ".join(out).split())
    b = body.encode("utf-8")
    if len(b) > max_bytes:
        b = b[:max_bytes]
        body = b.decode("utf-8", errors="ignore")
        cut = body.rfind(" ")
        if cut > max_bytes // 2:
            body = body[:cut]
        body += " …"
    return body


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS intros USING fts5(title, body, url UNINDEXED, tokenize='porter unicode61')")
    conn.execute("CREATE TABLE IF NOT EXISTS done(file TEXT PRIMARY KEY, rows INTEGER, seconds REAL)")
    return conn


def fts_query(q: str) -> str:
    """Every word quoted, so user text cannot inject FTS5 operators; implicit AND."""
    toks = [t for t in TOKEN.findall(q.lower()) if len(t) > 1][:12]
    return " ".join(f'"{t}"' for t in toks) or '""'


def search(conn: sqlite3.Connection, query: str, k: int = 3, snippet_bytes: int = 300) -> List[dict]:
    """Top-k intros by BM25 (title weighted 5x); falls back to OR-matching when AND finds nothing."""
    sql = "SELECT title, body, url FROM intros WHERE intros MATCH ? ORDER BY bm25(intros, 5.0, 1.0) LIMIT ?"
    q = fts_query(query)
    rows = conn.execute(sql, (q, k)).fetchall()
    if not rows and " " in q:
        rows = conn.execute(sql, (q.replace(" ", " OR "), k)).fetchall()
    hits = []
    for title, body, url in rows:
        b = body.encode("utf-8")[:snippet_bytes]
        snippet = b.decode("utf-8", errors="ignore")
        if len(body.encode("utf-8")) > snippet_bytes:
            snippet = snippet[: snippet.rfind(" ")] + " …" if " " in snippet else snippet + " …"
        hits.append({"title": title, "snippet": snippet, "url": url})
    return hits


def format_results(hits: List[dict], max_bytes: int = 1024) -> str:
    """The bytes the model reads: `1. Title — snippet` per line, capped; the miss case is explicit."""
    if not hits:
        return "(no results)"
    lines, used = [], 0
    for i, h in enumerate(hits, 1):
        ln = f"{i}. {h['title']} — {h['snippet']}"
        n = len(ln.encode("utf-8")) + 1
        if used + n > max_bytes and lines:
            break
        lines.append(ln)
        used += n
    return "\n".join(lines)


def build(out: Path, files: int, max_bytes: int, keep: bool, cache_dir: Path) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    names = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("data/enwiki/"))[:files]
    conn = open_db(out)
    finished = {r[0] for r in conn.execute("SELECT file FROM done")}
    total = conn.execute("SELECT COALESCE(SUM(rows), 0) FROM done").fetchone()[0]
    print(f"{len(names)} files, {len(finished)} done, {total} articles so far", flush=True)
    for name in names:
        if name in finished:
            continue
        t0 = time.time()
        local = hf_hub_download(REPO, name, repo_type="dataset", local_dir=cache_dir)
        pf = pq.ParquetFile(local)
        n = 0
        batch_rows: List[tuple] = []
        for batch in pf.iter_batches(batch_size=4000, columns=["title", "text", "url"]):
            for title, text, url in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist(), batch.column("url").to_pylist()):
                body = intro_of(text or "", max_bytes)
                if len(body.encode("utf-8")) < 40:
                    continue
                batch_rows.append((title or "", body, url or ""))
            if len(batch_rows) >= 20000:
                conn.executemany("INSERT INTO intros(title, body, url) VALUES (?, ?, ?)", batch_rows)
                conn.commit()
                n += len(batch_rows)
                batch_rows = []
        if batch_rows:
            conn.executemany("INSERT INTO intros(title, body, url) VALUES (?, ?, ?)", batch_rows)
            n += len(batch_rows)
        conn.execute("INSERT INTO done(file, rows, seconds) VALUES (?, ?, ?)", (name, n, time.time() - t0))
        conn.commit()
        total += n
        if not keep:
            try:
                os.remove(local)
            except OSError:
                pass
        print(f"{name}: {n} intros in {time.time() - t0:.0f}s, {total} total", flush=True)
    conn.execute("INSERT INTO intros(intros) VALUES('optimize')")
    conn.commit()
    conn.close()
    print(f"done: {total} articles -> {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/wiki_intros.sqlite")
    ap.add_argument("--files", type=int, default=15, help="how many of the 15 enwiki parquet files")
    ap.add_argument("--max-bytes", type=int, default=1024)
    ap.add_argument("--keep", action="store_true", help="keep the downloaded parquet files")
    ap.add_argument("--cache-dir", default="data/_finewiki_tmp")
    ap.add_argument("--query", default=None, help="search the existing index instead of building")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.query:
        conn = open_db(out)
        hits = search(conn, args.query, args.k)
        print(format_results(hits))
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    build(out, args.files, args.max_bytes, args.keep, Path(args.cache_dir))


if __name__ == "__main__":
    main()
