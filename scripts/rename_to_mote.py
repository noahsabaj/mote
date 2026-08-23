"""One-shot rename Morpheme -> Mote (package, CLI, env vars, state dir, docs, web), 2026-08-23.

    python scripts/rename_to_mote.py --dry-run     # counts per file, and the lines that look linguistic
    python scripts/rename_to_mote.py --apply       # git mv + text replacements + .morpheme -> .mote

Run with a clean tree and with nothing importing the package (no training, no studio service). Skips
brand/ (the README deliberately keeps the old name), docs/research/ and docs/results/ (dated records),
and the plural "morphemes" / "a morpheme" (the linguistic term). After --apply:
    .venv\\Scripts\\pip install -e .      # the editable install is registered under the old name
    .\\mote build                         # tests, web bundle, service restart
    .\\mote service install              # new login item; delete the old "Morpheme Studio.vbs" by hand
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = ("brand/", "docs/research/", "docs/results/", "web/node_modules/", "web/dist/", "runs/", "data/")
SKIP_FILES = {"scripts/rename_to_mote.py"}
TEXT_EXT = {".py", ".md", ".toml", ".cmd", ".json", ".ts", ".svelte", ".html", ".css", ".txt", ".sh", ".webmanifest", ".yml", ".yaml", ".cfg", ".ini", ".gitignore"}
LINGUISTIC = re.compile(r"(?i)\bmorphemes\b|\ba morpheme\b|\bmorpheme-like\b|\bmorphemic\b")
PATTERNS = [
    (re.compile(r"MORPHEME(?!S\b)"), "MOTE"),
    (re.compile(r"Morpheme(?!s\b)"), "Mote"),
    (re.compile(r"morpheme(?!s\b)"), "mote"),
]


def tracked_text_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    for f in out:
        if not f or any(f.startswith(d) for d in SKIP_DIRS) or f in SKIP_FILES:
            continue
        p = ROOT / f
        if p.suffix.lower() in TEXT_EXT or p.name in (".gitignore",):
            yield f


def plan():
    per_file, flagged = {}, []
    for f in tracked_text_files():
        s = (ROOT / f).read_text(encoding="utf-8", errors="surrogateescape")
        n = sum(len(p.findall(s)) for p, _ in PATTERNS)
        if n:
            per_file[f] = n
        for i, line in enumerate(s.split("\n"), 1):
            if LINGUISTIC.search(line):
                flagged.append(f"{f}:{i}: {line.strip()[:120]}")
    return per_file, flagged


def apply():
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    dirty = [l for l in status.split("\n") if l.strip() and not l.endswith(".claude/")]
    if dirty:
        sys.exit("working tree not clean:\n" + "\n".join(dirty))
    subprocess.run(["git", "mv", "morpheme", "mote"], cwd=ROOT, check=True)
    subprocess.run(["git", "mv", "morpheme.cmd", "mote.cmd"], cwd=ROOT, check=True)
    subprocess.run(["git", "mv", "tests/test_morpheme_model.py", "tests/test_mote_model.py"], cwd=ROOT, check=True)
    changed = 0
    for f in tracked_text_files():
        p = ROOT / f
        s = p.read_text(encoding="utf-8", errors="surrogateescape")
        t = s
        for pat, rep in PATTERNS:
            t = pat.sub(rep, t)
        if t != s:
            p.write_text(t, encoding="utf-8", errors="surrogateescape")
            changed += 1
    state_old, state_new = ROOT / ".morpheme", ROOT / ".mote"
    if state_old.exists() and not state_new.exists():
        state_old.rename(state_new)
        print("renamed .morpheme/ -> .mote/")
    print(f"rewrote {changed} files; package moved to mote/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    per_file, flagged = plan()
    if args.dry_run or not args.apply:
        print(f"{sum(per_file.values())} replacements in {len(per_file)} files")
        for f, n in sorted(per_file.items(), key=lambda x: -x[1])[:25]:
            print(f"  {n:4d}  {f}")
        print("\nlinguistic uses left untouched:" if flagged else "\nno linguistic uses found")
        for l in flagged:
            print("  " + l)
        return
    apply()


if __name__ == "__main__":
    main()
