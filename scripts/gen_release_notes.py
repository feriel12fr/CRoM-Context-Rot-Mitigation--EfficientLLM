#!/usr/bin/env python3
from __future__ import annotations
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
OUT = ROOT / "release_notes.md"

def main(tag: str) -> None:
    version = tag.lstrip("v").strip()
    if not CHANGELOG.exists():
        OUT.write_text(f"# Release {tag}\n\n(CHANGELOG.md not found)\n", encoding="utf-8")
        return
    text = CHANGELOG.read_text(encoding="utf-8")
    pat = re.compile(rf"^##\s*[[^{re.escape(version)}]]?[^\n]*$", re.MULTILINE)
    m = pat.search(text)
    if not m:
        OUT.write_text(
            f"# Release {tag}\n\nSection for {version} not found in CHANGELOG.\n\n" + text,
            encoding="utf-8",
        )
        return
    start = m.end()
    m2 = re.search(r"^##\s+", text[start:], re.MULTILINE)
    end = start + (m2.start() if m2 else len(text) - start)
    section = text[m.start():end].strip()
    body = f"# Release {tag}\n\n{section}\n\n— generated from [CHANGELOG.md](CHANGELOG.md)"
    OUT.write_text(body, encoding="utf-8")

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_REF_NAME", "")
    if not tag:
        print("Usage: gen_release_notes.py vX.Y.Z", file=sys.stderr)
        sys.exit(2)
    main(tag)
