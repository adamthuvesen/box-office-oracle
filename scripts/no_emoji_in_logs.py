#!/usr/bin/env python3
"""Pre-commit gate for emoji glyphs inside logger.* / print() calls."""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

CALL = re.compile(r"(logger\.[a-z]+|print)\(.*", re.IGNORECASE)


def has_emoji_glyph(line: str) -> bool:
    for ch in line:
        cat = unicodedata.category(ch)
        if cat in ("So", "Sk"):
            return True
        if ord(ch) > 0x2000 and cat.startswith(("S", "C")) and cat != "Cc":
            return True
    return False


def main(paths: list[str]) -> int:
    offenders: list[str] = []
    for path in map(Path, paths):
        if not path.is_file():
            continue
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if CALL.search(line) and has_emoji_glyph(line):
                offenders.append(f"{path}:{i}: {line.rstrip()}")
    if offenders:
        print("Emoji glyph found in logger/print call (block per project policy):")
        for o in offenders:
            print(f"  {o}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
