#!/usr/bin/env python3
"""Regenerate fastvideo.toml's `[[models]]` block from the catalog's
fastvideo models (everything else in the file is kept)."""

import json
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
catalog = json.loads((here.parents[1] / "catalog" / "catalog.json").read_text())
path = here / "fastvideo.toml"
toml = path.read_text()
block = ""
for m in catalog["models"]:
    if m["engine"] == "fastvideo":
        block += f"# {m['label']} ({m['family']}; trees: {', '.join(m['trees'])})\n[[models]]\nvariant = \"{m['id']}\"\nresident = false\n\n"
new = re.sub(r"(?s)(?:#[^\n]*\n)?\[\[models\]\].*?(?=\[protocols\])", block, toml, count=1)
path.write_text(new)
print(f"{path.name}: {block.count('[[models]]')} models", file=sys.stderr)
