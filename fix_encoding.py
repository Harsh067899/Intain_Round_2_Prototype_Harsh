"""Patch all src/*.py files to use encoding='utf-8' on file writes."""
import os, re

for root, dirs, files in os.walk("src"):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        fp = os.path.join(root, fn)
        with open(fp, "r", encoding="utf-8") as f:
            txt = f.read()
        # Replace open(..., "w") with open(..., "w", encoding="utf-8")
        new = txt.replace(', "w")', ', "w", encoding="utf-8")')
        new = new.replace(", 'w')", ", 'w', encoding='utf-8')")
        if new != txt:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(new)
            print(f"Patched: {fp}")
