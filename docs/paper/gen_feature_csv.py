#!/usr/bin/env python3
"""Generate LaTeX table rows for the Feature Comparison supplementary table.

Reads xlsx2csv output from stdin, strips the General section (anything
after the first blank line), and writes LaTeX `column & column & ... \\`
rows to stdout. Parenthetical remarks in tool cells (e.g. "yes (3D volume)")
become \textsuperscript{letter} markers, with a notes block at the bottom.

Usage: xlsx2csv Feature_Comparison.xlsx | python3 gen_feature_csv.py
"""

import csv
import re
import sys


def tex_escape(s):
    """Escape LaTeX special characters."""
    return re.sub(r"([_%&$#_{}~^])", r"\\\1", s)


def parse_note(val, col_idx, tool_names, notes):
    """If val = 'yes (comment)', record footnote letter; return 'yes\\textsuperscript{X}'."""
    m = re.match(r"^(yes|no|nope)\s*\((.+)\)$", val.strip())
    if not m:
        return tex_escape(val)
    base, comment = m.group(1), m.group(2).strip()
    tool = tool_names[col_idx - 2]
    letters = "abcdefghijklmnopqrstuvwxyz"
    key = (tool, comment)
    if key not in notes:
        notes[key] = letters[len(notes)]
    return tex_escape(base) + r"\textsuperscript{" + notes[key] + r"}"


# ── Read CSV rows until first blank line ──────────────────────────────
rows = []
for row in csv.reader(sys.stdin):
    if not any(cell.strip() for cell in row):
        break
    rows.append(row)

# rows[0] = header (Feature, Description, REAVER, …, VesSkel)
# rows[1:] = feature data (78 rows, 9 columns each)
tool_names = rows[0][2:]
notes = {}

for row in rows[1:]:
    cells = [
        tex_escape(row[0]).replace(
            r"\_", r"\_\allowbreak "
        ),  # Feature — allow breaks at _
        tex_escape(row[1]),  # Description
    ]
    for i in range(2, 9):
        cells.append(parse_note(row[i], i, tool_names, notes))
    print(" & ".join(cells) + r" \\")

# ── Notes block ──────────────────────────────────────────────────────
if notes:
    print(r"\midrule")
    print(r"\multicolumn{9}{l}{\footnotesize \textbf{Notes:}} \\")
    for (tool, comment), letter in sorted(notes.items(), key=lambda x: x[1]):
        print(
            r"\multicolumn{9}{l}{\footnotesize "
            + tex_escape(f"{letter}) {tool}: {comment}")
            + r"} \\"
        )
