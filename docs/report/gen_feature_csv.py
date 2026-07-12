#!/usr/bin/env python3
"""Generate LaTeX table rows for the Feature Comparison supplementary table.

Reads the Feature Comparison spreadsheet via xlsx2csv, strips the General
section (anything after the first blank line), and writes LaTeX
`column & column & ... \\` rows to stdout. Parenthetical remarks in tool
cells (e.g. "yes (3D volume)") become \textsuperscript{letter} markers,
with a notes block at the bottom.

Usage: python3 gen_feature_csv.py
       python3 gen_feature_csv.py <path/to/Feature_Comparison.xlsx>
"""

import csv
import os
import re
import subprocess
import sys


def tex_escape(s):
    """Escape LaTeX special characters."""
    return re.sub(r"([_%&$#_{}~^])", r"\\\1", s)


def normalize_val(val):
    """Normalise 'omitted' to 'no' for uniform display."""
    v = val.strip()
    if v.lower().startswith("omitted"):
        return "no" + v[len("omitted") :]
    return v


def parse_note(val, col_idx, tool_names, notes):
    """If val = 'yes (comment)', record footnote letter; return 'yes\\textsuperscript{X}'."""
    m = re.match(r"^(yes|no|nope|omitted)\s*\((.+)\)$", val.strip())
    if not m:
        return tex_escape(normalize_val(val))
    base, comment = m.group(1), m.group(2).strip()
    if base == "omitted":
        base = "no"
    tool = tool_names[col_idx - 2]
    letters = "abcdefghijklmnopqrstuvwxyz"
    key = (tool, comment)
    if key not in notes:
        notes[key] = letters[len(notes)]
    return tex_escape(base) + r"\textsuperscript{" + notes[key] + r"}"


# ── Locate spreadsheet ────────────────────────────────────────────────
if len(sys.argv) > 1:
    xlsx_path = sys.argv[1]
else:
    xlsx_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "Feature_Comparison.xlsx",
    )

# ── Read CSV rows until first blank line ──────────────────────────────
with subprocess.Popen(
    ["xlsx2csv", xlsx_path], stdout=subprocess.PIPE, text=True
) as proc:
    rows = []
    for row in csv.reader(proc.stdout):
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
