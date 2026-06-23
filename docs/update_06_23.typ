#set page(
  width: 28.8cm,
  height: 16.2cm,
  margin: (top: 1.4cm, bottom: 1cm, left: 1.8cm, right: 1.8cm),
  fill: white,
  numbering: "1",
  number-align: right,
)

#set text(font: "New Computer Modern", size: 16pt)
#set par(justify: true, leading: 0.65em)

#let accent = rgb("#1d4ed8")
#let accent-light = rgb("#eff6ff")
#let accent-mid = rgb("#93c5fd")
#let muted = rgb("#6b7280")
#let dark = rgb("#111827")
#let green-bg = rgb("#f0fdf4")
#let green-border = rgb("#bbf7d0")
#let green-text = rgb("#166534")

#let slide(title, body) = {
  pagebreak()
  block(
    width: 100%,
    inset: (top: 8pt, bottom: 10pt, left: 0pt, right: 0pt),
    stroke: (bottom: 2.5pt + accent),
  )[
    #text(size: 24pt, fill: accent, weight: "bold")[#title]
  ]
  v(0.5cm)
  body
}

#let highlight(body) = {
  block(
    width: 100%,
    fill: accent-light,
    inset: 14pt,
    radius: 6pt,
    stroke: 1pt + accent-mid,
  )[
    #body
  ]
}

#let proofbox(body) = {
  block(
    width: 100%,
    fill: green-bg,
    inset: 14pt,
    radius: 6pt,
    stroke: 1pt + green-border,
  )[
    #body
  ]
}

#let color-cell(c, label) = {
  block(
    width: 1.2em,
    height: 1.2em,
    fill: c,
    inset: 0pt,
    align(center + horizon, text(size: 0.55em, weight: "bold", fill: if label == "D" { black } else { white }, label)),
  )
}

#let color-row(colors, labels, sz: 1.2em) = {
  let cells = ()
  for i in range(colors.len()) {
    cells.push(block(width: sz, height: sz, fill: colors.at(i), inset: 0pt))
  }
  grid(columns: (sz,) * colors.len(), rows: (sz,), gutter: 0pt, ..cells)
}

#align(center + horizon)[
  #block(width: 80%)[
    #align(center)[
      #text(size: 34pt, fill: accent, weight: "bold")[
        VesSkel\
      ]
      #v(1.0cm)
      #text(size: 20pt, fill: dark)[
        Vessel Skeletonization and Graph-Based\
        Phenotype Analysis in Retinal Fundus Images
      ]
      #v(1.2cm)
      #line(length: 40%, stroke: 1.5pt + accent)
      #v(0.8cm)
      #text(size: 16pt, fill: muted)[
        Simon Wittmann

        Supervisor: Anna Möller
      ]
      #v(0.3cm)
      #text(size: 14pt, fill: muted)[
        23. Juni 2026
      ]
    ]
  ]
]

#slide("Implementation Progress")[
  #v(-0.3cm)
  + preprocessing: binary closing & hole filling
  + new features & updated Feature Comparison Table
  + 3D thinning performance (#sym.arrow parallel scanning, epoch removal tracking)
  + benchmark results (2D + 3D vs skimage & VesselVio)
  + nicer CLI UX (shorter subcommands, worker feedback)
  + paper draft started
  + prediction notebook refactored
]

#slide("Preprocessing: Binary Closing & Hole Filling")[
  *Binary closing* bridges small gaps in segmented vessels before thinning

  *Hole filling* fills enclosed background regions inside vessels

  #v(0.3cm)
  #table(
    columns: (auto, auto, 1fr),
    stroke: 0.5pt + rgb("#e5e7eb"),
    inset: 7pt,
    fill: (x, y) => if y == 0 { accent-light } else if calc.odd(y) { rgb("#f9fafb") },
    [*Config Key*], [*Default*], [*Description*],
    [`closing_iterations`], [`0`], [binary closing iterations (0 = off)],
    [`fill_holes`], [`false`], [fill enclosed background regions],
    [`max_hole_size`], [`0`], [max filled hole area (0 = unlimited)],
    [`show_preprocessed`], [`false`], [show preprocessed binary in napari],
  )

  #v(0.3cm)
  #highlight[
    Supporting tests: in `test_preprocessing.py` covering closing, hole filling, size thresholding.
  ]
]

#slide("New Features")[
  *Key additions since last update:*
  + Mean segment volume ($V = pi sum r_i^2$ for each segment)
  + Mean segment surface area ($A = 2 pi sum r_i$)
  + Added to `per_segment_radii` output and summary features
  + Used in downstream classification

  #v(0.3cm)
  *Feature Table Update:*
  + Re-mapped which features are feasible (many marked "nope")
  + only TODO left: per-node metrics

  SHOW TABLE
]

#slide("3D Thinning Performance Improvements")[
  #columns(2, gutter: 1.5em)[
    *Epoch-Based Removal Tracking*

    Problem: vanilla Lee94 re-checks simplicity (DFS on 26-neighbor graph) for *every* candidate before removal

    Insight: `_mark_removable_candidates` already verified all candidates before any removals in this batch. If none of a candidate's 26 neighbors have been removed yet, the old verdict is still valid - skip the DFS.


    #colbreak()
    Mechanism:
    + `epoch` increments per batch; stamped at each removed voxel
    + Scan 26 neighbors for stamp == current epoch (cheap, 26 int comparisons)
    + If no neighbor stamped → remove immediately
    + If neighbor stamped → re-run DFS to verify still simple
    + Never reset; monotonically increasing epoch makes old stamps invisible
  ]
]

#slide("Benchmark Results")[
  #v(-0.3cm)
  #columns(2, gutter: 1.5em)[

    *2D (HRF dataset, 45 images)*
    #table(
      columns: (auto, auto, auto),
      stroke: 0.5pt + rgb("#e5e7eb"),
      inset: 5pt,
      fill: (x, y) => if y == 0 { accent-light } else if calc.odd(y) { rgb("#f9fafb") },
      [], [*Mean (s)*], [*vs vesskel*],
      [vesskel], [0.084], [-],
      [skimage Zhang], [0.157], [1.87x slower],
      [skimage Lee], [0.438], [5.21x slower],
      [VesselVio Lee], [0.796], [9.46x slower],
    )

    #colbreak()
    *3D (vessap test volume)*
    #table(
      columns: (auto, auto, auto),
      stroke: 0.5pt + rgb("#e5e7eb"),
      inset: 5pt,
      fill: (x, y) => if y == 0 { accent-light } else if calc.odd(y) { rgb("#f9fafb") },
      [], [*Mean (s)*], [*vs vesskel*],
      [vesskel], [0.730], [-],
      [skimage Lee], [1.024], [1.4x slower],
      [VesselVio Lee], [0.804], [1.10x slower],
    )
  ]

  *Key Takeaways*
  #highlight[
    + 2D is highly optimized via LUT-based simplicity check
    + 3D competitive with skimage and VesselVio
    + Pure Python + Numba - no Cython, no compiled extensions
  ]
]

#slide("CLI Improvements")[
  *Shorter Subcommands*
  - `vesskel init config.json` instead of `vesskel config-init --out config.json`
  - `vesskel validate config.json` instead of `vesskel validate-config --config config.json`

  *Instant Worker Feedback*
  - `vesskel run (-j N)` now prints: `Spawning N worker processes...`

  *Default Config Changed*
  - `branches`, `branch_text`, `summary` now default `false`
]

#slide("Paper Draft")[
  *First Table of Content draft started*

  #v(0.3cm)
  *also: Prediction Notebook Refactored*
  + `analysis/HRF_Prediction.ipynb` restructured
  + Pipeline runs within the notebook
  + Cleaner separation between analysis and reporting
  + No notable Performance improvements, even though cleanup ran and more features available #sym.arrow more investigation needed
]

#slide("Summary & Next Steps")[
  #columns(2, gutter: 1.5em)[
    *What I did:*
    + Preprocessing (closing, hole filling)
    + Per-segment volume & surface area
    + 3D thinning speedups (parallel scanning, epoch tracking)
    + Benchmark scripts (2D + 3D)
    + CLI polish (shorter cmds, worker feedback)
    + Table of Content draft
    + Feature table finalized
    + Prediction notebook refactored

    #colbreak()

    *Next steps:*
    + per_node features finalization
    + finalize Phenotype Prediction
    + nice documentation
    + Write the paper
    + (optional) more tests
  ]
]
