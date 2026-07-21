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
#let amber-bg = rgb("#fffbeb")
#let amber-border = rgb("#fde68a")
#let amber-text = rgb("#92400e")

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

#let warnbox(body) = {
  block(
    width: 100%,
    fill: amber-bg,
    inset: 14pt,
    radius: 6pt,
    stroke: 1pt + amber-border,
  )[
    #body
  ]
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
        21. Juli 2026
      ]
    ]
  ]
]

#slide("CLI Throughput: Lung Scans Parallel Thinning")[

  Added `.mhd` + `.raw` (MetaImage) file loading via ITK

  *Benchmark:* `vesskel run` on the 20 VESSEL12 lung CT masks (loaded via MHD support)

  *Hardware:* 12-core CPU (24 threads) -- homogenous CPU Architecture, no thermal throttling

  #v(0.3cm)
  #table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + rgb("#e5e7eb"),
    inset: 6pt,
    fill: (x, y) => if y == 0 { accent-light } else if calc.odd(y) { rgb("#f9fafb") },
    [*Mode*], [*Wall Time*], [*User CPU*], [*Sys CPU*], [*CPU Util*],
    [`-j 12` (default)], [`3:28.94`], [`2376.86s`], [`34.05s`], [1153%],
    [`-j 1` (seq.)], [`4:53.63`], [`1773.63s`], [`16.75s`], [609%],
  )

  #v(0.3cm)
  #highlight[
    Parallel is *~28% faster* in wall time (3:29 vs 4:54).
    Sequential uses *25% less total CPU* (1774s vs 2377s user).
  ]
]

#slide("3D Lung Thinning: VesSkel vs other Implementations")[
  #v(-0.3cm)

  *Benchmark:* `benchmark/3d_lung_comparison.py`: Lee94 thinning only (preprocessing excluded from timing)

  *Dataset:* 20 VESSEL12 lung CT angiography masks (loaded via MHD)

  #v(0.3cm)
  #table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + rgb("#e5e7eb"),
    inset: 6pt,
    fill: (x, y) => if y == 0 { accent-light } else if calc.odd(y) { rgb("#f9fafb") },
    [*Stat*], [*vesskel*], [*skimage Lee*], [*VesselVio Lee*], [*Speedup (sk / vv)*],
    [TOTAL], [335.39s], [604.99s], [679.38s], [1.80x / 2.03x],
    [MEAN], [16.77s], [30.25s], [33.97s], [1.80x / 2.03x],
    [MEDIAN], [16.95s], [28.85s], [32.66s], [1.70x / 1.93x],
  )

  #v(0.3cm)
  #highlight[
    *Reminder*: Prior test on a synthetic volume (vessap test volume) showed modest gains: 1.4x vs skimage, 1.1x vs VesselVio
  ]
]

#slide("Napari UX: Conditional Toggles")[
  #v(-0.3cm)

  *Problem:* Previously all output options were always enableable -- confusing when extraction was off -> would fail/skip silently

  #v(0.3cm)
  *Solution:* Output widgets auto-disable when their extraction toggle is off:

  #table(
    columns: (auto, auto, auto),
    stroke: 0.5pt + rgb("#e5e7eb"),
    inset: 6pt,
    fill: (x, y) => if y == 0 { accent-light } else if calc.odd(y) { rgb("#f9fafb") },
    [*Output option*], [*Requires*], [*Behavior*],
    [`Write skeleton`], [always on], [always enabled],
    [`Write branch CSV`], [`Extract branches`], [disabled when off],
    [`Write node CSV`], [`Extract nodes`], [disabled when off],
    [`Write radius`], [`Vessel radius`], [disabled when off],
    [`Write summary CSV`], [`Extract summary`], [disabled when off],
  )

  #v(0.3cm)
  *Cleanup group* separated from Advanced Features -- `Show preprocessed` auto-enables only when closing/hole-fill is active
]

#slide("Branch Coloring by Feature")[
  #v(-0.3cm)

  *Before:* Branches always colored by tortuosity (hardcoded)

  *After:* Dropdown lets user pick any numeric property -- tortuosity, straightness, branch distance, radius stats, etc.

  #v(0.3cm)
  #highlight[
    Live recoloring on dropdown change -- no re-analysis needed.
  ]

  #v(0.3cm)
  *Demo time?*
]

#slide("Code Architecture & Quality")[
  #v(-0.3cm)

  *IO extraction*
  + `_load_image`, `_save_skeleton`, `_save_radius`, `_write_csv` moved from `_batch.py` -> new `_io.py`
  + New `save_analysis_outputs()` unifies all output writing in one entry point
  + `write_summary` flag suppresses per-image CSV in batch mode (aggregated summary at top level)

  #v(0.3cm)
  *Config refactoring*
  + `from_dict` now uses `dataclass fields` introspection -- no manual field mapping
  + Schema version bumped 2 -> 3
  + New field `branch_color_property` added

  #v(0.3cm)
  *Default changes*
  + `extract_summary` now defaults to `True` (was `False`) -- CLI and napari always compute summary features

  #v(0.3cm)
  *Batch CLI robustness*
  + Graceful Ctrl+C shutdown: `KeyboardInterrupt` -> `ex.kill_workers()` -> no hanging workers
]

#slide("Report and Presentation Progress")[

  #v(0.3cm)
  #highlight[
    Report is structurally complete -- sections: Intro, Results, Discussion, Methods.
    Needs final polish pass + formatting tweaks.
  ]
]

#slide("Summary & Next Steps")[
  #columns(2, gutter: 1.5em)[
    *Benchmarks*
    + 20-image VESSEL12 lung CT benchmark
    + MHD support is nice for clinical data
    + 1.8x-2.0x speedup vs skimage/VesselVio
    + Parallel scaling does not perform too well

    *Napari UX*
    + Conditional toggles: self-documenting widget
    + Branch coloring dropdown + live recoloring
    + Cleanup group separated from advanced

    *Code quality*
    + Config introspection, schema v3
    + Graceful Ctrl+C in batch CLI
    + Summary on by default

    #colbreak()

    #v(0.3cm)
    *Next steps:*
    + Report and Presentation final polish
  ]
]
