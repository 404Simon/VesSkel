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
        19. Mai 2026
      ]
    ]
  ]
]

#slide("Implementation Progress")[
  #v(-0.3cm)
  - Experiment: Removing correlated features did not improve the best classifier
  - Refactoring, better Docstrings and propagate plugin version to `napari.yaml`
  - Unified `PipelineConfig` (JSON-serializable, shared napari + CLI)
  - `vesskel`-CLI with `run` / `config-init` / `validate-config`
  - Refactored napari widgets onto a single widget using shared `analyze_binary_image` pipeline
  - Feature comparison table (REAVER, VesselVio, VesselExpress, TWOMBLI, VesSAP, Skan)
]


#slide("Unified Pipeline: napari + CLI")[
  The core analysis is now a single shared function `analyze_binary_image()` used by both the napari widget and the new standalone CLI.

  #v(0.3cm)
  #highlight[
    *CLI usage:* `vesskel run --input "data/*.png" --config config.json --out results/`

    *Config management:* `vesskel config-init` (starter JSON), `vesskel validate-config` (check validity)
  ]

  #v(0.3cm)
  *Config as contract:* same file can be exported/imported inside napari

  *Current Limited Config-Options:*
  + Branch extraction toggle, branch text labels, summary features, fractal dimension
  + Output controls: skeleton .npy/.png, summary.csv, per-image branches.csv
]

#slide("Feature Comparison: Tools Overview")[
  #v(-0.5cm)

  See `Feature_Comparison.xlsx`

  #v(0.3cm)
  #set text(size: 14pt)
  #highlight[
    \* I only took features that can be e2e-extracted using the Tool, e.g. Vessap uses their DL Network, then go through Allen Cell Registration (external Tool) and then analyze the Results using Matlab code.
  ]
  #highlight[
    *Vipar* is not included in the comparison: it is not an open-source package and cannot be used programmatically.
  ]
  #highlight[
    *Skan* is built around powerful pandas dataframes. Features are often not directly outputted but can be easily computed, e.g. `endpoint_count	= np.sum(skel.degrees == 1)` or `min_segment_length = min(summarize()['branch_distance'])`. I treated features that can be easily computed as present.

    This extensibility is a strong argument for further relying on the library.
  ]
]

#slide("Summary & Next Steps")[

  #columns(2, gutter: 1.5em)[
    *What I did:*
    + Unified pipeline
    + Standalone `vesskel` CLI (`run`, `config-init`, `validate-config`)
    + Configurable pipeline via JSON
    + Feature comparison table across 6 other tools

    #colbreak()

    *Next steps:*
    + Add more features
    + Pipeline Performance Improvements?
    + Group Feature Table into Categories
    + Integrate original image for intensity-based features
    + Add Preprocessing-Options (Clique Removal, etc.)
  ]
]
