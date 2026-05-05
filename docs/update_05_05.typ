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
        05. Mai 2026
      ]
    ]
  ]
]

#slide("Implementation Progress")[
  #v(-0.3cm)
  #table(
    columns: (auto, 1fr),
    stroke: 0.5pt + rgb("#e5e7eb"),
    inset: 7pt,
    fill: (x, y) => if y == 0 { accent-light } else if calc.odd(y) { rgb("#f9fafb") },
    [*Date*], [*Milestone*],
    [Apr 24], [VesSAP 3D volume thinning & analysis],
    [Apr 28], [Napari: branch feature extraction widget],
    [Apr 29], [Fractal dimension],
    [Apr 28], [Feature analysis and Phenotype Prediction],
    [Apr 30], [Literature review (VesselExpress, TWOMBLI, VesSAP, REAVER)],
  )
]

#slide("Graph Construction & Features")[
  Skeletons are transformed into graph representations via `skan.Skeleton`

  #columns(2, gutter: 1.5em)[
    *Topology*
    + Number of nodes / edges
    + Number of endpoints / bifurcations
    + Number of connected components
    + Mean & max node degree

    #colbreak()

    *Geometry*
    + Total / mean / std / min / max segment length
    + Mean & std tortuosity
    + Fractal dimension + $R^2$
  ]

  #v(1.3cm)
  => 16 features in total
]

#slide("Napari Plugin: Two Widgets")[
  `vesskel` now ships two napari widgets for interactive exploration:

  #v(0.3cm)
  + *2D/3D Thinning*
  + *Feature Extraction*
    - Colored branch paths (tortuosity heatmap)
    - (Per-branch text labels (id, length, tortuosity))
    - Summary layer with all 16 global features
      - Open in Feature Table for inspection
]

#slide("VesSAP 3D Volume Testing")[
  Applied `vesskel` thinning + feature extraction to a real 3D volume (VesSAP test volume)

  *Demo Time*
]

#slide("Feature Analysis and Phenotype Prediction")[
  *Notebook Demo Time*
]

#slide("Literature Review")[
  *go through md file*
]

#slide("Open Questions")[
  *go through md file*
]

