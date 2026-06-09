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
        09. Juni 2026
      ]
    ]
  ]
]

#slide("Implementation Progress")[
  #v(-0.3cm)
  + Euclidean Distance Transformation #sym.arrow Radius, Diameter
  + 39 features implemented (87 in comparison table)
  + 2D Simple Point LUT
  + Warning for unknown config settings
  + lots of new tests (9 files, #sym.arrow 150+ tests)
  + (not) running regression tests
  + parallelized batching (`-j` / `--jobs N`)
  + fast! shell completions (800 ms #sym.arrow 81 ms)
  + wavefront experiment
  + graph cleanup
]

#slide("Vessel Radius & Diameter via EDT")[
  The *Euclidean Distance Transform* (EDT) computes the distance from each foreground pixel to the nearest background pixel.
  Sampling the EDT at skeleton positions yields the *local vessel radius* at every centerline point.

  #v(0.5cm)
  #table(
    columns: (1fr, 1fr),
    stroke: 0.5pt + rgb("#e5e7eb"),
    inset: 8pt,
    fill: (x, y) => if y == 0 { accent-light } else if calc.odd(y) { rgb("#f9fafb") },
    [*Global Statistics*], [*Per-Segment Statistics*],
    [mean, std, min, max radius], [mean, std, min, max radius],
    [mean, std, min, max diameter], [mean, std, min, max diameter],
  )

  #v(0.4cm)
  - Napari radius layer
  - used for junction cleanup diameter estimation
  - toggleable via `"vessel_radius": true` in config
]

#slide("New Features")[

  *Key additions since last update:*
  + Per-segment radius & diameter stats (mean, std, min, max)
    - highest feature importance in random forest
    - plain SVM now achieves highest score
  + Per-segment tortuosity & straightness
  + Vessel area & vessel area fraction
  + HGU (hyphal growth unit = total length / endpoint count)
]

#slide("2D Simple Point LUT")[
  The simple point check used a flood-fill DFS for each candidate pixel, walking the 8-neighborhood to count connected foreground components. This is called *millions of times* during thinning.

  *Solution:* Precompute the simple point status for all 256 possible 8-neighborhood patterns.

  #v(0.3cm)
  #highlight[
    The LUT is built at module level and shared across all thinning calls via `@njit(cache=True)`.
  ]

  #v(0.3cm)
  The same approach cannot be used for 3D (26-neighborhood = $2^26$ entries = 67 million -- too large). The 3D implementation keeps the octant-based Euler check.
]

#slide("Junction Triangle Cleanup I")[
  *Problem:* Wide vessel junctions appear as thick blobs to the thinning algorithm. Instead of producing a single junction pixel, the skeleton contains small triangle- or diamond-shaped cycles:

  #align(center)[
    ```text
         o                        ---o---
        / \                      /   |   \
       /   \        ->          /    |    \
      o-----o                  /     |     \
     /   |   \
    ```
  ]

  These artifacts inflate bifurcation counts and distort topology metrics.

  *Simple Algorithm:*
  + Build junction graph via skan #sym.arrow find cycles via networkx
  + Compute perimeter of each cycle (sum of Euclidean edge lengths)
  + Estimate local vessel diameter from EDT at cycle node positions
  + Filter: collapse cycles where $"perimeter" < "threshold_factor" times "diameter"$
  + Merge overlapping cycles #sym.arrow collapse to centroid
  + Reconnect external vessel arms

]

#slide("Junction Triangle Cleanup II")[

  #v(0.3cm)
  *New Config Values:*
  - `"junction_cleanup": true`
  - `"cleanup_threshold_factor": 5.0` (range 2.5–10)

  #highlight[
    When cleanup is enabled, the cleaned skeleton replaces the original for *every* downstream stage: saved files, summary features, napari layers.
  ]
]

#slide("CLI Improvements")[
  #columns(2, gutter: 1.5em)[

    *Parallel Batch Processing*
    + `vesskel run --jobs N` / `-j N`
    + `ProcessPoolExecutor` spawns workers
    + Each worker processes one image independently
    + Errors are collected, processing continues
    + full HRF now takes #sym.approx 23s instead of before \ #sym.approx 50s on my machine

    *Config quality-of-life*
    + Unknown keys in config JSON produce a warning
    #colbreak()

    *Shell Completions*
    + Generated via `argcomplete`
    + naive: #sym.approx 800 ms (numpy, PIL, pipeline imported eagerly)
    + After: #sym.approx 81 ms (heavy imports deferred after argcomplete guard)
    + Supports bash, zsh, powershell
    + Regression test enforces completion speed
  ]
]

#slide("Testing")[
  #table(
    columns: (auto, auto, 1fr),
    stroke: 0.5pt + rgb("#e5e7eb"),
    inset: 7pt,
    fill: (x, y) => if y == 0 { accent-light } else if calc.odd(y) { rgb("#f9fafb") },
    [*File*], [*Tests*], [*Scope*],
    [`test_cli.py`], [~25], [completions, input discovery, batch (seq + par)],
    [`test_config.py`], [~20], [serialization, validation, unknown keys],
    [`test_features.py`], [~20], [tortuosity, fractal dim, radii, graph],
    [`test_pipeline.py`], [~25], [full pipeline, all option toggles],
    [`test_napari_layers.py`], [~18], [layer generation, toggle combos],
    [`test_utils.py`], [~12], [`to_binary` behavior],
    [`test_2d_thinning_regression.py`], [45], [HRF images vs saved baselines],
    [`test_3d_thinning_regression.py`], [1], [brain volume vs saved baselines],
    [`test_3d_skimage_comparison.py`], [1], [vesskel = skimage bit-identical],
  )

  + Regression tests now marked `@pytest.mark.slow`
  + Removed parallel test runner, not really needed anymore
]

#slide("Wavefront Experiment")[
  #columns(2, gutter: 1.5em)[

    *Problem:* The recheck-removal phase is sequential -- removing pixel A invalidates B's precomputed simplicity check.

    *Key insight:* Two pixels can be processed in parallel if they are *not* 8-neighbors. The 8-neighbor graph on a 2D grid has chromatic number $4$.

    #sym.arrow Color pixels by $(c % 2, r % 2)$:
    #grid(
      columns: (1.2em, 1.2em, 1.2em, 1.2em, 1.2em, 1.2em, 1.2em, 1.2em),
      rows: (1.2em, 1.2em, 1.2em, 1.2em),
      gutter: 0pt,
      stroke: 0.3pt + rgb("#cccccc"),
      align: center + horizon,
      ..range(32).map(i => {
        let r = calc.quo(i, 8)
        let c = calc.rem(i, 8)
        let cpar = calc.rem(c, 2)
        let rpar = calc.rem(r, 2)
        if cpar == 0 and rpar == 0 {
          color-cell(rgb("#4A90D9"), "A")
        } else if cpar == 1 and rpar == 0 {
          color-cell(rgb("#E67E22"), "B")
        } else if cpar == 0 and rpar == 1 {
          color-cell(rgb("#27AE60"), "C")
        } else {
          color-cell(rgb("#F1C40F"), "D")
        }
      })
    )

    #colbreak()

    *Results:*
    + neglegible performance improvements (x% - 1x% faster) (second phase is not that time consuming in general)
    + different output, missing branches, not medial axis thinning anymore
  ]
]

#slide("Summary & Next Steps")[

  #columns(2, gutter: 1.5em)[
    *What I did:*
    + EDT-based vessel radius & diameter (global, per-segment)
    + more new features (now 39)
    + 2D Simple Point LUT #sym.arrow faster thinning
    + junction cleanup
    + cli improvements (parallel batching, completions, config warnings)
    + more tests
    + Wavefront experiment (4-coloring insight)

    #colbreak()

    *Next steps:*
    + Group Feature Table into Categories
    + more features
    + 3D experiments with graph cleanup
    + more preprocessing options? simple hole filling could help on some cases in HRF
    + revisit phenotype classification
  ]
]
