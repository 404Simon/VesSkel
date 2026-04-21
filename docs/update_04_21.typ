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
        21. April 2026
      ]
    ]
  ]
]

#slide("Project Overview")[
  #columns(2, gutter: 1.5em)[
    *Objectives*
    + Efficient numba-parallelized Lee94 skeletonization (2D + 3D)
    + Application to HRF retinal vessel masks
    + Graph-based feature extraction
    + Phenotype differentiation

    #colbreak()

    *Key Deliverables*
    + `vesskel` Python package
    + Optimized 2D & 3D thinning
    + (kinda slow) Graph construction + basic feature extraction
    + Napari plugin for interactive use
    + Regression test suite (45 HRF + 3D brain)
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
    [Apr 01], [Project setup, HRF dataset integration],
    [Apr 03], [Initial 3D Lee94 adapted for 2D images],
    [Apr 03], [First regression test (skeletonization)],
    [Apr 04], [Proof: Euler invariant is redundant in 2D],
    [Apr 04], [Parallelized 2D implementation (without Euler)],
    [Apr 09], [Numba parallelization for 2D thinning],
    [Apr 14], [Graph construction + feature extraction],
    [Apr 16], [Feature regression tests],
    [Apr 18], [3D implementation (skimage port)],
    [Apr 18], [3D Parallelization (candidate marking + adj LUTs)],
    [Apr 18], [Napari Plugin],
    [Apr 18], [CI and PyPi Release],
  )
]

#slide("Phase 1: 3D -> 2D Adaptation")[
  The initial implementation embedded 2D images into a 3D volume and ran the full Lee94 algorithm unchanged:

  $ V_(p,r,c) = cases(I_(r-1,c-1) "if" p = 1, 0 "otherwise") $

  #v(0.3cm)

  Observations:
  + The 26-neighborhood collapses to an effective 8-neighborhood in the $p=1$ plane
  + All neighbors in the $p=0$ and $p=2$ planes are identically zero
  + Every foreground pixel is automatically a "border point" since the $p$-direction neighbors are always background. The 2 extra border check directions provides no constraining power in 2D.
  + Topology is preserved -- but at the cost of carrying redundant 3D machinery
]

#slide("Key Insight: Euler Check is Redundant in 2D")[
  Lee94 preserves topology through four conditions:

  #v(0.3cm)
  #table(
    columns: (auto, 1fr, auto),
    stroke: 0.5pt + rgb("#e5e7eb"),
    inset: 7pt,
    align: (center, left, center),
    fill: (x, y) => if y == 0 { accent-light },
    [*Nr.*], [*Condition*], [*Role*],
    [1], [Border point], [Restricts deletion to boundary],
    [2], [Endpoint], [Preserves line endings],
    [3], [Euler invariant ($delta chi = 0$)], [Prevents hole/tunnel changes],
    [4], [Simple point ($O = 1$)], [Prevents disconnection],
  )

  #v(0.4cm)
  In 3D, conditions 3 and 4 are *complementary*:\
  Euler detects holes/tunnels, Simple detects disconnection.

  #v(0.3cm)
  #highlight[
    *In 2D, the Euler check is provably redundant.*\
    The Simple-Point-Check alone suffices for topology preservation.
  ]
]

#slide("Proof: Impossibility of Hole Creation")[
  #proofbox[
    *Lemma 1:* _In 2D Lee94 thinning, no border point removal can create a hole ($delta H > 0$)._
  ]

  #v(0.3cm)
  *Proof.* For a hole to be created by removing pixel $x$, all 4-neighbors must be foreground:

  $ forall y in N_4(x) : V_y = 1 quad arrow quad x "fails border point condition" $

  If any 4-neighbor is background, it provides a 4-connected path from the newly-background $x$ to the exterior, so no new isolated background component (hole) is formed:

  $ exists y in N_4(x) : V_y = 0 quad arrow quad "no hole created" $

  Since border point candidates must have at least one background 4-neighbor, hole creation is impossible. $square$
]

#slide("Proof: Impossibility of Hole Elimination & Theorem")[
  #proofbox[
    *Lemma 2:* _No simple point removal can eliminate a hole ($delta H < 0$)._
  ]

  #v(0.2cm)
  If removing $x$ would eliminate a hole, foreground pixels in $N(x)$ must form $>= 2$ disconnected components wrapping around separate background regions. But the Simple-Point-Check requires $O(S inter N(x)) = 1$. Contradiction. $square$

  #v(0.4cm)

  #block(
    width: 100%,
    fill: rgb("#fef3c7"),
    inset: 14pt,
    radius: 6pt,
    stroke: 1pt + rgb("#fde68a"),
  )[
    *Theorem:* _For 2D Lee94 thinning, the Euler invariant check is redundant._

    #v(0.2cm)
    Since $delta H > 0$ is impossible (Lemma 1) and $delta H < 0$ is impossible (Lemma 2):
    $ delta H = 0 quad arrow quad delta chi = delta O - delta H = delta O $
    The Simple-Point-Check ensures $delta O = 0$, so $delta chi = 0$ is automatically satisfied. $square$
  ]
]

#slide("Practical Implications of the Proof")[
  The proof enables significant simplifications in the 2D implementation:

  #v(0.3cm)
  + *No 3D embedding required* -- operate directly on padded 2D image $I in {0,1}^((H+2) times (W+2))$
  + *Border detection simplifies* from 6-neighbor 3D checks to 4-neighbor 2D checks (N, S, E, W)
  + *Euler-Invariant-Check omitted entirely* -- no octree construction, no Euler LUT
  + *Simple-Point-Check via direct flood fill* -- count 8-connected foreground components with DFS (no octrees)

  #v(0.4cm)
  The thinning loop iterates over 4 directional sub-iterations, each:
  1. Collecting border candidates along one direction in parallel
  2. Sequential recheck for simple-point preservation
]

#slide("Phase 2: Native 2D Implementation")[
  Refactored from 3D volume to direct 2D processing (`thin_2d.py`):

  #v(0.3cm)
  #table(
    columns: (1fr, 1fr),
    stroke: 0.5pt + rgb("#e5e7eb"),
    inset: 8pt,
    fill: (x, y) => if y == 0 { accent-light },
    [*3D Embedded (before)*], [*2D Native (after)*],
    [3D volume $3 times (H+2) times (W+2)$], [Padded 2D $(H+2) times (W+2)$],
    [6 border directions], [4 border directions (N, S, E, W)],
    [26-neighborhood lookup], [8-neighborhood lookup],
    [Euler check + Simple check], [Simple check *only*],
    [Octree-based connectivity], [Direct flood-fill DFS],
  )

  #v(0.4cm)
  #highlight[
    *Performance:* Thinning all 45 HRF Images takes $approx 9 "seconds" (approx 0.197 s "per image")$. \
  ]
]

#slide("Phase 3: Numba Parallelization")[
  The thinning loop has an inherent data dependency -- deletions in one iteration affect the next. But *candidate marking* is data-parallel:

  #v(0.3cm)

  #highlight[
    *Strategy: parallel candidate marking, sequential deletion*
  ]

  #v(0.3cm)
  + *Parallel phase* (`prange` over rows): each row independently marks candidates that pass border + endpoint + simple-point checks
  + *Merge phase*: collect per-row results into a flat candidate array
  + *Sequential phase*: iterate candidates, recheck simple-point condition, delete if still valid
]

#slide("Phase 4: Full 3D Implementation")[
  Ported the scikit-image Cython implementation to pure Python + Numba (`thin_3d.py`):

  #v(0.3cm)
  + Faithful port of `skimage.morphology._skeletonize_lee_cy`
  + Euler LUT + octant index table for Euler invariant computation
  + 26-neighborhood adjacency via pre-computed lookup tables
  + All checks active: border, endpoint, Euler invariant, simple point

  #v(0.3cm)
  #highlight[
    *Validated:* The output is *bit-identical* to `skimage.morphology.skeletonize` on the
    scikit-image brain volume test case.
  ]

  Further Refinement: `_mark_removable_candidates()` marks deletable pixels in parallel; sequential `_apply_removals()` rechecks and deletes
]

#slide("Phase 5: Graph Construction & Feature Extraction")[
  Skeletons are transformed into graph representations via `skan.Skeleton`

  #columns(2, gutter: 1.5em)[
    *Topology*
    + Number of nodes
    + Number of edges
    + Number of endpoints
    + Number of bifurcations
    + Number of connected components
    + Mean node degree
    + Maximum node degree

    #colbreak()

    *Geometry*
    + Total vessel length
    + Mean segment length
    + Std segment length
    + Max segment length
    + Min segment length
    + Mean tortuosity
    + Std tortuosity
  ]

  #v(0.4cm)
  *Example:* Sample `01_h` from HRF:
  #table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt + rgb("#e5e7eb"),
    inset: 6pt,
    fill: (x, y) => if y == 0 { accent-light },
    [*Nodes*], [*Edges*], [*Bifurcations*], [*Mean tortuosity*], [*Components*],
    [720], [765], [404], [1.086], [14],
  )
]

#slide("Napari Plugin: Interactive Skeletonization")[
  `vesskel` now also ships a Napari plugin for visual exploration:

  #v(0.3cm)
  + Registered via `napari.yaml` manifest + `napari.manifest` entry point
  + *Lee94 Thinning* widget: select an image layer, run thinning, see the skeleton as a new labels layer


  #v(0.3cm)
  #highlight[
    The Dispatcher (`thin.py`) auto-selects 2D or 3D based on `img.ndim`, so it works for both.
  ]
]

#slide("Some Testing")[
  Three-tier regression test suite with baseline comparison:

  #v(0.3cm)
  #table(
    columns: (auto, 1fr, auto),
    stroke: 0.5pt + rgb("#e5e7eb"),
    inset: 7pt,
    fill: (x, y) => if y == 0 { accent-light },
    [*Test*], [*What it checks*], [*Scope*],
    [2D Regression], [Thinning + features on HRF vs. saved baselines], [45 samples],
    [3D Regression], [Thinning + features on brain volume vs. saved baselines], [1 volume],
    [3D Comparison], [vesskel vs. `skimage.morphology.skeletonize` -- bit-identical output], [1 volume],
  )
]

#slide("Package Structure")[
  ```text
     vesskel
    ├──  __init__.py   # public API (currently only exposes thin)
    ├──  _napari.py    # napari widget
    ├──  benchmark     # some benchmark files
    ├──  features.py   # graph construction & feature extraction
    ├──  hrf.py        # dataset loader
    ├──  napari.yaml   # plugin manifest
    ├──  thin.py       # dispatcher with lazy imports based on ndim
    ├──  thin_2d.py    # 2D Lee
    └──  thin_3d.py    # 3D Lee
  ```
]

#slide("Summary & Next Steps")[

  #columns(2, gutter: 1.5em)[
    *What I did:*
    + Native 2D thinning without Euler invariant check (with proof)
    + Full 3D thinning -- bit-identical to scikit-image
    + Numba parallelization for both 2D and 3D
    + Rudimentary graph-based feature extraction
    + Rudimentary Napari plugin
    + Regression test suite
    + CI/CD with PyPi releases

    #colbreak()

    *Next steps:*
    + More Graph Features (e.g. fractal dimension)
    + DIY Graph Assembly?
    + Statistical analysis of phenotype differentiation
    + 3D Benchmarking with huge volumes: `vesskel` vs. `vesselvio`
    + Parallelize rechecking phase? (maybe too much overhead)
    + Toggleable Feature extraction in Napari
    + Configurable/Savable/Loadable Batch jobs in Napari
  ]
]

