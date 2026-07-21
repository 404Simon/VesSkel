# VesSkel

Vessel Skeletonization and Graph-Based Phenotype Analysis in Retinal Fundus Images

## Installation

```sh
uv sync                  # core only
uv sync --extra dev      # + test tools
uv sync --extra napari   # + napari GUI
uv sync --all-extras     # everything
```

## Napari

```sh
uv sync --extra napari && napari
```

Open a `manual1` TIFF from the HRF folder, then run **Lee94 Thinning** from the VesSkel plugin menu to see the skeleton.

Inside the **Analyze Vessels** widget, tune extraction settings and use **Save Config** to export a reusable JSON preset.

## CLI

Use the same JSON preset exported from napari to batch-process images.

```sh
vesskel init config.json
vesskel validate config.json
vesskel run --input HRF/manual1 --config config.json --out outputs
```

CLI outputs:

- `outputs/summary.csv` with one feature row per image
- Optional per-image skeleton outputs (default: `.npy`)
- Optional per-image branch tables when `output.write_branch_csv=true`
- Optional per-image node tables when `output.write_node_csv=true`

## Configuration

Extraction and output settings are defined in a JSON config file (e.g. the one exported from napari or written by hand).

```json
{
  "schema_version": 3,
  "extraction": {
    "branches": false,
    "branch_color_property": "tortuosity",
    "branch_text": false,
    "nodes": false,
    "summary": true,
    "fractal_dimension": false,
    "vessel_radius": false,
    "junction_cleanup": false,
    "cleanup_threshold_factor": 2.5,
    "closing_iterations": 0,
    "fill_holes": false,
    "max_hole_size": 0,
    "show_preprocessed": false
  },
  "output": {
    "write_skeleton_npy": true,
    "write_skeleton_png": false,
    "write_summary_csv": true,
    "write_branch_csv": false,
    "write_node_csv": false,
    "write_radius": false
  }
}
```

| Key | Type | Default | Description |
|---|---|---|---|
| `extraction.branches` | bool | `false` | Extract per-branch features for CSV export or napari visualization |
| `extraction.branch_color_property` | str | `"tortuosity"` | Branch property used to color the napari shapes layer; one of `tortuosity`, `straightness`, `mean_radius`, `std_radius`, `volume`, `surface_area`, ... |
| `extraction.branch_text` | bool | `false` | Display branch ID, length, and tortuosity labels on the napari branch layer |
| `extraction.nodes` | bool | `false` | Extract per-node features for CSV export or napari visualization |
| `extraction.summary` | bool | `true` | Compute summary features  |
| `extraction.fractal_dimension` | bool | `false` | Compute fractal dimension of the skeleton |
| `extraction.vessel_radius` | bool | `false` | Estimate vessel radius using EDT from the segmentation |
| `extraction.junction_cleanup` | bool | `false` | Clean up ambiguous junction pixels after thinning |
| `extraction.cleanup_threshold_factor` | float | `2.5` | Sensitivity for junction cleanup (higher = larger cycles get collapsed) |
| `extraction.closing_iterations` | int | `0` | Morphological closing iterations applied before thinning (0 = disabled) |
| `extraction.fill_holes` | bool | `false` | Fill holes in the binary segmentation before thinning |
| `extraction.max_hole_size` | int | `0` | Maximum hole area (px) to fill when `fill_holes` is true; 0 = fill all |
| `extraction.show_preprocessed` | bool | `false` | Show preprocessed binary layer (after closing and hole filling) in the napari viewer |
| `output.write_skeleton_npy` | bool | `true` | Save skeleton as `.npy` (NumPy array) per image |
| `output.write_skeleton_png` | bool | `false` | Save binary skeleton mask as `.png` per image |
| `output.write_summary_csv` | bool | `true` | Write aggregated per-image features to `summary.csv` |
| `output.write_branch_csv` | bool | `false` | Write per-branch CSV tables (requires `extraction.branches`) |
| `output.write_node_csv` | bool | `false` | Write per-node CSV tables (requires `extraction.nodes`) |
| `output.write_radius` | bool | `false` | Write per-pixel radius matrix as `.npy` (requires `extraction.vessel_radius`) |

### Shell completions

```sh
# zsh
eval "$(vesskel completions zsh)"

# bash
eval "$(vesskel completions bash)"

# PowerShell
vesskel completions powershell | Out-String | Invoke-Expression
```

Add the appropriate line to your shell rc for persistent tab-completion.

## Tests

```sh
uv sync --extra dev && pytest                     # all tests
uv sync --extra dev && pytest -m "not slow"       # skip regression tests
```

- **2D regression** - thinning + feature extraction on all 45 HRF samples, compared against saved baselines
- **3D regression** - thinning + features on a brain volume (from scikit-image), same baseline approach
- **3D comparison** - vesskel `lee94_thin` vs `skimage.morphology.skeletonize` on the brain volume, asserting identical output

First run (or `--update-baseline`) generates baselines in `tests/skeletons/` and `tests/features/`.

## Dataset

This project uses the High-Resolution Fundus (HRF) Image Database, established by a collaborative research group to support comparative studies on automatic segmentation algorithms on retinal fundus images.

The database contains 45 images total:

- 15 images of healthy patients
- 15 images of patients with diabetic retinopathy
- 15 images of glaucomatous patients

Binary gold standard vessel segmentation images and field of view (FOV) masks are available for each image.

### License

> Budai, Attila; Bock, Rüdiger; Maier, Andreas; Hornegger, Joachim; Michelson, Georg.
> Robust Vessel Segmentation in Fundus Images.
> International Journal of Biomedical Imaging, vol. 2013, 2013

The HRF dataset is released under the **Creative Commons 4.0 Attribution License**.

For more information, visit the [HRF Image Database](https://www5.cs.fau.de/research/data/fundus-images/).
