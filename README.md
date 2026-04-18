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
uv sync --extra napari && uv run napari
```

Open a `manual1` TIFF from the HRF folder, then run **Lee94 Thinning** from the VesSkel plugin menu to see the skeleton.

## Tests

```sh
uv sync --extra dev && uv run pytest
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

