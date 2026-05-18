# Glossary

- segment: a centerline piece between two junctions or a junction and an endpoint
- filament: a whole connected vessel path or connected component of the skeleton
- curvature: mean angular change along the skeleton trace per step
- tortuosity: ratio of actual path length to end-to-end straight-line distance (segment_length / e2e_distance)
- straightness: inverse of tortuosity (e2e_distance / segment_length)
- lacunarity: measures the gappiness or heterogeneity of a spatial pattern. Computed by sliding a box of varying sizes across the image and measuring the variance in pixel occupancy. High lacunarity = large, uneven gaps (clumpy, patchy coverage). Low lacunarity = small, evenly distributed gaps (homogeneous, uniform coverage). Two patterns can have the same area fraction (i.e. same vessel density) but very different lacunarity depending on how the gaps are distributed.
- hgu: hyphal growth unit = total vessel length / endpoint count (μm). Represents the average uninterrupted fibre length before hitting an endpoint; higher HGU means longer fibre stretches with fewer breaks
- box_counting_fractal_dimension: fractal dimension of the fibre mask computed by counting how many boxes of varying sizes are needed to cover the pattern. Ranges ~1–2 for 2D images. Values near 2 mean fibres nearly fill the plane; values near 1 mean sparse, line-like coverage
- high_density_matrix: fraction of image pixels (0–1) classified as especially dense ECM. An extra intensity threshold is applied on top of the fibre mask to isolate the brightest/most tightly packed fibre regions from the general mask.
- alignment_coherency: percentage (0–100%) measuring how uniformly fibres point in the same direction. 0% = isotropic (fibres point in all directions equally with no preferred orientation, like scattered toothpicks). 100% = all fibres are perfectly parallel (like combed hair). Computed via OrientationJ dominant direction analysis.
- gap_area: area of an inscribed circle (px) fitted into the empty spaces between fibres. Captures the size of holes in the fibre network. Reported as summary statistics (mean, sd, 5th/50th/95th percentiles) and as a per-gap array suitable for histograms or distribution analysis
