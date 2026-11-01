# Data provenance

**Organiser files** come from Source Cooperative, `humane-intelligence/bias-bounty-mapping-equity-challenge`, `reference/` and `strata/`. They are named as they appear in our feature-kernel log, `derived/bb-features-v1.log`:

- `<region>-overture-buildings.parquet`
- `<region>-overture-roads.parquet`
- `<region>-overture-pois.parquet`
- `<region>-census-acs-housing.parquet`
- `<region>-census-tracts.parquet`
- `<region>-strata-tract-table.parquet`

None of them are redistributed here.

**Our derived files** were produced on Kaggle, DuckDB 1.5.6, from those organiser files only.

| File | Source | Content |
|---|---|---|
| `derived/<region>-bldg-tract.parquet` | Overture buildings | per tract × `src` × `subtype` × `class`: `n`, `n_multisrc`, area sums |
| `derived/pois_tract_summary.parquet` | Overture places | per tract: POI count, mean confidence, low-confidence share, `meta` share, fire-department presence |
| `rb3.parquet` | Overture roads + tract polygons | road lengths per tract; see `code/geometry_step.sql` |
| `roadfeat2.parquet` | `rb3.parquet` | per-tract road features |
| `scored_vectors.parquet`, `scores.csv` | our submissions | all named scored submissions and their Zindi public/private scores |

The scripts' `<bundle_dir>` argument needs only three files per region:

- `<region>-tracts.parquet`: the organiser tract table with `bbox` in place of geometry
- `<region>-strata.parquet`: the organiser strata table, 227 columns
- `<region>-acs.parquet`: the organiser ACS housing table

The scripts fall back to `derived/` for building and POI aggregates.

Tract polygons are the organiser-provided Census tracts. These are a permitted input, not the TIGER road layer used for the reference.
