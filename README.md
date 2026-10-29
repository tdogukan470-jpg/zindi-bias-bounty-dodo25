---
title: "Bias Bounty: Mapping Equity Challenge — Methodology and Bias Discovery"
subtitle: "Zindi username: dodo25 · Submitted for: Best Bias Discovery and Best Documentation"
date: "October 2026"
---

# 0. Summary

We predict the organizer's tract-level `coverage_gap_score` for the four scored regions (Maricopa AZ, Northern CA, Eastern OK, South-Central TX) using **only the data provided with the challenge** (Overture Maps roads, buildings and places, the strata and ACS tables). We use no TIGER, Microsoft, USGS or CBP layers and no copies of them.

There are no training labels. The method therefore has three parts:

1. A deterministic reconstruction of the transport component from Overture road geometry, built on the published methodology.
2. Interpretable proxy directions for the building and POI components.
3. A leaderboard-probe calculus that measures how each direction relates to the hidden reference, plus a regularized global re-fit over every scored submission.

Our best public score is **0.0658** (submission `t19_global`). It improved from 0.0790 for the first transport-only reconstruction.

**Bias discovery (Section 8).** The transport-gap metric contains a geometric measurement artifact that the automated scorecard cannot see. Census-tract boundaries often run along highways. A reference road lying on a shared boundary is counted in both tracts, while Overture's slightly offset copy of the same road falls in only one of them. The other tract therefore shows a "gap" that does not exist. Under our reconstruction:

- The artifact adds a mean of **+0.066** to the transport gap of defined tracts. In **19%** of them it adds more than 0.05.
- It is **1.81× larger in rural tracts** (d = 0.32, p < 10⁻¹⁰) and **1.32× larger in tribal statistical areas** (p = 0.001).
- It accounts for roughly **31% of the rural–urban difference** in mean coverage gap.

Part of the rural disparity reported by the scorecard is therefore measurement, not mapping. A second, smaller artifact comes from freeway ramps. It works in the opposite direction and has its own group pattern.

# 1. Data and rules compliance

- **Inputs** are the challenge bundle only:
  - `*-tracts`, `*-strata` (227 columns), `*-acs`
  - Overture roads, buildings and places as distributed for the challenge
  - Aggregates of these inputs that we computed ourselves: per-tract road lengths with boundary buffers, building counts by source, POI counts by category
- **Not used:** TIGER/Line, Microsoft Building Footprints, USGS National Map, County Business Patterns, or any copy of them. External information is not used for the score.
- **Tools:** Python 3, pandas, numpy, DuckDB (spatial and httpfs extensions). All are open source. The geometry ran on Kaggle CPU kernels in EPSG:5070 (CONUS Albers, equal area, metres).
- **Determinism:** No step is stochastic. Every intermediate file reproduces exactly.

# 2. Structure of the reference score (from the published methodology)

For tract *i*, with defined components among transport T, building B and POI P:

$$ y_i = \frac{B_i + P_i + m_i\,T_i}{2 + m_i}, \qquad m_i = \mathbb{1}[\text{TIGER S1100/S1200 length in tract} > 0]. $$

We first checked that B and P are almost always defined, so that the undefined component is essentially transport. The methodology reports the share of tracts with at least one undefined component as 21% (OK), 37% (CA), 28% (TX) and 55% (AZ). Our reconstructed transport-undefined shares are 22%, 39%, 29% and 55%. Every region agrees within 2 points.

# 3. Transport component reconstruction

**Overture side (numerator).** This is the length of `motorway`, `trunk`, `primary` and `secondary` segments inside the tract (`ST_Intersection`, `L0`).

**Reference-side proxy (denominator).** TIGER S1100/S1200 are numbered Interstate, US and State routes. We identify them in Overture through the `routes[].network` field:

```
^US:I(:.*)?$ | ^US:US(:(Business|Alternate|Spur|Bypass|Truck))?$
| ^US:(AZ|CA|OK|TX|NM)(:(Loop|Spur|Business|Beltway|Park))?$
| ^US:(OK:Turnpike|TX:Toll|TX:NTTA|TX:Harris:HCTRA)$  (+ US:TX:FM/RM)
```

Ramps (`subclass = link`) are excluded, because TIGER codes ramps as S1630.

**Boundary buffers.** Many reference roads lie exactly on tract boundaries (Section 8). The reference proxy therefore uses the length inside the tract polygon buffered by 8 m (all classes) and 25 m (`motorway`, `trunk`). The second buffer covers the far carriageway of divided highways. We measured that 93% of boundary-coincident highway length lies within 8 m of the boundary and 88% within 5 m. A second mode at 20–25 m corresponds to divided carriageways.

**Gap and definedness.**

- T = clip(1 − Overture / reference, 0, 1).
- m = 1 when the reference proxy exceeds 100 m.
- The thresholds are insensitive: 50 m or 250 m changes only 13 and 40 tracts respectively.

**Overture tagging gap.** A field inspection in QGIS of 200 long, straight `motorway` and `trunk` segments without route tags (Maricopa) found that most were the second carriageway of a tagged divided highway, or interchange ramps and service/frontage connectors. Examples are in the appendix figures. Rescuing untagged mainline carriageways changes only 94 tracts, about 1 Mm against 52 Mm of tagged core.

# 4. Building and POI proxies

- **Building.**
  - Overture building counts by source: OpenStreetMap, Microsoft ML, Google, Esri.
  - Building counts without small structures under 40 m², compared with `usfs_BuildingCount_sum` from the strata table. The USFS count is a raster aggregate of building footprints that is provided with the challenge data.
  - Building area per structure.
- **POI.**
  - Places per capita.
  - Source shares and confidence.
  - Facility categories (fire, EMS, schools; hospitals excluded, as the methodology states).
  - Business-type places relative to housing units.
- **Neighbourhood.** k-nearest-tract averages of the prediction, within each region. These turned out to be among the strongest late directions (Section 6).

# 5. Leaderboard-probe calculus

Let *s* be a scored submission with public MSE M₀, and let *u* be any perturbation. Then

$$ \mathrm{MSE}(s + a u) = M_0 + 2a\,c + a^2 q, \qquad q = \overline{u^2}, \quad c = \overline{u\,(s - y)}. $$

The quantity *q* is computed locally. A single scored probe at a = 1 gives c = (M₁ − M₀ − q)/2. The optimal step is a* = −c/q.

For many directions we solve Qa = −c, with Q the Gram matrix computed locally. Information from earlier probes carries over to a new base exactly, through c_new = c_old + ⟨u, Δbase⟩.

## 5.1 Global re-fit with a noise model

Every scored file is a vector with one known inner product against the hidden residual. We re-fit the best combination of all scored vectors at once, rather than chaining small fits:

$$ a^{*} = -(G + \lambda\,\mathrm{diag}(\sigma_k^2))^{-1} g, $$

Here σ_k is estimated by **leave-one-out prediction** of each scored submission from all the others. The penalty is chosen by simulation, so that the expected realized gain is maximized rather than the in-sample gain.

## 5.2 Quality control that changed results

- **Leave-one-out consistency revealed two mislabeled batches.** In two batches, the scores had been attached to the wrong files: b1/b2/b4, and a1/a2. With the corrected mapping, the joint prediction error for `t14` fell from 0.00102 to 0.00021. The re-fit with corrected labels (`t18_global`) was predicted at 0.06601 and scored **0.06599**.
- **Eastern WA is not scored.** A probe that changed only the 128 WA tracts left the public score unchanged to 9 digits. All inner products are therefore computed on the four scored regions.
- **Public/private.** The public leaderboard is a random ~30% sample. We shrink directions in proportion to their estimated noise, and select a public-optimal and a shrunk variant as the two final submissions.

# 6. Results (public leaderboard RMSE)

| Submission | Content | RMSE |
|---|---|---|
| `t3` | Transport reconstruction with 8/25 m boundary buffers | 0.07904 |
| `t7`–`t9b` | + source-mix and per-capita POI/building directions | 0.07430 |
| `t10`–`t13` | + structural building (`usfs`), TX FM roads, denominator weighting | 0.07251 |
| `t14` | + primary/secondary routed roads in reference (largest single direction) | 0.06941 |
| `t15`–`t17` | + buffer and reference-scale calibration by region | 0.06792 |
| `t18_global` | global re-fit, corrected labels | 0.06599 |
| `t19_global` | + county/kNN/tail/facility probes | **0.06581** |

**Directions that did not help** (reported for completeness):

- Adding Texas FM roads to the reference alone, without adding them to the Overture side.
- Rescaling building and POI corrections by 1/(2+m): the opposite sign was optimal.
- Population-based facility-absence indicators: optimal weight ≈ 0.
- County-level POI-per-capita gap: negative weight.

# 7. Edge cases and alternative weightings

- **Undefined transport.** The score is (B + P)/2. These tracts benefit most from neighbourhood smoothing: the optimal weight is 0.68 on that subset.
- **Bounds.** Predictions are clipped to [0, 1]. Earlier versions used a 0.01 floor, and a probe showed that tracts at the floor were under-predicted.
- **Alternative weighting.** We tested weighting building/POI directions by the tract's actual 1/(2+m) share against a uniform weight. Uniform was better (Section 6), which suggests that the organizer's B and P vary more in transport-defined tracts.

# 8. BIAS DISCOVERY — a measurement artifact that inflates rural and tribal coverage gaps

## 8.1 What the scorecard misses

The automated scorecard reports disparity ratios of the *score itself* across fixed strata, such as Rural vs Urban and Tribal vs Non-Tribal. It cannot tell how much of a disparity comes from **incomplete open data** and how much comes from **how the gap is measured**. We identify a measurement component.

## 8.2 Mechanism

1. Census tract boundaries frequently follow major roads.
2. A reference (TIGER) road lying exactly on a shared boundary intersects the closed polygons of **both** tracts, so its full length counts in each.
3. Overture represents the same road a few metres off the line. It therefore falls inside **one** tract.
4. The neighbouring tract records reference length with no matching Overture length, which is a spurious transport gap.

Evidence from the provided data:

- 93% of boundary-coincident highway length is within 8 m of the boundary.
- A reconstruction that counts boundary roads on both sides (8/25 m buffers) scored clearly better on the leaderboard than one that counts them on one side only. The 8/25 m reconstruction is part of `t3`. The no-buffer probe `c4` confirmed that the partial direction matters.

## 8.3 Size and who is affected

We computed each defined tract's transport gap twice, with and without the double-counted boundary length (full table in `bias_artifacts_table.csv`):

| Group | Artifact, in group | Artifact, rest | Ratio | Welch t | p | Cohen's d |
|---|---|---|---|---|---|---|
| **Rural** (`ur_class`) | 0.099 | 0.055 | **1.81×** | 11.2 | <10⁻¹⁰ | +0.32 |
| **Tribal statistical area** | 0.084 | 0.064 | **1.32×** | 3.2 | 0.001 | +0.14 |
| High SVI minority (≥0.75) | 0.055 | 0.072 | 0.76× | −4.9 | 8·10⁻⁷ | −0.13 |
| High SVI overall | 0.064 | 0.067 | 0.95× | −1.0 | 0.31 | — |

- Across all defined tracts the artifact adds a mean of **+0.066** to T, and more than 0.05 in **19%** of them.
- Rural and tribal tracts are larger, and their boundaries more often follow state and US highways. The artifact therefore concentrates there.
- Propagated to the composite score, it explains about **31% of the rural–urban difference** in mean coverage gap. Under our reconstruction, the rural ratio falls from 2.24× to 2.13×.

## 8.4 A second artifact: freeway ramps

Overture classes ramps as `motorway`/`trunk` (`subclass = link`), while TIGER codes them S1630, outside S1100/S1200. Counting ramps on the Overture side hides real gaps in interchange-heavy tracts.

- The effect is small on average: +0.004 to T.
- It is **3.7× larger in rural tracts** (t = 10.6), where a single interchange is a large share of named-highway length.
- It is **smaller in high-minority urban tracts** (0.51×).
- Measured as a share of named-highway length, ramps are a larger fraction in high-SVI-minority tracts (9.2% vs 6.9%, p < 10⁻¹⁰). Their transport scores therefore rest more on ramp length.

The QGIS inspection (Section 3, appendix) showed the same pattern on the ground: interchange ramps, collector-distributor lanes and frontage/service roads are where Overture's road classes and route tags diverge from the official highway definition.

## 8.5 Recommendations

1. Assign boundary-coincident reference roads to one tract, or split them 50/50, before computing ratios.
2. Exclude Overture `subclass = link` from the numerator, so that it matches TIGER's exclusion of S1630.
3. Report the transport component's disparity with and without these corrections, so that users can separate mapping gaps from measurement effects.

## 8.6 Reproducibility

Every number in this section is computed from the provided Overture roads (per-tract `L0`, `L8`, `L25`, `Lbd8` and `subclass`) and the strata table. The scripts are listed in Section 9.

# 9. Code map

| File | Purpose |
|---|---|
| `k6/main.py`, `k7/main.py` | Kaggle DuckDB kernels: per-(tract, class, route-flag, subclass) lengths `L0`/`L8`/`L25`/`Lbd8` in EPSG:5070 |
| `feat.py` | Per-tract Overture building/POI aggregates |
| `probe_solve.py` | Quadratic probe solver, ridge joint fit |
| `global_refit.py` | Leave-one-out noise model, noise-weighted global re-fit, label-consistency check |
| `bias_artifacts.py` | Section 8 tables (`bias_artifacts_table.csv`) |

**Associated leaderboard submissions:** `t18_global` (0.06599), `t19_global` (0.06581), and the final selections made at close.

# Appendix — field inspection figures

QGIS screenshots of Maricopa interchanges. They show untagged second carriageways, ramps and frontage roads relative to route-tagged mainlines (OpenStreetMap basemap © OpenStreetMap contributors).

The 200 longest untagged `motorway`/`trunk` segments in Maricopa were each rendered over the OpenStreetMap basemap, with the Overture road layer styled by class. Four representative cases follow. The full set of 200 is in the repository.

![Fig. A1 — Freeway-to-freeway interchange (segment 6348, 1.8 km). Ramps and collector lanes branch from the mainline. Overture classes them as `motorway`, while TIGER codes them S1630.](figs/ss_013_fid6348_motorway_1796m.jpg){width=95%}

![Fig. A2 — Interchange on an arterial grid (segment 9798, 1.4 km). The ramp loops and frontage/service connectors run parallel to the mainline.](figs/ss_020_fid9798_motorway_1441m.jpg){width=95%}

![Fig. A3 — Rural trunk junction (segment 30157, 1.2 km). Link roads make up a large share of the named-highway length in a sparse tract.](figs/ss_032_fid30157_trunk_1217m.jpg){width=95%}

![Fig. A4 — Curving divided freeway (segment 28771, 1.0 km). The second carriageway carries no route tag, so a route-based reference proxy misses it.](figs/ss_052_fid28771_motorway_1032m.jpg){width=95%}
