-- Reconstruction of the geometry step that produced data/rb3.parquet (run on Kaggle, DuckDB 1.x with the
-- spatial extension). The original kernel was not preserved; this query documents the computation and
-- is the specification against which rb3.parquet should be checked. One row per (tract GEOID, Overture
-- class, core flag, fm flag, subclass) with summed lengths in metres (EPSG:5070, CONUS Albers).
--   L0   : length of the segment strictly inside the tract polygon
--   L8   : length inside the tract polygon buffered by 8 m
--   L25  : length inside the tract polygon buffered by 25 m
--   Lbd8 : length within 8 m of the tract boundary (inside the 8 m buffer of ST_Boundary)
--   n    : number of segments
-- core/fm flags come from routes[].network (regex in code/routes4.py; fm = US:TX:FM or US:TX:RM).
-- Inputs are the organiser's files from Source Cooperative, reference/<region>/ (the same names appear in our
-- Kaggle feature-kernel log, data/derived/bb-features-v1.log): <region>-census-tracts.parquet (tract polygons)
-- and <region>-overture-roads.parquet (Overture transportation segments, release 2026-08-19.0).
-- Overture layers are lon/lat (OGC:CRS84): read them as such before ST_Transform, see the challenge README.
-- Notes on the output columns:
--   * L25 is computed for every class; the model uses it for motorway/trunk only.
--   * Lbd8 uses a buffer on BOTH sides of the boundary line, so it includes road just outside the tract;
--     the boundary shares in the writeup (28.9%) are Lbd8 / L8 on that basis.
--   * rb3.parquet also carries lng6 / lng25 (exploratory flags for segments running along the boundary,
--     not used in any result) and region (the region the file was built for).
INSTALL spatial; LOAD spatial;
CREATE TABLE tracts AS
  SELECT GEOID, ST_Transform(ST_GeomFromWKB(geometry), 'OGC:CRS84', 'EPSG:5070') AS g
  FROM read_parquet('<region>-census-tracts.parquet');
CREATE TABLE roads AS
  SELECT id, class, subclass,
         ST_Transform(ST_GeomFromWKB(geometry), 'OGC:CRS84', 'EPSG:5070') AS g,
         list_transform(routes, r -> r.network) AS networks
  FROM read_parquet('<region>-overture-roads.parquet')
  WHERE class IN ('motorway','trunk','primary','secondary');
CREATE TABLE flagged AS
  SELECT *,
         list_bool_or(list_transform(networks, n -> regexp_matches(n, '^US:I(:.*)?$') AND NOT contains(n, 'Future')
                                                     OR regexp_matches(n, '^US:US(:(Business|Alternate|Spur|Bypass|Truck))?$')
                                                     OR regexp_matches(n, '^US:(AZ|CA|OK|TX)(:(Loop|Spur|Business|Beltway|Park))?$')
                                                     OR regexp_matches(n, '^US:(OK:Turnpike|TX:Toll|TX:NTTA|TX:Harris:HCTRA)$'))) AS core,
         list_bool_or(list_transform(networks, n -> n IN ('US:TX:FM','US:TX:RM'))) AS fm
  FROM roads;
CREATE TABLE rb3 AS
  SELECT t.GEOID, '<region>' AS region, r.class, coalesce(r.core,false) AS core, coalesce(r.fm,false) AS fm, coalesce(r.subclass,'') AS sub,
         SUM(ST_Length(ST_Intersection(r.g, t.g)))                                   AS L0,
         SUM(ST_Length(ST_Intersection(r.g, ST_Buffer(t.g, 8))))                     AS L8,
         SUM(ST_Length(ST_Intersection(r.g, ST_Buffer(t.g, 25))))                    AS L25,
         SUM(ST_Length(ST_Intersection(r.g, ST_Buffer(ST_Boundary(t.g), 8))))        AS Lbd8,
         COUNT(*) AS n
  FROM flagged r JOIN tracts t ON ST_Intersects(r.g, ST_Buffer(t.g, 25))
  GROUP BY 1,2,3,4,5,6;
COPY rb3 TO 'rb3-<region>.parquet' (FORMAT PARQUET);
