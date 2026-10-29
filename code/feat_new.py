import numpy as np,pandas as pd
sp='/tmp/claude-0/-home-claude/14f09c49-bb96-5f73-8ddf-c3b7c11d4322/scratchpad/'
R=["maricopa-az","northern-ca","eastern-ok","south-central-tx","eastern-wa"]
T18=pd.read_csv('/mnt/user-data/outputs/t18_global.csv',dtype={'GEOID':str})
idx=T18.GEOID; s=T18.coverage_gap_score.values; reg=T18.region.values
TR=pd.concat([pd.read_parquet(f'/home/claude/bb/data/{r}-tracts.parquet') for r in R]).drop_duplicates('GEOID').set_index('GEOID').reindex(idx)
ST=pd.concat([pd.read_parquet(f'/home/claude/bb/data/{r}-strata.parquet') for r in R]).drop_duplicates('GEOID').set_index('GEOID').reindex(idx)
AC=pd.concat([pd.read_parquet(f'/home/claude/bb/data/{r}-acs.parquet') for r in R]).drop_duplicates('GEOID').set_index('GEOID').reindex(idx)
P=pd.concat([pd.read_parquet(f'/home/claude/bb/data/{r}-pois-tract.parquet') for r in R])
f=pd.read_parquet('/home/claude/bb/roadfeat2.parquet').set_index('GEOID').reindex(idx)
m=(f.coreB>100).astype(float).values; w=1/(2+m)
pop=TR.pop_total.fillna(0).values
cx=np.array([(b['xmin']+b['xmax'])/2 for b in TR.bbox]); cy=np.array([(b['ymin']+b['ymax'])/2 for b in TR.bbox])
county=(TR.STATEFP+TR.COUNTYFP).values
