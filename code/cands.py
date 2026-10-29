import pandas as pd, numpy as np
D=pd.read_parquet("feat.parquet"); H=pd.read_pickle("H.pkl")
c4=["motorway","trunk","primary","secondary"]
H["c4"]=H["class"].isin(c4)
g=H.groupby("GEOID").apply(lambda h: pd.Series({"Lcore":h.len_m[h.core].sum(),"L4":h.len_m[h.c4].sum()}),include_groups=False)
D=D.drop(columns=[c for c in ["Lcore","L4"] if c in D]).merge(g,left_on="GEOID",right_index=True,how="left")
D[["Lcore","L4"]]=D[["Lcore","L4"]].fillna(0)
D["county"]=D.GEOID.str[:5]
X=pd.DataFrame({"GEOID":D.GEOID})
X["t_def"]=(D.Lcore>20).astype(float)
X["t_gap15"]=np.where(D.Lcore>20,(1-D.L4/(1.5*D.Lcore.clip(lower=1))).clip(0,1),0)
usfs=D.usfs_BuildingCount_sum.fillna(0)
X["b_gap"]=np.where(usfs>0,(1-D.ov_b/usfs.clip(lower=1)).clip(0,1),0)
X["fire0"]=(D.ov_fire==0).astype(float)
X["school0"]=(D.ov_school==0).astype(float)
cty_p=D.groupby("county").ov_p.transform("sum"); cty_pop=D.groupby("county").pop_total.transform("sum")
share=(D.ov_p/cty_p.clip(lower=1))/(D.pop_total/cty_pop.clip(lower=1)).replace(0,np.nan)
X["cbp_pop"]=(1-share.fillna(1)).clip(0,1)
X["rural"]=(D.pct_urban.fillna(0)<0.5).astype(float)
for r in D.region.unique(): X["reg_"+r]=(D.region==r).astype(float)
X["region"]=D.region.values
X.to_parquet("X.parquet"); D.to_parquet("feat2.parquet")
print(X.drop(columns=["GEOID","region"]).describe().T[["mean","std"]].round(3))
