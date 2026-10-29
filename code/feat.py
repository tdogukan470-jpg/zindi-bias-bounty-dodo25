import pandas as pd, numpy as np
R=["maricopa-az","northern-ca","eastern-ok","south-central-tx","eastern-wa"]
def load(r):
    sub=pd.read_csv(f"{r}-sample-submission.csv",dtype={"GEOID":str})[["GEOID"]]
    st=pd.read_parquet(f"{r}-strata.parquet"); tr=pd.read_parquet(f"{r}-tracts.parquet"); acs=pd.read_parquet(f"{r}-acs.parquet")
    b=pd.read_parquet(f"{r}-bldg-tract.parquet"); rd=pd.read_parquet(f"{r}-roads-tract.parquet"); p=pd.read_parquet(f"{r}-pois-tract.parquet")
    df=sub.merge(st,on="GEOID",how="left").merge(acs[["GEOID","housing_units"]],on="GEOID",how="left")
    df=df.merge(tr[["GEOID","area_m2_calc"]+[c for c in tr.columns if c in("ALAND","AWATER")and c not in df.columns]],on="GEOID",how="left")
    bt=b.groupby("GEOID").agg(ov_b=("n","sum"),ov_b_area=("area_sum","sum"),ov_b_small=("n_small","sum"),ov_b_large=("n_large","sum"))
    bs=b.pivot_table(index="GEOID",columns="src",values="n",aggfunc="sum").add_prefix("ov_b_src_")
    df=df.merge(bt,on="GEOID",how="left").merge(bs,on="GEOID",how="left")
    rp=rd.pivot_table(index="GEOID",columns="class",values="len_m",aggfunc="sum").add_prefix("rd_")
    df=df.merge(rp,on="GEOID",how="left")
    p["fire"]=p.cat.eq("fire_department"); p["ems"]=p.cat.eq("ambulance_and_ems_services")
    p["school"]=p.cat.isin(["elementary_school","middle_school","high_school","school","private_school","public_school"])
    p["hosp"]=p.cat.eq("hospital")
    pt=p.groupby("GEOID").agg(ov_p=("id","size"),ov_fire=("fire","sum"),ov_ems=("ems","sum"),ov_school=("school","sum"),ov_hosp=("hosp","sum"),p_conf=("confidence","mean"))
    ps=p.pivot_table(index="GEOID",columns="src",values="id",aggfunc="size").add_prefix("ov_p_src_")
    df=df.merge(pt,on="GEOID",how="left").merge(ps,on="GEOID",how="left")
    df["region"]=r
    return df
D=pd.concat([load(r) for r in R],ignore_index=True)
for c in D.columns:
    if c.startswith(("ov_","rd_")): D[c]=D[c].fillna(0)
D.to_parquet("/home/claude/bb/feat.parquet")
print(D.shape)
