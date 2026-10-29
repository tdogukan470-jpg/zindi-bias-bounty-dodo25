import pandas as pd, re, numpy as np
H=pd.read_pickle("/home/claude/bb/H.pkl")
def core(L):
    for n in L:
        if re.fullmatch(r"US:I(:.*)?",n) and "Future" not in n: return True
        if re.fullmatch(r"US:US(:(Business|Alternate|Spur|Bypass|Truck))?",n): return True
        if re.fullmatch(r"US:(AZ|CA|OK|TX)(:(Loop|Spur|Business|Beltway|Park))?",n): return True
        if re.fullmatch(r"US:(OK:Turnpike|TX:Toll|TX:NTTA|TX:Harris:HCTRA)",n): return True
    return False
H["core"]=H.nl.map(core)
H["c4"]=H["class"].isin(["motorway","trunk","primary","secondary"])
for r,h in H.groupby("region"):
    Lc=h.len_m[h.core].sum(); L4=h.len_m[h.c4].sum()
    print(r, "ov4/core=%.2f"%(L4/Lc), " core by class:", (h[h.core].groupby("class").len_m.sum()/Lc).round(3).to_dict())
H.to_pickle("/home/claude/bb/H.pkl")
