"""fetch_pbr.py <out_dir>: download the CC0 Poly Haven texture sets used by the 'realistic' style (1k diffuse jpg,
roughness jpg, displacement png) and write <out_dir>/index.json with each set's mean linear albedo."""
import json, os, sys, urllib.request
import numpy as np
from PIL import Image
from pbr_sets import ALL_SETS
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
def get(u, p):
    if os.path.exists(p) and os.path.getsize(p) > 0: return
    req = urllib.request.Request(u, headers={"User-Agent": "solar-bgen/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(p + ".part", "wb") as f: f.write(r.read())
    os.replace(p + ".part", p)
idx = {}
for sid in ALL_SETS:
    req = urllib.request.Request(f"https://api.polyhaven.com/files/{sid}", headers={"User-Agent": "solar-bgen/1.0"})
    fl = json.load(urllib.request.urlopen(req, timeout=60))
    d = os.path.join(out, sid); os.makedirs(d, exist_ok=True)
    dif = fl.get("Diffuse") or fl.get("diffuse") or fl[sorted(k for k in fl if k.lower().startswith("col"))[0]]
    get(dif["1k"]["jpg"]["url"], f"{d}/diff.jpg")
    get(fl["Rough"]["1k"]["jpg"]["url"], f"{d}/rough.jpg")
    if "Displacement" in fl: get(fl["Displacement"]["1k"]["png"]["url"], f"{d}/disp.png")
    else: Image.open(f"{d}/diff.jpg").convert("L").save(f"{d}/disp.png")   # no height map: diffuse luminance
    a = np.asarray(Image.open(f"{d}/diff.jpg").convert("RGB").resize((128, 128)), dtype=np.float64) / 255
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    idx[sid] = dict(mean=[round(float(x), 4) for x in lin.reshape(-1, 3).mean(0)])
    print(sid, idx[sid], flush=True)
json.dump(idx, open(os.path.join(out, "index.json"), "w"), indent=1)
print("FETCH_OK", len(idx))
