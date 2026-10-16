"""audit.py: summarise clips in storage (diversity + checks).  python3 audit.py <clips prefix>   (STORAGE_CLI, default solar-storage)"""
import json, subprocess, sys, os, tempfile, collections
STORE = os.environ.get("STORAGE_CLI", "solar-storage")
P = sys.argv[1]
t = tempfile.mkdtemp()
subprocess.run([STORE, "cp", P + "/*.json", t + "/"], stdout=subprocess.DEVNULL)
C = collections.Counter
cnt = {k: C() for k in ("biome", "cam", "tod", "gen", "nchar", "chars", "hero_actions", "sky")}
rows = []
for f in sorted(os.listdir(t)):
    m = json.load(open(os.path.join(t, f))); r = m["recipe"]
    cnt["biome"][r["biome"]] += 1; cnt["cam"][m["camera"]["mode"]] += 1; cnt["tod"][r["time_of_day"]] += 1
    cnt["gen"][m["generator"]] += 1; cnt["nchar"][len(r["characters"])] += 1; cnt["sky"][r["lighting"]["hdri"]] += 1
    for c in r["characters"]: cnt["chars"][c["name"]] += 1
    cnt["hero_actions"]["+".join(a["cls"] for a in r["characters"][0]["actions"])] += 1
    rows.append((m["clip_id"], m["generator"], r["biome"], m["camera"]["mode"], r["time_of_day"], m["checks"]["player_visible_frac"],
                 [c["name"] + ("/" + c["behaviour"]) for c in r["characters"]]))
for r_ in rows: print(*r_)
print("N", len(rows))
for k, v in cnt.items(): print(k, dict(v))
