"""pick.py SRC AUDIT.jsonl JOBS.jsonl OUTDIR: seeded random pick (50 kept, 15 dropped) from the decision file of decide.py,
writes OUTDIR/<SRC>_pick.jsonl (input of review.py) and OUTDIR/solar-audit-<SRC>.html (videos side by side, flat names)."""
import sys, json, random, collections, html
SEED = 20261005
SRC, AUDIT, JOBS, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rows = [json.loads(l) for l in open(AUDIT)]
jobs = {json.loads(l)["id"]: json.loads(l) for l in open(JOBS)}
keep = [r for r in rows if r["keep"] is True]
drop = [r for r in rows if r["keep"] is False]
rng = random.Random(SEED)
pk = rng.sample(keep, min(50, len(keep)))
# dropped: one per reason first (rarest first), then random, 15 in all
by = collections.defaultdict(list)
for r in drop:
    for w in r["reasons"]:
        by[w].append(r)
pd, seen = [], set()
for w in sorted(by, key=lambda w: len(by[w])):
    cand = [r for r in by[w] if r["id"] not in seen]
    if cand and len(pd) < 15:
        r = rng.choice(cand); pd.append(r); seen.add(r["id"])
rest = [r for r in drop if r["id"] not in seen]
pd += rng.sample(rest, max(0, min(15 - len(pd), len(rest))))
cnt = collections.Counter(w for r in drop for w in r["reasons"])


def fname(i):
    return f"sa_{SRC}_" + i.split("/", 1)[1].replace("/", "_") + ".mp4"


with open(f"{OUT}/{SRC}_pick.jsonl", "w") as f:
    for r in pk + pd:
        j = jobs[r["id"]]
        f.write(json.dumps({"id": r["id"], "final": j["final"], "cond": j["cond"], "file": fname(r["id"])}) + "\n")


def facts(r):
    s = [f"motion corr lag0 {r.get('mcorr_0')}", f"best lag {r.get('mcorr_best_lag')}", f"edge recall {r.get('erec_0')} / mirrored {r.get('erec_base')}",
         f"dup {r['dup_final']:.2f}/{r['dup_cond']:.2f}", f"luma {r['luma_mean']:.0f}", f"contrast {r['contrast']:.0f}", f"cond cover {r['cond_cover_mean']:.2f}"]
    return " · ".join(s)


cond = {"magpie": "grey-box white model (white_box)", "carla": "semantic segmentation (semantic)"}[SRC]
nk = sum(1 for r in rows if r["keep"] is True); nd = len(drop); na = sum(1 for r in rows if r["keep"] is None)
h = [f"""<!doctype html><meta charset=utf-8><title>solar-audit-{SRC}</title>
<style>body{{font:14px/1.45 -apple-system,Helvetica,sans-serif;margin:20px;background:#fafafa;color:#222}}
.c{{margin:0 0 22px}} video{{width:100%;max-width:1344px;display:block;background:#000}} .m{{color:#555;font-size:12px}}
.r{{color:#b00;font-weight:600}} h2{{margin-top:36px}} code{{background:#eee;padding:1px 4px}}</style>
<h1>SolarRenderer audit spot check: {SRC}</h1>
<p>Condition: {cond}. Each clip shows three panels: final | condition | Canny edges of the condition (red) over the final. Magpie clips are 60 fps; every second frame is shown here (30 fps).</p>
<p>All {len(rows)} clips: kept {nk}, dropped {nd}{f', not audited {na}' if na else ''}. Random seed <code>{SEED}</code> (Python <code>random.Random({SEED})</code>):
50 clips drawn from the kept clips, then one dropped clip per reason (rarest reason first), topped up at random to 15.</p>
<p>Drop reasons: {' · '.join(f'{k} {v}' for k, v in cnt.most_common())}</p>
<h2>Kept clips (random 50)</h2>"""]
for r in pk:
    h.append(f"<div class=c><b>{html.escape(r['id'])}</b><div class=m>{facts(r)}</div><video src='{fname(r['id'])}' controls muted loop preload=none></video></div>")
h.append("<h2>Dropped clips (15)</h2>")
for r in pd:
    h.append(f"<div class=c><b>{html.escape(r['id'])}</b> <span class=r>dropped: {', '.join(r['reasons'])}</span><div class=m>{facts(r)}</div>"
             f"<video src='{fname(r['id'])}' controls muted loop preload=none></video></div>")
open(f"{OUT}/solar-audit-{SRC}.html", "w").write("\n".join(h))
print(SRC, "picked", len(pk), len(pd))
