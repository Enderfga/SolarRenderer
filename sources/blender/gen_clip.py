"""gen_clip.py -- build + render one procedural game-style clip in Blender.

blender -b --gpu-backend opengl --factory-startup --python gen_clip.py -- <seed> <out_dir> [--frames N] [--samples S]
          [--scale PCT] [--no-render]
Writes <out_dir>/render/####.png, <out_dir>/white_box/####.png, <out_dir>/meta.json.
Same seed -> same clip (all randomness from random.Random(seed, attempt)).
"""
import bpy, bmesh, sys, os, json, math, random, re, time, glob
import numpy as np
from mathutils import Vector, Matrix, Euler
from bpy_extras.object_utils import world_to_camera_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import biomes as BM

ASSETS = os.environ.get("BGEN_ASSETS", "gen_assets")
# absolute asset directory recorded in catalog.json keys when catalog.py was run (remapped to ASSETS below)
CATALOG_ROOT = os.environ.get("BGEN_CATALOG_ROOT", os.path.abspath(ASSETS))
W, H, FPS = 1344, 768, 24

argv = sys.argv[sys.argv.index("--") + 1:]
SEED, OUT = int(argv[0]), argv[1]
def opt(name, default, typ=int):
    return typ(argv[argv.index(name) + 1]) if name in argv else default
NF = opt("--frames", 192)
SAMPLES = opt("--samples", 32)
PCT = opt("--scale", 100)
NO_RENDER = "--no-render" in argv
# Headless EEVEE renders on the first GPU whatever CUDA_VISIBLE_DEVICES says, so a machine running
# several processes per card piles them all onto card 0. Cycles runs on CUDA and honours it; only the cheap
# white-box pass is left on card 0.
ENGINE = opt("--engine", "eevee", str)
FORCE_BIOME = opt("--biome", None, str)
if FORCE_BIOME: FORCE_BIOME = FORCE_BIOME.replace("_", " ")   # "city_street" == "city street"
FORCE_CAM = opt("--cam", None, str)
# Render styles: extra frame-aligned final streams of the same scene.  'toon' is composited from the light passes of
# the main Cycles render (no extra render); 'realistic' is a second Cycles render with Poly Haven PBR surfaces;
# 'pixel' is made by post.py from the toon frames.  Style choice uses its own RNG, so the scene of a seed is unchanged.
STYLES = [x for x in opt("--styles", "toon,realistic,pixel", str).split(",") if x]
REAL_FRAC = opt("--real-frac", 0.5, float)        # fraction of clips that get the realistic render (it costs ~1.15x the main render)
REAL_SAMPLES = opt("--real-samples", 32)          # Cycles samples of the realistic render (0 = same as --samples)
SRNG = random.Random(SEED * 7919 + 17)
if "realistic" in STYLES and SRNG.random() >= REAL_FRAC: STYLES.remove("realistic")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ asset index
CAT = json.load(open(os.path.join(ASSETS, "catalog.json")))
INDEX = {}  # pack -> {stem: (abs_path, info)}
for k, v in CAT.items():
    rel = k.split("packs/", 1)[1] if "packs/" in k else k.split("gen_assets/", 1)[-1]
    for pk, pref in BM.PACKS.items():
        if rel.startswith(pref) or rel.startswith("packs/" + pref):
            stem = os.path.basename(k)
            stem = re.sub(r"(\.gltf)?\.(glb|gltf)$", "", stem)
            path = os.path.join(ASSETS, "packs", rel) if not rel.startswith("quat/") else os.path.join(ASSETS, rel)
            if not os.path.exists(path):
                path = k.replace(CATALOG_ROOT, ASSETS)
            INDEX.setdefault(pk, {}).setdefault(stem, (path, v))
            break

def resolve(ref, sub=None):
    rx = ref["rx"] if sub is None else re.sub(re.escape(sub[0]), sub[1], ref["rx"])
    out = []
    for stem, (p, info) in sorted(INDEX.get(ref["pack"], {}).items()):
        if not re.fullmatch(rx, stem): continue
        if "error" in info or info.get("n_mesh", 0) == 0: continue
        sf = ref.get("sizefilter")
        if sf:
            dx, dy, dz = [info["max"][i] - info["min"][i] for i in range(3)]
            if not (sf[0] <= dx <= sf[1] and sf[2] <= dy <= sf[3] and sf[4] <= dz <= sf[5]): continue
        out.append(dict(stem=stem, path=p, info=info, pack=ref["pack"], s=ref.get("s", 1.0), ref=ref))
    return out

# ------------------------------------------------------------------ small helpers
def lerp(a, b, t): return a + (b - a) * t
def wchoice(rng, d):
    ks = list(d.keys()); ws = [d[k] for k in ks]
    return rng.choices(ks, weights=ws, k=1)[0]

def chaikin(pts, it=3):
    for _ in range(it):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            new.append(0.75 * a + 0.25 * b); new.append(0.25 * a + 0.75 * b)
        new.append(pts[-1]); pts = new
    return pts

class Path:
    """Arc-length parametrised 2D polyline."""
    def __init__(self, pts):
        P = np.array(pts, dtype=float)
        # resample at 0.25 unit
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        S = np.concatenate([[0], np.cumsum(seg)])
        n = max(2, int(S[-1] / 0.25))
        s = np.linspace(0, S[-1], n)
        self.P = np.stack([np.interp(s, S, P[:, 0]), np.interp(s, S, P[:, 1])], 1)
        self.S = s; self.L = s[-1]
        d = np.gradient(self.P, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        self.D = d
    def pos(self, s):
        s = np.clip(s, 0, self.L)
        return np.array([np.interp(s, self.S, self.P[:, 0]), np.interp(s, self.S, self.P[:, 1])])
    def dir(self, s):
        s = np.clip(s, 0, self.L)
        d = np.array([np.interp(s, self.S, self.D[:, 0]), np.interp(s, self.S, self.D[:, 1])])
        return d / (np.linalg.norm(d) + 1e-9)
    def normal(self, s):
        d = self.dir(s); return np.array([-d[1], d[0]])
    def dist(self, xy):
        xy = np.atleast_2d(xy)
        return np.min(np.linalg.norm(self.P[None, :, :] - xy[:, None, :], axis=2), axis=1)

# ------------------------------------------------------------------ scene building
class Builder:
    def __init__(self, rng):
        self.rng = rng
        self.protos = {}
        self.circles = []   # (x, y, r) occupied
        self.lamps = []     # (x, y, z)
        self.used = {}      # pack -> set(stems)
        self.counts = {}
        sc = bpy.context.scene
        self.lib = bpy.data.collections.new("_lib"); sc.collection.children.link(self.lib)
        self.world_col = bpy.data.collections.new("world"); sc.collection.children.link(self.world_col)

    def proto(self, a):
        key = a["path"]
        if key in self.protos: return self.protos[key]
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=a["path"])
        new = [o for o in bpy.data.objects if o not in before]
        col = bpy.data.collections.new("P_" + a["stem"]); self.lib.children.link(col)
        for o in new:
            for c in list(o.users_collection): c.objects.unlink(o)
            col.objects.link(o)
            if o.type == "MESH" and o.name.startswith("Icosphere"): o.hide_render = True
        self.protos[key] = col
        return col

    def place(self, a, x, y, rz, s=None, z=0.0, tag="prop"):
        s = a["s"] if s is None else s
        e = bpy.data.objects.new(a["stem"], None)
        e.instance_type = "COLLECTION"; e.instance_collection = self.proto(a)
        e.location = (x, y, z); e.rotation_euler = (0, 0, rz); e.scale = (s, s, s)
        e["bgen_stem"] = a["stem"]; e["bgen_tag"] = tag   # read back by build_proxies (category of the proxy)
        self.world_col.objects.link(e)
        self.used.setdefault(a["pack"], set()).add(a["stem"])
        self.counts[tag] = self.counts.get(tag, 0) + 1
        if a["ref"].get("lamp") and (not a["ref"].get("lampfilter") or a["ref"]["lampfilter"] in a["stem"]):
            hgt = (a["info"]["max"][2] - a["info"]["min"][2]) * s
            self.lamps.append((x, y, z + a["ref"]["lamp"] * hgt + 0.3))
        return e

    @staticmethod
    def footprint(a, s=None):
        s = a["s"] if s is None else s
        i = a["info"]
        return (i["max"][0] - i["min"][0]) * s, (i["max"][1] - i["min"][1]) * s, (i["max"][2] - i["min"][2]) * s, \
               ((i["max"][0] + i["min"][0]) / 2 * s, (i["max"][1] + i["min"][1]) / 2 * s)

    def free(self, x, y, r):
        for (cx, cy, cr) in self.circles:
            if (cx - x) ** 2 + (cy - y) ** 2 < (cr + r) ** 2: return False
        return True

    def lockout(self, x, y, r): self.circles.append((x, y, r))

    def finalize(self):
        self.lib.hide_render = True; self.lib.hide_viewport = True
        bpy.context.view_layer.layer_collection.children["_lib"].exclude = True


def material(name, c1, c2, scale=0.08, rough=0.9):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = rough
    tc = nt.nodes.new("ShaderNodeTexCoord"); nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = scale; nz.inputs["Detail"].default_value = 6
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*c1, 1); ramp.color_ramp.elements[1].color = (*c2, 1)
    ramp.color_ramp.elements[0].position = 0.35; ramp.color_ramp.elements[1].position = 0.65
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"]); nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def ground_plane(b, c1, c2, size=600):
    me = bpy.data.meshes.new("ground")
    h = size / 2
    me.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
    o = bpy.data.objects.new("ground", me); b.world_col.objects.link(o)
    o.data.materials.append(material("ground_mat", c1, c2, 0.05))
    return o


def ribbon(b, path, half, color, z=0.03):
    vs, fs = [], []
    for i in range(0, len(path.P), 2):
        p = path.P[i]; d = path.D[i]; n = np.array([-d[1], d[0]])
        w = half * (1 + 0.08 * math.sin(i * 0.37))
        vs += [(*(p + n * w), z), (*(p - n * w), z)]
    for i in range(len(vs) // 2 - 1):
        fs.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
    me = bpy.data.meshes.new("pathmesh"); me.from_pydata(vs, [], fs)
    o = bpy.data.objects.new("path", me); b.world_col.objects.link(o)
    c2 = tuple(max(0, x * 0.75) for x in color)
    o.data.materials.append(material("path_mat", color, c2, 0.3))
    return o

# ---- layouts --------------------------------------------------------------------------------

def grid_path(rng, ncells_min, allow_turns=True):
    """Manhattan path on integer cells, no self-intersection.  Returns list of (i, j)."""
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for _ in range(200):
        cells = [(0, 0)]; d = 0; seen = {(0, 0)}
        ok = True
        while len(cells) < ncells_min:
            run = rng.randint(3, 7)
            for _ in range(run):
                c = (cells[-1][0] + dirs[d][0], cells[-1][1] + dirs[d][1])
                if any((c[0] + a, c[1] + bb) in seen and (c[0] + a, c[1] + bb) != cells[-1]
                       for a, bb in dirs):
                    ok = False; break
                cells.append(c); seen.add(c)
            if not ok: break
            if allow_turns: d = (d + rng.choice([1, 3])) % 4
        if ok: return cells
    return [(0, j) for j in range(ncells_min)]


def build_city(b, bio, rng, need_len):
    T = bio["T"]
    n = int(need_len / T) + 3
    cells = grid_path(rng, n, allow_turns=rng.random() < 0.8)
    road = set(cells)
    pts = [np.array([i * T, j * T]) for i, j in cells]
    path = Path(chaikin(pts, 2))
    roads = resolve(bio["road"][0]); cross = resolve(bio["road"][1]); junc = resolve(bio["junction"])[0]
    side = resolve(bio["sidewalk"])[0]
    blds = [resolve(r) for r in bio["buildings"]]
    blds = [x for x in blds if x]
    style = rng.sample(range(len(blds)), k=min(len(blds), rng.choice([1, 2, 2, 3])))
    # roads
    for k, (i, j) in enumerate(cells):
        prv = cells[k - 1] if k > 0 else None; nxt = cells[k + 1] if k + 1 < len(cells) else None
        dirv = [(c[0] - i, c[1] - j) for c in (prv, nxt) if c]
        straight = len(dirv) == 2 and dirv[0][0] == -dirv[1][0] and dirv[0][1] == -dirv[1][1]
        if not straight and len(dirv) == 2:
            b.place(junc, i * T, j * T, 0, tag="road")
        else:
            horiz = any(abs(dx) == 1 for dx, dy in dirv)
            a = rng.choice(cross) if rng.random() < 0.12 else rng.choice(roads)
            b.place(a, i * T, j * T, math.pi / 2 if horiz else 0, tag="road")
    # sidewalks (8-neighbourhood) + edge props, buildings ring 2..4
    side_cells = set()
    for (i, j) in cells:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                c = (i + di, j + dj)
                if c not in road: side_cells.add(c)
    edge_pool = []
    for r in bio["edge"]:
        for a in resolve(r): edge_pool.append((a, r.get("w", 1) / max(1, len(resolve(r)))))
    for (i, j) in sorted(side_cells):
        b.place(side, i * T, j * T, 0, tag="sidewalk")
        # nearest road cell -> direction
        nb = [(i + di, j + dj) for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)) if (i + di, j + dj) in road]
        for (ri, rj) in nb:
            if rng.random() < 0.75:
                a = rng.choices([e[0] for e in edge_pool], weights=[e[1] for e in edge_pool])[0]
                dx, dy = ri - i, rj - j
                along = rng.uniform(-0.35, 0.35) * T
                px = i * T + dx * 0.32 * T + (along if dx == 0 else 0)
                py = j * T + dy * 0.32 * T + (along if dy == 0 else 0)
                rz = math.atan2(dx, -dy) + (0 if "light" in a["stem"] else rng.choice([0, 0, math.pi / 2, rng.uniform(0, 6.28)]))
                b.place(a, px, py, rz, z=side["info"]["max"][2] * side["s"], tag="edge")
    # buildings ring
    ring = set()
    for (i, j) in side_cells:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                c = (i + di, j + dj)
                if c not in road and c not in side_cells: ring.add(c)
    ring2 = set()
    for (i, j) in ring:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                c = (i + di, j + dj)
                if c not in road and c not in side_cells and c not in ring: ring2.add(c)
    for (i, j) in sorted(ring | ring2):
        # face the closest road cell
        best = min(cells, key=lambda c: (c[0] - i) ** 2 + (c[1] - j) ** 2)
        dx, dy = best[0] - i, best[1] - j
        if abs(dx) >= abs(dy): dx, dy = (1 if dx > 0 else -1), 0
        else: dx, dy = 0, (1 if dy > 0 else -1)
        pool = blds[rng.choice(style)]
        a = rng.choice(pool)
        fw, fd, fh, _ = Builder.footprint(a)
        s = a["s"] * min(1.0, 0.92 * T / max(fw, fd, 1e-3))
        if a["ref"].get("base"): b.place(side, i * T, j * T, 0, tag="base")
        b.place(a, i * T, j * T, math.atan2(dx, -dy), s=s, tag="building")
    # parked cars on straight road cells, offset to the side
    cars = resolve(bio["vehicles"][0])
    for k, (i, j) in enumerate(cells[1:-1], 1):
        if rng.random() < 0.3:
            d = path.dir(k * T) if False else None
            prv, nxt = cells[k - 1], cells[k + 1]
            if prv[0] == nxt[0]:   # vertical road
                sgn = rng.choice([-1, 1]); b.place(rng.choice(cars), i * T + sgn * 0.3 * T, j * T + rng.uniform(-1.5, 1.5), 0 if sgn < 0 else math.pi, tag="vehicle")
            elif prv[1] == nxt[1]:
                sgn = rng.choice([-1, 1]); b.place(rng.choice(cars), i * T + rng.uniform(-1.5, 1.5), j * T + sgn * 0.3 * T, math.pi / 2 if sgn > 0 else -math.pi / 2, tag="vehicle")
    ground_plane(b, bio["ground"], bio["ground2"])
    info = dict(lane=3.0, cells=len(cells), turns=sum(1 for k in range(1, len(cells) - 1)
                                                        if (cells[k + 1][0] - cells[k - 1][0]) and (cells[k + 1][1] - cells[k - 1][1])))
    return path, info


def build_dungeon(b, bio, rng, need_len):
    T = bio["T"]
    n = int(need_len / T) + 3
    cells = grid_path(rng, n)
    floor = set(cells)
    # widen: corridor + random rooms
    for (i, j) in cells:
        for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            floor.add((i + di, j + dj))
    for (i, j) in rng.sample(cells, k=max(1, len(cells) // 6)):
        r = rng.randint(1, 2)
        for di in range(-r - 1, r + 2):
            for dj in range(-r - 1, r + 2):
                floor.add((i + di, j + dj))
    pts = [np.array([i * T, j * T]) for i, j in cells]
    path = Path(chaikin(pts, 2))
    fl = resolve(rng.choice(bio["floors"]))
    fl2 = resolve(rng.choice(bio["floors"]))
    walls = resolve(bio["walls"][0])
    plain = [w for w in walls if w["stem"] in ("wall", "wall_cracked", "wall_arched", "wall_broken")] or walls
    pillars = resolve(bio["pillar"][0])
    for (i, j) in sorted(floor):
        a = rng.choice(fl if rng.random() < 0.85 else fl2)
        b.place(a, i * T, j * T, rng.choice([0, 1, 2, 3]) * math.pi / 2, tag="floor")
    # walls on floor boundary edges; wall model long axis = X, thickness Y
    edges = []
    for (i, j) in sorted(floor):
        for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            if (i + di, j + dj) not in floor:
                cx, cy = (i + di * 0.5) * T, (j + dj * 0.5) * T
                rz = math.atan2(-di, dj) if False else (0.0 if dj != 0 else math.pi / 2)
                # face inward: model -Y faces floor
                rz = math.atan2(-di, dj) * 0 + {(0, 1): math.pi, (0, -1): 0.0, (1, 0): math.pi / 2, (-1, 0): -math.pi / 2}[(di, dj)]
                a = rng.choice(plain) if rng.random() < 0.7 else rng.choice(walls)
                b.place(a, cx, cy, rz, tag="wall")
                edges.append((cx, cy, di, dj))
    # pillars at outer corners of the floor
    corners = set()
    for (i, j) in floor:
        for ci in (-0.5, 0.5):
            for cj in (-0.5, 0.5):
                cnt = sum(((int(round(i + ci + a)), int(round(j + cj + bb))) in floor) for a in (-0.5, 0.5) for bb in (-0.5, 0.5))
                if cnt in (1, 3): corners.add((i + ci, j + cj))
    for (ci, cj) in sorted(corners):
        if pillars and rng.random() < 0.6: b.place(rng.choice(pillars), ci * T, cj * T, 0, tag="pillar")
    # props next to walls, kept off the walking lane
    pool = []
    for r in bio["edge"]:
        res = resolve(r)
        for a in res: pool.append((a, r.get("w", 1) / max(1, len(res))))
    for (cx, cy, di, dj) in edges:
        if rng.random() < 0.45:
            a = rng.choices([e[0] for e in pool], weights=[e[1] for e in pool])[0]
            fw, fd, fh, _ = Builder.footprint(a)
            off = 0.5 + max(fw, fd) / 2 + 0.2
            px, py = cx - di * off + (rng.uniform(-1, 1) if di == 0 else 0), cy - dj * off + (rng.uniform(-1, 1) if dj == 0 else 0)
            if path.dist(np.array([px, py]))[0] < 3.2 + max(fw, fd) / 2: continue
            if not b.free(px, py, max(fw, fd) / 2): continue
            b.lockout(px, py, max(fw, fd) / 2)
            b.place(a, px, py, math.atan2(-di, dj) + rng.choice([0, math.pi / 2, math.pi, -math.pi / 2]), tag="prop")
    # wall torches: point lights along walls
    tm = resolve(dict(pack="kk_dun", rx="torch_mounted", s=1.0))
    rzmap = {(0, 1): math.pi, (0, -1): 0.0, (1, 0): math.pi / 2, (-1, 0): -math.pi / 2}
    for (cx, cy, di, dj) in edges:
        if tm and rng.random() < 0.25:
            b.place(tm[0], cx - di * 0.5, cy - dj * 0.5, rzmap[(di, dj)], z=2.3, tag="torch")
            b.lamps.append((cx - di * 1.0, cy - dj * 1.0, 3.0))
    ground_plane(b, bio["ground"], bio["ground2"])
    info = dict(lane=2.8, cells=len(floor))
    return path, info


def free_path(rng, need_len):
    pts = [np.array([0.0, 0.0])]; h = rng.uniform(0, 2 * math.pi); L = 0
    while L < need_len:
        seg = rng.uniform(10, 24)
        pts.append(pts[-1] + seg * np.array([math.cos(h), math.sin(h)])); L += seg
        h += rng.choice([-1, 1]) * rng.uniform(0.15, 1.0) * (1 if rng.random() < 0.8 else 0)
    return Path(chaikin(pts, 4))


def build_free(b, bio, rng, need_len, variant):
    path = free_path(rng, need_len)
    hw = rng.uniform(*bio["hw"]); lane = bio["lane"]
    sub = None
    if variant == "autumn": sub = ("(_dark)?", "_fall")
    lining = [x for r in bio["lining"] for x in [(resolve(r, sub) or resolve(r), r)]]
    def pick(lst):
        ws = [r.get("w", 1.0) for _, r in lst]
        res, r = rng.choices(lst, weights=ws)[0]
        return rng.choice(res) if res else None
    # path itself is locked out up to hw
    def corridor_ok(x, y, fw, fd, rz, margin):
        c, s_ = math.cos(rz), math.sin(rz)
        pts = [(x, y)] + [(x + c * dx - s_ * dy, y + s_ * dx + c * dy) for dx in (-fw / 2, fw / 2) for dy in (-fd / 2, fd / 2)]
        return np.all(path.dist(np.array(pts)) >= margin)
    # 1. lining on both sides
    for side in (-1, 1):
        s = rng.uniform(0, 4)
        while s < path.L:
            a = pick(lining)
            if a is None: break
            jit = 1 + rng.uniform(-1, 1) * a["ref"].get("jit", 0)
            sc_ = a["s"] * jit
            fw, fd, fh, (ox, oy) = Builder.footprint(a, sc_)
            d = path.dir(s); n = path.normal(s) * side
            off = hw + fd / 2 + rng.uniform(0.0, 1.5)
            x, y = path.pos(s) + n * off
            rz = math.atan2(-n[0], n[1]) if a["ref"].get("jit", 0) < 0.2 else rng.uniform(0, 2 * math.pi)
            r = 0.5 * max(fw, fd) * 0.85
            if corridor_ok(x, y, fw, fd, rz, hw) and b.free(x, y, r):
                b.place(a, x, y, rz, s=sc_, tag="lining"); b.lockout(x, y, r)
                s += max(fw * 0.8, 2.0) + rng.uniform(0, 3)
            else:
                s += 1.5
    # 2. background fill
    fills = [x for r in bio["fill"] for x in [(resolve(r, sub) or resolve(r), r)]]
    nfill = int(path.L * 1.4)
    for _ in range(nfill):
        a = pick(fills)
        if a is None: break
        s = rng.uniform(-10, path.L + 10); side = rng.choice([-1, 1])
        off = hw + rng.uniform(4, 45)
        x, y = path.pos(s) + path.normal(s) * side * off + path.dir(s) * (min(s, 0) + max(s - path.L, 0))
        sc_ = a["s"] * (1 + rng.uniform(-1, 1) * a["ref"].get("jit", 0))
        fw, fd, fh, _ = Builder.footprint(a, sc_)
        r = 0.5 * max(fw, fd) * 0.8; rz = rng.uniform(0, 2 * math.pi)
        if corridor_ok(x, y, fw, fd, rz, hw + 1) and b.free(x, y, r):
            b.place(a, x, y, rz, s=sc_, tag="fill"); b.lockout(x, y, r)
    # 3. edge props between lane and lining
    edges = [x for r in bio["edge"] for x in [(resolve(r, sub) or resolve(r), r)]]
    nedge = int(path.L * rng.uniform(0.35, 0.7))
    for _ in range(nedge):
        a = pick(edges)
        if a is None: break
        s = rng.uniform(0, path.L); side = rng.choice([-1, 1])
        sc_ = a["s"] * (1 + rng.uniform(-1, 1) * a["ref"].get("jit", 0))
        fw, fd, fh, _ = Builder.footprint(a, sc_)
        r = 0.5 * max(fw, fd)
        lo = lane + 0.4 + r
        if lo > hw + 2.5: continue
        off = rng.uniform(lo, max(lo, hw + 1.0))
        x, y = path.pos(s) + path.normal(s) * side * off
        rz = rng.uniform(0, 2 * math.pi)
        if corridor_ok(x, y, fw, fd, rz, lane + 0.4) and b.free(x, y, r * 0.9):
            b.place(a, x, y, rz, s=sc_, tag="edge"); b.lockout(x, y, r * 0.9)
    # 4. small scatter (ignores occupancy except the lane)
    if bio.get("scatter"):
        sc_list = [x for r in bio["scatter"] for x in [(resolve(r), r)]]
        for _ in range(int(path.L * 2.5)):
            a = pick(sc_list)
            s = rng.uniform(-5, path.L + 5); side = rng.choice([-1, 1])
            off = rng.uniform(lane + 0.6, hw + 25)
            x, y = path.pos(s) + path.normal(s) * side * off
            b.place(a, x, y, rng.uniform(0, 6.3), s=a["s"] * (1 + rng.uniform(-0.3, 0.3)), tag="scatter")
    ground_plane(b, bio["ground"], bio["ground2"])
    if bio.get("path"): ribbon(b, path, lane + 0.6, bio["path"])
    return path, dict(lane=lane, half_width=round(hw, 2))

# ------------------------------------------------------------------ characters

def import_character(name, rng):
    pack, stem, kind, scale, anims, extra = BM.CHARACTERS[name]
    a = INDEX[pack][stem]
    before_o = set(bpy.data.objects); before_a = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=a[0])
    objs = [o for o in bpy.data.objects if o not in before_o]
    acts = {ac.name.split(".")[0] if ac.name.split(".")[-1].isdigit() else ac.name: ac for ac in bpy.data.actions if ac not in before_a}
    # strip blender's '.001' style suffix
    acts = {re.sub(r"\.\d{3}$", "", k): v for k, v in acts.items()}
    arm = next(o for o in objs if o.type == "ARMATURE")
    if arm.parent is not None:
        mw = arm.matrix_world.copy(); arm.parent = None; arm.matrix_world = mw
    arm.rotation_mode = "XYZ"
    weapons = []
    if extra.get("weapons"):
        weapons = rng.choice(extra["weapons"])
        body = extra["body"]
        for o in objs:
            if o.type == "MESH" and not o.name.startswith(body) and re.sub(r"\.\d{3}$", "", o.name) not in weapons:
                bpy.data.objects.remove(o, do_unlink=True)
        if pack == "kk_skel":  # skeleton weapons are separate assets: attach to hand slots
            for k, wname in enumerate(weapons):
                if wname not in INDEX["kk_skel"]: continue
                bo = set(bpy.data.objects)
                bpy.ops.import_scene.gltf(filepath=INDEX["kk_skel"][wname][0])
                for o in [o for o in bpy.data.objects if o not in bo]:
                    if o.parent is None:
                        bone = "handslot.r" if (k == 0 and "Shield" not in wname) else "handslot.l"
                        if bone in arm.data.bones:
                            o.parent = arm; o.parent_type = "BONE"; o.parent_bone = bone
                            o.matrix_parent_inverse = Matrix.Identity(4)
                            o.location = (0, -arm.data.bones[bone].length, 0)
    for o in [o for o in bpy.data.objects if o not in before_o]:
        if o.type == "MESH" and o.name.startswith("Icosphere"):
            bpy.data.objects.remove(o, do_unlink=True)
    objs = [o for o in bpy.data.objects if o not in before_o]
    meshes = [o for o in objs if o.type == "MESH"]
    # pick actions per class
    chosen = {}
    for cls, cands in anims.items():
        have = [c for c in cands if c in acts]
        if have: chosen[cls] = have
    return dict(base_scale=tuple(arm.scale), name=name, kind=kind, arm=arm, objs=objs, meshes=meshes, acts=acts, classes=chosen,
                scale=scale, pack=pack, stem=stem, weapons=weapons, animal=bool(extra.get("animal")))


def measure_action(ch, act):
    """In-place locomotion speed + forward direction (armature local XY) from stance-foot motion."""
    arm = ch["arm"]; sc = bpy.context.scene
    ad = arm.animation_data_create(); ad.action = act
    try:
        if act.slots and ad.action_slot is None: ad.action_slot = act.slots[0]
    except Exception: pass
    f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
    names = [b.name for b in arm.data.bones]
    P = []
    for f in range(f0, f1 + 1):
        sc.frame_set(f)
        mw = arm.matrix_world
        P.append([list(mw @ arm.pose.bones[n].tail) for n in names])
    P = np.array(P)  # F x B x 3 (armature space)
    ad.action = None
    zmin = P[:, :, 2].min()
    low = [k for k in range(len(names)) if P[:, k, 2].min() < zmin + 0.12 * (P[:, :, 2].max() - zmin + 1e-6)]
    vs = []
    for k in low:
        z = P[:, k, 2]; thr = z.min() + 0.04 * (P[:, :, 2].max() - zmin + 1e-6)
        for f in range(len(z) - 1):
            if z[f] < thr and z[f + 1] < thr: vs.append(P[f + 1, k, :2] - P[f, k, :2])
    if not vs: return 0.0, None
    v = np.mean(vs, axis=0); sp = float(np.linalg.norm(v))
    return sp, (-v / (sp + 1e-9))


def setup_character(ch, plan, traj_xy, heading, ground_z, frames):
    """plan: list of (cls, action_name, f_start, f_end).  traj_xy/heading per frame."""
    arm = ch["arm"]; sc = bpy.context.scene
    root = bpy.data.objects.new("Root_" + ch["name"], None); sc.collection.objects.link(root)
    arm.parent = root; arm.matrix_parent_inverse = Matrix.Identity(4)
    arm.location = (0, 0, 0)
    arm.rotation_mode = "XYZ"
    arm.rotation_euler = (arm.rotation_euler.x, arm.rotation_euler.y, arm.rotation_euler.z + ch["face_rot"])
    s = ch["scale"]; arm.scale = tuple(x * s for x in ch["base_scale"])
    ad = arm.animation_data_create(); ad.action = None
    for t in list(ad.nla_tracks): ad.nla_tracks.remove(t)
    tracks = [ad.nla_tracks.new(), ad.nla_tracks.new()]
    B = 8
    for k, (cls, an, fs, fe) in enumerate(plan):
        act = ch["acts"][an]
        L = max(1.0, act.frame_range[1] - act.frame_range[0])
        start = fs - (B if k > 0 else 0) - (ch["phase"] if k == 0 else 0)
        end = fe + (B if k + 1 < len(plan) else 4)
        st = tracks[k % 2].strips.new(f"{an}_{k}", int(start), act)
        st.extrapolation = "NOTHING"; st.use_auto_blend = False
        st.repeat = max(0.05, (end - int(start)) / L)
        if k % 2 == 1: st.blend_in = B
        if k % 2 == 1 and k + 1 < len(plan): st.blend_out = B
    for f in frames:
        x, y = traj_xy[f - 1]; root.location = (x, y, ground_z[f - 1])
        root.rotation_euler = (0, 0, heading[f - 1] - math.pi / 2)
        root.keyframe_insert("location", frame=f); root.keyframe_insert("rotation_euler", frame=f)
    ch["root"] = root

# ------------------------------------------------------------------ main build

def ray_ignore(sc, dg, origin, direction, dist, ignore):
    o = Vector(origin); d = Vector(direction).normalized(); left = dist
    for _ in range(8):
        hit, loc, nrm, idx, ob, mat = sc.ray_cast(dg, o, d, distance=left)
        if not hit: return None
        if ob is not None and ob.name in ignore:
            step = (loc - o).length + 0.02; o = loc + d * 0.02; left -= step
            if left <= 0: return None
            continue
        return (loc - Vector(origin)).length, ob
    return None


def unwrap(a):
    return np.unwrap(np.array(a))


def smooth(arr, tau_frames):
    """Zero-phase exponential smoothing (forward-backward)."""
    a = np.array(arr, dtype=float); k = 1.0 / max(1.0, tau_frames)
    out = a.copy()
    for i in range(1, len(a)): out[i] = out[i - 1] + k * (out[i] - out[i - 1])
    for i in range(len(a) - 2, -1, -1): out[i] = out[i + 1] + k * (out[i] - out[i + 1])
    return out


def build(seed, attempt):
    rng = random.Random(seed * 1000 + attempt)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = FPS; sc.frame_start, sc.frame_end = 1, NF
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, PCT
    frames = list(range(1, NF + 1))
    R = dict(seed=seed, attempt=attempt)

    biome = FORCE_BIOME or rng.choices(list(BM.BIOMES), weights=[v["weight"] for v in BM.BIOMES.values()])[0]
    bio = BM.BIOMES[biome]; R["biome"] = biome
    tod = wchoice(rng, bio["tod"]); R["time_of_day"] = tod
    cam_mode = FORCE_CAM or rng.choices(["follow", "first_person", "top_down", "orbit"], weights=[0.55, 0.12, 0.15, 0.18])[0]
    R["camera_mode"] = cam_mode
    variant = None
    if biome == "forest": variant = rng.choice([None, None, "autumn"])
    R["variant"] = variant

    # ---- characters: hero + NPCs
    hero_name = rng.choice(bio["chars"]["hero"])
    n_npc = rng.choices([0, 1, 2, 3], weights=[0.2, 0.35, 0.3, 0.15])[0]
    if cam_mode == "first_person": n_npc = max(1, n_npc)
    npc_names = [rng.choice(bio["chars"]["npc"]) for _ in range(n_npc)]
    chars = []
    for k, nm in enumerate([hero_name] + npc_names):
        ch = import_character(nm, rng); ch["id"] = k; ch["role"] = "player" if k == 0 else "npc"
        ch["phase"] = rng.randint(0, 20)
        chars.append(ch)
    # measure locomotion
    for ch in chars:
        ch["speeds"] = {}
        face = None
        for cls in ("walk", "run"):
            for an in ch["classes"].get(cls, []):
                sp, fw = measure_action(ch, ch["acts"][an])
                v = sp * FPS
                default = (1.7 if cls == "walk" else 4.2) * (ch["arm"].dimensions.z / 2.4 if ch["animal"] else 1.0)
                if not (0.25 * default < v < 3.5 * default): v = default
                ch["speeds"][an] = v  # world units / s at base scale
                if fw is not None and face is None and sp > 0.003: face = fw
        if ch["animal"]:  # quadrupeds: forward = tail -> head at rest
            arm = ch["arm"]; bpy.context.view_layer.update(); mw = arm.matrix_world
            hb = [b for b in arm.data.bones if b.name.lower() == "head"]
            tb = [b for b in arm.data.bones if b.name.lower() in ("body", "torso", "back")]
            if hb and tb:
                v = (mw @ hb[0].head_local) - (mw @ tb[0].head_local)
                if math.hypot(v.x, v.y) > 1e-4: face = np.array([v.x, v.y]) / math.hypot(v.x, v.y)
        if ch["kind"] == "kn": face = np.array([0.0, -1.0])  # Kenney rigs: stance measure unreliable, model faces -Y
        if face is None: face = np.array([0.0, 1.0])
        # rotate so that forward = local +Y of root
        ch["face_rot"] = math.pi / 2 - math.atan2(face[1], face[0])
    for ch in chars:
        ch["arm"].animation_data_create().action = None

    # ---- player plan
    hero = chars[0]
    cls_avail = hero["classes"]
    templates = [["run"], ["walk"], ["walk", "run"], ["run", "walk"], ["walk", "idle", "walk"], ["run", "jump", "run"],
                 ["walk", "interact", "walk"], ["run", "attack"], ["walk", "cheer"], ["run", "idle"]]
    tw = [3, 3, 2, 2, 1, 1.2, 1, 1, 0.6, 0.8]
    if cam_mode == "orbit": tw = [1, 2, 1, 1, 2, 0.5, 2, 2, 1, 2]
    while True:
        tpl = rng.choices(templates, weights=tw)[0]
        if all(c in cls_avail for c in tpl): break
    # durations
    if len(tpl) == 1: cuts = [0, NF]
    else:
        stat = [c for c in tpl if c not in ("walk", "run")]
        cuts = [0]
        for k, c in enumerate(tpl[:-1]):
            if c in ("walk", "run"): ln = rng.randint(48, 96)
            elif c == "jump": ln = 26
            else: ln = rng.randint(40, 72)
            cuts.append(min(NF - 30, cuts[-1] + ln))
        cuts.append(NF)
    plan = []
    for k, c in enumerate(tpl):
        an = rng.choice(cls_avail[c]); plan.append((c, an, cuts[k] + 1, cuts[k + 1]))
    s_char = hero["scale"]

    def speed_of(ch, cls, an):
        if cls in ("walk", "run"): return ch["speeds"].get(an, 0) * ch["scale"]
        if cls == "jump":
            rs = [ch["speeds"][a] for a in ch["classes"].get("run", []) if a in ch["speeds"]]
            return (max(rs) if rs else 0) * ch["scale"]
        return 0.0
    def speed_curve(ch, plan_):
        v = np.zeros(NF)
        for (c, an, fs, fe) in plan_: v[fs - 1:fe] = speed_of(ch, c, an)
        return smooth(v, 4)
    hv = speed_curve(hero, plan)
    travel = float(hv.sum() / FPS)
    need = travel + 45 + 30

    # ---- world
    b = Builder(rng)
    if bio["layout"] == "grid": path, linfo = build_city(b, bio, rng, need)
    elif bio["layout"] == "dungeon": path, linfo = build_dungeon(b, bio, rng, need)
    else: path, linfo = build_free(b, bio, rng, need, variant)
    b.finalize()
    R["layout"] = linfo
    lane = linfo["lane"]
    s0 = min(26.0, path.L * 0.25)
    hs = s0 + np.concatenate([[0], np.cumsum(hv[:-1] / FPS)])
    hero["s"] = hs; hero["lat"] = np.zeros(NF)

    # ---- NPC plans
    for ch in chars[1:]:
        cav = ch["classes"]
        if cam_mode == "first_person": mode = rng.choice(["stand", "stand", "walk_toward"])
        else: mode = rng.choice(["stand", "stand", "walk_toward", "walk_ahead"])
        if not any(c in cav for c in ("walk", "run")): mode = "stand"
        side = rng.choice([-1, 1]); lat = side * rng.uniform(1.5, max(1.6, lane - 0.4))
        if mode == "stand":
            opts = [c for c in ("idle", "attack", "cast", "interact", "cheer", "block", "sit") if c in cav]
            c = rng.choice(opts); an = rng.choice(cav[c])
            ch["plan"] = [(c, an, 1, NF)]
            s_at = float(np.clip(hs[min(NF - 1, rng.randint(NF // 3, NF - 1))] + rng.uniform(1, 8), 1, path.L - 1))
            if cam_mode == "first_person": s_at = float(np.clip(hs[0] + rng.uniform(6, 16), 1, path.L - 1))
            ch["s"] = np.full(NF, s_at); ch["lat"] = np.full(NF, lat)
            face_player = rng.random() < 0.7
            ch["face_mode"] = "player" if face_player else "random"; ch["face_r"] = rng.uniform(0, 2 * math.pi)
        else:
            c = rng.choice([x for x in ("walk", "run") if x in cav] if rng.random() < 0.4 else [x for x in ("walk",) if x in cav] or [x for x in ("run",) if x in cav])
            an = rng.choice(cav[c])
            ch["plan"] = [(c, an, 1, NF)]
            v = speed_of(ch, c, an)
            if mode == "walk_toward":
                st = hs[0] + rng.uniform(18, 40); ch["s"] = st - np.arange(NF) * v / FPS; ch["dirsign"] = -1
            else:
                st = hs[0] + rng.uniform(4, 10); ch["s"] = st + np.arange(NF) * v / FPS; ch["dirsign"] = 1
            ch["s"] = np.clip(ch["s"], 0.5, path.L - 0.5)
            ch["lat"] = np.full(NF, lat); ch["face_mode"] = "path"
        ch["mode"] = mode
    hero["plan"] = plan; hero["mode"] = "player"; hero["face_mode"] = "path"; hero["dirsign"] = 1

    # trajectories
    dg = bpy.context.evaluated_depsgraph_get()
    char_obj_names = set()
    for ch in chars:
        xy = np.array([path.pos(s) + path.normal(s) * l for s, l in zip(ch["s"], ch["lat"])])
        if ch["face_mode"] == "path":
            hd = [math.atan2(*(path.dir(s) * ch.get("dirsign", 1))[::-1]) for s in ch["s"]]
            # stationary hero segments keep last heading
            hd = smooth(unwrap(hd), 6)
        elif ch["face_mode"] == "player":
            hd = [math.atan2(hero_xy[1] - p[1], hero_xy[0] - p[0]) for p, hero_xy in zip(xy, chars[0]["xy"])]
            hd = smooth(unwrap(hd), 10)
        else:
            hd = np.full(NF, ch["face_r"])
        ch["xy"] = xy; ch["hd"] = np.array(hd)
        for o in ch["objs"]: char_obj_names.add(o.name)
    # ground height by raycast (ignore characters)
    for ch in chars:
        zs = []
        for x, y in ch["xy"]:
            hit = ray_ignore(sc, dg, (x, y, 60), (0, 0, -1), 120, char_obj_names)
            zs.append(60 - hit[0] if hit else 0.0)
        zs = np.array(zs)
        zs = np.clip(zs, -0.5, 2.0)
        ch["z"] = smooth(zs, 3)
    for ch in chars:
        setup_character(ch, ch["plan"], ch["xy"], ch["hd"], ch["z"], frames)
        for o in ch["meshes"]: o.pass_index = ch["id"] + 1

    # ---- camera
    cam_d = bpy.data.cameras.new("Cam"); cam_d.sensor_width = 36
    cam = bpy.data.objects.new("Cam", cam_d); sc.collection.objects.link(cam); sc.camera = cam
    cam.rotation_mode = "XYZ"
    hxy = hero["xy"]; hz = hero["z"]
    hhead = 2.0 * hero["scale"] if not hero["animal"] else 1.6
    cp = {}
    if cam_mode == "follow":
        dist = rng.uniform(4.5, 8.0); hgt = rng.uniform(2.6, 4.2); latoff = rng.uniform(-1.2, 1.2)
        cam_d.lens = rng.uniform(22, 32); look_ahead = rng.uniform(2, 5)
        hd = smooth(unwrap(hero["hd"]), 14)
        desired = [np.array([hxy[i][0] - math.cos(hd[i]) * dist - math.sin(hd[i]) * latoff * 0 + math.sin(hd[i]) * latoff,
                             hxy[i][1] - math.sin(hd[i]) * dist - math.cos(hd[i]) * latoff, hz[i] + hgt]) for i in range(NF)]
        targets = [np.array([hxy[i][0] + math.cos(hd[i]) * look_ahead, hxy[i][1] + math.sin(hd[i]) * look_ahead, hz[i] + hhead * 0.6]) for i in range(NF)]
        cp = dict(distance=round(dist, 2), height=round(hgt, 2), lateral=round(latoff, 2))
    elif cam_mode == "first_person":
        cam_d.lens = rng.uniform(18, 24)
        hd = smooth(unwrap(hero["hd"]), 8)
        eye = hhead * 0.95
        bob = 0.04 * np.sin(np.arange(NF) * 2 * math.pi / 14)
        desired = [np.array([hxy[i][0] + math.cos(hd[i]) * 0.3, hxy[i][1] + math.sin(hd[i]) * 0.3, hz[i] + eye + bob[i]]) for i in range(NF)]
        pitch = rng.uniform(-0.12, 0.0)
        targets = [desired[i] + np.array([math.cos(hd[i]) * 10, math.sin(hd[i]) * 10, 10 * pitch]) for i in range(NF)]
        for o in hero["meshes"]: o.hide_render = True
        cp = dict(eye_height=round(eye, 2))
    elif cam_mode == "top_down":
        el = math.radians(rng.uniform(40, 70)); D = rng.uniform(9, 15); az = rng.uniform(0, 2 * math.pi)
        cam_d.lens = rng.uniform(28, 45)
        cx = smooth(hxy[:, 0], 8); cy = smooth(hxy[:, 1], 8)
        desired = [np.array([cx[i] + D * math.cos(el) * math.cos(az), cy[i] + D * math.cos(el) * math.sin(az), hz[i] + D * math.sin(el)]) for i in range(NF)]
        targets = [np.array([cx[i], cy[i], hz[i] + 0.8]) for i in range(NF)]
        cp = dict(elevation_deg=round(math.degrees(el), 1), distance=round(D, 2))
    else:  # orbit
        rad = rng.uniform(5, 9); hgt = rng.uniform(1.8, 4.5); w = rng.choice([-1, 1]) * math.radians(rng.uniform(15, 40))
        a0 = rng.uniform(0, 2 * math.pi); cam_d.lens = rng.uniform(24, 35)
        cx = smooth(hxy[:, 0], 10); cy = smooth(hxy[:, 1], 10)
        desired = [np.array([cx[i] + rad * math.cos(a0 + w * i / FPS), cy[i] + rad * math.sin(a0 + w * i / FPS), hz[i] + hgt]) for i in range(NF)]
        targets = [np.array([cx[i], cy[i], hz[i] + hhead * 0.5]) for i in range(NF)]
        cp = dict(radius=round(rad, 2), height=round(hgt, 2), deg_per_s=round(math.degrees(w), 1))
    # spring-arm collision (pull camera toward the pivot if geometry blocks the view)
    if cam_mode != "first_person":
        frac = []
        for i in range(NF):
            piv = np.array([hxy[i][0], hxy[i][1], hz[i] + hhead * 0.75])
            v = desired[i] - piv; L = float(np.linalg.norm(v))
            hit = ray_ignore(sc, dg, piv, v, L + 0.6, char_obj_names)
            frac.append(1.0 if hit is None else max(0.15, (hit[0] - 0.6) / L))
        frac = np.array(frac)
        # fast in, slow out
        f2 = frac.copy()
        for i in range(1, NF): f2[i] = min(frac[i], f2[i - 1] + 0.02)
        for i in range(NF - 2, -1, -1): f2[i] = min(f2[i], f2[i + 1] + 0.06)
        f2 = smooth(f2, 2)
        for i in range(NF):
            piv = np.array([hxy[i][0], hxy[i][1], hz[i] + hhead * 0.75])
            desired[i] = piv + (desired[i] - piv) * f2[i]
        R["spring_arm_min"] = round(float(f2.min()), 3)
    # keyframe camera
    cam_keys = []
    for i, f in enumerate(frames):
        loc = Vector(desired[i]); tg = Vector(targets[i])
        rot = (tg - loc).to_track_quat("-Z", "Y").to_euler("XYZ")
        if i > 0:  # euler continuity
            prev = cam.rotation_euler.copy(); rot.make_compatible(prev)
        cam.location = loc; cam.rotation_euler = rot
        cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_euler", frame=f)
    R["camera"] = dict(mode=cam_mode, lens_mm=round(cam_d.lens, 2), **cp)

    # ---- lighting
    tp = BM.TOD[tod]
    sky_group = rng.choice(tp["sky"]); hdri = rng.choice(BM.HDRI[sky_group])
    wd = bpy.data.worlds.new("World"); sc.world = wd; wd.use_nodes = True
    nt = wd.node_tree; bg = nt.nodes["Background"]
    tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(os.path.join(ASSETS, "hdri", hdri + "_2k.hdr"))
    sun_az = rng.uniform(0, 2 * math.pi)
    mp.inputs["Rotation"].default_value = (0, 0, rng.uniform(0, 2 * math.pi))
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    tint = rng.choice(bio["sky_tint"]) if bio.get("sky_tint") else (1, 1, 1)
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0; mix.inputs[7].default_value = (*tint, 1)
    nt.links.new(env.outputs["Color"], mix.inputs[6]); nt.links.new(mix.outputs[2], bg.inputs["Color"])
    sky_strength = rng.uniform(*tp["sky_strength"]); bg.inputs["Strength"].default_value = sky_strength
    sun_d = bpy.data.lights.new("Sun", "SUN")
    sun_d.energy = rng.uniform(*tp["energy"]); sun_d.color = tp["color"]; sun_d.angle = math.radians(rng.uniform(0.5, 4))
    sun = bpy.data.objects.new("Sun", sun_d); sc.collection.objects.link(sun)
    elev = math.radians(rng.uniform(*tp["elev"]))
    sun.rotation_euler = (math.pi / 2 - elev, 0, sun_az)
    lamp_info = 0
    if tp["lamps"] and b.lamps:
        warm = rng.choice([(1.0, 0.75, 0.45), (1.0, 0.85, 0.6), (1.0, 0.65, 0.35)])
        lamp_e = rng.uniform(150, 400) if tod != "indoor" else rng.uniform(250, 600)
        lamps = b.lamps
        # keep the nearest lamps to the path (EEVEE light budget)
        lamps = sorted(lamps, key=lambda p: path.dist(np.array(p[:2]))[0])[:96]
        for k, (x, y, z) in enumerate(lamps):
            ld = bpy.data.lights.new(f"Lamp{k}", "POINT"); ld.energy = lamp_e; ld.color = warm
            ld.shadow_soft_size = 0.3; ld.use_shadow = False
            lo = bpy.data.objects.new(f"Lamp{k}", ld); lo.location = (x, y, max(z, 0.8)); sc.collection.objects.link(lo)
        lamp_info = len(lamps)
    # fog / haze via mist pass in the compositor
    fog = 0.0
    if rng.random() < (0.5 if tod in ("morning", "dusk", "night", "indoor") else 0.3):
        fog = rng.uniform(0.15, 0.6)
    fog_col = {"morning": (0.85, 0.82, 0.78), "noon": (0.8, 0.85, 0.9), "afternoon": (0.82, 0.83, 0.86),
               "dusk": (0.75, 0.5, 0.4), "night": (0.08, 0.1, 0.16), "indoor": (0.05, 0.04, 0.04)}[tod]
    fog_col = tuple(c * t for c, t in zip(fog_col, tint))
    wd.mist_settings.start = rng.uniform(5, 15); wd.mist_settings.depth = rng.uniform(40, 110); wd.mist_settings.falloff = "QUADRATIC"
    R["lighting"] = dict(hdri=hdri, sky=sky_group, sky_strength=round(sky_strength, 2), sky_tint=tint,
                         sun_elevation_deg=round(math.degrees(elev), 1), sun_azimuth_deg=round(math.degrees(sun_az), 1),
                         sun_energy=round(sun_d.energy, 2), lamps=lamp_info, fog_density=round(fog, 2))
    R["_fog"] = (fog, fog_col)

    # ---- visibility pre-check (cheap, before rendering)
    vis = {ch["id"]: [] for ch in chars}
    for i, f in enumerate(frames):
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        cpos = cam.matrix_world.translation.copy()
        for ch in chars:
            if cam_mode == "first_person" and ch["role"] == "player": vis[ch["id"]].append(False); continue
            arm = ch["arm"]; bones = ch["vis_bones"] if "vis_bones" in ch else None
            if bones is None:
                bl = [bb.name for bb in arm.data.bones]
                ch["vis_bones"] = bones = [bl[int(k)] for k in np.linspace(0, len(bl) - 1, min(8, len(bl)))]
            own = set(o.name for o in ch["objs"])
            ok = 0
            for bn in bones:
                wpt = arm.matrix_world @ arm.pose.bones[bn].head
                c = world_to_camera_view(sc, cam, wpt)
                if not (0 <= c.x <= 1 and 0 <= c.y <= 1 and c.z > 0.1): continue
                d = wpt - cpos; L = d.length
                hit = sc.ray_cast(dg, cpos, d.normalized(), distance=max(0.0, L - 0.25))
                if (not hit[0]) or (hit[4] is not None and hit[4].name in own): ok += 1
            vis[ch["id"]].append(ok >= 2)
    R["_vis"] = vis
    # camera-inside-geometry check: short rays in 6 directions
    inside = 0
    for i, f in enumerate(frames[::8]):
        sc.frame_set(f); dg = bpy.context.evaluated_depsgraph_get()
        cpos = cam.matrix_world.translation.copy(); close = 0
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            h = sc.ray_cast(dg, cpos, Vector(d), distance=0.25)
            if h[0] and (h[4] is None or h[4].name not in char_obj_names): close += 1
        if close >= 1: inside += 1
    R["_cam_close_frac"] = inside / len(frames[::8])
    return R, chars, cam, b, path


def visible_frac(R, chars, cam_mode):
    if cam_mode == "first_person":
        fr = [max(R["_vis"][ch["id"]][i] for ch in chars[1:]) for i in range(NF)] if len(chars) > 1 else [False]
    else:
        fr = R["_vis"][0]
    return float(np.mean(fr))


def describe(R, chars, b):
    bio = BM.BIOMES[R["biome"]]
    tod = R["time_of_day"]
    tods = {"morning": "in the early morning with low warm sunlight", "noon": "at midday under a high sun",
            "afternoon": "in the afternoon", "dusk": "at dusk with an orange sunset sky and lit lamps",
            "night": "at night under a dark sky, lit by lamps", "indoor": "lit by torches and candles"}[tod]
    sky = R["lighting"]["sky"]
    skyw = {"clear": "a clear sky", "cloudy": "a partly cloudy sky", "overcast": "an overcast sky", "sunset": "a sunset sky", "night": "a night sky"}[sky]
    def act_word(cls):
        return {"walk": "walking", "run": "running", "idle": "standing idle", "attack": "attacking", "cast": "casting a spell",
                "interact": "interacting with something", "cheer": "cheering", "block": "blocking with a raised guard",
                "sit": "sitting on the ground", "jump": "jumping"}.get(cls, cls)
    hero = chars[0]
    seq = [act_word(c) for c, *_ in hero["plan"]]
    hero_txt = f"a {hero['name']}" + (" " + " then ".join(seq) if seq else "")
    if hero["weapons"]: hero_txt += " carrying " + " and ".join(w.replace("_", " ").replace("1H ", "one-handed ").replace("2H ", "two-handed ").lower() for w in hero["weapons"])
    cam = R["camera_mode"]
    camtxt = {"follow": "A third-person camera follows behind", "first_person": "A first-person view from the eyes of",
              "top_down": "A high top-down camera tracks", "orbit": "A camera orbits around"}[cam]
    s1 = f"{camtxt} {hero_txt} along {bio['words'][1]}" + (" (autumn colours)" if R.get("variant") == "autumn" else "") + "."
    others = []
    for ch in chars[1:]:
        c = ch["plan"][0][0]
        if ch["mode"] == "walk_toward": others.append(f"a {ch['name']} {act_word(c)} toward the camera")
        elif ch["mode"] == "walk_ahead": others.append(f"a {ch['name']} {act_word(c)} ahead")
        else: others.append(f"a {ch['name']} {act_word(c)} by the path")
    s2 = ("Nearby, " + ", ".join(others[:-1]) + (" and " if len(others) > 1 else "") + others[-1] + ".") if others else "No other characters are present."
    s3 = f"The scene is set {tods}" + (f" with {skyw}" if tod != "indoor" else "") + \
         (", with light haze" if 0 < R["lighting"]["fog_density"] < 0.35 else ", with thick fog" if R["lighting"]["fog_density"] >= 0.35 else "") + "."
    packs = sorted({BM.PACK_CREDIT[p] for p in b.used} | {BM.PACK_CREDIT[ch["pack"]] for ch in chars})
    s4 = "Stylised low-poly game assets: " + ", ".join(packs) + "."
    if cam == "first_person": s1 = s1.replace("from the eyes of a", "from the eyes of a")
    return " ".join([s1, s2, s3, s4])


def setup_eevee(R):
    sc = bpy.context.scene
    if ENGINE == "cycles":
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "OPTIX"
        cy = sc.cycles
        cy.device = "GPU"; cy.samples = SAMPLES; cy.use_adaptive_sampling = True
        cy.use_denoising = True; cy.denoiser = "OPTIX"
        cy.max_bounces = 4; cy.diffuse_bounces = 2; cy.glossy_bounces = 2; cy.transmission_bounces = 4
        cy.use_persistent_data = True
    else:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    e = sc.eevee
    e.taa_render_samples = SAMPLES
    try:
        e.use_raytracing = True; e.ray_tracing_options.resolution_scale = "2"
        e.use_fast_gi = True
    except Exception: pass
    try: e.use_shadows = True; e.shadow_ray_count = 1; e.shadow_step_count = 4
    except Exception: pass
    sc.view_settings.view_transform = "AgX"
    try: sc.view_settings.look = "AgX - Medium High Contrast"
    except Exception: pass
    build_compositor(R, toon=ENGINE == "cycles" and "toon" in STYLES)


def _add_fog_bloom(tree, R, rl, last):
    fog, fog_col = R["_fog"]
    if fog > 0:
        mul = tree.nodes.new("CompositorNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = fog
        mx = tree.nodes.new("CompositorNodeMixRGB"); mx.inputs[2].default_value = (*fog_col, 1)
        tree.links.new(rl.outputs["Mist"], mul.inputs[0]); tree.links.new(mul.outputs[0], mx.inputs[0])
        tree.links.new(last, mx.inputs[1]); last = mx.outputs[0]
    if R["time_of_day"] in ("night", "dusk", "indoor"):
        try:
            gl = tree.nodes.new("CompositorNodeGlare"); gl.glare_type = "BLOOM"
            try: gl.quality = "MEDIUM"
            except Exception: pass
            tree.links.new(last, gl.inputs[0]); last = gl.outputs[0]
        except Exception: pass
    return last


def build_compositor(R, toon=False):
    sc = bpy.context.scene
    fog, fog_col = R["_fog"]
    sc.use_nodes = True; tree = sc.node_tree
    for n in list(tree.nodes): tree.nodes.remove(n)
    vl = bpy.context.view_layer
    if fog > 0 or toon: vl.use_pass_mist = True
    if toon:
        for p in ("use_pass_emit", "use_pass_environment", "use_pass_normal", "use_pass_z", ):
            setattr(vl, p, True)
        vl.cycles.denoising_store_passes = True
    rl = tree.nodes.new("CompositorNodeRLayers"); comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(_add_fog_bloom(tree, R, rl, rl.outputs["Image"]), comp.inputs["Image"])
    if toon: add_toon(tree, rl, R, os.path.join(OUT, "render_toon") + "/")
    sc.render.use_compositing = True


# ------------------------------------------------------------------ style: toon (cel shading + ink lines, from passes)
TOON_STEP = 1.25    # light quantised in steps of this many stops
def add_toon(tree, rl, R, out_dir):
    N, L = tree.nodes.new, tree.links.new
    def put(sock, x):
        if isinstance(x, bpy.types.NodeSocket): L(x, sock)
        elif hasattr(sock, "default_value") and not isinstance(sock.default_value, float) and isinstance(x, (int, float)):
            sock.default_value = (x, x, x, 1)
        else: sock.default_value = x
    def mix(mode, a, b, fac=1.0):
        m = N("CompositorNodeMixRGB"); m.blend_type = mode; m.use_clamp = False
        put(m.inputs[0], fac); put(m.inputs[1], a); put(m.inputs[2], b); return m.outputs[0]
    def math(op, a, b=0.0):
        m = N("CompositorNodeMath"); m.operation = op; m.use_clamp = False
        put(m.inputs[0], a); put(m.inputs[1], b); return m.outputs[0]
    def bw(x):
        m = N("CompositorNodeRGBToBW"); L(x, m.inputs[0]); return m.outputs[0]
    def sobel(x):
        m = N("CompositorNodeFilter"); m.filter_type = "SOBEL"; L(x, m.inputs["Image"]); return m.outputs[0]
    o = rl.outputs
    # shading = (denoised image - emission - background) / albedo; works for every BSDF (metallic too)
    alb = o["Denoising Albedo"]
    lit = mix("SUBTRACT", mix("SUBTRACT", o["Image"], o["Emit"]), o["Env"])
    light = mix("DIVIDE", lit, mix("ADD", alb, 0.03))
    lum = math("MAXIMUM", bw(light), 1e-4)
    q = math("POWER", 2.0, math("MULTIPLY", math("ROUND", math("DIVIDE", math("LOGARITHM", lum, 2.0), TOON_STEP)), TOON_STEP))
    lq = mix("MULTIPLY", light, math("DIVIDE", q, lum))
    base = mix("ADD", mix("ADD", mix("MULTIPLY", alb, lq), o["Emit"]), o["Env"])
    hs = N("CompositorNodeHueSat"); L(base, hs.inputs["Image"])
    try: hs.inputs["Saturation"].default_value = 1.2
    except Exception: hs.color_saturation = 1.2
    base = hs.outputs[0]
    # ink: normal creases (> ~40 deg) and depth steps (> ~12 % of the distance), faded with distance
    en = math("MULTIPLY", bw(sobel(o["Normal"])), 1.0 / 0.7)
    ez = math("MULTIPLY", bw(sobel(math("LOGARITHM", math("MAXIMUM", o["Depth"], 1e-3), 2.718281828))), 1.0 / 0.18)
    ink = math("GREATER_THAN", math("MAXIMUM", en, ez), 1.0)
    ink = math("MULTIPLY", ink, math("SUBTRACT", 1.0, math("MULTIPLY", o["Mist"], 0.85)))
    out = mix("MIX", base, mix("MULTIPLY", alb, 0.12), fac=ink)
    out = _add_fog_bloom(tree, R, rl, out)
    fo = N("CompositorNodeOutputFile"); fo.base_path = out_dir
    fo.format.file_format = "PNG"; fo.format.color_mode = "RGB"; fo.format.compression = 15
    fo.file_slots[0].path = ""
    L(out, fo.inputs[0])


# ------------------------------------------------------------------ style: realistic (Poly Haven PBR surfaces)
import pbr_sets as PS
PBR_DIR = os.path.join(ASSETS, "pbr")
_IMG = {}
def _pbr_img(sid, kind):
    key = (sid, kind)
    if key not in _IMG:
        im = bpy.data.images.load(os.path.join(PBR_DIR, sid, {"diff": "diff.jpg", "rough": "rough.jpg", "disp": "disp.png"}[kind]))
        if kind != "diff": im.colorspace_settings.name = "Non-Color"
        _IMG[key] = im
    return _IMG[key]


class _SG:  # tiny shader-graph builder
    def __init__(self, nt): self.nt = nt; self.N = nt.nodes.new; self.L = nt.links.new
    def put(self, sock, x):
        if isinstance(x, bpy.types.NodeSocket): self.L(x, sock)
        else: sock.default_value = x
    def m(self, op, a, b=0.0, c=0.0):
        n = self.N("ShaderNodeMath"); n.operation = op
        self.put(n.inputs[0], a); self.put(n.inputs[1], b); self.put(n.inputs[2], c); return n.outputs[0]
    def v(self, op, a, b=(0, 0, 0), scale=None):
        n = self.N("ShaderNodeVectorMath"); n.operation = op; self.put(n.inputs[0], a); self.put(n.inputs[1], b)
        if scale is not None: self.put(n.inputs["Scale"], scale)
        return n.outputs["Vector"]
    def mixc(self, fac, a, b):
        n = self.N("ShaderNodeMix"); n.data_type = "RGBA"; self.put(n.inputs["Factor"], fac)
        self.put(n.inputs[6], a); self.put(n.inputs[7], b); return n.outputs[2]
    def mixf(self, fac, a, b):
        n = self.N("ShaderNodeMix"); n.data_type = "FLOAT"; self.put(n.inputs["Factor"], fac)
        self.put(n.inputs[2], a); self.put(n.inputs[3], b); return n.outputs[0]
    def tex(self, sid, kind, vec, box=False):
        n = self.N("ShaderNodeTexImage"); n.image = _pbr_img(sid, kind)
        if box: n.projection = "BOX"; n.projection_blend = 0.25
        self.L(vec, n.inputs["Vector"]); return n.outputs["Color"]
    def planar_uv(self, tile):
        """World-space planar UV along the dominant axis of the shading normal (low-poly faces: one axis per face)."""
        g = self.N("ShaderNodeNewGeometry")
        sn = self.N("ShaderNodeSeparateXYZ"); self.L(g.outputs["Normal"], sn.inputs[0])
        sp = self.N("ShaderNodeSeparateXYZ"); self.L(g.outputs["Position"], sp.inputs[0])
        ax, ay, az = (self.m("ABSOLUTE", sn.outputs[i]) for i in range(3))
        mz = self.m("MULTIPLY", self.m("GREATER_THAN", az, ax), self.m("GREATER_THAN", az, ay))
        mx = self.m("MULTIPLY", self.m("SUBTRACT", 1.0, mz), self.m("GREATER_THAN", ax, ay))
        Px, Py, Pz = sp.outputs[0], sp.outputs[1], sp.outputs[2]
        u = self.m("MULTIPLY_ADD", self.m("SUBTRACT", Py, Px), mx, Px)
        v = self.m("MULTIPLY_ADD", self.m("SUBTRACT", Py, Pz), mz, Pz)
        c = self.N("ShaderNodeCombineXYZ"); self.L(u, c.inputs[0]); self.L(v, c.inputs[1])
        return self.v("SCALE", c.outputs[0], scale=1.0 / tile), mz


def _lum(g, c):
    n = g.N("ShaderNodeRGBToBW"); g.L(c, n.inputs[0]); return n.outputs[0]


def _pbr_set(g, sid, vec, C, lumC, means, k, box=False):
    """-> (albedo, roughness, height) of set sid.  albedo = mix(C x texture luminance detail, texture colour at C's
    brightness, k): k = 0 keeps the asset colour, k = 1 takes the photo's colour."""
    mu = means[sid]["mean"]; ml = max(1e-3, 0.2126 * mu[0] + 0.7152 * mu[1] + 0.0722 * mu[2])
    d = g.tex(sid, "diff", vec, box)
    a1 = g.v("SCALE", C, scale=g.m("DIVIDE", _lum(g, d), ml))
    a2 = g.v("SCALE", d, scale=g.m("DIVIDE", lumC, ml))
    return g.mixc(k, a1, a2), g.tex(sid, "rough", vec, box), g.tex(sid, "disp", vec, box)


def _finish(g, bs, alb, rough, hgt, sat=0.85, bump=0.45, dist=0.03, rlo=0.35):
    hsv = g.N("ShaderNodeHueSaturation"); hsv.inputs["Saturation"].default_value = sat; g.L(alb, hsv.inputs["Color"])
    g.L(hsv.outputs[0], bs.inputs["Base Color"])
    for lk in list(bs.inputs["Roughness"].links): g.nt.links.remove(lk)
    g.L(g.m("MULTIPLY_ADD", rough, 1.0 - rlo, rlo), bs.inputs["Roughness"])
    if not bs.inputs["Normal"].is_linked:
        bp = g.N("ShaderNodeBump"); bp.inputs["Strength"].default_value = bump; bp.inputs["Distance"].default_value = dist
        g.L(hgt, bp.inputs["Height"]); g.L(bp.outputs["Normal"], bs.inputs["Normal"])
    try: bs.inputs["Specular IOR Level"].default_value = 0.5
    except Exception: pass


def apply_pbr(R):
    """Re-surface every material of the scene with Poly Haven textures.  Asset materials (palette atlases or flat
    colours) keep their colour; a per-pixel class from that colour (grey -> stone, green -> foliage, brown/orange ->
    wood, other -> painted plaster; 'metal*' materials -> metal) picks the texture that modulates it, projected
    planar in world space along the face's dominant axis (stone: wall set on vertical faces, floor set on horizontal
    ones).  Characters (pass_index > 0) get a fabric/leather set in their rest-pose (Generated) coordinates.
    The ground plane and path ribbon take the texture's own colour."""
    means = json.load(open(os.path.join(PBR_DIR, "index.json")))
    pick = {k: SRNG.choice(v) for k, v in PS.CLASSES.items()}
    gsets, psets = PS.GROUND.get(R["biome"], PS.GROUND["_default"])
    pick["ground"] = SRNG.choice(gsets); pick["path"] = SRNG.choice(psets)
    for e in bpy.data.objects:   # vehicles: metal set (read in the shader from the instancer)
        if e.get("bgen_stem") and categorize(e["bgen_stem"], e.get("bgen_tag", "prop")) == "vehicle": e["bgen_metal"] = 1.0
    n_mat = 0
    for mat in list(bpy.data.materials):
        if not mat.use_nodes or mat.users == 0: continue
        bs = next((n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
        if bs is None: continue
        g = _SG(mat.node_tree)
        if mat.name.startswith(("ground_mat", "path_mat")):
            sid = pick["ground" if mat.name.startswith("ground") else "path"]
            uv, _ = g.planar_uv(7.0 if mat.name.startswith("ground") else 3.0)
            for lk in list(bs.inputs["Base Color"].links): g.nt.links.remove(lk)
            bio = BM.BIOMES[R["biome"]]; gc = bio["ground"] if mat.name.startswith("ground") else (bio.get("path") or bio["ground"])
            mu = means[sid]["mean"]; ml = 0.2126 * mu[0] + 0.7152 * mu[1] + 0.0722 * mu[2]
            gain = min(2.5, max(0.5, ((0.2126 * gc[0] + 0.7152 * gc[1] + 0.0722 * gc[2]) / max(ml, 1e-3)) ** 0.5))
            alb = g.v("SCALE", g.tex(sid, "diff", uv), scale=gain)
            _finish(g, bs, alb, g.tex(sid, "rough", uv), g.tex(sid, "disp", uv), sat=0.95, bump=0.6, dist=0.05, rlo=0.5)
            n_mat += 1; continue
        bc = bs.inputs["Base Color"]
        if bc.is_linked:
            C = bc.links[0].from_socket; g.nt.links.remove(bc.links[0])
        else:
            rgb = g.N("ShaderNodeRGB"); rgb.outputs[0].default_value = tuple(bc.default_value); C = rgb.outputs[0]
        uv, mz = g.planar_uv(2.6)
        lumC = _lum(g, C)
        if "metal" in mat.name.lower():
            alb, rough, hgt = _pbr_set(g, pick["metal"], uv, C, lumC, means, 0.15)
        else:
            hsv = g.N("ShaderNodeSeparateColor"); hsv.mode = "HSV"; g.L(C, hsv.inputs[0])
            h, sat_, val = hsv.outputs[0], hsv.outputs[1], hsv.outputs[2]
            stone = g.m("MAXIMUM", g.m("LESS_THAN", sat_, 0.35), g.m("LESS_THAN", val, 0.035))
            nst = g.m("SUBTRACT", 1.0, stone)
            fol = g.m("MULTIPLY", nst, g.m("MULTIPLY", g.m("GREATER_THAN", h, 0.16), g.m("LESS_THAN", h, 0.52)))
            wood = g.m("MULTIPLY", g.m("SUBTRACT", nst, fol), g.m("MULTIPLY", g.m("GREATER_THAN", h, 0.025), g.m("LESS_THAN", h, 0.16)))
            layers = [(None, pick["painted"], 0.2), (wood, pick["wood"], 0.45), (fol, pick["foliage"], 0.35),
                      (g.m("MULTIPLY", stone, g.m("SUBTRACT", 1.0, mz)), pick["stone_wall"], 0.6), (g.m("MULTIPLY", stone, mz), pick["stone_floor"], 0.6)]
            alb = rough = hgt = None
            for w, sid, k in layers:
                a, r, d = _pbr_set(g, sid, uv, C, lumC, means, k)
                if w is None: alb, rough, hgt = a, r, d
                else: alb, rough, hgt = g.mixc(w, alb, a), g.mixf(w, rough, r), g.mixf(w, hgt, d)
        at = g.N("ShaderNodeAttribute"); at.attribute_type = "INSTANCER"; at.attribute_name = "bgen_metal"
        a, r, d = _pbr_set(g, pick["metal"], uv, C, lumC, means, 0.2)
        alb, rough, hgt = g.mixc(at.outputs["Fac"], alb, a), g.mixf(at.outputs["Fac"], rough, r), g.mixf(at.outputs["Fac"], hgt, d)
        # characters: fabric in rest-pose coordinates (does not swim on moving limbs)
        oi = g.N("ShaderNodeObjectInfo"); is_ch = g.m("GREATER_THAN", oi.outputs["Object Index"], 0.5)
        tc = g.N("ShaderNodeTexCoord")
        a, r, d = _pbr_set(g, pick["fabric"], g.v("SCALE", tc.outputs["Generated"], scale=3.0), C, lumC, means, 0.1, box=True)
        alb, rough, hgt = g.mixc(is_ch, alb, a), g.mixf(is_ch, rough, r), g.mixf(is_ch, hgt, d)
        _finish(g, bs, alb, rough, hgt)
        n_mat += 1
    R["pbr"] = dict(sets=pick, materials=n_mat)


def setup_realistic(R):
    apply_pbr(R)
    sc = bpy.context.scene
    sc.cycles.samples = REAL_SAMPLES or SAMPLES
    build_compositor(R, toon=False)


def setup_workbench():
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "STUDIO"; sh.color_type = "SINGLE"; sh.single_color = (0.62, 0.62, 0.62)
    sh.show_shadows = False; sh.show_cavity = False; sh.show_specular_highlight = False
    sh.show_object_outline = False
    sc.world.color = (0.08, 0.08, 0.08)
    sc.display.render_aa = "8"
    sc.view_settings.view_transform = "Standard"; sc.view_settings.look = "None"
    sc.render.use_compositing = False
    sc.use_nodes = False


def render_to(sub):
    sc = bpy.context.scene
    d = os.path.join(OUT, sub); os.makedirs(d, exist_ok=True)
    sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.compression = 15
    sc.render.filepath = d + "/"
    t = time.time(); bpy.ops.render.render(animation=True)
    return time.time() - t


STYLE_TEXT = {
    "render": "stylised low-poly 3D game look: simple faceted models with flat palette colours, soft path-traced light and shadows",
    "render_toon": "cel-shaded anime game look: flat colour fills, light and shadow in a few hard-edged bands, dark ink outlines",
    "render_realistic": "realistic look: the same low-poly shapes covered with photographic surface textures (stone, wood, "
                        "plaster, foliage, bare ground) with fine surface relief and varied roughness under natural image-based lighting",
    "render_pixel": "retro pixel-art game look: chunky square pixels at quarter resolution, a limited colour palette, hard pixel edges",
}


def style_meta(R):
    """stream -> short style description (post.py drops streams it did not produce)."""
    out = {"render": STYLE_TEXT["render"]}
    if ENGINE == "cycles" and "toon" in STYLES: out["render_toon"] = STYLE_TEXT["render_toon"]
    if ENGINE == "cycles" and "realistic" in STYLES and "pbr" in R: out["render_realistic"] = STYLE_TEXT["render_realistic"]
    if "pixel" in STYLES: out["render_pixel"] = STYLE_TEXT["render_pixel"]
    return out


def export_meta(R, chars, cam, b, path, timings, objects=None):
    sc = bpy.context.scene
    Wr, Hr = W, H
    camf = []
    skip = ("IK", "control", "handslot", "kneeIK", "elbowIK", "heelIK")
    for ch in chars:
        arm = ch["arm"]
        bones = [bb.name for bb in arm.data.bones if not any(k in bb.name for k in skip) and bb.name != "root"]
        if any(not arm.data.bones[bn].use_deform for bn in bones):
            bones = [bn for bn in bones if arm.data.bones[bn].use_deform]
        bones = [bn for bn in bones if not any(k in bn.lower() for k in ("pole", "target", "ik"))]
        joints = [(bn, "head") for bn in bones]
        leaf = [bn for bn in bones if not [c for c in arm.data.bones[bn].children if c.name in bones]]
        joints += [(bn, "tail") for bn in leaf]
        jn = [bn if e == "head" else bn + "_end" for bn, e in joints]
        idx = {n: i for i, n in enumerate(jn)}
        edges = [[idx[arm.data.bones[bn].parent.name], idx[bn]] for bn in bones
                 if arm.data.bones[bn].parent is not None and arm.data.bones[bn].parent.name in bones]
        edges += [[idx[bn], idx[bn + "_end"]] for bn in leaf]
        ch["_joints"], ch["_jn"], ch["_edges"], ch["_fr"] = joints, jn, edges, []
    for i, f in enumerate(range(1, NF + 1)):
        sc.frame_set(f)
        mw = cam.matrix_world
        q = mw.to_quaternion()
        camf.append(dict(frame=f, location=[round(x, 4) for x in mw.translation], rotation_euler_xyz=[round(x, 5) for x in mw.to_euler("XYZ")],
                         quaternion_wxyz=[round(x, 6) for x in q]))
        for ch in chars:
            arm = ch["arm"]; j2, j3 = [], []
            for bn, e in ch["_joints"]:
                pb = arm.pose.bones[bn]
                w = arm.matrix_world @ (pb.head if e == "head" else pb.tail)
                c = world_to_camera_view(sc, cam, w)
                j3.append([round(w.x, 4), round(w.y, 4), round(w.z, 4)])
                j2.append([round(c.x * Wr, 2), round((1 - c.y) * Hr, 2), round(c.z, 3), int(0 <= c.x <= 1 and 0 <= c.y <= 1 and c.z > 0)])
            ch["_fr"].append(dict(frame=f, j2d=j2, j3d=j3, visible=bool(R["_vis"][ch["id"]][i])))
    cd = cam.data
    fov_h = 2 * math.degrees(math.atan(cd.sensor_width / (2 * cd.lens)))
    fov_v = 2 * math.degrees(math.atan(cd.sensor_width * H / W / (2 * cd.lens)))
    meta = dict(
        clip_id=f"bg{SEED:08d}", seed=SEED, attempt=R["attempt"], generator="blender_gen v6", blender=bpy.app.version_string,
        width=W, height=H, fps=FPS, frames=NF, units="KayKit units (humanoid ~2.4 tall, 1 unit ~ 0.75 m)",
        camera=dict(mode=R["camera_mode"], lens_mm=round(cd.lens, 3), sensor_width_mm=cd.sensor_width,
                    fov_h_deg=round(fov_h, 3), fov_v_deg=round(fov_v, 3), params=R["camera"],
                    convention="Blender camera: looks along -Z, up +Y; world Z up", frames=camf),
        recipe=dict(biome=R["biome"], variant=R.get("variant"), time_of_day=R["time_of_day"], layout=R["layout"],
                    lighting=R["lighting"], asset_packs={BM.PACK_CREDIT[p]: sorted(v) for p, v in b.used.items()},
                    pbr=R.get("pbr"), object_counts=b.counts, path_length=round(float(path.L), 2),
                    characters=[dict(id=ch["id"], role=ch["role"], name=ch["name"], asset=f"{BM.PACK_CREDIT[ch['pack']]}/{ch['stem']}",
                                     weapons=ch["weapons"], behaviour=ch["mode"],
                                     actions=[dict(cls=c, action=an, frame_start=fs, frame_end=fe) for c, an, fs, fe in ch["plan"]],
                                     hidden_in_render=bool(R["camera_mode"] == "first_person" and ch["role"] == "player"))
                                for ch in chars]),
        description=R["description"],
        characters=[dict(id=ch["id"], name=ch["name"], joint_names=ch["_jn"], edges=ch["_edges"],
                         format="j2d: [x_px, y_px (top-left origin), cam_depth, in_frame]; j3d: world xyz",
                         frames=ch["_fr"]) for ch in chars],
        checks=dict(player_visible_frac=R["_visfrac"], camera_near_geometry_frac=R["_cam_close_frac"]),
        timings=timings,
    )
    if objects is not None: meta["objects"] = objects
    meta["styles"] = style_meta(R)
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), separators=(",", ":"))


# ------------------------------------------------------------------ proxy streams: box (coarse primitives) + class-ID map
# Every visible object is stood in for by one primitive: static objects by the oriented bounding box of their asset
# (the placement is a z-rotation + uniform scale, so the box stays a box), characters by a capsule (humanoids) or a box
# (animals) that follows the root and the hips each frame, with no limbs; the ground is one flat plane. The box pass
# renders these with the white-box look; the class pass renders them unlit in a fixed per-category colour.
CATEGORIES = ["sky", "ground", "building", "tree_plant", "prop", "vehicle", "character", "animal"]
PALETTE = {"sky": (0, 0, 0), "ground": (128, 64, 128), "building": (70, 70, 70), "tree_plant": (107, 142, 35),
           "prop": (220, 220, 0), "vehicle": (0, 0, 142), "character": (220, 20, 60), "animal": (255, 140, 0)}
_CAT_TAG = {"road": "ground", "sidewalk": "ground", "base": "ground", "floor": "ground",
            "wall": "building", "pillar": "building", "building": "building", "vehicle": "vehicle"}
_CAT_RX = [  # first match wins
    ("building", r"stone-wall|^wall|building|crypt|tower|hangar|basemodule|cargodepot|^structure|drill_structure|lander|"
                 r"windturbine|castle|^arch$|shrine|tent|rocket"),
    ("vehicle", r"^car_|spacetruck|rover|craft_|^cart|siege-"),
    ("prop", r"^mountain|rock|stone|cliff|meteor|tree-log|^log|resource"),
    ("tree_plant", r"(^|[_-])trees?([_-]|$)|^pine|bush|plant|flower|mushroom|grass|hedge|stump"),  # not street*light*
]


def categorize(stem, tag):
    if tag in _CAT_TAG: return _CAT_TAG[tag]
    for cat, rx in _CAT_RX:
        if re.search(rx, stem): return cat
    return "prop"


def _lin(c):  # palette (sRGB 8-bit) -> Object.color; the class pass uses the Raw view transform, so stored value = output
    return (c[0] / 255, c[1] / 255, c[2] / 255, 1.0)


def _capsule_mesh(name, r, h):
    """Upright capsule of radius r and total height h centred on the origin (odd ring count: no vertex ring at z=0,
    so the band across the equator becomes the cylinder)."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=11, radius=r)
    hh = max(0.0, h / 2 - r)
    for v in bm.verts: v.co.z += hh if v.co.z > 0 else -hh
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = True
    return me


def _body_points(ch, dg, Rinv):
    pts = []
    for o in ch["body"]:
        oe = o.evaluated_get(dg); me = oe.to_mesh()
        co = np.empty(len(me.vertices) * 3, dtype=np.float64); me.vertices.foreach_get("co", co)
        oe.to_mesh_clear()
        M = np.array(Rinv @ oe.matrix_world)
        pts.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def build_proxies(R, chars, cam, b):
    sc = bpy.context.scene
    pc = bpy.data.collections.new("proxy"); sc.collection.children.link(pc)
    pc.hide_render = True   # shown only for the box / class passes
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=2.0)
    cube = bpy.data.meshes.new("proxy_cube"); bm.to_mesh(cube); bm.free()
    tracks, statics = [], []
    for ch in chars:  # character tracks first: track id == character id
        tracks.append(None)
    # ---- static objects
    pbox = {}
    for e in list(b.world_col.objects):
        if e.type == "MESH":
            if e.name.startswith("ground"):   # the ground plane: one flat plane, shared mesh
                o = bpy.data.objects.new("px_ground", e.data); o.matrix_world = e.matrix_world.copy(); pc.objects.link(o)
                o.color = _lin(PALETTE["ground"])
            continue   # the path ribbon lies on the ground plane: dropped
        if e.instance_type != "COLLECTION" or e.instance_collection is None: continue
        col = e.instance_collection
        if col.name not in pbox:
            mn = np.full(3, 1e9); mx = np.full(3, -1e9)
            for o in col.all_objects:
                if o.type != "MESH" or o.hide_render: continue
                for c in o.bound_box:
                    w = o.matrix_world @ Vector(c); mn = np.minimum(mn, w); mx = np.maximum(mx, w)
            pbox[col.name] = (mn, mx) if (mx >= mn).all() else None
        if pbox[col.name] is None: continue
        mn, mx = pbox[col.name]
        stem, tag = e.get("bgen_stem", e.name), e.get("bgen_tag", "prop")
        cat = categorize(stem, tag)
        half = np.maximum((mx - mn) / 2, 0.01); ctr = (mx + mn) / 2
        M = e.matrix_world @ Matrix.Translation(Vector(ctr)) @ Matrix.Diagonal((*half, 1.0))
        o = bpy.data.objects.new("px", cube); o.matrix_world = M; pc.objects.link(o); o.color = _lin(PALETTE[cat])
        if cat == "ground": continue   # ground tiles: drawn, but not tracked
        s = e.scale[0]; cw = e.matrix_world @ Vector(ctr)
        statics.append(dict(id=None, category=cat, category_id=CATEGORIES.index(cat), asset=stem, placement=tag,
                            box3d=[round(cw.x, 3), round(cw.y, 3), round(cw.z, 3), round(2 * half[0] * s, 3),
                                   round(2 * half[1] * s, 3), round(2 * half[2] * s, 3), round(e.rotation_euler.z % (2 * math.pi), 4)]))
    # ---- characters: size from the posed body at a few frames (root space), position from root + hips each frame
    samp = sorted(set(int(x) for x in np.linspace(1, NF, 6)))
    for ch in chars:
        arm = ch["arm"]
        ch["body"] = [o for o in ch["objs"] if o.type == "MESH" and o.parent_type != "BONE"
                      and re.sub(r"\.\d{3}$", "", o.name) not in ch["weapons"]] or [o for o in ch["objs"] if o.type == "MESH"]
        names = {bn.name.lower(): bn.name for bn in arm.data.bones}
        ch["hip_bone"] = next((names[k] for k in ("hips", "pelvis", "torso", "body") if k in names), arm.data.bones[0].name)
        ch["_bb"], ch["_hip"] = [], []
    for f in samp:
        sc.frame_set(f); dg = bpy.context.evaluated_depsgraph_get()
        for ch in chars:
            Rinv = ch["root"].matrix_world.inverted()
            P = _body_points(ch, dg, Rinv)
            if len(P): ch["_bb"].append(np.concatenate([P.min(0), P.max(0)]))
            ch["_hip"].append(np.array(Rinv @ (ch["arm"].matrix_world @ ch["arm"].pose.bones[ch["hip_bone"]].head)))
    for ch in chars:
        bb = np.median(np.array(ch["_bb"]), axis=0); mn, mx = bb[:3], bb[3:]
        ch["_hip_ref"] = np.median(np.array(ch["_hip"]), axis=0)
        ctr = (mn + mx) / 2; dx, dy, dz = mx - mn
        if ch["animal"]:
            size = np.array([dx, dy, dz]); o = bpy.data.objects.new("px_" + ch["name"], cube)
            o.scale = tuple(size / 2); kind = "box"
        else:
            r = 0.25 * (dx + dy); size = np.array([2 * r, 2 * r, max(dz, 2 * r)])
            o = bpy.data.objects.new("px_" + ch["name"], _capsule_mesh("capsule_" + ch["name"], r, size[2])); kind = "capsule"
        o.rotation_mode = "XYZ"; pc.objects.link(o)
        o.color = _lin(PALETTE["animal" if ch["animal"] else "character"])
        hidden = R["camera_mode"] == "first_person" and ch["role"] == "player"
        if hidden: o.hide_render = True
        ch["proxy"], ch["_ctr0"], ch["_psize"] = o, ctr, size
        cat = "animal" if ch["animal"] else "character"
        tracks[ch["id"]] = dict(id=ch["id"], category=cat, category_id=CATEGORIES.index(cat), asset=f"{BM.PACK_CREDIT[ch['pack']]}/{ch['stem']}",
                                placement=ch["role"], character_id=ch["id"], proxy=kind, hidden_in_render=hidden,
                                size=[round(float(x), 3) for x in size], box3d_frames=[])
    for k, t in enumerate(statics): t["id"] = len(chars) + k
    tracks += statics
    # ---- per-frame bake: character proxies follow root + hips; camera matrices kept for the 2D boxes
    cams = []
    for f in range(1, NF + 1):
        sc.frame_set(f)
        cams.append(np.array(cam.matrix_world))
        for ch in chars:
            Rw = ch["root"].matrix_world; Rinv = Rw.inverted()
            hip = np.array(Rinv @ (ch["arm"].matrix_world @ ch["arm"].pose.bones[ch["hip_bone"]].head))
            off = hip - ch["_hip_ref"]; off[2] = max(off[2], -0.25 * ch["_psize"][2])
            Mw = Rw @ Matrix.Translation(Vector(ch["_ctr0"] + off))
            o = ch["proxy"]; prev = o.rotation_euler.copy()
            o.location = Mw.translation; rot = Mw.to_euler("XYZ")
            if f > 1: rot.make_compatible(prev)
            o.rotation_euler = rot
            o.keyframe_insert("location", frame=f); o.keyframe_insert("rotation_euler", frame=f)
            c = Mw.translation
            tracks[ch["id"]]["box3d_frames"].append([round(c.x, 3), round(c.y, 3), round(c.z, 3), round(Mw.to_euler("XYZ").z % (2 * math.pi), 4)])
    return dict(col=pc, tracks=tracks, cams=cams)


_BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def _box_corners(c, size, yaw):
    """c: N x 3 centres, size: N x 3, yaw: N -> N x 8 x 3 world corners (corner k: bit0 x, bit1 y, bit2 z)."""
    s = np.array([[(k & 1) * 2 - 1, ((k >> 1) & 1) * 2 - 1, ((k >> 2) & 1) * 2 - 1] for k in range(8)], dtype=float)
    L = s[None] * (size[:, None, :] / 2)
    cy, sy = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    X = cy * L[..., 0] - sy * L[..., 1]; Y = sy * L[..., 0] + cy * L[..., 1]
    return np.stack([X, Y, L[..., 2]], -1) + c[:, None, :]


def project_tracks(px, cam):
    """Per frame: [[track_id, x0, y0, x1, y1, min_depth], ...] for every tracked box whose projection (near-plane
    clipped) overlaps the image and lies within the camera clip range.  Not occlusion-tested."""
    cd = cam.data; fpx = cd.lens / cd.sensor_width * W; cs, ce = cd.clip_start, cd.clip_end
    tr = px["tracks"]; st = [t for t in tr if "box3d" in t]
    sid = np.array([t["id"] for t in st]); sb = np.array([t["box3d"] for t in st]).reshape(-1, 7)
    scorn = _box_corners(sb[:, :3], sb[:, 3:6], sb[:, 6]) if len(st) else np.zeros((0, 8, 3))
    movers = [t for t in tr if "box3d_frames" in t and not t["hidden_in_render"]]
    out = []
    for i, Mc in enumerate(px["cams"]):
        ids, corn = sid, scorn
        if movers:
            mb = np.array([t["box3d_frames"][i] for t in movers]); ms = np.array([t["size"] for t in movers])
            corn = np.concatenate([corn, _box_corners(mb[:, :3], ms, mb[:, 3])]); ids = np.concatenate([ids, [t["id"] for t in movers]])
        if not len(ids): out.append([]); continue
        Minv = np.linalg.inv(Mc)
        P = corn @ Minv[:3, :3].T + Minv[:3, 3]; d = -P[..., 2]         # N x 8 camera space, depth along view
        pts = [np.where((d >= cs)[..., None], P, np.nan)]
        for a, bb in _BOX_EDGES:
            da, db = d[:, a] - cs, d[:, bb] - cs
            t_ = np.where(da * db < 0, da / (da - db + 1e-12), np.nan)
            pts.append((P[:, a] + t_[:, None] * (P[:, bb] - P[:, a]))[:, None, :])
        Q = np.concatenate(pts, 1); dq = np.maximum(-Q[..., 2], cs)
        u = W / 2 + fpx * Q[..., 0] / dq; v = H / 2 - fpx * Q[..., 1] / dq
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            x0, x1, y0, y1 = np.nanmin(u, 1), np.nanmax(u, 1), np.nanmin(v, 1), np.nanmax(v, 1)
            dmin = np.nanmin(np.where(d >= cs, d, np.nan), 1)
        dmin = np.where(np.isnan(dmin), cs, dmin)
        ok = (d >= cs).any(1) & (dmin < ce)
        x0, x1 = np.clip(x0, 0, W), np.clip(x1, 0, W); y0, y1 = np.clip(y0, 0, H), np.clip(y1, 0, H)
        ok &= ((x1 - x0) >= 2) & ((y1 - y0) >= 2)
        ok &= ~(np.isnan(x0) | np.isnan(y0))
        fr = [[int(ids[k]), int(math.floor(x0[k])), int(math.floor(y0[k])), int(math.ceil(x1[k])), int(math.ceil(y1[k])),
               round(float(dmin[k]), 1)] for k in np.nonzero(ok)[0]]
        out.append(fr)
    return out


def objects_meta(px, cam):
    return dict(categories=CATEGORIES, palette={k: list(v) for k, v in PALETTE.items()},
                format=dict(track="id; category; category_id (index in categories); asset; placement. Static objects: box3d = "
                                  "[cx, cy, cz, size_x, size_y, size_z, yaw_rad] (world, Z up, yaw about +Z, oriented box of the "
                                  "asset as placed). Characters/animals (id == characters[].id): size + box3d_frames[f] = "
                                  "[cx, cy, cz, yaw_rad] of the proxy capsule/box per frame.",
                            frames="frames[f] = [[track_id, x0, y0, x1, y1, min_depth], ...]: the 3D box projected to the image "
                                   "(integer pixels, outward-rounded, top-left origin, near-plane clipped, clipped to the frame), for boxes in front of the "
                                   "camera and within its clip range; not occlusion-tested. Ground tiles are drawn but not tracked.",
                            class_video="_class.mp4: flat colour per category from palette (sRGB), sky/background black, no AA."),
                tracks=px["tracks"], frames=project_tracks(px, cam))


def setup_proxy_pass(px, chars, b, kind):
    """kind: 'box' (white-box look) or 'class' (flat unlit category colour).  Hides the real geometry for the rest of the run."""
    setup_workbench()
    b.world_col.hide_render = True
    for ch in chars:
        for o in ch["objs"]:
            if o.type == "MESH": o.hide_render = True
    px["col"].hide_render = False
    sc = bpy.context.scene
    # a box can be larger than its mesh (tree canopies) and swallow the camera: cull back faces so the camera sees out
    sc.display.shading.show_backface_culling = True
    if kind == "class":
        sh = sc.display.shading
        sh.light = "FLAT"; sh.color_type = "OBJECT"
        sc.world.color = (0, 0, 0)
        sc.display.render_aa = "OFF"
        sc.view_settings.view_transform = "Raw"
        sc.render.dither_intensity = 0.0


# ------------------------------------------------------------------ run
T0 = time.time()
for attempt in range(6):
    R, chars, cam, b, path = build(SEED, attempt)
    vf = visible_frac(R, chars, R["camera_mode"])
    R["_visfrac"] = round(vf, 3)
    print(f"ATTEMPT {attempt} biome={R['biome']} cam={R['camera_mode']} visible={vf:.2f} camclose={R['_cam_close_frac']:.2f}", flush=True)
    if vf >= 0.6 and R["_cam_close_frac"] <= 0.1: break
R["description"] = describe(R, chars, b)
print("DESC", R["description"])
timings = dict(build_s=round(time.time() - T0, 2))
STILLS = opt("--stills", None, str)
def stills(sub):
    sc = bpy.context.scene
    for f in [int(x) for x in STILLS.split(",")]:
        sc.frame_set(f); sc.render.filepath = os.path.join(OUT, f"{sub}_{f:04d}.png")
        sc.render.image_settings.file_format = "PNG"; bpy.ops.render.render(write_still=True)
t = time.time(); PX = build_proxies(R, chars, cam, b); timings["proxy_build_s"] = round(time.time() - t, 2)
if "realistic" in STYLES and not os.path.exists(os.path.join(PBR_DIR, "index.json")):
    print("WARN no PBR textures under", PBR_DIR, "- skipping the realistic style", flush=True); STYLES.remove("realistic")
REAL = ENGINE == "cycles" and "realistic" in STYLES
if STILLS:
    t = time.time(); setup_eevee(R); stills("render"); timings["render_s"] = round(time.time() - t, 2)
    if REAL: t = time.time(); setup_realistic(R); stills("render_realistic"); timings["realistic_s"] = round(time.time() - t, 2)
    setup_workbench(); stills("white_box")
    setup_proxy_pass(PX, chars, b, "box"); stills("box")
    setup_proxy_pass(PX, chars, b, "class"); stills("class")
elif not NO_RENDER:
    setup_eevee(R); timings["render_s"] = round(render_to("render"), 2)
    if REAL: setup_realistic(R); timings["realistic_s"] = round(render_to("render_realistic"), 2)
    setup_workbench(); timings["white_box_s"] = round(render_to("white_box"), 2)
    setup_proxy_pass(PX, chars, b, "box"); timings["box_s"] = round(render_to("box"), 2)
    setup_proxy_pass(PX, chars, b, "class"); timings["class_s"] = round(render_to("class"), 2)
t = time.time(); OBJ = objects_meta(PX, cam); timings["tracks_s"] = round(time.time() - t, 2)
export_meta(R, chars, cam, b, path, timings, OBJ)
timings["total_s"] = round(time.time() - T0, 2)
print("GEN_DONE", json.dumps(timings), flush=True)
