"""Record engine-native paired clips from an UnrealZoo (UE5.7) server through UnrealCV.

Synchronous capture: the game stays paused; per frame we read the character poses, place the camera,
capture lit / object mask / normal / depth (+ bones) of that frozen world state, then advance exactly one
tick (resume+pause = one 1/30 s tick; `slomo 1.25` makes it 1/24 s). Every stream of frame t therefore
shows the same world state.

usage: uz_record.py --port 9000 --env Old_Town --clips 2 --out DIR [--modes tps,fps] [--seed 0]
needs bridge.py running for the port (see up.sh).
"""
import argparse, io, json, math, os, random, subprocess, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uc import UC

W, H, FPS, NF = 1344, 768, 24, 192
CHAR = '/Game/SmartLocomotion/Blueprints/BP_Character.BP_Character_C'
BONES = ['pelvis', 'spine_02', 'spine_03', 'neck_01', 'Head', 'clavicle_l', 'upperarm_l', 'lowerarm_l', 'hand_l',
         'clavicle_r', 'upperarm_r', 'lowerarm_r', 'hand_r', 'thigh_l', 'calf_l', 'foot_l', 'ball_l',
         'thigh_r', 'calf_r', 'foot_r', 'ball_r']
EDGES = [('pelvis', 'spine_02'), ('spine_02', 'spine_03'), ('spine_03', 'neck_01'), ('neck_01', 'Head'),
         ('spine_03', 'clavicle_l'), ('clavicle_l', 'upperarm_l'), ('upperarm_l', 'lowerarm_l'), ('lowerarm_l', 'hand_l'),
         ('spine_03', 'clavicle_r'), ('clavicle_r', 'upperarm_r'), ('upperarm_r', 'lowerarm_r'), ('lowerarm_r', 'hand_r'),
         ('pelvis', 'thigh_l'), ('thigh_l', 'calf_l'), ('calf_l', 'foot_l'), ('foot_l', 'ball_l'),
         ('pelvis', 'thigh_r'), ('thigh_r', 'calf_r'), ('calf_r', 'foot_r'), ('foot_r', 'ball_r')]
COLORS = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 220, 60), (255, 80, 255), (60, 255, 255)]


def ffmpeg_writer(path):
    return subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE)


def floats(s):
    return [float(v) for v in s.split()]


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def basis(pitch, yaw):
    """UE camera basis (x fwd, y right, z up; left-handed), roll = 0."""
    p, y = math.radians(pitch), math.radians(yaw)
    f = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    r = np.array([-math.sin(y), math.cos(y), 0.0])
    u = np.array([-math.sin(p) * math.cos(y), -math.sin(p) * math.sin(y), math.cos(p)])
    return f, r, u


def project(P, C, pitch, yaw, fpx):
    f, r, u = basis(pitch, yaw)
    d = np.asarray(P, float) - C
    z = d @ f
    x = W / 2 + fpx * (d @ r) / np.maximum(z, 1e-3)
    y = H / 2 - fpx * (d @ u) / np.maximum(z, 1e-3)
    return np.stack([x, y, z], -1)


def whitebox(d_cm, fpx):
    """Clay shading from engine planar depth (same recipe as the CARLA generator). Camera frame x fwd, y right, z up."""
    d = np.minimum(d_cm / 100.0, 2000.0).astype(np.float32)
    ds = cv2.GaussianBlur(d, (5, 5), 0)
    uu = (np.arange(W, dtype=np.float32) - W / 2) / fpx
    vv = (np.arange(H, dtype=np.float32) - H / 2) / fpx
    P = np.stack([ds, ds * uu[None, :], -ds * vv[:, None]], -1)
    du = np.zeros_like(P); dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1] = P[2:] - P[:-2]
    n = np.cross(dv, du)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    s = np.sign(-(n * P).sum(-1, keepdims=True)); s[s == 0] = 1; n *= s
    L1 = np.array([-0.45, -0.35, 0.82], np.float32); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.2, 0.6, 0.3], np.float32); L2 /= np.linalg.norm(L2)
    shade = 0.30 + 0.55 * np.clip(n @ L1, 0, 1) + 0.18 * np.clip(n @ L2, 0, 1) + 0.12 * np.clip(n[..., 2], 0, 1)
    lg = np.log(d + 1e-3)
    edge = np.clip((np.abs(cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(lg, cv2.CV_32F, 0, 1, ksize=3))) * 2.0, 0, 0.35)
    g = np.clip(shade - edge, 0, 1) * 215
    fog = np.clip((d - 150) / 600, 0, 1)
    g = g * (1 - fog) + 205 * fog
    g[d >= 1999] = 225
    return np.repeat(g.astype(np.uint8)[:, :, None], 3, -1)


def depth_vis(d_cm):
    m = np.clip(d_cm / 100.0, 0.1, 1000.0)
    g = 255 * (1 - np.log(1 + m) / np.log(1 + 1000.0))
    return np.repeat(g.astype(np.uint8)[:, :, None], 3, -1)


def dec_bmp(b):
    a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    return np.ascontiguousarray(a[:, :, :3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=9000)
    ap.add_argument('--env', required=True)
    ap.add_argument('--clips', type=int, default=2)
    ap.add_argument('--out', required=True)
    ap.add_argument('--modes', default='tps,fps')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--walkers', type=int, default=3)
    ap.add_argument('--prefix', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed * 7919 + a.port)
    u = UC(a.port)
    assert u('vget /level/name').strip().lower() == a.env.lower(), u('vget /level/name')
    # spawn points: the curated safe_start list of the gym Track config; fall back to engine reachable points
    cfg = os.path.join(os.environ.get('UZ_HOME', os.path.dirname(os.path.abspath(__file__))), 'unrealzoo-gym/gym_unrealcv/envs/setting/Track', a.env + '.json')
    pts = np.array(json.load(open(cfg)).get('safe_start', []), float) if os.path.exists(cfg) else np.zeros((0, 3))
    src = 'safe_start'
    if len(pts) < 2:
        pts = np.array(json.loads(u('vget /reachablepoints'))['points'], float); src = 'reachablepoints'
    print(a.env, src, len(pts), flush=True)
    u('vset /action/game/resume')
    u('vrun slomo 1.25')            # one tick = 1/30 s * 1.25 = 1/24 s
    cams = u('vget /cameras').split()
    u('vset /cameras/spawn')
    cid = len(u('vget /cameras').split()) - 1
    assert cid >= len(cams), 'camera spawn failed'
    # fast capture returns the PREVIOUS capture of lit/mask/normal (one-frame lag, depth is current); turn it off
    print('fast capture off:', u(f'vset /camera/{cid}/use_fast_capture 0'), flush=True)
    modes = a.modes.split(',')
    done = 0; tries = 0
    while done < a.clips and tries < a.clips * 4:
        k = tries; tries += 1
        mode = modes[done % len(modes)]
        clip = f"{a.prefix}{a.env}_{a.seed:02d}_{done:02d}_{mode}"
        names = []
        try:
            ok = record_clip(a, u, rng, cid, k, mode, clip, pts, names)
        except Exception as e:
            import traceback; traceback.print_exc(); ok = False
        for w_ in list(OPEN_WRITERS):
            try: w_.stdin.close(); w_.wait()
            except Exception: pass
        OPEN_WRITERS.clear()
        u('vset /action/game/resume')
        for n in names:
            u(f'vset /object/{n}/destroy')
        if ok: done += 1
        else:
            for f in os.listdir(a.out):
                if f.startswith(clip + '_') or f == clip + '.json': os.remove(os.path.join(a.out, f))
            print(f'{clip} attempt {k} failed, retrying', flush=True)


OPEN_WRITERS = []


def record_clip(a, u, rng, cid, k, mode, clip, pts, names):
        t0 = time.time()
        p0 = pts[rng.randrange(len(pts))]
        hero = f'hero{k}'
        u(f'vset /objects/spawn_from_path {CHAR} {hero} {p0[0]} {p0[1]} {p0[2] + 60}')
        names.append(hero)
        app = rng.randrange(0, 12)
        u(f'vbp {hero} set_app {app}')
        speed = rng.choice([150, 250, 350]) if mode == 'tps' else rng.choice([150, 250])
        u(f'vbp {hero} set_speed {speed}')
        def new_goal():
            u(f'vbp {hero} nav_random 4000 1')
        new_goal()
        for w in range(a.walkers):
            q = p0 + np.array([rng.uniform(-500, 500), rng.uniform(-500, 500), 60.0])
            nm = f'npc{k}_{w}'
            r = u(f'vset /objects/spawn_from_path {CHAR} {nm} {q[0]} {q[1]} {q[2]}')
            if r.strip() != nm: continue
            names.append(nm)
            u(f'vbp {nm} set_app {rng.randrange(0, 12)}')
            u(f'vbp {nm} set_speed {rng.choice([120, 180, 300])}')
            u(f'vbp {nm} nav_random 2500 1')
        # let them start walking in real time, then freeze
        time.sleep(1.5)
        u('vset /action/game/pause')
        # some safe_start points are metres above the floor: accept the spawn once the character has landed
        chk = u(f'vget /object/{hero}/location')
        if not chk.startswith('error'):
            u('vset /action/game/resume'); time.sleep(1.0); u('vset /action/game/pause')
            chk2 = u(f'vget /object/{hero}/location')
        if chk.startswith('error') or chk2.startswith('error') or abs(float(chk2.split()[2]) - float(chk.split()[2])) > 30:
            print(clip, 'hero lost/falling after spawn', chk, p0.tolist(), flush=True); return False
        fov = rng.uniform(75, 90) if mode == 'tps' else rng.uniform(80, 95)
        fpx = (W / 2) / math.tan(math.radians(fov) / 2)
        u(f'vset /camera/{cid}/size {W} {H}'); u(f'vset /camera/{cid}/fov {fov:.3f}')
        if mode == 'fps':
            u(f'vset /object/{hero}/hide')
        # camera rig params
        arm = rng.uniform(280, 420); hgt = rng.uniform(50, 110); side = rng.choice([-1, 1]) * rng.uniform(0, 55)
        orbit0 = rng.choice([0, 0, rng.uniform(-35, 35)]); orbit_rate = rng.choice([0, 0, rng.uniform(-0.25, 0.25)])
        arm_eff = arm; cyaw = None; cpitch = -8.0
        hero_color = None
        writers = {s: ffmpeg_writer(os.path.join(a.out, f'{clip}_{s}.mp4'))
                   for s in ['rgb', 'whitebox', 'mask', 'depth', 'normals', 'skeleton']}
        OPEN_WRITERS.extend(writers.values())
        frames_meta = []; prev_depth = None; still_frames = 0; last_loc = None; last_cam = None
        WARM = 10
        bone_arg = ','.join(BONES)
        for t in range(-WARM, NF):
            st = u.batch([f'vget /object/{hero}/location', f'vget /object/{hero}/rotation'])
            hl = np.array(floats(st[0])); hr = floats(st[1]); cur_loc = hl
            hyaw = hr[1]
            if cyaw is None: cyaw = hyaw
            if mode == 'tps':
                cyaw = cyaw + 0.07 * wrap(hyaw - cyaw)
                yaw = cyaw + orbit0 + orbit_rate * max(t, 0)
                f, r, up = basis(0, yaw)
                target = hl + np.array([0, 0, 55.0])
                # spring arm: shorten when the previous frame shows an occluder between camera and character
                if prev_depth is not None and last_cam is not None:
                    pj = project(target, last_cam, last_pitch, last_yaw, fpx)
                    x, y = int(pj[0]), int(pj[1])
                    if 0 <= x < W and 0 <= y < H:
                        dd = prev_depth[max(0, y - 6):y + 7, max(0, x - 6):x + 7].min()
                        if dd < pj[2] - 60: arm_eff = max(110.0, arm_eff * 0.8)
                        else: arm_eff = arm_eff + 0.05 * (arm - arm_eff)
                cam = target - f * arm_eff + r * side + np.array([0, 0, hgt])
                look = target + f * 120.0
                dv = look - cam
                pitch = math.degrees(math.atan2(dv[2], np.linalg.norm(dv[:2])))
                yaw_c = math.degrees(math.atan2(dv[1], dv[0]))
            else:
                hb = json.loads(u(f'vget /object/{hero}/bones Head world'))
                head = np.array([hb[0]['transform'][c] for c in 'XYZ'])
                cyaw = cyaw + 0.25 * wrap(hyaw - cyaw)
                f, r, up = basis(0, cyaw)
                cam = head + f * 18.0 + np.array([0, 0, 6.0])
                yaw_c = cyaw; pitch = -6.0 + 2.0 * math.sin(max(t, 0) / 24.0 * 1.3)
            last_cam, last_pitch, last_yaw = cam, pitch, yaw_c
            cmds = [f'vset /camera/{cid}/location {cam[0]:.3f} {cam[1]:.3f} {cam[2]:.3f}',
                    f'vset /camera/{cid}/rotation {pitch:.4f} {yaw_c:.4f} 0']
            if t < 0:
                cmds += [f'vget /camera/{cid}/lit bmp', f'vget /camera/{cid}/depth npy']
                res = u.batch(cmds)
                prev_depth = np.load(io.BytesIO(res[3]))
            else:
                cmds += [f'vget /camera/{cid}/lit bmp', f'vget /camera/{cid}/object_mask bmp',
                         f'vget /camera/{cid}/normal bmp', f'vget /camera/{cid}/depth npy']
                cmds += [f'vget /object/{n}/bones {bone_arg} world' for n in names]
                res = u.batch(cmds)
                rgb, mask, nrm = dec_bmp(res[2]), dec_bmp(res[3]), dec_bmp(res[4])
                dep = np.load(io.BytesIO(res[5])).astype(np.float32); prev_depth = dep
                chars = {}
                skel = np.zeros((H, W, 3), np.uint8)
                for ci, (n, rb) in enumerate(zip(names, res[6:])):
                    try: bl = json.loads(rb)
                    except Exception: continue
                    J = {b['bone_name']: [b['transform'][c] for c in 'XYZ'] for b in bl}
                    if not J: continue
                    keys = list(J)
                    pj = project(np.array([J[b] for b in keys]), cam, pitch, yaw_c, fpx)
                    vis = {}
                    for b, (x, y, z) in zip(keys, pj):
                        xi, yi = int(round(x)), int(round(y))
                        inside = z > 1 and 0 <= xi < W and 0 <= yi < H
                        vis[b] = bool(inside and dep[yi, xi] > z - 25)
                    j2 = {b: [float(x), float(y), float(z)] for b, (x, y, z) in zip(keys, pj)}
                    chars[n] = {'joints3d': J, 'joints2d': j2, 'visible': vis}
                    if mode == 'fps' and n == hero: continue
                    col = COLORS[ci % len(COLORS)]
                    for b1, b2 in EDGES:
                        if b1 in j2 and b2 in j2 and j2[b1][2] > 1 and j2[b2][2] > 1:
                            th = 3 if (vis.get(b1) and vis.get(b2)) else 1
                            cv2.line(skel, (int(j2[b1][0]), int(j2[b1][1])), (int(j2[b2][0]), int(j2[b2][1])), col, th, cv2.LINE_AA)
                    for b in keys:
                        if j2[b][2] > 1: cv2.circle(skel, (int(j2[b][0]), int(j2[b][1])), 4 if vis[b] else 2, (255, 255, 255), -1)
                wb = whitebox(dep, fpx)
                for s, im in [('rgb', rgb), ('whitebox', wb), ('mask', mask), ('depth', depth_vis(dep)), ('normals', nrm), ('skeleton', skel)]:
                    writers[s].stdin.write(im.tobytes())
                if t == NF // 2:
                    np.savez_compressed(os.path.join(a.out, f'{clip}_mid_depth.npz'), depth_m=np.minimum(dep / 100.0, 60000).astype(np.float16))
                    cv2.imwrite(os.path.join(a.out, f'{clip}_still.jpg'), np.vstack([np.hstack([rgb, wb]), np.hstack([mask, depth_vis(dep)])])[::2, ::2])
                frames_meta.append({'frame': t, 'cam_loc': cam.tolist(), 'cam_rot_pyr': [pitch, yaw_c, 0.0],
                                    'hero_loc': hl.tolist(), 'hero_rot_pyr': hr, 'characters': chars})
            # navigation upkeep: new goal if the hero stalls
            if last_loc is not None and np.linalg.norm(hl - last_loc) < 0.5: still_frames += 1
            else: still_frames = 0
            last_loc = hl
            if still_frames > 12:
                u('vset /action/game/resume'); new_goal(); u('vset /action/game/pause'); still_frames = 0
            u.batch(['vset /action/game/resume', 'vset /action/game/pause'])   # advance exactly one tick
        for w_ in writers.values():
            w_.stdin.close(); w_.wait()
        OPEN_WRITERS.clear()
        # quality gate: the character must stay on the ground and the camera must actually travel
        hz = np.array([f['hero_loc'][2] for f in frames_meta]); cl = np.array([f['cam_loc'] for f in frames_meta])
        static = float((np.linalg.norm(np.diff(cl, axis=0), axis=1) < 0.5).mean())
        if hz[0] - hz.min() > 300 or static > 0.4:
            print(f'{clip} rejected: z drop {hz[0] - hz.min():.0f} cm, static frac {static:.2f}', flush=True); return False
        meta = {'env': a.env, 'clip': clip, 'mode': mode, 'fps': FPS, 'size': [W, H], 'fov_deg': fov,
                'K': [[fpx, 0, W / 2], [0, fpx, H / 2], [0, 0, 1]], 'tick_dt': 1 / 24, 'hero': hero, 'hero_app': app,
                'hero_speed': speed, 'arm_cm': arm, 'characters': names, 'depth_units': 'cm, planar (UnrealCV depth npy)',
                'streams': ['rgb', 'whitebox', 'mask', 'depth', 'normals', 'skeleton'],
                'whitebox': 'clay shading synthesised from engine depth', 'frames': frames_meta}
        json.dump(meta, open(os.path.join(a.out, f'{clip}.json'), 'w'))
        print(f'{clip} done {time.time() - t0:.1f}s', flush=True)
        return True


if __name__ == '__main__':
    main()
