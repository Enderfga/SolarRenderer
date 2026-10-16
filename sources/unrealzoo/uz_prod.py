"""Production recorder: one UE server (already up via up.sh) -> N clips of one map, each uploaded as soon
as it is finished. The capture loop is the first recorder's (uz_record.py): game paused, one 1/24 s tick per frame,
use_fast_capture 0 so lit/mask/normal belong to the same tick as depth.

Differences to the first recorder:
  * every clip has its own RNG, seeded by (map, seed, index, attempt), so a restarted task skips the clips already
    in storage and records the remaining ones exactly as it would have;
  * stream names render / white_box / mask / depth / normals / skeleton;
  * gates: hero falls > 3 m, camera static > 40 %, lens blocked (>30 % of the frame closer than 60 cm) in > 25 % of
    frames, another character within 1 m of the camera in > 20 % of frames, mean luma < 15, hero walked < 1 m in 8 s,
    a character visible in its joints but absent from the object mask in >= 5 frames and > 10 % of its visible frames
    -> rejected (uploaded under rejected/); unlabeled mask fraction
    is recorded only; a hero that vanishes mid-clip is a failed attempt (nothing uploaded);
  * every spawned character is annotated explicitly (`vset /annotation/object/N`, see spawn_character); its mask colour
    goes into the json (`mask_colors`) and its mask pixel count into every frame (`characters.N.mask_px`);
  * upload: <dest>/clips/<clip_id>_<stream>.mp4, _mid_depth.npz, then <clip_id>.json last (the json is the done marker);
    a still goes to <dest>/stills/. Local files are deleted after upload.

usage: uz_prod.py --port 9000 --env Venice --seed 46000003 --clips 8 --out WORK/prod/out/g0 \
                  --dest <storage>/<prefix>/unrealzoo --gym WORK/unrealzoo-gym [--nf 192] [--modes tps,fps] [--results FILE]
exit 0 = at least one clip accepted (or all already done); 3 = no clip accepted; other = crash (server gone etc).
"""
import argparse, io, json, math, os, queue, random, re, socket, subprocess, sys, threading, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uc import UC
from uz_record import (W, H, FPS, CHAR, BONES, EDGES, COLORS, ffmpeg_writer, floats, wrap, basis, project,
                       whitebox, depth_vis, dec_bmp)

STORE = os.environ.get('STORAGE_CLI', 'solar-storage')
STREAMS = ['render', 'white_box', 'mask', 'depth', 'normals', 'skeleton']
LICENSE = 'UnrealZoo Apache-2.0 code; Fab marketplace assets, research use only'
GATES = {'max_z_drop_cm': 300, 'max_cam_static_frac': 0.4, 'max_lens_block_frac': 0.25, 'min_luma_mean': 15,
         'min_hero_path_m_per_8s': 1.0, 'max_npc_at_lens_frac': 0.2,
         'npc_at_lens_def': 'any joint of another character within 1 m of the camera',
         'lens_block_def': '>30% of pixels with depth < 60 cm', 'mask_unlabeled': 'recorded only, never rejects'}
GATES.update({'max_char_mask_missing_frac': 0.1, 'min_char_mask_missing_frames': 5,
              'char_mask_missing_def': 'frames with >= 2 visible torso joints of a character but no mask pixel of its '
                                       'colour (the fps hero, hidden on purpose, is not checked); a character fails if '
                                       'it is missing in >= 5 frames and in > 10 % of the frames it is visible'})
TORSO = ['pelvis', 'spine_02', 'spine_03', 'neck_01', 'Head']
OPEN_WRITERS = []
HERO_N = [0]


class HeroLost(Exception):
    """The hero actor disappeared mid-clip (fell out of the world): a failed attempt, not a server problem."""


def spawn_character(u, name, loc, app):
    """Spawn a BP_Character at loc, set its appearance, then annotate it for the object mask; returns its mask colour
    (r, g, b), or None if the spawn failed or the actor was already gone. `app` is an int or a callable drawn only after a successful spawn.
    UnrealCV 5.0.0 auto-annotates at spawn through a component cache; after any actor has been destroyed a later spawn
    can hit a stale entry and is then not drawn in object_mask at all (the wall behind it shows through) although
    `vget /object/N/color` still answers. An explicit `vset /annotation/object/N` after the spawn always draws it."""
    r = u(f'vset /objects/spawn_from_path {CHAR} {name} {loc[0]} {loc[1]} {loc[2]}')
    if r.strip() != name: return None
    u(f'vbp {name} set_app {app() if callable(app) else app}')
    a = u(f'vset /annotation/object/{name}')
    if 'can not find object' in a.lower(): return None       # fell out of the world right after the spawn
    if 'error' in a.lower(): raise RuntimeError(f'annotation of {name} failed: {a[:200]}')
    c = u(f'vget /object/{name}/color')
    m = re.findall(r'[RGB]=(\d+)', c)
    if len(m) < 3: raise RuntimeError(f'no mask colour for {name}: {c[:200]}')
    return tuple(int(v) for v in m[:3])


def mask_px(mask_bgr, rgb):
    """Number of mask pixels carrying exactly this actor colour (the mask is a lossless BMP, BGR order)."""
    if rgb is None: return 0
    return int((mask_bgr == np.array(rgb[::-1], np.uint8)).all(-1).sum())


def mask_missing(frames):
    """frames: [(visible dict, mask px)] -> (frames missing from the mask, frames with >= 2 visible torso joints)."""
    seen = [px for vis, px in frames if sum(bool(vis.get(b)) for b in TORSO) >= 2]
    return sum(px == 0 for px in seen), len(seen)


def mask_gate_fails(missing, seen):
    return missing >= GATES['min_char_mask_missing_frames'] and missing > GATES['max_char_mask_missing_frac'] * seen


def store(args, tries=3):
    for i in range(tries):
        r = subprocess.run([STORE, '--numworkers', '4'] + args, capture_output=True, text=True)
        if r.returncode == 0: return True
        print('storage command failed', args[:2], r.stderr[-300:], flush=True); time.sleep(5 * (i + 1))
    return False


def exists_remote(url):
    return subprocess.run([STORE, 'ls', url], capture_output=True).returncode == 0


class Uploader(threading.Thread):
    """Uploads finished clips in the background so recording never waits in storage."""
    def __init__(self, out, dest):
        super().__init__(daemon=True); self.out, self.dest, self.q, self.failed = out, dest, queue.Queue(), 0
        self.start()

    def put(self, clip): self.q.put(clip)

    def run(self):
        while True:
            clip = self.q.get()
            try: self.upload(clip)
            except Exception as e:
                self.failed += 1; print(f'UPLOAD ERROR {clip}: {e!r}', flush=True)
            finally: self.q.task_done()

    def upload(self, clip):
        jp = os.path.join(self.out, clip + '.json')
        meta = json.load(open(jp))
        passed = meta['quality_gate']['passed']
        dst = f"{self.dest}/clips" if passed else f"{self.dest}/rejected"
        # <clip>_a<k>_* are rejected attempts of the same index; never ship them with the accepted clip
        files = [f for f in os.listdir(self.out) if f.startswith(clip + '_') and not f.endswith('_still.jpg')
                 and not re.match(r'a\d+_', f[len(clip) + 1:]) and not f.endswith('.cmds')]
        cmds = [f'cp "{os.path.join(self.out, f)}" "{dst}/{f}"' for f in files]
        still = os.path.join(self.out, clip + '_still.jpg')
        if os.path.exists(still) and passed:
            cmds.append(f'cp "{still}" "{self.dest}/stills/{clip}.jpg"')
        cf = os.path.join(self.out, clip + '.cmds')
        open(cf, 'w').write('\n'.join(cmds) + '\n')
        ok = store(['run', cf]) and store(['cp', jp, f'{dst}/{clip}.json'])
        if ok:
            for f in files + [clip + '.json', clip + '.cmds', clip + '_still.jpg']:
                p = os.path.join(self.out, f)
                if os.path.exists(p): os.remove(p)
            print(f'uploaded {clip} -> {dst}', flush=True)
        else:
            self.failed += 1; print(f'UPLOAD FAILED {clip} (kept locally, swept on next start)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=9000)
    ap.add_argument('--env', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--clips', type=int, default=8)
    ap.add_argument('--nf', type=int, default=192)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dest', default='', help='prefix; empty = keep local (screening)')
    ap.add_argument('--modes', default='tps,fps')
    ap.add_argument('--walkers', type=int, default=3)
    ap.add_argument('--attempts', type=int, default=3, help='attempts per clip index')
    ap.add_argument('--results', default='', help='append one json line per attempt')
    ap.add_argument('--gym', default=os.path.join(os.environ.get('UZ_HOME', '/opt/uzoo'), 'unrealzoo-gym'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    up = Uploader(a.out, a.dest) if a.dest else None
    # sweep: finished but not uploaded clips from an earlier run go up; partial clips (no json) are dropped
    names_local = os.listdir(a.out)
    for f in names_local:
        if f.endswith('.json') and up: up.put(f[:-5])
    done_json = {f[:-5] for f in names_local if f.endswith('.json')}
    for f in names_local:
        if not any(f.startswith(c + '_') or f == c + '.json' or f == c + '.cmds' for c in done_json):
            os.remove(os.path.join(a.out, f))
    u = UC(a.port)
    lvl = u('vget /level/name').strip()
    assert lvl.lower() == a.env.lower(), lvl
    cfg = os.path.join(a.gym, 'gym_unrealcv/envs/setting/Track', a.env + '.json')
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
    r = u(f'vset /camera/{cid}/use_fast_capture 0')
    print('fast capture off:', r, flush=True)
    assert 'error' not in r.lower(), 'cannot turn fast capture off'
    modes = a.modes.split(',')
    accepted = skipped = 0
    for idx in range(a.clips):
        mode = modes[(idx + a.seed) % len(modes)]
        clip = f"uz_{a.env}_s{a.seed:08d}_{idx:02d}_{mode}"
        if a.dest and exists_remote(f'{a.dest}/clips/{clip}.json'):
            print(clip, 'already in storage, skip', flush=True); skipped += 1; continue
        for att in range(a.attempts):
            rng = random.Random(f'{a.env}|{a.seed}|{idx}|{att}')
            names = []; t0 = time.time(); res = None
            lost = False
            try:
                res = record_clip(a, u, rng, cid, mode, clip, idx, att, pts, names)
            except HeroLost as e:
                print(clip, 'hero lost mid-clip:', e, flush=True); res = None; lost = True
            except Exception:
                import traceback; traceback.print_exc()
                raise                                   # server/bridge trouble: let the launcher restart the server
            finally:
                for w_ in list(OPEN_WRITERS):
                    try: w_.stdin.close(); w_.wait()
                    except Exception: pass
                OPEN_WRITERS.clear()
            u('vset /action/game/resume')
            for n in names: u(f'vset /object/{n}/destroy')
            status = ('hero_lost' if lost else 'spawn_fail') if res is None else ('ok' if res['passed'] else 'reject')
            line = {'t': time.strftime('%Y-%m-%dT%H:%M:%S'), 'host': socket.gethostname(), 'port': a.port,
                    'env': a.env, 'seed': a.seed, 'idx': idx, 'attempt': att, 'mode': mode, 'clip': clip,
                    'status': status, 'secs': round(time.time() - t0, 1), 'gate': res}
            if a.results:
                with open(a.results, 'a') as fh: fh.write(json.dumps(line) + '\n')
            print(json.dumps({k: v for k, v in line.items() if k != 'host'}), flush=True)
            if res is None:
                for f in os.listdir(a.out):
                    if f.startswith(clip + '_'): os.remove(os.path.join(a.out, f))
                continue
            if up: up.put(clip if res['passed'] else f'{clip}_a{att}')
            if res['passed']:
                accepted += 1; break
    if up: up.q.join()
    print(f'TASK_DONE env={a.env} seed={a.seed} accepted={accepted} skipped={skipped} upload_failed={up.failed if up else 0}', flush=True)
    sys.exit(0 if (accepted or skipped == a.clips) else 3)


def record_clip(a, u, rng, cid, mode, clip, idx, att, pts, names):
        """Returns None if the spawn failed (nothing written), else the gate stats dict (files + json written).
        Rejected clips are written under the name <clip>_a<attempt> so they never collide with the accepted one."""
        t0 = time.time()
        HERO_N[0] += 1; k = HERO_N[0]
        p0 = pts[rng.randrange(len(pts))]
        hero = f'hero{k}'
        names.append(hero)
        app = rng.randrange(0, 12)
        colors = {hero: spawn_character(u, hero, (p0[0], p0[1], p0[2] + 60), app)}
        speed = rng.choice([150, 250, 350]) if mode == 'tps' else rng.choice([150, 250])
        u(f'vbp {hero} set_speed {speed}')
        def new_goal():
            u(f'vbp {hero} nav_random 4000 1')
        new_goal()
        for w in range(a.walkers):
            ang, rad = rng.uniform(0, 2 * math.pi), rng.uniform(300, 600)   # ring around the hero, not on top of it
            q = p0 + np.array([rad * math.cos(ang), rad * math.sin(ang), 60.0])
            nm = f'npc{k}_{w}'
            c = spawn_character(u, nm, q, lambda: rng.randrange(0, 12))
            if c is None: continue
            names.append(nm); colors[nm] = c
            u(f'vbp {nm} set_speed {rng.choice([120, 180, 300])}')
            u(f'vbp {nm} nav_random 2500 1')
        time.sleep(1.5)
        u('vset /action/game/pause')
        chk = u(f'vget /object/{hero}/location'); chk2 = chk
        if not chk.startswith('error'):
            u('vset /action/game/resume'); time.sleep(1.0); u('vset /action/game/pause')
            chk2 = u(f'vget /object/{hero}/location')
        if chk.startswith('error') or chk2.startswith('error') or abs(float(chk2.split()[2]) - float(chk.split()[2])) > 30:
            print(clip, 'hero lost/falling after spawn', chk, p0.tolist(), flush=True); return None
        fov = rng.uniform(75, 90) if mode == 'tps' else rng.uniform(80, 95)
        fpx = (W / 2) / math.tan(math.radians(fov) / 2)
        u(f'vset /camera/{cid}/size {W} {H}'); u(f'vset /camera/{cid}/fov {fov:.3f}')
        if mode == 'fps':
            u(f'vset /object/{hero}/hide')
        arm = rng.uniform(280, 420); hgt = rng.uniform(50, 110); side = rng.choice([-1, 1]) * rng.uniform(0, 55)
        orbit0 = rng.choice([0, 0, rng.uniform(-35, 35)]); orbit_rate = rng.choice([0, 0, rng.uniform(-0.25, 0.25)])
        arm_eff = arm; cyaw = None
        name = clip  # renamed to <clip>_a<att> at the end if rejected
        writers = {s: ffmpeg_writer(os.path.join(a.out, f'{name}_{s}.mp4')) for s in STREAMS}
        OPEN_WRITERS.extend(writers.values())
        frames_meta = []; prev_depth = None; still_frames = 0; last_loc = None; last_cam = None
        unl, luma, block, npc_near = [], [], [], []
        mask_seen = {}               # character -> [(visible joints, mask px)] per frame
        WARM = 10; NF = a.nf
        bone_arg = ','.join(BONES)
        for t in range(-WARM, NF):
            st = u.batch([f'vget /object/{hero}/location', f'vget /object/{hero}/rotation'])
            if st[0].startswith('error') or st[1].startswith('error'): raise HeroLost(f'frame {t}: {st[0][:80]}')
            hl = np.array(floats(st[0])); hr = floats(st[1])
            hyaw = hr[1]
            if cyaw is None: cyaw = hyaw
            if mode == 'tps':
                cyaw = cyaw + 0.07 * wrap(hyaw - cyaw)
                yaw = cyaw + orbit0 + orbit_rate * max(t, 0)
                f, r, up_ = basis(0, yaw)
                target = hl + np.array([0, 0, 55.0])
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
                f, r, up_ = basis(0, cyaw)
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
                unl.append(float((mask.max(-1) < 3).mean()))
                luma.append(float(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).mean()))
                block.append(float((dep < 60).mean()))
                chars = {}; fnear = 1e9      # nearest joint of any other character to the camera, cm
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
                    px = mask_px(mask, colors.get(n))
                    chars[n] = {'joints3d': J, 'joints2d': j2, 'visible': vis, 'mask_px': px}
                    if not (mode == 'fps' and n == hero): mask_seen.setdefault(n, []).append((vis, px))
                    if n != hero:
                        fnear = min(fnear, float(np.linalg.norm(np.array([J[b] for b in keys]) - cam, axis=1).min()))
                    if mode == 'fps' and n == hero: continue
                    col = COLORS[ci % len(COLORS)]
                    for b1, b2 in EDGES:
                        if b1 in j2 and b2 in j2 and j2[b1][2] > 1 and j2[b2][2] > 1:
                            th = 3 if (vis.get(b1) and vis.get(b2)) else 1
                            cv2.line(skel, (int(j2[b1][0]), int(j2[b1][1])), (int(j2[b2][0]), int(j2[b2][1])), col, th, cv2.LINE_AA)
                    for b in keys:
                        if j2[b][2] > 1: cv2.circle(skel, (int(j2[b][0]), int(j2[b][1])), 4 if vis[b] else 2, (255, 255, 255), -1)
                npc_near.append(fnear)
                wb = whitebox(dep, fpx)
                for s, im in [('render', rgb), ('white_box', wb), ('mask', mask), ('depth', depth_vis(dep)),
                              ('normals', nrm), ('skeleton', skel)]:
                    writers[s].stdin.write(im.tobytes())
                if t == NF // 2:
                    np.savez_compressed(os.path.join(a.out, f'{name}_mid_depth.npz'), depth_m=np.minimum(dep / 100.0, 60000).astype(np.float16))
                    cv2.imwrite(os.path.join(a.out, f'{name}_still.jpg'), np.vstack([np.hstack([rgb, wb]), np.hstack([mask, depth_vis(dep)])])[::2, ::2])
                frames_meta.append({'frame': t, 'cam_loc': cam.tolist(), 'cam_rot_pyr': [pitch, yaw_c, 0.0],
                                    'hero_loc': hl.tolist(), 'hero_rot_pyr': hr, 'characters': chars})
            if last_loc is not None and np.linalg.norm(hl - last_loc) < 0.5: still_frames += 1
            else: still_frames = 0
            last_loc = hl
            if still_frames > 12:
                u('vset /action/game/resume'); new_goal(); u('vset /action/game/pause'); still_frames = 0
            u.batch(['vset /action/game/resume', 'vset /action/game/pause'])   # advance exactly one tick
        for w_ in writers.values():
            w_.stdin.close(); w_.wait()
        OPEN_WRITERS.clear()
        hz = np.array([f['hero_loc'][2] for f in frames_meta]); cl = np.array([f['cam_loc'] for f in frames_meta])
        hxy = np.array([f['hero_loc'] for f in frames_meta])
        static = float((np.linalg.norm(np.diff(cl, axis=0), axis=1) < 0.5).mean())
        blk = float((np.array(block) > 0.3).mean())
        npc_frac = float((np.array(npc_near) < 100).mean())
        g = {'z_drop_cm': round(float(hz[0] - hz.min()), 1), 'cam_static_frac': round(static, 3),
             'lens_block_frac': round(blk, 3), 'npc_at_lens_frac': round(npc_frac, 3),
             'npc_min_dist_m': round(min(npc_near) / 100.0, 2) if min(npc_near) < 1e9 else None, 'luma_mean': round(float(np.mean(luma)), 1),
             'mask_unlabeled_frac_mean': round(float(np.mean(unl)), 4), 'mask_unlabeled_frac_max': round(float(np.max(unl)), 4),
             'hero_path_m': round(float(np.linalg.norm(np.diff(hxy, axis=0), axis=1).sum() / 100.0), 2),
             'hero_disp_m': round(float(np.linalg.norm(hxy[-1] - hxy[0]) / 100.0), 2),
             'char_mask_missing': {n: list(mask_missing(v)) for n, v in mask_seen.items()}}
        reasons = []
        if g['z_drop_cm'] > GATES['max_z_drop_cm']: reasons.append('fall')
        if static > GATES['max_cam_static_frac']: reasons.append('static_camera')
        if blk > GATES['max_lens_block_frac']: reasons.append('lens_blocked')
        if npc_frac > GATES['max_npc_at_lens_frac']: reasons.append('npc_at_lens')
        if g['luma_mean'] < GATES['min_luma_mean']: reasons.append('dark')
        if g['hero_path_m'] < GATES['min_hero_path_m_per_8s'] * NF / 192: reasons.append('hero_stuck')
        if any(mask_gate_fails(*v) for v in g['char_mask_missing'].values()):
            reasons.append('char_not_in_mask')
        g['passed'] = not reasons; g['reasons'] = reasons
        clip_id = clip if g['passed'] else f'{clip}_a{att}'
        if clip_id != name:
            for f in os.listdir(a.out):
                if f.startswith(name + '_'): os.rename(os.path.join(a.out, f), os.path.join(a.out, clip_id + f[len(name):]))
        meta = {'clip_id': clip_id, 'map': a.env, 'camera_mode': mode, 'seed': a.seed, 'clip_index': idx,
                'attempt': att, 'rng_key': f'{a.env}|{a.seed}|{idx}|{att}',
                'source': 'UnrealZoo UE5.7 Linux v3.1.0 (HF UnrealZoo/UnrealZoo_UE5), unrealzoo-gym 85f4b72 Track safe_start',
                'license': LICENSE, 'host': socket.gethostname(), 'port': a.port,
                'quality_gate': g, 'gate_thresholds': GATES,
                'fps': FPS, 'size': [W, H], 'num_frames': NF, 'fov_deg': fov,
                'K': [[fpx, 0, W / 2], [0, fpx, H / 2], [0, 0, 1]], 'tick_dt': 1 / 24, 'hero': hero, 'hero_app': app,
                'hero_speed': speed, 'arm_cm': arm, 'characters': names, 'mask_colors': {n: list(c) for n, c in colors.items() if c},
                'depth_units': 'cm, planar (UnrealCV depth npy)',
                'mid_depth_npz': 'depth_m float16, metres, frame num_frames//2',
                'streams': STREAMS,
                'stream_notes': {'render': 'final lit game frame', 'white_box': 'clay shading synthesised from engine depth',
                                 'mask': 'UnrealCV object mask, one colour per actor (characters: mask_colors, RGB); black = unlabeled (terrain/foliage on some maps); the fps hero is hidden',
                                 'depth': 'log-scaled visualisation of planar depth', 'normals': 'engine world normals (incl. normal-map detail)',
                                 'skeleton': '21 joints per character, thick = visible'},
                'recorded_secs': round(time.time() - t0, 1), 'frames': frames_meta}
        json.dump(meta, open(os.path.join(a.out, f'{clip_id}.json'), 'w'))
        print(f'{clip_id} {"done" if g["passed"] else "REJECTED " + ",".join(reasons)} {time.time() - t0:.1f}s', flush=True)
        return g


if __name__ == '__main__':
    main()
