#!/usr/bin/env python
"""Generate one CARLA 0.9.16 clip with frame-aligned engine-native structure streams.

usage: carla_clip.py --seed S --slots N --port P --out DIR
  S     clip seed; the clip id is cc%09d % S
  N     number of worker slots in the seed space (num_shards * slots per shard); the map is held fixed for
        4 consecutive seeds of one slot so a server reloads its map only every 4th clip

Streams, all from one camera mount and one simulation tick (synchronous mode, dt = 1/24 s):
  rgb, semantic (CityScapes palette), instance, depth (inverse depth, near = bright), normals,
  whitebox (clay shading computed from engine depth: packaged CARLA cannot switch materials),
  skeleton (walker bones from the engine + vehicle 3D boxes, occlusion-tested against depth).
Writes <id>_<stream>.mp4 and, last, <id>.json; a clip that fails a check writes <id>.rejected.json
instead and exits 3.
"""
import argparse, json, math, multiprocessing as mp, os, queue, random, subprocess, sys, time
import numpy as np
import cv2
import carla

W, H, FPS = 1344, 768, 24
STREAMS = ["rgb", "whitebox", "semantic", "instance", "depth", "normals", "skeleton"]

MAPS = [("Town01_Opt", 1.0), ("Town02_Opt", 0.8), ("Town03_Opt", 1.5), ("Town04_Opt", 1.0), ("Town05_Opt", 1.5),
        ("Town06_Opt", 0.8), ("Town07_Opt", 1.0), ("Town10HD_Opt", 2.0), ("Town15", 1.0)]  # Town11/12/13 segfault
WEATHER = {
    "day": ["ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "SoftRainNoon", "MidRainyNoon", "HardRainNoon"],
    "sunset": ["ClearSunset", "CloudySunset", "WetSunset", "WetCloudySunset", "SoftRainSunset", "MidRainSunset", "HardRainSunset"],
    "night": ["ClearNight", "CloudyNight", "WetNight", "WetCloudyNight", "SoftRainNight", "MidRainyNight", "HardRainNight"],
    "dust": ["DustStorm"],
}
TOD_P = [("day", 0.45), ("sunset", 0.25), ("night", 0.25), ("dust", 0.05)]
CAM_P = [("chase", 0.30), ("high_chase", 0.20), ("hood", 0.20), ("ped_follow", 0.30)]
VEH_DENSITY = [("low", 20), ("mid", 50), ("high", 100)]
WALK_DENSITY = [("low", 40), ("mid", 100), ("high", 200)]

SKEL = [  # (parent, child, BGR colour)
    ("crl_hips__C", "crl_spine__C", (0, 255, 255)), ("crl_spine__C", "crl_spine01__C", (0, 255, 255)),
    ("crl_spine01__C", "crl_neck__C", (0, 200, 255)), ("crl_neck__C", "crl_Head__C", (0, 128, 255)),
    ("crl_spine01__C", "crl_shoulder__L", (255, 128, 0)), ("crl_shoulder__L", "crl_arm__L", (255, 160, 0)),
    ("crl_arm__L", "crl_foreArm__L", (255, 200, 0)), ("crl_foreArm__L", "crl_hand__L", (255, 255, 0)),
    ("crl_spine01__C", "crl_shoulder__R", (0, 128, 255)), ("crl_shoulder__R", "crl_arm__R", (0, 160, 255)),
    ("crl_arm__R", "crl_foreArm__R", (0, 200, 128)), ("crl_foreArm__R", "crl_hand__R", (0, 255, 0)),
    ("crl_hips__C", "crl_thigh__L", (255, 0, 128)), ("crl_thigh__L", "crl_leg__L", (255, 0, 200)),
    ("crl_leg__L", "crl_foot__L", (255, 0, 255)), ("crl_foot__L", "crl_toe__L", (200, 0, 255)),
    ("crl_hips__C", "crl_thigh__R", (128, 0, 255)), ("crl_thigh__R", "crl_leg__R", (64, 0, 255)),
    ("crl_leg__R", "crl_foot__R", (0, 0, 255)), ("crl_foot__R", "crl_toe__R", (0, 0, 200)),
]
JOINTS = sorted({j for a, b, _ in SKEL for j in (a, b)})
BOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def wchoice(rng, pairs):
    names, w = zip(*pairs)
    return rng.choices(names, weights=w, k=1)[0]


def job_for(seed, slots):
    q, r = divmod(seed, slots)
    map_name = wchoice(random.Random(f"map-{r}-{q // 4}"), MAPS)
    rng = random.Random(seed)
    tod = wchoice(rng, TOD_P)
    j = {"seed": seed, "map": map_name, "tod": tod, "weather": rng.choice(WEATHER[tod]),
         "cam": wchoice(rng, CAM_P), "veh_density": wchoice(rng, [(n, 1) for n, _ in VEH_DENSITY]),
         "walk_density": wchoice(rng, [(n, 1) for n, _ in WALK_DENSITY])}
    j["sun_alt"] = {"day": rng.uniform(25, 80), "sunset": rng.uniform(3, 15), "night": None, "dust": None}[tod]
    j["sun_azimuth"] = rng.uniform(0, 360)
    j["fog"] = rng.uniform(5, 35) if rng.random() < 0.15 else 0.0
    j["fov"] = rng.choice([75, 90, 90, 100])
    j["n_veh"] = dict(VEH_DENSITY)[j["veh_density"]]
    j["n_walk"] = dict(WALK_DENSITY)[j["walk_density"]]
    return j


def cam_transform(name, ext):
    if name == "chase":
        return carla.Transform(carla.Location(x=-(ext.x + 4.5), z=2 * ext.z + 1.0), carla.Rotation(pitch=-12))
    if name == "high_chase":
        return carla.Transform(carla.Location(x=-(ext.x + 7.0), z=2 * ext.z + 4.0), carla.Rotation(pitch=-22))
    if name == "hood":
        return carla.Transform(carla.Location(x=0.3 * ext.x, z=2 * ext.z + 0.25), carla.Rotation(pitch=-5))
    # ped_follow: third-person over-the-shoulder behind a walker (walker origin is at its capsule centre)
    return carla.Transform(carla.Location(x=-3.2, y=0.4, z=0.9), carla.Rotation(pitch=-12))


# ---------------------------------------------------------------- writer process (whitebox + encode)
def ffmpeg_writer(path):
    return subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
         "-threads", "4", path], stdin=subprocess.PIPE)


def whitebox(d, f):
    """Clay shading from planar depth d (metres). Camera frame: x fwd, y right, z up."""
    ds = cv2.GaussianBlur(d, (5, 5), 0)
    u = (np.arange(W, dtype=np.float32) - W / 2) / f
    v = (np.arange(H, dtype=np.float32) - H / 2) / f
    P = np.stack([ds, ds * u[None, :], -ds * v[:, None]], -1)
    du = np.zeros_like(P); dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1] = P[2:] - P[:-2]
    n = np.cross(dv, du)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    s = np.sign(-(n * P).sum(-1, keepdims=True)); s[s == 0] = 1; n *= s
    L1 = np.array([-0.45, -0.35, 0.82], np.float32); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.2, 0.6, 0.3], np.float32); L2 /= np.linalg.norm(L2)
    shade = 0.30 + 0.55 * np.clip(n @ L1, 0, 1) + 0.18 * np.clip(n @ L2, 0, 1) + 0.12 * np.clip(n[..., 2], 0, 1)
    ld = np.log(d + 1e-3)
    edge = np.clip((np.abs(cv2.Sobel(ld, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(ld, cv2.CV_32F, 0, 1, ksize=3))) * 2.0, 0, 0.35)
    g = np.clip(shade - edge, 0, 1) * 215
    fog = np.clip((d - 150) / 600, 0, 1)
    g = g * (1 - fog) + 205 * fog
    g[d > 990] = 225
    return np.repeat(g.astype(np.uint8)[:, :, None], 3, -1)


def inst_vis(a):
    key = a[:, :, 2].astype(np.uint32) * 65536 + a[:, :, 1].astype(np.uint32) * 256 + a[:, :, 0]
    h = (key * 2654435761) & 0xFFFFFF
    out = np.stack([(h & 255), (h >> 8) & 255, (h >> 16) & 255], -1).astype(np.uint8)
    out[key == 0] = 0
    return out


def writer_main(q, out, cid, f):
    os.nice(5)
    wr = {n: ffmpeg_writer(os.path.join(out, f"{cid}_{n}.mp4")) for n in STREAMS}
    while True:
        item = q.get()
        if item is None:
            break
        rgb, d, sem, inst, nrm, sk = item
        dvis = (np.clip(3.0 / np.maximum(d, 0.5), 0, 1) ** 0.5 * 255).astype(np.uint8)
        frames = {"rgb": rgb, "whitebox": whitebox(d, f), "semantic": sem, "instance": inst_vis(inst),
                  "depth": np.repeat(dvis[:, :, None], 3, -1), "normals": nrm, "skeleton": sk}
        for n in STREAMS:
            wr[n].stdin.write(np.ascontiguousarray(frames[n]).tobytes())
    rc = 0
    for w in wr.values():
        w.stdin.close(); rc |= w.wait()
    sys.exit(rc)


# ---------------------------------------------------------------- helpers
def bgr(img):
    return np.frombuffer(img.raw_data, np.uint8).reshape(H, W, 4)[:, :, :3].copy()


def depth_m(img):
    a = np.frombuffer(img.raw_data, np.uint8).reshape(H, W, 4).astype(np.float32)
    return (a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 65536.0) / (256.0 ** 3 - 1) * 1000.0


def project(pts_w, inv, f):
    p = np.c_[pts_w, np.ones(len(pts_w))] @ inv.T
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return W / 2 + f * y / x, H / 2 - f * z / x, x


def tf_dict(t):
    return {"x": t.location.x, "y": t.location.y, "z": t.location.z,
            "pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll}


def nav_loc(world, tries=20):
    """Random navmesh point; the navmesh can return None for a while after a map load."""
    for _ in range(tries):
        l = world.get_random_location_from_navigation()
        if l is not None:
            return l
    return None


def tf_r(t):
    return {k: round(v, 3) for k, v in tf_dict(t).items()}


def clear_world(client, world):
    acts = world.get_actors()
    ids = [a.id for a in acts.filter("sensor.*")] + [a.id for a in acts.filter("controller.*")] + \
          [a.id for a in acts.filter("vehicle.*")] + [a.id for a in acts.filter("walker.*")]
    for a in acts.filter("sensor.*"):
        try: a.stop()
        except Exception: pass
    if ids:
        client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=192)
    ap.add_argument("--warmup", type=int, default=48, help="ticks before the cameras exist (traffic gets moving)")
    ap.add_argument("--settle", type=int, default=16, help="ticks with cameras before recording (auto exposure)")
    a = ap.parse_args()
    job = job_for(a.seed, a.slots)
    cid = "cc%09d" % a.seed
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed % (2 ** 32))
    t_start = time.time()

    client = carla.Client("localhost", a.port); client.set_timeout(240)
    world = client.get_world()
    if not world.get_map().name.endswith(job["map"]):
        world = client.load_world(job["map"])
    else:
        s = world.get_settings(); s.synchronous_mode = False; world.apply_settings(s)
        clear_world(client, world)
    t_map = time.time()
    s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(s)
    tm = client.get_trafficmanager(a.port + 6000)
    tm.set_synchronous_mode(True); tm.set_random_device_seed(a.seed % (2 ** 31))
    tm.set_global_distance_to_leading_vehicle(2.5)
    world.set_pedestrians_seed(a.seed % (2 ** 31))

    wp = getattr(carla.WeatherParameters, job["weather"])
    if job["sun_alt"] is not None:
        wp.sun_altitude_angle = job["sun_alt"]
    wp.sun_azimuth_angle = job["sun_azimuth"]
    if job["fog"] > 0:
        wp.fog_density = max(wp.fog_density, job["fog"]); wp.fog_distance = 10.0
    world.set_weather(wp)
    night = wp.sun_altitude_angle < 0

    bl = world.get_blueprint_library()
    actors, sensors, ctrls, walkers = [], [], [], []
    wq = None; wproc = None
    try:
        spawns = world.get_map().get_spawn_points(); random.shuffle(spawns)
        vbps = list(bl.filter("vehicle.*"))
        cars = [b for b in vbps if b.has_attribute("base_type") and b.get_attribute("base_type").as_str() == "car"] or vbps
        ego = None
        for sp in spawns:
            ego = world.try_spawn_actor(random.choice(cars), sp)
            if ego: break
        actors.append(ego)
        for sp in spawns[1:job["n_veh"] + 1]:
            v = world.try_spawn_actor(random.choice(vbps), sp)
            if v: actors.append(v)
        for v in actors:
            v.set_autopilot(True, tm.get_port())
            if night:
                v.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
        tm.ignore_lights_percentage(ego, 100); tm.ignore_signs_percentage(ego, 100)
        tm.vehicle_percentage_speed_difference(ego, -15)

        wbps = bl.filter("walker.pedestrian.*")
        ego_loc = ego.get_location()
        locs = [l for l in (world.get_random_location_from_navigation() for _ in range(job["n_walk"] * 4)) if l is not None]
        locs.sort(key=lambda l: l.distance(ego_loc))
        half = job["n_walk"] // 2
        rest = locs[half:]
        for l in locs[:half] + random.sample(rest, min(len(rest), job["n_walk"] - half)):
            b = random.choice(wbps)
            if b.has_attribute("is_invincible"): b.set_attribute("is_invincible", "false")
            wk = world.try_spawn_actor(b, carla.Transform(l + carla.Location(z=1.0)))
            if wk: walkers.append(wk)
        world.tick()
        cbp = bl.find("controller.ai.walker")
        for wk in walkers:
            ctrls.append(world.spawn_actor(cbp, carla.Transform(), wk))
        world.tick()
        world.set_pedestrians_cross_factor(0.2)
        for c in ctrls:
            c.start()
            dest = nav_loc(world)
            if dest is not None:
                c.go_to_location(dest)
            c.set_max_speed(random.uniform(1.0, 2.0))
        actors += walkers

        target = ego
        if job["cam"] == "ped_follow":
            if not walkers:
                raise RuntimeError("ped_follow: no walkers on this map")
            target = min(walkers, key=lambda w: w.get_location().distance(ego_loc))
            c = ctrls[walkers.index(target)]
            cands = [l for l in (nav_loc(world) for _ in range(8)) if l is not None]
            if cands:
                c.go_to_location(max(cands, key=lambda l: l.distance(target.get_location())))
            c.set_max_speed(random.uniform(1.3, 1.8))

        for _ in range(a.warmup):  # no cameras yet: cheap
            world.tick()

        cam_tf = cam_transform(job["cam"], target.bounding_box.extent)
        qs = {}
        for k in ["rgb", "semantic_segmentation", "instance_segmentation", "depth", "normals"]:
            b = bl.find("sensor.camera." + k)
            b.set_attribute("image_size_x", str(W)); b.set_attribute("image_size_y", str(H)); b.set_attribute("fov", str(job["fov"]))
            if k == "rgb" and b.has_attribute("motion_blur_intensity"):
                b.set_attribute("motion_blur_intensity", "0.0")
            sen = world.spawn_actor(b, cam_tf, attach_to=target, attachment_type=carla.AttachmentType.Rigid)
            q = queue.Queue(); sen.listen(q.put); qs[k] = q; sensors.append(sen)
        f = W / (2 * math.tan(math.radians(job["fov"]) / 2))
        for _ in range(a.settle):
            world.tick()
        for q in qs.values():
            while not q.empty(): q.get()

        wq = mp.get_context("spawn").Queue(maxsize=24)
        wproc = mp.get_context("spawn").Process(target=writer_main, args=(wq, a.out, cid, f), daemon=True)
        wproc.start()

        meta = {"id": cid, "engine": "carla", "carla_version": client.get_server_version(), "job": job,
                "map": job["map"], "weather": job["weather"], "sun_altitude": wp.sun_altitude_angle,
                "sun_azimuth": wp.sun_azimuth_angle, "fog_density": wp.fog_density, "camera_mount": job["cam"],
                "fps": FPS, "width": W, "height": H, "fov": job["fov"],
                "K": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "camera_rel_to_target": tf_dict(cam_tf),
                "target_id": target.id, "target_type": target.type_id, "ego_vehicle_id": ego.id,
                "streams": STREAMS, "skeleton_joints": JOINTS,
                "walker_format": "joints3d[j] = world xyz of skeleton_joints[j]; joints2d[j] = [u, v, in_image, visible (not occluded per depth)]", "skeleton_edges": [[p, c] for p, c, _ in SKEL],
                "coords": "UE left-handed: x fwd, y right, z up, metres, degrees",
                "whitebox_note": "clay shading computed from engine depth (packaged CARLA has no material/view-mode override)",
                "frames": []}
        mismatch = 0; t_loop = time.time(); tick_t = 0.0; lum = []
        for i in range(a.frames):
            t0 = time.time(); fid = world.tick()
            imgs = {}
            for k, q in qs.items():
                im = q.get(timeout=60)
                while im.frame < fid:
                    im = q.get(timeout=60)
                imgs[k] = im
                if im.frame != fid: mismatch += 1
            tick_t += time.time() - t0
            snap = world.get_snapshot()
            ctf = imgs["rgb"].transform
            inv = np.array(ctf.get_inverse_matrix())
            rgb = bgr(imgs["rgb"]); lum.append(float(rgb[::8, ::8].mean()))
            d = depth_m(imgs["depth"])
            imgs["semantic_segmentation"].convert(carla.ColorConverter.CityScapesPalette)
            sem = bgr(imgs["semantic_segmentation"])
            inst = bgr(imgs["instance_segmentation"])
            nrm = bgr(imgs["normals"])
            sk = np.zeros((H, W, 3), np.uint8)
            fr = {"index": i, "frame_id": fid, "sim_time": snap.timestamp.elapsed_seconds,
                  "sensor_frame_ids": {k: v.frame for k, v in imgs.items()},
                  "camera_world": tf_r(ctf), "world_to_camera": np.round(inv, 6).tolist(), "vehicles": [], "walkers": []}
            cpos = np.array([ctf.location.x, ctf.location.y, ctf.location.z])
            for act in world.get_actors().filter("vehicle.*"):
                st = snap.find(act.id)
                if st is None: continue
                t = st.get_transform()
                if np.linalg.norm(np.array([t.location.x, t.location.y, t.location.z]) - cpos) > 120: continue
                pw = np.array([[p.x, p.y, p.z] for p in act.bounding_box.get_world_vertices(t)])
                u, v, x = project(pw, inv, f)
                vis = bool((x > 0.5).all() and ((u > 0) & (u < W) & (v > 0) & (v < H)).any())
                if not vis and np.linalg.norm(np.array([t.location.x, t.location.y, t.location.z]) - cpos) > 30: continue
                vel = st.get_velocity()
                fr["vehicles"].append({"id": act.id, "type": act.type_id, "transform": tf_r(t),
                                       "velocity": [round(vel.x, 2), round(vel.y, 2), round(vel.z, 2)], "bbox_world": pw.round(2).tolist(),
                                       "bbox_2d": np.c_[u, v].round(1).tolist() if vis else None})
                if vis and not (act.id == target.id and job["cam"] != "ped_follow"):
                    for e0, e1 in BOX_EDGES:
                        cv2.line(sk, (int(u[e0]), int(v[e0])), (int(u[e1]), int(v[e1])), (180, 180, 180), 2, cv2.LINE_AA)
            for wk in walkers:
                st = snap.find(wk.id)
                if st is None: continue
                t = st.get_transform()
                if np.linalg.norm(np.array([t.location.x, t.location.y, t.location.z]) - cpos) > 80: continue
                ru, rv, rx = project(np.array([[t.location.x, t.location.y, t.location.z]]), inv, f)
                if rx[0] < 0.3 or not (-0.25 * W < ru[0] < 1.25 * W and -0.5 * H < rv[0] < 1.5 * H): continue
                bones = {b.name: b.world.location for b in wk.get_bones().bone_transforms}
                jn = [j for j in JOINTS if j in bones]
                if len(jn) != len(JOINTS): continue
                pw = np.array([[bones[j].x, bones[j].y, bones[j].z] for j in jn])
                u, v, x = project(pw, inv, f)
                inside = (x > 0.3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                visible = np.zeros(len(jn), bool)
                for k_ in np.where(inside)[0]:
                    visible[k_] = x[k_] <= d[int(v[k_]), int(u[k_])] + 0.6
                jd = {n: (float(u[k_]), float(v[k_]), bool(inside[k_]), bool(visible[k_])) for k_, n in enumerate(jn)}
                if not inside.any(): continue
                fr["walkers"].append({"id": wk.id, "type": wk.type_id, "transform": tf_r(t), "target": wk.id == target.id,
                                      "joints3d": pw.round(2).tolist(),
                                      "joints2d": [[round(jd[n][0], 1), round(jd[n][1], 1), int(jd[n][2]), int(jd[n][3])] for n in jn]})
                if visible.sum() >= 3:
                    for p0, p1, col in SKEL:
                        if p0 in jd and p1 in jd and jd[p0][2] and jd[p1][2] and (jd[p0][3] or jd[p1][3]):
                            cv2.line(sk, (int(jd[p0][0]), int(jd[p0][1])), (int(jd[p1][0]), int(jd[p1][1])), col, 3, cv2.LINE_AA)
                    for n in jn:
                        if jd[n][3]:
                            cv2.circle(sk, (int(jd[n][0]), int(jd[n][1])), 3, (255, 255, 255), -1, cv2.LINE_AA)
            meta["frames"].append(fr)
            wq.put((rgb, d, sem, inst, nrm, sk))
            if not wproc.is_alive():
                raise RuntimeError("writer process died")
        t_rec = time.time()
        wq.put(None); wproc.join(300)
        if wproc.exitcode != 0:
            raise RuntimeError(f"writer exit {wproc.exitcode}")
        t_end = time.time()

        path = sum(math.dist([a_["camera_world"][c] for c in "xyz"], [b_["camera_world"][c] for c in "xyz"])
                   for a_, b_ in zip(meta["frames"][:-1], meta["frames"][1:]))
        reasons = []
        if mismatch: reasons.append(f"sensor frame mismatch x{mismatch}")
        if path < 3.0: reasons.append(f"camera nearly static ({path:.1f} m)")
        if np.mean(lum) < 6: reasons.append(f"rgb almost black (mean {np.mean(lum):.1f})")
        meta["alignment"] = {"sensor_frame_mismatches": mismatch, "all_streams_same_tick": mismatch == 0}
        meta["camera_path_m"] = path
        meta["rgb_mean"] = float(np.mean(lum))
        meta["n_vehicles"] = sum(1 for x in actors if x.type_id.startswith("vehicle"))
        meta["n_walkers"] = len(walkers)
        meta["timing_s"] = {"total": t_end - t_start, "map_load": t_map - t_start, "setup": t_loop - t_map,
                            "record": t_rec - t_loop, "tick_readback": tick_t, "writer_drain": t_end - t_rec,
                            "loop_fps": a.frames / (t_rec - t_loop)}
        meta["rejected"] = reasons
        name = f"{cid}.rejected.json" if reasons else f"{cid}.json"
        with open(os.path.join(a.out, name + ".tmp"), "w") as fh:
            json.dump(meta, fh, separators=(",", ":"))
        os.replace(os.path.join(a.out, name + ".tmp"), os.path.join(a.out, name))
        print(json.dumps({"id": cid, "map": job["map"], "cam": job["cam"], "weather": job["weather"],
                          "rejected": reasons, **{k: round(v, 1) for k, v in meta["timing_s"].items()}}), flush=True)
        return 3 if reasons else 0
    finally:
        if wproc is not None and wproc.is_alive():
            wproc.kill()
        for sen in sensors:
            try: sen.stop()
            except Exception: pass
        try:
            s = world.get_settings(); s.synchronous_mode = False; s.fixed_delta_seconds = None
            world.apply_settings(s); tm.set_synchronous_mode(False)
            clear_world(client, world)
        except Exception as e:
            print("cleanup failed:", e, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
