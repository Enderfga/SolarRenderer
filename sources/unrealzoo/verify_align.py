"""verify_align.py DIR CLIP: frame counts of all streams, and the edge-overlap shift test.
Edges of RGB frame t are compared with edges of the whitebox / mask / normals frames t+k, k in -2..2.
With a moving camera the overlap must peak at k = 0. Also reports per-frame camera motion so a static
stretch (where the test is uninformative) is visible."""
import cv2, json, sys, os, numpy as np

def frames(path):
    c = cv2.VideoCapture(path); out = []
    while True:
        ok, im = c.read()
        if not ok: break
        out.append(im)
    return out

d, clip = sys.argv[1], sys.argv[2]
meta = json.load(open(os.path.join(d, clip + '.json')))
S = {s: frames(os.path.join(d, f'{clip}_{s}.mp4')) for s in meta['streams']}
counts = {s: len(v) for s, v in S.items()}
# first-recorder names (rgb, whitebox) and production names (render, white_box)
REF = 'render' if 'render' in S else 'rgb'
WB = 'white_box' if 'white_box' in S else 'whitebox'
def edges(im):
    g = cv2.cvtColor(cv2.resize(im, (672, 384)), cv2.COLOR_BGR2GRAY)
    return cv2.dilate(cv2.Canny(g, 60, 160), np.ones((3, 3), np.uint8)) > 0
res = {}
for other in [WB, 'mask', 'normals']:
    score = {k: [] for k in range(-2, 3)}
    for t in range(4, len(S[REF]) - 4, 8):
        er = edges(S[REF][t])
        for k in score:
            eo = edges(S[other][t + k])
            score[k].append((er & eo).sum() / max(1, eo.sum()))
    res[other] = {k: round(float(np.mean(v)), 4) for k, v in score.items()}
best = {o: max(r, key=r.get) for o, r in res.items()}
cam = np.array([f['cam_loc'] for f in meta['frames']])
step = np.linalg.norm(np.diff(cam, axis=0), axis=1)
print(json.dumps({'clip': clip, 'frame_counts': counts, 'json_frames': len(meta['frames']),
                  'edge_overlap_by_offset': res, 'best_offset': best,
                  'cam_step_cm_mean': round(float(step.mean()), 2), 'cam_static_frac': round(float((step < 0.5).mean()), 3)}))
