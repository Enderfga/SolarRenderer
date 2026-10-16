# CARLA

Input: the [CARLA](https://carla.org) 0.9.16 simulator with its AdditionalMaps package. Every stream comes from the
engine itself, captured from one camera mount on the same simulation tick (synchronous mode, dt = 1/24 s).

`carla_clip.py --seed S --slots N --port P --out DIR` records one clip against a running CARLA server: a map,
weather, time of day, traffic density and camera rig (chase, high chase, hood, pedestrian follow) drawn from the
seed. It writes `cc<seed>_{rgb,whitebox,semantic,instance,depth,normals,skeleton}.mp4` (1344x768, 24 fps, 192
frames = 8 s) and `cc<seed>.json` (camera, per-frame poses, checks). The white box is clay shading computed from
engine depth, since packaged CARLA cannot switch materials; the skeleton stream draws walker bones and vehicle 3D
boxes, occlusion-tested against depth. A clip that fails a check writes `<id>.rejected.json` and exits 3.

For many clips: `build_assets.sh` once (packs CARLA and the client wheels into `CGEN_ASSETS`), then on each GPU
machine `CGEN_ASSETS=<storage>/... CGEN_OUT=<storage>/.../clips CGEN_REJECT=<storage>/.../rejected bash run.sh <shard> <num_shards>`.
`run.sh` installs everything with `deploy.sh` (as root; servers run as a non-root `carla` user with headless Vulkan
through the NVIDIA EGL library), starts one server per GPU and uploads each finished clip, json last.
