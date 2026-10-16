# Release scripts

How the Hub dataset (`Enderfga/SolarRenderer-Data`) was packed. Run them in this order.

- `pack_train.py <k> <n> --index <train.jsonl> --enc <enc_solar_v2.jsonl> --out <release root>` — shard k of n. Re-makes every released training clip exactly as the encoder saw it, with the encoder's own functions: 24 fps, the first 158 frames, LANCZOS to 1344x768, then H.264 at CRF 18 (render) / 15 (condition). Writes `train/<source>/<clip>/{render.mp4, condition.mp4, first_frame.png}` and `index/train_<k>.jsonl`; restartable. Covers Magpie (30,597 clips), CARLA (25,317) and UnrealZoo (1,824). It loads `tools/omni_preencode.py` of the `main` branch: pass `--preencode <path>` (default `tools/omni_preencode.py`, i.e. run from a `main` checkout) or set `SOLAR_PREENCODE`; `main`'s `src/` is put on the import path, add more with `SOLAR_PYTHONPATH`.
- `pack_shards.py <stage dir> --src <release root>` — cuts those clips into WebDataset shards of 170 clips (about 2 GB): `<stage>/train/<source>/<source>-<k>.tar` holding `<key>.render.mp4`, `<key>.condition.mp4`, `<key>.first_frame.png`, `<key>.json`, plus `<stage>/train/index.jsonl`.
- `upload_hf.py <stage dir> [--repo Enderfga/SolarRenderer-Data]` — uploads the staged folder with `upload_large_folder` (resumable). Reads `HF_TOKEN` from the environment.

Environment variables: `STORAGE_CLI` (storage command, default `solar-storage`), `SOLAR_RELEASE_ROOT` (default for `--out` / `--src`), `SOLAR_PREENCODE`, `SOLAR_PYTHONPATH`, `SOLAR_SHM` (scratch for shard packing, default `/dev/shm`), `TMPDIR`, `HF_TOKEN`. `ffmpeg` must be on PATH.
