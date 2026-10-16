# Black bars and overlays

This step finds black bars and overlays on each target clip, such as watermarks, logos, text, credits and game HUDs. For each clip it outputs either a crop box that removes them or a decision to drop the clip.

**Input.** `make_jobs.py CATALOG_PREFIX OUTDIR` writes job lists `OUTDIR/jobs_<source>.jsonl` from the clip catalog. Each row is `{"id", "source", "target": url, "streams": {name: url}}`. `streams` holds aligned observations and is usually empty. When a `mask` stream is present it marks the subject.

**Detection (CPU).** Run `run_detect.sh JOBS.jsonl NAME SHARD NSHARDS` with shards 0..n-1 on any number of machines. It calls `detect.py`, which samples 24 frames per clip and looks for evidence in three ways:
- black bars: dark lines along an edge whose inner boundary is straight;
- static edges: strong edges that stay in place while the picture moves under them;
- OCR (RapidOCR): text that recurs at the same place.

It writes `NAME.s<k>.jsonl` and uploads three JPEG frames per clip to `$SOLAR_CLEAN_WORK/frames/` for the next steps.

**VLM review (GPU).** `vlm_serve.sh MODEL_DIR` serves Qwen3.5-397B-A17B-FP8 with vLLM on 8 GPUs. `vlm.py DET.jsonl OUT.jsonl` shows the model two frames per clip, at 15 % and 85 % of the clip, and asks it to list overlays with boxes and to say whether the clip has letterbox bars. `vlm_loop.sh NAME NSHARDS OUT_DIR` repeats this while detection is still running and stops once every detection row has been reviewed. `vlm2.py` is an optional second pass. It shows six frames with the OCR text boxes numbered and asks, for each box, whether the text is an overlay or part of the scene. It adds to `vlm.py` and does not replace it.

**Decision.** `decide.py SOURCE DET VLM OUT [--theta 0.70] [--v2 VLM2]` fuses the evidence. A VLM overlay is accepted when OCR or the static-edge map supports it, when it appears in the same place in both frames, or when the model is very confident and read text in it. The script then searches for the largest box with the 1344:768 canvas aspect ratio inside the bars that keeps clear of every accepted overlay plus a margin. If the clip has a subject mask, the box must also keep the subject. The clip is dropped when no such box keeps at least `theta` (70 %) of the default crop area.

**Review and counts.** `review.py OUTDIR DEC_DIR DET_DIR` writes an HTML page with sample clips from each outcome (cropped, dropped, unsure, clean), with every box drawn. `stats.py DEC_DIR` prints the counts for each source. In spot checks of 200 clips each from Magpie, CARLA and UnrealZoo, the pass found no overlays.

**Environment.** `SOLAR_CLEAN_WORK` (storage prefix for frames, detections and reviews; required), `VLM_URL` (default `http://127.0.0.1:8000`), `VLM_MODEL` (default `qwen3.5`), `STORAGE_CLI` (default `solar-storage`), `PYTHON`, `CLEAN_TMP`, `WORKERS`, `PULL`, `CONC`, and for the server `VLLM`, `PORT` and `VLLM_LOG`. Requires `ffmpeg`, `numpy`, `opencv-python`, `rapidocr_onnxruntime` and `requests`.
