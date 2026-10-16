# Alignment audit

This step decides, clip by clip, whether a finished video and its structure condition are aligned well enough to train on. It runs on CPU. Every clip gets a list of reasons, and a clip with no reasons is kept.

**Input.** `make_jobs.py CATALOG_PREFIX OUTDIR [N]` reads the clip catalog `CATALOG_PREFIX/<source>.jsonl` for `magpie` and `carla`. It writes one job list per source, `OUTDIR/<source>.jsonl`, with one row per clip: `{"id", "source", "final": url, "cond": url, "extra": {name: url}, "seconds"}`. The `extra` entries are small JSON side files; CARLA uses them for scene metadata. Passing `N` writes a seeded random sample of N clips instead of the full list.

**Metrics.** `run_audit.sh JOBS_DIR OUT_PREFIX SHARD NSHARDS` runs `audit.py` on shard k of n for each source. Start it with k = 0..n-1 on any number of machines. Each shard is uploaded to `OUT_PREFIX/<source>/shard_<k>.jsonl` every 200 clips, and a rerun resumes from the uploaded file. For each clip, `audit.py` decodes both streams to grey and records:
- the correlation of frame-to-frame motion between condition and target, at lag 0 and at the best lag in -6..6;
- the share of condition Canny edges that lie within 2 px of a target edge, compared with the same measure against a left-right mirrored target;
- repeated frames, scene cuts and unmatched jumps;
- brightness, contrast and black or flat frames;
- how much of each condition frame is covered;
- a dHash of the target at four frames, used to find near-duplicates.

**Decision.** `decide.py SRC JOBS.jsonl METRICS_PREFIX [--nofetch] [--upload DEST_URL]` applies the rules and writes `<SRC>_audit.jsonl` into `$AUDIT_METRICS_DIR`. Each row has `id`, the scalar metrics, `keep` and `reasons`. The thresholds for each source are set at the top of the file. Possible reasons are wrong format, too short, static, time offset, weak sync, frozen or empty condition, shot cut, unmatched jump, black or flat frames, too dark, overexposed, low contrast, structure mismatch and near-duplicate. Near-duplicates are looked for only among clips that pass every other rule.

**Review pages.** `pick.py SRC AUDIT.jsonl JOBS.jsonl OUTDIR` draws a seeded sample of 50 kept clips and 15 dropped clips (one per reason first) and writes an HTML page. `review.py PICK.jsonl OUTDIR NPROC` renders a three-panel video for each sampled clip: target, condition, and the condition's edges drawn over the target.

**Result on Magpie.** 12,239 of 44,266 clips were dropped. Counts by reason: shot cuts 3,269; static 2,678; structure mismatch 2,415; too dark 2,366; near-duplicates 1,372; black frames 720; unmatched jumps 602. Weak sync, frozen or empty condition and time offset each account for fewer than 200. A clip can have more than one reason.

**Environment.** `STORAGE_CLI` (default `solar-storage`), `PYTHON` (default `python3`), `AUDIT_TMP`, `AUDIT_METRICS_DIR` and `REVIEW_TMP` (local scratch directories), `NPROC_MAGPIE` / `NPROC_OTHER` / `NDL` (process and download-thread counts). Requires `ffmpeg`/`ffprobe`, `numpy` and `opencv-python`.
