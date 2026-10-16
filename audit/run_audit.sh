#!/bin/bash
# run_audit.sh JOBS_DIR OUT_PREFIX SHARD NSHARDS [SOURCE ...]
# Runs shard SHARD of NSHARDS of the audit for each source (default: magpie carla), one background driver per
# source, logs in ./logs. OUT_PREFIX is the storage prefix the metrics shards go to (OUT_PREFIX/<source>/shard_<k>.jsonl).
# Run the same command with SHARD = 0 .. NSHARDS-1 on as many machines as you like; a rerun resumes from the
# uploaded shard. Analysis processes per source: magpie 60, others 18 (override with NPROC_MAGPIE / NPROC_OTHER);
# 4 download threads per driver (NDL).
# env: PYTHON (default python3), STORAGE_CLI, AUDIT_TMP (see audit.py)
set -e
[ $# -ge 4 ] || { echo "usage: $0 JOBS_DIR OUT_PREFIX SHARD NSHARDS [SOURCE ...]"; exit 1; }
JOBS=$1; OUT=$2; S=$3; N=$4; shift 4
SOURCES=${*:-magpie carla}
PY=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p logs
for src in $SOURCES; do
  if [ "$src" = magpie ]; then P=${NPROC_MAGPIE:-60}; else P=${NPROC_OTHER:-18}; fi
  nohup $PY "$HERE/audit.py" "$JOBS/$src.jsonl" "$S" "$N" "$OUT" "$P" "${NDL:-4}" >> "logs/${src}_$S.log" 2>&1 &
  echo "started $src shard $S/$N pid $!"
done
