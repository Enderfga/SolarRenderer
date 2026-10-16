#!/bin/bash
# run_detect.sh JOBS.jsonl NAME SHARD NSHARDS : detection for shard SHARD of NSHARDS of one job list (make_jobs.py output),
# written as NAME.s<SHARD>.jsonl (see detect.py). Run SHARD = 0 .. NSHARDS-1 on as many machines as you like;
# a rerun on the same machine resumes from the local output file.
# env: SOLAR_CLEAN_WORK (required, see detect.py), PYTHON (default python3), WORKERS (80), PULL (16), STORAGE_CLI, CLEAN_TMP
set -e
[ $# -ge 4 ] || { echo "usage: $0 JOBS.jsonl NAME SHARD NSHARDS"; exit 1; }
: "${SOLAR_CLEAN_WORK:?set SOLAR_CLEAN_WORK}"
HERE=$(cd "$(dirname "$0")" && pwd)
export WORKERS=${WORKERS:-80} PULL=${PULL:-16}
nice -n 5 "${PYTHON:-python3}" "$HERE/detect.py" "$1" "$3" "$4" "$2"
