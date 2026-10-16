#!/bin/bash
# vlm_loop.sh NAME NSHARDS OUT_DIR : VLM review (vlm.py) of the detections NAME.s0 .. NAME.s<NSHARDS-1> while detection
# is still running. Every round it pulls the finished shard files from $SOLAR_CLEAN_WORK/det/ (and uses the growing
# local file in $CLEAN_TMP/det/ for a shard that is not finished and runs on this machine), reviews every row not yet
# reviewed (vlm.py resumes), uploads OUT_DIR/vlm_NAME.jsonl to $SOLAR_CLEAN_WORK/vlm/, and stops once every shard is
# finished and every detection row without an error has a review row.
# env: SOLAR_CLEAN_WORK (required), PYTHON (default python3), STORAGE_CLI, CLEAN_TMP, CONC (128), VLM_URL, VLM_MODEL
[ $# -ge 3 ] || { echo "usage: $0 NAME NSHARDS OUT_DIR"; exit 1; }
NAME=$1; NS=$2; O=$3
HERE=$(cd "$(dirname "$0")" && pwd); P=${PYTHON:-python3}; V=$HERE/vlm.py; STORE=${STORAGE_CLI:-solar-storage}
W=${SOLAR_CLEAN_WORK:?set SOLAR_CLEAN_WORK}; W=${W%/}
LOCAL=${CLEAN_TMP:-${TMPDIR:-/tmp}/solar_clean}/det
R=$O/det_remote; mkdir -p "$R"
while true; do
  files=""; fin=0
  for k in $(seq 0 $((NS - 1))); do
    if $STORE cp "$W/det/$NAME.s$k.jsonl" "$R/$NAME.s$k.jsonl" >/dev/null 2>&1; then
      fin=$((fin + 1)); files="$files,$R/$NAME.s$k.jsonl"
    elif [ -s "$LOCAL/$NAME.s$k.jsonl" ]; then
      files="$files,$LOCAL/$NAME.s$k.jsonl"
    fi
  done
  files=${files#,}
  [ -n "$files" ] && CONC=${CONC:-128} $P $V "$files" "$O/vlm_$NAME.jsonl"
  [ -s "$O/vlm_$NAME.jsonl" ] && $STORE cp "$O/vlm_$NAME.jsonl" "$W/vlm/vlm_$NAME.jsonl" >/dev/null 2>&1
  n=$(cat $(echo "$files" | tr , ' ') /dev/null | grep -vc '"error"')
  m=$(cat "$O/vlm_$NAME.jsonl" 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date +%T) round: det_ok=$n vlm=$m shards_finished=$fin/$NS"
  if [ "$fin" = "$NS" ] && [ "$m" -ge "$n" ]; then echo VLM_LOOP_DONE; break; fi
  sleep 120
done
