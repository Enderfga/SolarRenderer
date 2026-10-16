#!/bin/bash
# convert.sh <i> <N> [part/sample ...]
#   Converts the complete WildWorld samples (all 4 files in storage: rgb.mp4, depth.mkv, state.csv, skeleton.npz) with
#   crc32(name) % N == i (stable as the list grows) into SolarRenderer clips (ww_clip.py): 8 s non-overlapping windows,
#   24 fps, 1344x768, {render,white_box,skeleton}.mp4 + json at $WW_DST, or a json with the fail reasons at
#   .../rejected/.  Explicit samples override the shard list.  Resumable: samples with .../samples/<id>.json are
#   skipped; within a sample, windows whose json is in clips/ or rejected/ are skipped.
# env: WW_SRC (storage prefix of the WildWorld copy, <part>/<sample_id>/<file>, required),
#      WW_DST (storage prefix for accepted clips, ending in /clips, required),
#      WW_WORK (local scratch, default ./wildworld_work), STORAGE_CLI (default solar-storage), PYTHON (default python3),
#      WW_LOOP=1: re-list every WW_LOOP_S (default 1800) seconds while the download is growing; stops after the first
#      pass that started once the file $WW_WORK/upload_stable exists (touch it when the download is complete).
# A sample that failed 3 times (any shard on this machine, $WW_WORK/failed.txt) is skipped; a shard gives up after 60
# failed attempts.  CPU only. Launch (or launch.sh):
#   setsid nohup bash convert.sh I N > wildworld_work/conv_I.log 2>&1 < /dev/null &
IDX=$1; NS=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
export STORAGE_CLI=${STORAGE_CLI:-solar-storage}; STORE=$STORAGE_CLI
export WW_SRC=${WW_SRC:?set WW_SRC to the storage prefix of the WildWorld copy}
export WW_DST=${WW_DST:?set WW_DST to the storage prefix for the clips}
export WW_WORK=${WW_WORK:-$PWD/wildworld_work}
DONE=${WW_DST%/*}/samples      # <sample_id>.json: all windows of that sample done
PY=${PYTHON:-python3}
mkdir -p $WW_WORK/logs
PROG=$WW_WORK/progress_${IDX}_${NS}.log
L=$WW_WORK/samples_${IDX}_${NS}.txt
attempts=0          # failed attempts this run (any rc other than 0/3); give up at 60
while :; do
  if [ $# -gt 0 ]; then printf "%s\n" "$@" > $L
  else
    $STORE ls "$WW_SRC/*" | awk '{print $NF}' | grep -E '/(rgb\.mp4|depth\.mkv|state\.csv|skeleton\.npz)$' \
      | sed 's#/[^/]*$##' | sort | uniq -c | awk '$1==4{print $2}' \
      | $PY -c "import sys, zlib; [print(l.strip()) for l in sys.stdin if zlib.crc32(l.strip().encode()) % $NS == $IDX]" > $L.all
    $STORE ls "$DONE/*.json" 2>/dev/null | awk '{print $NF}' | sed 's/\.json$//' | sort -u > $L.done
    awk -F/ 'FILENAME==ARGV[1]{d[$1]=1; next} !($NF in d)' $L.done $L.all > $L     # ARGV test: .done may be empty
  fi
  stable=0; [ -f $WW_WORK/upload_stable ] && stable=1
  n=$(grep -c . $L)
  echo "$(date -u +%FT%TZ) shard $IDX/$NS: $(grep -c . $L.all 2>/dev/null) complete samples, $n to do, upload_stable=$stable" | tee -a $PROG
  fails=0
  while read -r smp; do
    [ -z "$smp" ] && continue
    nf=$(grep -cxF -- "$smp" $WW_WORK/failed.txt 2>/dev/null); [ "${nf:-0}" -ge 3 ] && continue   # 3 failed attempts: skip it
    t0=$(date +%s); log=$WW_WORK/logs/$(echo "$smp" | tr / _).log
    timeout -k 30 12h $PY $HERE/ww_clip.py "$smp" >> "$log" 2>&1 < /dev/null; rc=$?
    dt=$(( $(date +%s) - t0 ))
    echo "$(date -u +%FT%TZ) $smp rc $rc ${dt}s $(grep -h 'sample done' "$log" | tail -1 | cut -c1-200)" >> $PROG
    if [ $rc = 0 ] || [ $rc = 3 ]; then fails=0; continue; fi
    echo "$smp" >> $WW_WORK/failed.txt
    attempts=$((attempts + 1)); fails=$((fails + 1))
    [ $attempts -ge 60 ] && { echo "$(date -u +%FT%TZ) GIVEUP after $attempts failed attempts (last rc $rc)" | tee -a $PROG; exit 2; }
    # fast failures (broken env, storage outage): back off
    [ $dt -lt 60 ] && sleep $(( fails * 20 < 600 ? fails * 20 : 600 ))
  done < $L
  echo "$(date -u +%FT%TZ) shard $IDX/$NS pass complete" | tee -a $PROG
  [ "${WW_LOOP:-0}" = 1 ] && [ $# = 0 ] || break
  if [ $stable = 1 ]; then echo "$(date -u +%FT%TZ) download was stable before this pass, so the list was final: exit" | tee -a $PROG; break; fi
  sleep ${WW_LOOP_S:-1800}
done
