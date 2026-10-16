#!/bin/bash
# UnrealZoo production on one machine: one UE server + one recorder per GPU, cycling through the kept maps.
#
#   prod.sh start      start the 8 GPU loops (skips GPUs whose loop is already running)
#   prod.sh drain      let every GPU finish its current task, then stop
#   prod.sh stop       stop now (servers, bridges, recorders, loops); unfinished clips are re-recorded later
#   prod.sh status     per-GPU state and clip counts
#   prod.sh run-gpu G  (internal) the loop for GPU G
#
# env: UZ_OUT (storage prefix for the output, required for start), UZ_HOME (package from deploy.sh, default /opt/uzoo),
#      UZ_MACHINE (index of this machine among the producers, default 0), UZ_MAPS (maps file, default maps_prod.txt
#      next to this script), UZ_TASKS / UZ_QUEUE (task-list mode, below), STORAGE_CLI (default solar-storage).
#
# A task = (map, seed): load the map on a fresh server (up.sh), record CLIPS_PER_TASK clips (uz_prod.py), each
# uploaded to $UZ_OUT/clips (or $UZ_OUT/rejected) as soon as it is finished. Task k of GPU g on machine NN (UZ_MACHINE) uses
#   map  = MAPS[(NN*8 + g + k) % len(MAPS)]          seed = NN*1000000 + g*100000 + k
# k is kept in $P/state/g<G>.k. Restarting resumes at the same task; clips already in storage are skipped, so a machine
# that lost its local state just fast-forwards through tasks it already did.
# Failures: each task attempt counts. A failed attempt (server did not come up, recorder crashed, or no clip passed
# the gates) sleeps min(30*2^(n-1), 900) s, n = consecutive failed attempts on that GPU. A map that fails
# MAP_FAIL_CAP attempts on a GPU is dropped there. After MAX_CONSEC_FAIL consecutive failed attempts, or
# MAX_ATTEMPTS attempts in total, or UP_FAIL_CAP consecutive server start failures, the GPU loop gives up
# (state/g<G>.GAVEUP) and its server is stopped; `start` does not restart a GPU that gave up unless GAVEUP is removed.
# Server starts on a machine are serialised with >= 25 s between them (lock in up.sh).
#
# Task-list mode (UZ_TASKS=FILE, lines "MAP SEED", e.g. tasks_v2_catalog.txt): instead of the formula above, every
# GPU on every machine takes the next unclaimed line of FILE. A line is claimed by mkdir $UZ_QUEUE/claim/<line>
# (atomic on a shared filesystem), so machines that see the same UZ_QUEUE directory share one queue. The claimed line is kept in $P/state/g<G>.cur until the task ends, so a restarted loop
# resumes it first (clips already in storage are skipped). A task whose map hit MAP_FAIL_CAP on this GPU is written to
# $UZ_QUEUE/dropped and skipped. The loop exits when no line is left.
CODE=$(cd "$(dirname "$0")" && pwd)
R=${UZ_HOME:-/opt/uzoo}; P=$R/prod
DEST=${UZ_OUT:-}
MAPS_FILE=${UZ_MAPS:-$CODE/maps_prod.txt}
CLIPS_PER_TASK=${CLIPS_PER_TASK:-8}
MAP_FAIL_CAP=${MAP_FAIL_CAP:-3}
MAX_CONSEC_FAIL=${MAX_CONSEC_FAIL:-8}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5000}
UP_FAIL_CAP=${UP_FAIL_CAP:-2}      # server failed to come up this many times in a row -> this GPU is disabled
MACHINE_IDX=$((10#${UZ_MACHINE:-0}))
TASKS=${UZ_TASKS:-}
Q=${UZ_QUEUE:-$R/queue_$(basename "${TASKS:-none}" .txt)}
mkdir -p $P/logs $P/state $P/out

claim_task() {   # task-list mode: echo the line number of this GPU's task (resumed or newly claimed); empty = none left
  local g=$1 i n
  if [ -s $P/state/g$g.cur ]; then cat $P/state/g$g.cur; return; fi
  mkdir -p $Q/claim
  n=$(grep -c . $TASKS)
  for i in $(seq 1 $n); do
    if mkdir $Q/claim/$i 2> /dev/null; then echo "$(hostname) g$g $(date -Is)" > $Q/claim/$i/by; echo $i > $P/state/g$g.cur; echo $i; return; fi
  done
}

kill_gpu() {   # server + bridge + recorder of GPU $1
  local g=$1 p
  for p in $(pgrep -f "unrealzoo/up.sh $g "); do kill -9 $p; done
  for p in $(pgrep -f "uz_prod.py --port $((9000+g)) "); do kill -9 $p; done
  for p in $(pgrep -f "bridge.py $((9000+g))"); do kill -9 $p; done
  for p in $(pgrep -f "graphicsadapter=$g -ResX"); do kill -9 $p; done
}

run_gpu() {
  local g=$1
  local LOG=$P/logs/loop_g$g.log
  mapfile -t MAPS < <(grep -v '^#' $MAPS_FILE | awk 'NF {print $1}')
  local M=${#MAPS[@]}
  [ $M -gt 0 ] || { echo "no maps in $MAPS_FILE"; exit 1; }
  local k=$(cat $P/state/g$g.k 2>/dev/null || echo 0)
  local attempts=0 consec=0 upfail=0 rc
  declare -A mfail
  echo "$(date -Is) start gpu=$g machine=$MACHINE_IDX maps=$M k=$k"
  while :; do
    [ -f $P/STOP ] && { echo "$(date -Is) STOP file, exiting"; break; }
    local map line= seed
    if [ -n "$TASKS" ]; then
      line=$(claim_task $g)
      [ -n "$line" ] || { echo "$(date -Is) task list done"; break; }
      read -r map seed < <(grep . $TASKS | sed -n "${line}p")
      if [ "${mfail[$map]:-0}" -ge $MAP_FAIL_CAP ]; then
        echo "$map $seed $(hostname) g$g map failed $MAP_FAIL_CAP times" >> $Q/dropped; rm -f $P/state/g$g.cur; continue
      fi
    else
      map=${MAPS[$(( (MACHINE_IDX*8 + g + k) % M ))]}
    fi
    if [ -z "$TASKS" ] && [ "${mfail[$map]:-0}" -ge $MAP_FAIL_CAP ]; then
      k=$((k+1)); echo $k > $P/state/g$g.k
      local alive=0 m; for m in "${MAPS[@]}"; do [ "${mfail[$m]:-0}" -lt $MAP_FAIL_CAP ] && alive=1; done
      [ $alive = 1 ] || { echo "$(date -Is) GIVEUP: every map failed $MAP_FAIL_CAP times" | tee $P/state/g$g.GAVEUP; break; }
      continue
    fi
    attempts=$((attempts+1))
    if [ $attempts -gt $MAX_ATTEMPTS ]; then echo "$(date -Is) GIVEUP: $MAX_ATTEMPTS attempts" | tee $P/state/g$g.GAVEUP; break; fi
    [ -n "$TASKS" ] || seed=$((MACHINE_IDX*1000000 + g*100000 + k))
    echo "$(date -Is) task k=$k${line:+ line=$line} map=$map seed=$seed attempt=$attempts"
    timeout 600 $CODE/up.sh $g $map 300; rc=$?
    if [ $rc -ne 0 ]; then
      upfail=$((upfail+1))
      if [ $upfail -ge $UP_FAIL_CAP ]; then
        kill_gpu $g
        echo "$(date -Is) GIVEUP: server failed to start $upfail times in a row (rc=$rc, map=$map); GPU $g disabled" | tee $P/state/g$g.GAVEUP; break
      fi
    else
      upfail=0
    fi
    if [ $rc -eq 0 ]; then
      timeout $((CLIPS_PER_TASK*1200)) $R/venv/bin/python -u $CODE/uz_prod.py --port $((9000+g)) --env $map --seed $seed \
        --clips $CLIPS_PER_TASK --out $P/out/g$g --dest $DEST --results $P/results.jsonl >> $P/logs/rec_g$g.log 2>&1
      rc=$?
      [ $rc -eq 0 ] || rc=$((100+rc))
    fi
    if [ $rc -eq 0 ]; then
      consec=0; k=$((k+1)); echo $k > $P/state/g$g.k; rm -f $P/state/g$g.cur
      echo "$(date -Is) task ok"
    else
      consec=$((consec+1)); mfail[$map]=$(( ${mfail[$map]:-0} + 1 ))
      # rc 103 = the map produced no passing clip: move on; anything else: retry this task after the backoff
      [ $rc -eq 103 ] && { k=$((k+1)); echo $k > $P/state/g$g.k; rm -f $P/state/g$g.cur; }
      local back=$(( 30 * (1 << (consec-1)) )); [ $back -gt 900 ] && back=900
      echo "$(date -Is) task FAILED rc=$rc map=$map (map fails ${mfail[$map]}, consecutive $consec) backoff ${back}s"
      kill_gpu $g
      if [ $consec -ge $MAX_CONSEC_FAIL ]; then echo "$(date -Is) GIVEUP: $consec consecutive failures" | tee $P/state/g$g.GAVEUP; break; fi
      sleep $back
    fi
  done
  kill_gpu $g
}

case "$1" in
  start)
    [ -f $R/.pkg_ok ] || { echo "not deployed: run $CODE/deploy.sh first"; exit 1; }
    [ -n "$DEST" ] || { echo "set UZ_OUT to the storage prefix for the output"; exit 1; }
    rm -f $P/STOP
    for g in 0 1 2 3 4 5 6 7; do
      if pgrep -f "prod.sh run-gpu $g\$" > /dev/null; then echo "gpu $g already running"; continue; fi
      if [ -f $P/state/g$g.GAVEUP ]; then echo "gpu $g disabled: $(cat $P/state/g$g.GAVEUP)"; continue; fi
      (setsid nohup $CODE/prod.sh run-gpu $g >> $P/logs/loop_g$g.log 2>&1 < /dev/null &)
      echo "gpu $g started"
    done ;;
  run-gpu) run_gpu $2 ;;
  drain) touch $P/STOP; echo "draining: loops exit after their current task" ;;
  stop)
    touch $P/STOP
    for p in $(pgrep -f "prod.sh run-gpu "); do kill -9 $p; done
    for g in 0 1 2 3 4 5 6 7; do kill_gpu $g; done
    echo stopped ;;
  status)
    echo "machine $MACHINE_IDX ($(hostname))  STOP=$([ -f $P/STOP ] && echo yes || echo no)"
    for g in 0 1 2 3 4 5 6 7; do
      run=$(pgrep -f "prod.sh run-gpu $g\$" > /dev/null && echo run || echo ---)
      gu=$([ -f $P/state/g$g.GAVEUP ] && echo GAVEUP || echo "")
      echo "g$g $run k=$(cat $P/state/g$g.k 2>/dev/null || echo 0) $gu | $(grep -E 'task (k|ok|FAILED)' $P/logs/loop_g$g.log 2>/dev/null | tail -1 | cut -c1-110)"
    done
    python3 - $P/results.jsonl <<'EOF'
import json, sys, time, collections
try: L = [json.loads(l) for l in open(sys.argv[1])]
except FileNotFoundError: L = []
now = time.time()
def age(r): return now - time.mktime(time.strptime(r['t'], '%Y-%m-%dT%H:%M:%S'))
for name, win in [('last 1h', 3600), ('all', 1e12)]:
    c = collections.Counter(r['status'] for r in L if age(r) < win)
    print(f"{name:8s} ok={c['ok']} reject={c['reject']} spawn_fail={c['spawn_fail']}")
rej = collections.Counter(x for r in L if r['status'] == 'reject' for x in r['gate']['reasons'])
print('reject reasons', dict(rej))
EOF
    echo "pending local uploads: $(ls $P/out/g*/*.json 2>/dev/null | wc -l)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';'; echo ;;
  *) sed -n '2,8p' $0; exit 1 ;;
esac
