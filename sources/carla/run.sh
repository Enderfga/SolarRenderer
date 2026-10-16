#!/bin/bash
# run.sh <shard> <num_shards>   (one shard per machine; every machine needs the same GPU count x CGEN_PER_GPU)
#   One CARLA 0.9.16 server per GPU (CGEN_PER_GPU servers per GPU), each fed by its own worker loop.
#   Slot i (= gpu*PER + j) generates seeds  shard + num_shards*(i + NSLOT*k), k = 0,1,2,...
#   (NSLOT = #gpus * PER, must be the same on every machine), skipping seeds whose json is already in storage
#   (clips/ or rejected/), until this machine has uploaded ceil(CGEN_TARGET/num_shards) clips.
# env: CGEN_OUT / CGEN_REJECT (storage prefixes for accepted clips / rejected clips' json, required),
#      CGEN_ASSETS (see env.sh), CGEN_WORK (local work dir, default ./cgen_work),
#      CGEN_TARGET (total clips over all shards, default 100000), CGEN_GPUS (default "0 1 2 3 4 5 6 7"),
#      CGEN_PER_GPU (servers per GPU, default 1), STORAGE_CLI (default solar-storage).
# Works on a fresh machine: deploy.sh installs everything from $CGEN_ASSETS + apt.
# Launch:  (setsid nohup bash run.sh I N > cgen_pod.log 2>&1 < /dev/null &)
IDX=$1; NP=$2
HERE=$(cd "$(dirname "$0")" && pwd)
TARGET=${CGEN_TARGET:-100000}; GPUS=${CGEN_GPUS:-"0 1 2 3 4 5 6 7"}; PER=${CGEN_PER_GPU:-1}
QUOTA=$(( (TARGET + NP - 1) / NP ))
NSLOT=$(( $(echo $GPUS | wc -w) * PER )); SLOTS=$((NP * NSLOT))
W=${CGEN_WORK:-$PWD/cgen_work}; mkdir -p $W/logs
bash $HERE/deploy.sh > $W/deploy.log 2>&1 || { echo "deploy failed"; tail -5 $W/deploy.log; exit 1; }
source $HERE/env.sh
: ${CGEN_OUT:?set CGEN_OUT} ${CGEN_REJECT:?set CGEN_REJECT}
log() { echo "$(date -u +%FT%TZ) $*"; }
# seeds already done (any machine)
$STORE ls "$CGEN_OUT/*.json" 2>/dev/null | grep -oE 'cc[0-9]{9}' > $W/done.txt
$STORE ls "$CGEN_REJECT/*.json" 2>/dev/null | grep -oE 'cc[0-9]{9}' >> $W/done.txt
MINE=$(grep -oE '[0-9]+' $W/done.txt | sed 's/^0*//' | awk -v i=$IDX -v n=$NP '($1+0)%n==i' | wc -l)
echo $MINE > $W/uploaded.cnt
log "shard $IDX/$NP quota $QUOTA already $MINE  gpus [$GPUS] x $PER servers  slots $SLOTS"

server_port() { echo $((2000 + 10 * $1)); }
server_alive() { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null; }
stop_server() {  # $1 = port
  for p in $(pgrep -f "rpc-port=$1 -graphicsadapte[r]"); do kill -9 $p; done
}
start_server() {  # $1 = slot, $2 = gpu ; returns 0 when the server answers RPC
  local P=$(server_port $1); stop_server $P; sleep 2
  ( cd $CARLA_DIR && setsid runuser -u carla -- env VK_ICD_FILENAMES=$VKJSON ./CarlaUE4.sh -RenderOffScreen -nosound \
      -carla-rpc-port=$P -graphicsadapter=$2 > $W/logs/server_s$1.log 2>&1 < /dev/null & )
  for i in $(seq 1 40); do
    sleep 3
    pgrep -f "rpc-port=$P -graphicsadapte[r]" >/dev/null || return 1   # crashed during start-up
    timeout 20 $PY -c "import carla; c=carla.Client('localhost',$P); c.set_timeout(10); c.get_world()" 2>/dev/null && return 0
  done
  return 1
}
next_seed() {  # $1 = slot ; per-slot counter, prints next seed not yet done, or nothing when quota reached
  local up=$(cat $W/uploaded.cnt); [ "$up" -ge "$QUOTA" ] && return
  while :; do
    local k=$(cat $W/k_s$1.cnt); echo $((k + 1)) > $W/k_s$1.cnt
    local s=$((IDX + NP * ($1 + NSLOT * k))); local c=$(printf "cc%09d" $s)
    grep -qx $c $W/done.txt || { echo $s; return; }
  done
}
upload_clip() {  # $1 = out dir, $2 = id ; json last: its presence in storage marks the clip complete
  for st in rgb whitebox semantic instance depth normals skeleton; do
    $STORE cp $1/$2_$st.mp4 $CGEN_OUT/$2_$st.mp4 >/dev/null || return 1
  done
  $STORE cp $1/$2.json $CGEN_OUT/$2.json >/dev/null
}
worker() {  # $1 = slot, $2 = gpu
  local slot=$1 g=$2 P=$(server_port $1) fails=0 sfails=0 since=0 up=0
  local O=$W/out_s$slot; mkdir -p $O; [ -f $W/k_s$slot.cnt ] || echo 0 > $W/k_s$slot.cnt
  while :; do
    # (re)start the server when it is down (crash) or after 40 clips (guards against leaks)
    if [ $up = 0 ] || ! server_alive $P || [ $since -ge 40 ]; then
      [ $up = 1 ] && ! server_alive $P && log "slot $slot gpu $g: server on $P down, restarting"
      since=0
      if start_server $slot $g; then up=1; sfails=0; else
        up=0; fails=$((fails + 1)); sfails=$((sfails + 1))
        # a GPU whose renderer cannot start usually has a hung graphics engine (seen followed by an Xid 109 error
        # and a node reboot): stop launching on it instead of retrying
        if [ $sfails -ge 3 ]; then log "slot $slot gpu $g: server failed to start 3 times in a row, GPU disabled for this run"; break; fi
        if [ $fails -ge 50 ]; then log "slot $slot gpu $g: giving up after $fails consecutive failures"; break; fi
        log "slot $slot gpu $g: server did not come up ($fails in a row), backoff $((10 * fails))s"
        sleep $((10 * fails)); continue
      fi
    fi
    s=$(next_seed $slot); [ -z "$s" ] && break
    c=$(printf "cc%09d" $s)
    rm -f $O/$c*
    timeout -k 20 900 $PY $HERE/carla_clip.py --seed $s --slots $SLOTS --port $P --out $O >> $W/logs/clips_s$slot.log 2>&1
    rc=$?; since=$((since + 1))
    if [ $rc = 0 ]; then
      upload_clip $O $c && { ( flock 9; echo $(( $(cat $W/uploaded.cnt) + 1 )) > $W/uploaded.cnt ) 9>$W/lock; } || rc=4
    elif [ $rc = 3 ]; then
      $STORE cp $O/$c.rejected.json $CGEN_REJECT/$c.json >/dev/null 2>&1
    fi
    rm -f $O/$c*
    log "seed $s slot $slot gpu $g rc $rc" >> $W/progress.log
    if [ $rc = 0 ] || [ $rc = 3 ]; then fails=0; continue; fi
    fails=$((fails + 1))
    if [ $fails -ge 50 ]; then log "slot $slot gpu $g: giving up after $fails consecutive failures"; break; fi
    log "slot $slot gpu $g: seed $s failed rc $rc ($fails in a row), restarting server, backoff $((10 * fails))s"
    up=0; sleep $((10 * fails))
  done
  stop_server $P
}
slot=0
for g in $GPUS; do for j in $(seq 1 $PER); do worker $slot $g & slot=$((slot + 1)); sleep 2; done; done
wait
log "shard $IDX done: $(cat $W/uploaded.cnt) clips"
