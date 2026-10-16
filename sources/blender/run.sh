#!/bin/bash
# run.sh <shard> <num_shards>   (one shard per machine)
#   Generates clips with seeds  shard + k*num_shards  (k = 0,1,2,...), skipping seeds whose json is already in storage
#   (clips/ or rejected/), until this machine has uploaded ceil(BGEN_TARGET/num_shards) clips.
# env: BGEN_OUT (storage prefix for accepted clips, ending in /clips; rejected json go to the sibling rejected/, required),
#      BGEN_ASSETS (see deploy.sh), BGEN_HOME (install dir, default /opt/bgen), BGEN_WORK (default ./bgen_work),
#      BGEN_TARGET (total clips over all shards, default 20000), BGEN_GPUS (default "0 1 2 3 4 5 6 7"),
#      BGEN_PER_GPU (Blender processes per GPU, default 4), BGEN_ENGINE (eevee | cycles, default eevee),
#      BGEN_SAMPLES (render samples, default 32), BGEN_ARGS (extra gen_clip.py arguments), STORAGE_CLI (default solar-storage).
# Runs in the foreground; launch with:  nohup bash run.sh I N > bgen_pod.log 2>&1 &
IDX=$1; NP=$2
HERE=$(cd "$(dirname "$0")" && pwd)
TARGET=${BGEN_TARGET:-20000}; GPUS=${BGEN_GPUS:-"0 1 2 3 4 5 6 7"}; PER=${BGEN_PER_GPU:-4}
QUOTA=$(( (TARGET + NP - 1) / NP ))
W=${BGEN_WORK:-$PWD/bgen_work}; mkdir -p $W/logs
[ -x ${BGEN_HOME:-/opt/bgen}/blender-4.5.3-linux-x64/blender ] || bash $HERE/deploy.sh || { echo "deploy failed"; exit 1; }
source $HERE/env.sh
BGEN_OUT=${BGEN_OUT:?set BGEN_OUT to the storage prefix for accepted clips}
# seeds already done (any machine)
$STORE ls "$BGEN_OUT/*.json" 2>/dev/null | grep -oE 'bg[0-9]{8}' > $W/done.txt
$STORE ls "${BGEN_OUT%/clips}/rejected/*.json" 2>/dev/null | grep -oE 'bg[0-9]{8}' >> $W/done.txt
MINE=$(grep -oE '[0-9]+' $W/done.txt | sed 's/^0*//' | awk -v i=$IDX -v n=$NP '($1+0)%n==i' | wc -l)
echo $MINE > $W/uploaded.cnt   # counts done seeds of this shard (approximate quota accounting incl. rejects)
echo 0 > $W/k.cnt
echo "$(date -u +%FT%TZ) shard $IDX/$NP quota $QUOTA already $MINE  gpus [$GPUS] x $PER"
next_seed() {  # atomic k++ ; prints seed or nothing when quota reached
  ( flock 9
    up=$(cat $W/uploaded.cnt); [ "$up" -ge "$QUOTA" ] && exit 0
    while :; do
      k=$(cat $W/k.cnt); echo $((k + 1)) > $W/k.cnt
      s=$((IDX + k * NP)); c=$(printf "bg%08d" $s)
      grep -qx $c $W/done.txt || { echo $s; exit 0; }
    done ) 9>$W/lock
}
worker() {
  g=$1; j=$2
  while :; do
    s=$(next_seed); [ -z "$s" ] && break
    $HERE/run_clip.sh $s $g $W >> $W/logs/g${g}_${j}.log 2>&1; rc=$?
    if [ $rc = 0 ] || [ $rc = 3 ]; then ( flock 9; echo $(( $(cat $W/uploaded.cnt) + (rc == 0 ? 1 : 0) )) > $W/uploaded.cnt ) 9>$W/lock; fi
    echo "$(date -u +%FT%TZ) seed $s gpu $g rc $rc" >> $W/progress.log
  done
}
for g in $GPUS; do for j in $(seq 1 $PER); do worker $g $j & sleep 3; done; done
wait
echo "$(date -u +%FT%TZ) shard $IDX done: $(cat $W/uploaded.cnt) clips"
