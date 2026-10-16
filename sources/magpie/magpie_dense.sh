#!/usr/bin/env bash
# Second cutting pass over Magpie Lite kept in storage: the first pass took an 8 s window every 25 s (a third
# of the footage); this pass cuts the windows in between on the CPU, a dozen trajectories at a time on local disk.
#   magpie_dense.sh <shard> <num_shards>
# Shard k of n takes every n-th trajectory. New clips go beside the old ones in $MAGPIE_OUT/clips/, their windows
# to $MAGPIE_OUT/metadata/windows_dense.<shard>.<batch>.jsonl.
# env: MAGPIE_OUT   prefix holding scene*/<player>/{render,white_box}.mp4, metadata/windows.jsonl (first pass) (required)
#      WORK        local scratch directory (default ./magpie_dense)
#      STORAGE_CLI storage command (default solar-storage);  PYTHON  python with av + numpy (default python3)
set -uo pipefail
I=$1; N=$2
STORE=${STORAGE_CLI:-solar-storage}; PY=${PYTHON:-python3}
B=${MAGPIE_OUT:?set MAGPIE_OUT to the storage prefix of the Magpie Lite copy}
HERE=$(cd "$(dirname "$0")" && pwd)
W=${WORK:-$PWD/magpie_dense}; mkdir -p $W/raw $W/clips
$STORE cp "$B/metadata/windows.jsonl" $W/windows.jsonl > /dev/null
$STORE ls "$B/scene*/*/render.mp4" | awk '{print $NF}' | sed 's#/render.mp4##' | sort | awk -v i=$I -v n=$N 'NR%n==i' > $W/trajs.txt
echo "$(date -u +%T) $(wc -l < $W/trajs.txt) trajectories"
split -l 12 -d $W/trajs.txt $W/batch.
for f in $W/batch.*; do
  b=${f##*.}
  $STORE ls "$B/metadata/windows_dense.$I.$b.jsonl" > /dev/null 2>&1 && { echo "batch $b done before"; continue; }
  rm -rf $W/raw/* $W/clips/*
  for t in $(cat $f); do echo "cp $B/$t/render.mp4 $W/raw/$t/render.mp4"; echo "cp $B/$t/white_box.mp4 $W/raw/$t/white_box.mp4"; done > $W/dl.txt
  $STORE --numworkers 32 run $W/dl.txt > /dev/null || { echo "download failed batch $b"; continue; }
  sed "s#^#$W/raw/#" $f > $W/batch_local.txt
  MAGPIE_ROOT=$W/raw MAGPIE_CLIPS=$W/clips SPACING=8 EXISTING=$W/windows.jsonl TRAJS=$W/batch_local.txt \
    WINDOWS=windows_dense.$I.$b.jsonl $PY $HERE/magpie_cut.py 2>&1 | grep -E "CUT DONE|PROBLEM|Error"
  $STORE --numworkers 32 cp "$W/clips/*.mp4" "$B/clips/" > /dev/null || { echo "upload failed batch $b"; continue; }
  # The windows file goes last: its presence is the mark that the batch is complete.
  $STORE cp "$W/clips/windows_dense.$I.$b.jsonl" "$B/metadata/windows_dense.$I.$b.jsonl" > /dev/null
  echo "$(date -u +%T) batch $b: $(wc -l < $W/clips/windows_dense.$I.$b.jsonl) windows"
done
rm -rf $W/raw $W/clips
echo "$(date -u +%T) DENSE DONE"
