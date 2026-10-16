#!/bin/bash
# launch.sh [N=16]: start N looping converter shards on this machine (WW_LOOP=1), logs in $WW_WORK/conv_<i>.log.
# Needs WW_SRC and WW_DST in the environment (see convert.sh).
N=${1:-16}
HERE=$(cd "$(dirname "$0")" && pwd)
export WW_WORK=${WW_WORK:-$PWD/wildworld_work}
mkdir -p $WW_WORK
for i in $(seq 0 $((N - 1))); do
  WW_LOOP=1 setsid nohup bash $HERE/convert.sh $i $N > $WW_WORK/conv_$i.log 2>&1 < /dev/null &
done
