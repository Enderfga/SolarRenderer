#!/bin/bash
# usage: run_batch.sh OUTDIR SEED CLIPS_PER_ENV GPU:ENV [GPU:ENV ...]
# For each GPU:ENV pair: (re)start the UE server on that GPU with ENV (up.sh) and record CLIPS_PER_ENV
# clips with uz_record.py, all pairs in parallel. Other GPUs are left alone. Package, venv and logs in $UZ_HOME.
OUT=$1; SEED=$2; N=$3; shift 3
D=$(cd "$(dirname "$0")" && pwd)
cd ${UZ_HOME:-/opt/uzoo}
for pair in "$@"; do
  g=${pair%%:*}; e=${pair#*:}
  for x in $(pgrep -f "uz_record.py --port $((9000+g)) "); do kill -9 $x; done
  ( $D/up.sh $g $e > logs/up$g.log 2>&1 && \
    venv/bin/python -u $D/uz_record.py --port $((9000+g)) --env $e --clips $N --out $OUT --seed $SEED > logs/rec_${e}_$SEED.log 2>&1 ) &
done
wait
echo BATCH_DONE
