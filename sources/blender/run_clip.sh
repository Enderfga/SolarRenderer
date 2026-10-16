#!/bin/bash
# run_clip.sh <seed> <gpu> [workdir_root]: generate + post + upload one clip.  Exit 0 = uploaded, 3 = rejected.
set -o pipefail
SEED=$1; GPU=$2; ROOT=${3:-$PWD/bgen_work}
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/env.sh
DST=${BGEN_OUT:?set BGEN_OUT to the storage prefix for accepted clips (ending in /clips)}
# PBR textures for the realistic style (a machine deployed without them): fetch once per machine
[ -f $BGEN_ASSETS/pbr/index.json ] || ( flock 8; [ -f $BGEN_ASSETS/pbr/index.json ] || { T=$BGEN_ASSETS/.pbr_tmp; rm -rf $T; mkdir -p $T && $STORE cp ${BGEN_ASSETS:?set BGEN_ASSETS}/pbr.tgz $T/ >/dev/null && tar xzf $T/pbr.tgz -C $T && mv $T/pbr $BGEN_ASSETS/pbr; rm -rf $T; } ) 8>$BGEN_ASSETS/.pbr.lock
CID=$(printf "bg%08d" $SEED)
D=$ROOT/$CID; rm -rf $D; mkdir -p $D
export CUDA_VISIBLE_DEVICES=$GPU
$BLENDER -b --gpu-backend opengl --factory-startup --python $HERE/gen_clip.py -- $SEED $D --samples ${BGEN_SAMPLES:-32} --engine ${BGEN_ENGINE:-eevee} ${BGEN_ARGS} > $D/blender.log 2>&1
if ! grep -q GEN_DONE $D/blender.log; then echo "$CID GEN_FAIL"; tail -5 $D/blender.log; mkdir -p $ROOT/_failed; cp $D/blender.log $ROOT/_failed/$CID.log; rm -rf $D; exit 2; fi
R=$(python3 $HERE/post.py $D 2>&1 | tail -1)
echo "$CID $R $(grep GEN_DONE $D/blender.log)"
if [[ $R == POST_OK* ]]; then
  for s in $(python3 -c "import json; print(*json.load(open(\"$D/${CID}.json\"))[\"files\"])"); do $STORE cp $D/${CID}_$s.mp4 $DST/${CID}_$s.mp4 >/dev/null || { echo "$CID UPLOAD_FAIL"; rm -rf $D; exit 4; }; done
  # json last: its presence in storage marks the clip complete
  $STORE cp $D/${CID}.json $DST/${CID}.json >/dev/null || { echo "$CID UPLOAD_FAIL"; rm -rf $D; exit 4; }
  rm -rf $D; exit 0
else
  $STORE cp $D/${CID}.json ${DST%/clips}/rejected/${CID}.json >/dev/null 2>&1
  rm -rf $D; exit 3
fi
