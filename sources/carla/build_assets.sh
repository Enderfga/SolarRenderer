#!/bin/bash
# build_assets.sh [work_dir]: one-off, builds what deploy.sh downloads and puts it under $CGEN_ASSETS:
#   $CGEN_ASSETS/CARLA_0.9.16_merged.tar.gz  (CARLA 0.9.16 + AdditionalMaps, 12.8 GB)
#   $CGEN_ASSETS/wheels/                    (carla 0.9.16, opencv-python-headless, numpy for the local python)
# Work dir on local disk; needs ~75 GB.
set -e
D=${1:-/opt/carla_build}; mkdir -p $D && cd $D
STORE=${STORAGE_CLI:-solar-storage}; A=${CGEN_ASSETS:?set CGEN_ASSETS to a storage prefix}
which aria2c pigz >/dev/null || { apt-get update -qq; apt-get install -y -qq aria2 pigz; }
B=https://carla-releases.s3.us-east-005.backblazeb2.com/Linux
aria2c -x16 -s16 -k64M --file-allocation=none -q $B/CARLA_0.9.16.tar.gz
aria2c -x16 -s16 -k64M --file-allocation=none -q $B/AdditionalMaps_0.9.16.tar.gz
mkdir -p carla916 && pigz -dc CARLA_0.9.16.tar.gz | tar x -C carla916 && pigz -dc AdditionalMaps_0.9.16.tar.gz | tar x -C carla916
tar c -C carla916 . | pigz -p 64 -3 > CARLA_0.9.16_merged.tar.gz
$STORE --numworkers 32 cp CARLA_0.9.16_merged.tar.gz $A/
mkdir -p wheels && pip download -q -d wheels carla==0.9.16 opencv-python-headless numpy && $STORE cp "wheels/*" $A/wheels/
