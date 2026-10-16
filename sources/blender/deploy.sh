#!/bin/bash
# deploy.sh: install Blender 4.5.3 + GL libs + CC0 assets under $BGEN_HOME (default /opt/bgen, local disk) from
# $BGEN_ASSETS, a storage prefix holding:
#   blender-4.5.3-linux-x64.tar.xz  the official Linux build (https://download.blender.org/release/Blender4.5/)
#   libs.tgz                         extra X11/GL runtime libraries for headless rendering, as usr/lib/x86_64-linux-gnu/...
#   pbr.tgz                          pbr/ written by fetch_pbr.py (Poly Haven CC0 texture sets + index.json)
#   assets.tgz                       assets/catalog.json (catalog.py) + assets/packs/ (the CC0 packs listed in biomes.py)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
STORE=${STORAGE_CLI:-solar-storage}; SRC=${BGEN_ASSETS:?set BGEN_ASSETS to the storage prefix holding the tarballs}
BG=${BGEN_HOME:-/opt/bgen}; mkdir -p $BG && cd $BG
[ -x blender-4.5.3-linux-x64/blender ] || { $STORE cp $SRC/blender-4.5.3-linux-x64.tar.xz ./b.txz && tar xJf b.txz && rm -f b.txz; }
[ -d libs ] || { $STORE cp $SRC/libs.tgz ./libs.tgz && tar xzf libs.tgz && rm -f libs.tgz; }
[ -f assets/pbr/index.json ] || { mkdir -p assets && $STORE cp $SRC/pbr.tgz ./pbr.tgz && tar xzf pbr.tgz -C assets && rm -f pbr.tgz; }
[ -f assets/catalog.json ] || { $STORE cp $SRC/assets.tgz ./assets.tgz && tar xzf assets.tgz && rm -f assets.tgz; }
python3 -c "import PIL, numpy" 2>/dev/null || pip install -q pillow numpy
which ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
export BGEN_HOME=$BG; source $HERE/env.sh
$BLENDER -b --factory-startup --python-expr "import bpy; print('BLENDER_OK', bpy.app.version_string)" 2>&1 | grep BLENDER_OK
echo DEPLOY_OK
