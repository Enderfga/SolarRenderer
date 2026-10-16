#!/bin/bash
# deploy.sh: make this machine ready to run UnrealZoo servers (idempotent; safe to re-run).
#   1. non-root user `carla` (UE refuses to run as root)
#   2. package UnrealZoo_UE5_7_Linux_v3.1.0 from $UZ_PKG -> $UZ_HOME (tar, 73.6 GB; ~150 GB free needed while extracting)
#   3. unrealzoo-gym (Track configs with the safe_start spawn points) -> $UZ_HOME/unrealzoo-gym
#   4. python venv with UnrealCV client + OpenCV
#   5. one hard-linked instance per GPU (inst0..inst7) with its own unrealcv.ini (port 9000+GPU, 1344x768)
#   6. Vulkan ICD for headless EGL (vk/nv_egl.json), as in the CARLA generator
# UZ_PKG (storage prefix, required) holds $P.tar (the Linux/ tree of the release zip
# https://huggingface.co/datasets/UnrealZoo/UnrealZoo_UE5, under a top directory $P) and $GYM.tar
# (https://github.com/UnrealZoo/unrealzoo-gym at commit 85f4b72).
# usage: UZ_PKG=<storage>/... [UZ_HOME=/opt/uzoo] [UZ_PKG_BYTES=<size of $P.tar>] deploy.sh     (run as root)
set -euo pipefail
STORE=${STORAGE_CLI:-solar-storage}
PKG=${UZ_PKG:?set UZ_PKG to the storage prefix holding the package tars}
P=UnrealZoo_UE5_7_Linux_v3.1.0
PKG_BYTES=${UZ_PKG_BYTES:-73606819840}      # size of $P.tar in storage
PAK_BYTES=71380507170                       # size of Content/Paks/pakchunk0-Linux.pak
GYM=unrealzoo-gym_85f4b72
R=${UZ_HOME:-/opt/uzoo}
mkdir -p $R/logs $R/vk
log() { echo "[deploy $(hostname) $(date +%H:%M:%S)] $*"; }

# 1. user
id carla > /dev/null 2>&1 || { useradd -m -s /bin/bash carla; log "user carla created"; }
chmod o+x "$(dirname "$R")"                  # carla must traverse the parent directory to reach $R/inst*

# 2. package
BIN=$R/$P/Linux/UnrealZoo_UE5_7/Binaries/Linux/UnrealZoo_UE5_7
PAK=$R/$P/Linux/UnrealZoo_UE5_7/Content/Paks/pakchunk0-Linux.pak
if [ -f $R/.pkg_ok ] && [ -x $BIN ] && [ "$(stat -c %s $PAK 2>/dev/null)" = "$PAK_BYTES" ]; then
  log "package present"
else
  rm -rf $R/$P $R/inst? $R/.pkg_ok
  avail=$(df --output=avail -B1G $R | tail -1 | tr -d ' ')
  [ "$avail" -ge 150 ] || { log "only ${avail} GB free on $R, need 150"; exit 1; }
  log "downloading $P.tar (${avail} GB free)"
  $STORE --numworkers 32 cp --concurrency 32 "$PKG/$P.tar" $R/$P.tar
  [ "$(stat -c %s $R/$P.tar)" = "$PKG_BYTES" ] || { log "tar size mismatch"; exit 1; }
  log "extracting"
  tar -C $R -xf $R/$P.tar
  rm -f $R/$P.tar
  [ "$(stat -c %s $PAK)" = "$PAK_BYTES" ] || { log "pak size mismatch after extract"; exit 1; }
  chmod +x $BIN
  touch $R/.pkg_ok
  log "package ok"
fi

# 3. gym configs
if [ ! -d $R/unrealzoo-gym/gym_unrealcv/envs/setting/Track ]; then
  mkdir -p $R/unrealzoo-gym
  $STORE cat "$PKG/$GYM.tar" | tar -C $R/unrealzoo-gym -xf -
  log "gym configs ok ($(ls $R/unrealzoo-gym/gym_unrealcv/envs/setting/Track | wc -l) Track configs)"
fi

# 4. venv
if ! $R/venv/bin/python -c "import unrealcv, cv2, numpy" 2> /dev/null; then
  python3 -m venv --system-site-packages $R/venv
  $R/venv/bin/pip install -q unrealcv==1.3.2 opencv-python-headless==5.0.0.93 numpy==2.5.3
  log "venv ok"
fi

# 5. per-GPU instances (hard links: no extra disk) + UnrealCV config
for g in 0 1 2 3 4 5 6 7; do
  [ -d $R/inst$g ] || cp -al $R/$P/Linux $R/inst$g
  printf '[UnrealCV.Core]\nPort=%d\nWidth=1344\nHeight=768\n' $((9000 + g)) > $R/inst$g/UnrealZoo_UE5_7/Binaries/Linux/unrealcv.ini
done
chown -R carla $R/inst? $R/$P

# 6. headless Vulkan through the NVIDIA EGL ICD
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"/usr/lib64/libEGL_nvidia.so.0","api_version":"1.4.312"}}' > $R/vk/nv_egl.json
[ -e /usr/lib64/libEGL_nvidia.so.0 ] || log "WARNING: /usr/lib64/libEGL_nvidia.so.0 missing"
command -v ffmpeg > /dev/null || { log "ffmpeg missing"; exit 1; }
log "DEPLOY_OK"
