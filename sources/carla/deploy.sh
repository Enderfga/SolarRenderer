#!/bin/bash
# deploy.sh: make a fresh Linux GPU machine able to run the CARLA generator (run as root). Idempotent; everything
# it installs comes from $CGEN_ASSETS (see build_assets.sh) + apt.
#   - CARLA 0.9.16 + AdditionalMaps (one merged tree; Town11/12/13 are never selected) under $CGEN_HOME
#     (default /opt/cgen: outside a mode-700 home directory, so the non-root carla user can reach its libraries)
#   - user "carla" (Unreal refuses to run as root)
#   - Vulkan ICD file pointing at the NVIDIA EGL library (a GLX-based ICD fails in headless containers)
#   - python venv with the carla client from the wheels under $CGEN_ASSETS
#   - system packages: ffmpeg, libvulkan1, python3-venv, pigz
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/env.sh
mkdir -p $CG && chmod 755 $CG && cd $CG
need=""
which ffmpeg >/dev/null || need="$need ffmpeg"
which pigz >/dev/null || need="$need pigz"
ldconfig -p | grep -q 'libvulkan.so.1 ' || need="$need libvulkan1"
python3 -c "import ensurepip, venv" 2>/dev/null || need="$need python3-venv"
if [ -n "$need" ]; then
  apt-get install -y -qq $need >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq $need >/dev/null; }
fi
# NVIDIA EGL library = the Vulkan driver entry point we use
EGL=$(ldconfig -p | awk '/libEGL_nvidia.so.0 /{print $NF; exit}')
[ -n "$EGL" ] || EGL=$(ls /usr/lib64/libEGL_nvidia.so.0 /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 2>/dev/null | head -1)
[ -n "$EGL" ] || { echo "no libEGL_nvidia.so.0 on this machine (NVIDIA graphics libraries not installed or not mounted)"; exit 1; }
cat > $VKJSON <<J
{"file_format_version":"1.0.1","ICD":{"library_path":"$EGL","api_version":"1.4.312"}}
J
chmod 644 $VKJSON
if [ ! -x $CARLA_DIR/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping ]; then
  rm -rf $CARLA_DIR.tmp; mkdir -p $CARLA_DIR.tmp
  $STORE cp $CGEN_ASSETS/CARLA_0.9.16_merged.tar.gz ./carla.tgz >/dev/null
  pigz -dc carla.tgz | tar x -C $CARLA_DIR.tmp
  rm -f carla.tgz; rm -rf $CARLA_DIR; mv $CARLA_DIR.tmp $CARLA_DIR
fi
id carla >/dev/null 2>&1 || useradd -m -s /bin/bash carla
chown -R carla $CARLA_DIR
if ! $PY -c "import carla, cv2, numpy" 2>/dev/null; then
  rm -rf $CG/venv $CG/wheels; mkdir -p $CG/wheels
  $STORE cp "$CGEN_ASSETS/wheels/*" $CG/wheels/ >/dev/null
  python3 -m venv --system-site-packages $CG/venv
  $CG/venv/bin/pip install -q --no-index --find-links $CG/wheels carla==0.9.16 opencv-python-headless numpy 2>&1 | grep -v -i -e deprecat -e "requires numpy" -e incompatible || true
fi
$PY -c "import carla, cv2, numpy; print('CLIENT_OK', numpy.__version__, cv2.__version__)"
echo "DEPLOY_OK vulkan_icd=$EGL"
