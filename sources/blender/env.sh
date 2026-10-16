# Environment for the Blender generator (installed by deploy.sh under $BGEN_HOME, local disk)
BG=${BGEN_HOME:-/opt/bgen}
export BGEN_ASSETS=$BG/assets
export LD_LIBRARY_PATH=$BG/libs/usr/lib/x86_64-linux-gnu:/usr/lib64:$LD_LIBRARY_PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
BLENDER=$BG/blender-4.5.3-linux-x64/blender
STORE=${STORAGE_CLI:-solar-storage}
