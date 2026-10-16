# Environment for the CARLA generator. deploy.sh installs everything under $CGEN_HOME (local disk; keep it outside
# a mode-700 home directory so the non-root `carla` user can reach its own libraries).
CG=${CGEN_HOME:-/opt/cgen}
CARLA_DIR=$CG/carla916
PY=$CG/venv/bin/python
VKJSON=$CG/nv_egl.json
STORE=${STORAGE_CLI:-solar-storage}
# CGEN_ASSETS: storage prefix holding CARLA_0.9.16_merged.tar.gz and wheels/ (made by build_assets.sh)
# CGEN_OUT:     storage prefix for accepted clips; CGEN_REJECT: storage prefix for the rejected clips' json
CGEN_ASSETS=${CGEN_ASSETS:?set CGEN_ASSETS to the storage prefix written by build_assets.sh}
CGEN_OUT=${CGEN_OUT:-}
CGEN_REJECT=${CGEN_REJECT:-}
