#!/bin/bash
# usage: launch.sh GPU MAP [extra UE args]   -> UnrealCV port 9000+GPU
G=$1; M=$2; shift 2
R=${UZ_HOME:-/opt/uzoo}
export VK_ICD_FILENAMES=$R/vk/nv_egl.json
cd $R/inst$G
exec runuser -u carla -- env VK_ICD_FILENAMES=$VK_ICD_FILENAMES CUDA_VISIBLE_DEVICES=$G ./UnrealZoo_UE5_7/Binaries/Linux/UnrealZoo_UE5_7 UnrealZoo_UE5_7 $M -RenderOffScreen -nosound -unattended -graphicsadapter=$G -ResX=1344 -ResY=768 -windowed -FORCELOGFLUSH "$@"
