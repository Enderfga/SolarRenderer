"""catalog.py -- -- <list_file> <out_json>: per asset bbox / armature / actions."""
import bpy, sys, json, os
from mathutils import Vector
a = sys.argv[sys.argv.index("--") + 1:]
files = [l.strip() for l in open(a[0]) if l.strip()]
out = {}
for f in files:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        if f.endswith(".fbx"): bpy.ops.import_scene.fbx(filepath=f)
        else: bpy.ops.import_scene.gltf(filepath=f)
    except Exception as e:
        out[f] = dict(error=str(e)); continue
    bpy.context.view_layer.update()
    mn = Vector((1e9,) * 3); mx = Vector((-1e9,) * 3); nm = 0; nv = 0
    for o in bpy.data.objects:
        if o.type == "MESH":
            nm += 1; nv += len(o.data.vertices)
            for c in o.bound_box:
                w = o.matrix_world @ Vector(c)
                mn = Vector(map(min, mn, w)); mx = Vector(map(max, mx, w))
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    r = dict(min=[round(x, 3) for x in mn], max=[round(x, 3) for x in mx], n_mesh=nm, n_vert=nv,
             meshes=[o.name for o in bpy.data.objects if o.type == "MESH"][:40])
    if arms:
        r["armature"] = arms[0].name
        r["bones"] = [b.name for b in arms[0].data.bones]
        r["actions"] = {ac.name: [ac.frame_range[0], ac.frame_range[1]] for ac in bpy.data.actions}
    out[f] = r
json.dump(out, open(a[1], "w"))
print("CATALOG_DONE", len(out))
