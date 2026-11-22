# Turns a gtl_core ImportResult into a Blender scene.

import bpy

from . import builder


def _instance_kind(name):
    up = name.upper()
    if up in ("LFTIRE", "RFTIRE", "LRTIRE", "RRTIRE",
              "LFTIRE_EXPO", "RFTIRE_EXPO"):
        return "tire"
    if "STEERING" in up or up == "WHEEL":
        return "steering"
    return "part"


_GROUP_TITLES = {"MAX": "Max", "HIGH": "High", "MED": "Med", "LOW": "Low",
                 "COMMON": "Common"}


def _group_title(group):
    if group in _GROUP_TITLES:
        return _GROUP_TITLES[group]
    return "_".join(p.capitalize() if p != "DASHHIGH" else "DashHigh"
                    for p in group.split("_")).replace("Notspin", "NotSpin")


def build_scene(result, opts):
    """Create the scene from ImportResult.

    Tracks: ONE collection per import (all objects linked into it).
    Cars: root collection with one child collection per CAS group -
    Max / High / Med / Low (LODs) and Spin / NotSpin / Expo / ... (types).
    High, Med and Low are hidden in the viewport right away."""
    mat_cache = {}
    missing_textures = set()
    use_alpha = bool(getattr(opts, "use_texture_alpha", True))

    def report_missing(tex):
        missing_textures.add(tex)

    root = bpy.data.collections.new(result.collection_name[:63])
    bpy.context.scene.collection.children.link(root)

    group_cols = {}
    hide_in_view_layer = []
    light_col = []

    def collection_for(group):
        if not getattr(result, "groups", None):
            return root
        col = group_cols.get(group)
        if col is None:
            title = _group_title(group)
            col = bpy.data.collections.new(
                ("%s_%s" % (result.collection_name, title))[:63])
            root.children.link(col)
            if group in ("HIGH", "MED", "LOW"):
                hide_in_view_layer.append(col)
            group_cols[group] = col
        return col

    # create collections in a stable order (Max first)
    for g in getattr(result, "groups", []) or []:
        collection_for(g)

    tire_objects = []
    steering_objects = []

    for inst in result.instances:
        if inst.light is not None:
            if opts.import_lights:
                if not light_col:
                    lc = bpy.data.collections.new(
                        ("%s_Nightlights" % result.collection_name)[:63])
                    root.children.link(lc)
                    light_col.append(lc)
                builder.build_light(inst, light_col[0])
            continue
        for mesh in inst.meshes:
            ob = builder.build_mesh_object(
                mesh, mat_cache, collection_for(getattr(mesh, "group", "COMMON")),
                report_missing, use_alpha=use_alpha)
            kind = _instance_kind(inst.name)
            if kind == "tire":
                tire_objects.append(ob)
            elif kind == "steering":
                steering_objects.append(ob)

    # --- wheel origins on the hub centre (spin axis = lateral = X) --------
    # The geometry stays exactly where the game puts it; only the object
    # origin moves to the hub so a rotation around X rolls the wheel.
    for ob in tire_objects:
        _move_origin_to_bbox_center(ob)
        ob["gtl_spin_axis"] = "1.0, 0.0, 0.0 (X = lateral)"

    for col in hide_in_view_layer:
        _eye_off(col)

    root["gtl_missing_textures"] = len(missing_textures)
    root["gtl_missing_meshes"] = len(result.missing_meshes)
    for key, val in getattr(result, "driver_info", {}).items():
        root[key] = str(val)
    return root, missing_textures


def _move_origin_to_bbox_center(ob):
    import numpy as np
    me = ob.data
    n = len(me.vertices)
    if not n:
        return
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    center = co.mean(axis=0)
    me.vertices.foreach_set("co", (co - center).ravel())
    ob.location = (ob.location[0] + float(center[0]),
                   ob.location[1] + float(center[1]),
                   ob.location[2] + float(center[2]))
    me.update()


def _eye_off(col):
    """Hide a collection with the eye icon of the outliner (view layer
    'Hide in Viewport'); 'Disable in Viewports' stays off."""
    def find(lc):
        if lc.collection is col:
            return lc
        for ch in lc.children:
            hit = find(ch)
            if hit:
                return hit
        return None
    try:
        lc = find(bpy.context.view_layer.layer_collection)
        if lc:
            lc.hide_viewport = True
    except Exception:                                      # noqa: BLE001
        pass
