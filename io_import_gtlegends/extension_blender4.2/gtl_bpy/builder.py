# Blender side of the importer: materials (pink "missing texture"
# fallback) and mesh object construction.
#
# Material rules:
#   * Principled BSDF, Roughness 1.0, texture straight into Base Color
#     (no vertex color / multiply nodes).
#   * Blend mode is "Opaque", or "Alpha Clip" when the texture really has an
#     alpha channel (option 'Use texture alpha').  Never blend / hashed.
#
# Mesh data brought over from the GMT files: UVs, custom split normals and
# the game's vertex color ("Col"), tangent and binormal attributes.

import os

import bpy
import numpy as np

PINK = (1.0, 0.0, 1.0, 1.0)
VCOL_LAYER = "Col"


def _find_texture(path):
    try:
        img = bpy.data.images.load(path, check_existing=True)
        img.name = os.path.splitext(os.path.basename(path))[0]
        return img
    except Exception:                                      # noqa: BLE001
        return None


def _placeholder_image(name):
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.new(name, 4, 4)
        img.pixels = list(PINK) * 16
    return img


def _set_clip(mat, threshold=0.5):
    """Alpha Clip (Blender 4.0/4.1 blend_method CLIP; 4.2+ has no clip mode,
    the threshold node in the shader does it there)."""
    if hasattr(mat, "blend_method"):
        try:
            mat.blend_method = "CLIP"
            mat.alpha_threshold = threshold
        except Exception:                                  # noqa: BLE001
            pass
    if hasattr(mat, "shadow_method"):
        try:
            mat.shadow_method = "CLIP"
        except Exception:                                  # noqa: BLE001
            pass
    if hasattr(mat, "surface_render_method"):
        try:
            mat.surface_render_method = "DITHERED"
        except Exception:                                  # noqa: BLE001
            pass


def build_material(spec, mat_cache, report_missing, use_alpha=True):
    """ONE Principled BSDF material (Roughness 1.0) for a GMT material."""
    key = (spec.name.lower(), spec.texture.lower() if spec.texture else "",
           bool(use_alpha))
    if key in mat_cache:
        return mat_cache[key]

    mat = bpy.data.materials.new(name=spec.name[:63])
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Roughness"].default_value = 1.0

    img = _find_texture(spec.found_path) if spec.found_path else None

    if img is None and spec.texture:
        # texture file not found: keep the texture node (named after the
        # missing file) with a pink placeholder image, so it is visible
        # which texture the material wants
        report_missing(spec.texture)
        img = _placeholder_image(spec.texture)
        mat.diffuse_color = PINK

    if img is not None:
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.image = img
        tex_node.label = spec.texture or img.name
        tex_node.location = (0, 0)
        nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        if (use_alpha and spec.found_path
                and not getattr(spec, "alpha_mask", False)
                and (spec.tex_alpha or spec.alpha)):
            # texture alpha -> Principled alpha, blend mode Alpha Clip.
            # Very transparent textures (glass, alpha < 0.5 everywhere) get
            # a lower clip threshold so they do not vanish.
            top = getattr(spec, "alpha_max", 1.0)
            nt.links.new(tex_node.outputs["Alpha"], bsdf.inputs["Alpha"])
            _set_clip(mat, 0.5 if top >= 0.5 else max(0.01, top * 0.5))
    else:
        bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)

    mat_cache[key] = mat
    return mat


def _mesh_from_data(name, verts, faces, uvs, colors=None, tangents=None,
                    binormals=None, normals=None, uvs_light=None,
                    mat_index=None):
    """Mesh creation from numpy arrays (foreach_set, one update call).

      colors    -> color attribute "Col" (linear RGBA, game values)
      tangents  -> attribute "Tangent"  (game stream or analytic)
      binormals -> attribute "Binormal"
      normals   -> custom split normals (exact game shading)
      uvs_light -> second uv layer "UVMap_Light" (only when it differs)"""
    mesh = bpy.data.meshes.new(name[:63])
    nv = len(verts)
    nf = len(faces)
    mesh.vertices.add(nv)
    mesh.vertices.foreach_set("co", np.ascontiguousarray(
        verts, dtype=np.float32).ravel())
    mesh.loops.add(nf * 3)
    flat_i = np.ascontiguousarray(faces, dtype=np.int32).ravel()
    mesh.loops.foreach_set("vertex_index", flat_i)
    mesh.polygons.add(nf)
    mesh.polygons.foreach_set("loop_start", np.arange(0, nf * 3, 3,
                                                      dtype=np.int32))
    try:
        mesh.polygons.foreach_set("loop_total", np.full(nf, 3, np.int32))
    except Exception:                                      # noqa: BLE001
        pass
    mesh.polygons.foreach_set("use_smooth", np.ones(nf, dtype=np.bool_))
    if mat_index is not None:
        mesh.polygons.foreach_set("material_index", mat_index)
    if uvs is not None and len(uvs) == nv:
        mesh.uv_layers.new(name="UVMap").data.foreach_set(
            "uv", np.ascontiguousarray(uvs[flat_i], dtype=np.float32).ravel())
    if uvs_light is not None and len(uvs_light) == nv:
        mesh.uv_layers.new(name="UVMap_Light").data.foreach_set(
            "uv", np.ascontiguousarray(uvs_light[flat_i],
                                       dtype=np.float32).ravel())
    # edges are derived from the faces; no validate() (data comes from a
    # parser that already guarantees in-range indices)
    mesh.update(calc_edges=True)

    if colors is not None and len(colors) == nv:
        try:
            attr = mesh.color_attributes.new(name=VCOL_LAYER,
                                             type='FLOAT_COLOR',
                                             domain='POINT')
            attr.data.foreach_set("color", np.ascontiguousarray(
                colors, dtype=np.float32).ravel())
        except Exception:                                  # noqa: BLE001
            pass
    for attr_name, vals in (("Tangent", tangents), ("Binormal", binormals)):
        if vals is not None and len(vals) == nv:
            try:
                at = mesh.attributes.new(name=attr_name, type='FLOAT_VECTOR',
                                         domain='POINT')
                at.data.foreach_set("vector", np.ascontiguousarray(
                    vals, dtype=np.float32).ravel())
            except Exception:                              # noqa: BLE001
                pass

    # custom split normals: the game's exact shading (set LAST)
    if normals is not None and len(normals) == nv:
        try:
            nrm = np.asarray(normals, dtype=np.float64)
            ln = np.sqrt((nrm * nrm).sum(axis=1))
            bad = ln < 1e-9
            nrm = nrm / np.where(bad, 1.0, ln)[:, None]
            nrm[bad] = (0.0, 0.0, 1.0)
            mesh.normals_split_custom_set_from_vertices(nrm.tolist())
            if hasattr(mesh, "use_auto_smooth"):
                mesh.use_auto_smooth = True
        except Exception:                                  # noqa: BLE001
            pass
    return mesh


class MatSpecSafe:
    """Fallback spec when a submesh has no matching material block."""

    def __init__(self, base_name):
        self.name = base_name + "_mat"
        self.texture = ""
        self.alpha = False
        self.found_path = None
        self.alpha_mask = False
        self.tex_alpha = False
        self.alpha_max = 1.0


def build_mesh_object(obj_data, mat_cache, collection, report_missing,
                      use_alpha=True):
    """Create a Blender object from MeshObjectData.

    Objects with a pivot (steering wheel etc.) get their origin ON the pivot
    so that rotating the object animates the part around its axis."""
    verts = obj_data.verts
    if obj_data.pivot:
        verts = verts - np.array(obj_data.pivot)

    used = np.unique(obj_data.face_mats)
    lut = np.zeros(int(obj_data.face_mats.max()) + 1 if len(used) else 1,
                   np.int32)
    lut[used] = np.arange(len(used), dtype=np.int32)
    mat_index = lut[obj_data.face_mats] if len(used) else None

    mesh = _mesh_from_data(obj_data.name, verts, obj_data.faces, obj_data.uvs,
                           colors=obj_data.colors, tangents=obj_data.tangents,
                           binormals=obj_data.binormals,
                           normals=obj_data.normals,
                           uvs_light=obj_data.uvs_light, mat_index=mat_index)

    for mi in used.tolist():
        if mi < len(obj_data.materials):
            spec = obj_data.materials[mi]
        else:
            spec = MatSpecSafe(obj_data.name)
        mesh.materials.append(build_material(spec, mat_cache, report_missing,
                                             use_alpha=use_alpha))

    ob = bpy.data.objects.new(obj_data.name[:63], mesh)
    if obj_data.pivot:
        ob.location = obj_data.pivot
    else:
        ob.location = (0.0, 0.0, 0.0)

    for key, val in obj_data.custom_props.items():
        ob[key] = val
    ob["gtl_instance"] = obj_data.instance_name
    ob["gtl_source"] = obj_data.source

    collection.objects.link(ob)
    if getattr(obj_data, "hidden", False):          # Render=False
        # only the "eye" (temporarily hide in viewport); 'Show in
        # Viewports' stays enabled
        try:
            ob.hide_set(True)
        except Exception:                                  # noqa: BLE001
            pass
        ob.hide_render = True
    return ob


def build_light(instance_data, collection):
    info = instance_data.light
    light = bpy.data.lights.new(name=instance_data.name[:63], type="POINT")
    rng = info.get("range", 40.0) or 40.0
    inten = info.get("intensity", 0.85)
    light.energy = max(50.0, inten * rng * rng * 2.0)
    light.color = info.get("color", (1.0, 1.0, 1.0))
    light.shadow_soft_size = 1.0
    ob = bpy.data.objects.new(instance_data.name[:63], light)
    ob.location = info["pos"]
    collection.objects.link(ob)
    return ob
