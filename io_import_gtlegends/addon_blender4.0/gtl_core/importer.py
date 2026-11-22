# High level import orchestration.
#
# Produces a neutral ImportResult (vertices already converted to Blender's
# coordinate system) that the bpy layer turns into objects/collections.
# This module is deliberately free of any bpy imports so it can be tested
# headless.

import os
import re

import numpy as np

from . import asftext, assets
from .gmtmesh import parse_gmt

# isi/gMotor2 is a LEFT-handed DirectX-style system: X = left(+), Y = up,
# Z = rear(+).  Blender is right-handed (Z up), so a plain axis permutation
# would MIRROR the car (left <-> right, all lettering mirrored).  The
# conversion therefore negates X as well:
#     Blender = (-X, -Z, Y)       (front = +Y, left side of the car = -X)
# Verified by rendering the Cobra body: the "859" roundel and the Dunlop
# decal read correctly from outside only with this handedness.
# Face winding is taken from the file as-is: the mirror turns the DirectX
# clockwise front into Blender's counter-clockwise front.
def iso_to_blender(v):
    return (-v[0], -v[2], v[1])


def iso_to_blender_array(a):
    """(N, 3) isi -> Blender, same mapping as iso_to_blender()."""
    out = np.empty(a.shape, dtype=a.dtype)
    out[:, 0] = -a[:, 0]
    out[:, 1] = -a[:, 2]
    out[:, 2] = a[:, 1]
    return out


# Face winding is taken EXACTLY from the file (strip parity handled in the
# parser).  The game culls with the DirectX convention (front = clockwise in
# its left-handed space); the handedness-flipping conversion above turns that
# into Blender's counter-clockwise front - 98 % of all faces of Zolder and the
# Cobra then agree with the stored game normals and the rest are foliage /
# wall faces whose vertex normals are simply unreliable.  Flipping faces
# individually against those normals (older versions) inverted most of them.


MESH_NAME_RE = re.compile(r"^(.*?)(<([0-9]+)>)(\.[A-Za-z0-9]+)?$")


LOD_TAGS = ("MAX", "HIGH", "MED", "LOW")
TYPE_TAGS = ("SPIN", "NOTSPIN", "EXPO", "DASHHIGH")
GROUP_ORDER = ("MAX", "HIGH", "MED", "LOW", "COMMON")


def _lod_in(ref):
    """LODIn= (nearest camera distance of a mesh line); 0 when absent."""
    m = re.search(r"[-\d.]+", str(ref.params.get("LODIN", "0")))
    try:
        return float(m.group()) if m else 0.0
    except ValueError:
        return 0.0


def classify_tags(tags):
    """Collection group of a CAS mesh line from its <TAG> prefixes.
    Digits (<1> = GenString slot inside the file name) are not tags.
    LOD tag wins (Max / High / Med / Low), otherwise the type tags are
    combined (SPIN, NOTSPIN_DASHHIGH, EXPO), no tag -> COMMON."""
    tl = [t.upper() for t in tags if t and not t.isdigit()]
    for lod in LOD_TAGS:
        if lod in tl:
            return lod
    types = [t for t in tl if t in TYPE_TAGS] or \
            [t for t in tl if t not in LOD_TAGS]
    return "_".join(types) if types else "COMMON"


# driver meshes are modelled around the origin; the game places them at
# the position given in the driver's *.CAR file
DRIVER_POS_KEYS = {
    "EXTDRIVERBODY": "ExternalDriverPosition",
    "EXTDRIVERHEAD": "ExternalDriverPosition",
    "FPDRIVER": "FirstPersonDriverPosition",
}


class ImportOptions:
    def __init__(self):
        self.import_sky = False
        self.import_lights = True
        self.cache_mode = "TEMP"
        self.cache_dir = ""
        self.use_texture_alpha = True
        self.driver_file = ""         # explicit *.CAR chosen by the user


# isi wildcard paint materials (rFactor/GTL/GTR2 family): the game paints
# these per driver - wccarbody, wcwing, wcwindow, wcextra0..9, ...
WC_MATERIAL_RE = re.compile(r"^wc[a-z0-9_]{2,}$")


def read_driver_file(path):
    """Parse one driver/vehicle file (*.CAR): GenString, Number, Team,
    Graphics=, steering data and optional wildcard-paint assignments."""
    info = {"path": path, "number": "", "team": "", "driver": "",
            "genstring": "", "graphics": "", "wc_tex": {}, "extras": {}}
    try:
        with open(path, "r", encoding="latin-1", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return info

    def grab(key):
        m = re.search(r"%s\s*=\s*\(([^)]*)\)" % key, text, re.IGNORECASE)
        if m:
            parts = [p.strip() for p in m.group(1).split(",")]
            try:
                return [float(p) for p in parts]
            except ValueError:
                return None
        m = re.search(r"%s\s*=\s*([-\d.eE+]+)" % key, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    def grab_str(key):
        m = re.search(r'%s\s*=\s*"?([^"\r\n/]+)"?' % key, text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    info["genstring"] = grab_str("GenString")
    info["graphics"] = grab_str("Graphics")
    info["number"] = grab_str("Number")
    info["team"] = grab_str("Team")
    info["driver"] = grab_str("Driver") or grab_str("Driver1")

    # wildcard paint assignments:
    #   Texture=wccarbody=MyPaint.dds      (rF style)
    #   WcCarBody=MyPaint.dds              (direct key style)
    for line in text.splitlines():
        line = line.split("//")[0].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().lower()
        val = val.strip().strip('"')
        if not val:
            continue
        if key.startswith("texture") and "=" in val:
            mat, _, tex = val.partition("=")
            mat = mat.strip().lower()
            if WC_MATERIAL_RE.match(mat) and tex.strip():
                info["wc_tex"][mat] = tex.strip()
        elif WC_MATERIAL_RE.match(key):
            info["wc_tex"][key] = val

    info["extras"] = {
        "Eyepoint": grab("Eyepoint"),
        "CameraTarget": grab("CameraTarget"),
        "SteeringWheelAxis": grab("SteeringWheelAxis"),
        "SteeringWheelType": grab("SteeringWheelType"),
        "FirstPersonDriverPosition": grab("FirstPersonDriverPosition"),
        "ExternalDriverPosition": grab("ExternalDriverPosition"),
        "ShiftStickPosition": grab("ShiftStickPosition"),
    }
    return info


class MeshObjectData:
    """One Blender object (one GMT file).  Vertex data are numpy arrays,
    already in Blender's coordinate system."""

    def __init__(self, name, instance_name, tags):
        self.name = name
        self.instance_name = instance_name
        self.tags = tags
        self.verts = None          # (N, 3) float64, blender coords (global)
        self.normals = None        # (N, 3) float32
        self.uvs = None            # (N, 2) float32
        self.colors = None         # (N, 4) float32, RGBA
        self.tangents = None       # (N, 3) float32
        self.binormals = None      # (N, 3) float32
        self.uvs_light = None      # (N, 2) float32 or None (lightmap uv)
        self.tangents_stored = False   # True: game tangent stream was usable
        self.faces = None          # (M, 3) int32
        self.face_mats = None      # (M,) int32 material index per face
        self.materials = []        # list of MatSpec
        self.pivot = None          # blender coords or None
        self.group = "COMMON"      # LOD / type collection name (cars)
        self.hidden = False        # Render=False -> object hidden
        self.custom_props = {}
        self.source = ""


class MatSpec:
    __slots__ = ("name", "texture", "alpha",
                 "found_path", "alpha_mask", "tex_alpha", "alpha_max")

    def __init__(self, name, texture, alpha):
        self.name = name
        self.texture = texture
        self.alpha = alpha
        self.found_path = None
        self.alpha_mask = False     # texture alpha is a reflection mask
        self.tex_alpha = False      # texture file has real transparency
        self.alpha_max = 1.0        # highest alpha value in the texture


class InstanceData:
    def __init__(self, name):
        self.name = name
        self.meshes = []            # MeshObjectData
        self.light = None           # (pos_bl, color, range, type)


class ImportResult:
    def __init__(self):
        self.collection_name = ""
        self.instances = []
        self.missing_textures = set()
        self.missing_meshes = []
        self.parsed_files = 0
        self.info_lines = []
        self.is_track = False
        self.groups = []             # collection groups (cars)
        self.driver_info = {}        # gtl_number/team/driver/genstring/...

    def add_info(self, msg):
        self.info_lines.append(msg)


class GtlImporter:
    def __init__(self, options: ImportOptions, temp_dir_fn=None, log=None):
        self.opts = options
        self.log = log or (lambda m: None)
        provider = assets.default_cache_provider(
            options.cache_mode, options.cache_dir, temp_dir_fn)
        self.store = assets.AssetStore(provider, log=self.log)
        self.gen_string = ""
        self.car_id = ""
        self.car_extras = {}
        self.drivers = []            # parsed driver (*.CAR) files
        self.wc_tex_map = {}         # wc material -> texture from driver file
        self._mas_walk_cache = None  # one recursive scan per import, reused
        self._search_cache = None    # MASFile search folders
        self._scan_cache = {}        # folder -> (files, sub folders)

    # ------------------------------------------------------------- helpers
    def _log(self, msg):
        self.log(msg)

    def _resolve_mesh_name(self, mesh_file):
        """Resolve a MeshFile= reference (may contain <n> slots)."""
        base, ext = os.path.splitext(mesh_file)
        ext = ext or ".gmt"
        m = MESH_NAME_RE.match(os.path.basename(mesh_file))
        if not m:
            if self.store.has(mesh_file):
                return mesh_file
            return None
        stem, _slot, digits, _e = m.group(1), m.group(2), m.group(3), m.group(4)
        cands = asftext.placeholder_candidates(self.gen_string, digits)
        for cand in cands:
            name = "%s%s%s" % (stem, cand, ext)
            if self.store.has(name):
                return name
        # also try without any slot
        plain = stem + ext
        if self.store.has(plain):
            return plain
        return None

    def _load_gmt(self, resolved_name):
        """Parse a GMT straight from its archive (no extraction to disk)."""
        data, source = self.store.read_bytes(resolved_name)
        if data is None:
            return None, None
        try:
            return parse_gmt(data), source
        except Exception as exc:                          # noqa: BLE001
            self._log("GMT parse error %s: %s" % (resolved_name, exc))
            return None, None

    def _discover_driver_files(self, scene_dir):
        """All *.CAR driver/vehicle files under the mod folder (depth <= 2)."""
        out = []
        base = os.path.abspath(scene_dir)
        for root, _dirs, files in os.walk(base):
            if root[len(base):].count(os.sep) > 2:
                continue
            for f in files:
                if f.lower().endswith(".car"):
                    out.append(os.path.join(root, f))
        return sorted(out)

    def _select_driver(self, drivers):
        """First driver file (lowest number) - or the .CAR the user picked."""
        def sort_key(d):
            num = d["number"]
            return (0 if num.isdigit() else 1,
                    int(num) if num.isdigit() else 10 ** 9,
                    d["path"])
        return sorted(drivers, key=sort_key)[0]

    def _read_car_extras(self, scene_dir):
        """Read the driver (*.CAR) files: GenString, Number, Team, Graphics=
        and SteeringWheelAxis / SteerLock.  The selected driver defines the
        paint job (see WC_MATERIAL_RE) and the GenString substitution."""
        drivers = []
        paths = self._discover_driver_files(scene_dir)
        explicit = getattr(self.opts, "driver_file", "")
        if explicit and os.path.isfile(explicit) and not any(
                os.path.abspath(p) == os.path.abspath(explicit) for p in paths):
            paths.append(explicit)
        for p in paths:
            drivers.append(read_driver_file(p))
        drivers = [d for d in drivers if d]
        self.drivers = drivers

        if not drivers:
            self._log("No driver file (.CAR) found - using generic textures")
            self.gen_string = ""
            self.car_id = ""
            self.car_extras = {}
            self.wc_tex_map = {}
            self.result.driver_info = {}
            return

        chosen = self._select_driver(drivers)
        explicit = getattr(self.opts, "driver_file", "")
        if explicit:
            ex = os.path.abspath(explicit)
            for d in drivers:
                if os.path.abspath(d["path"]) == ex:
                    chosen = d
                    break
        self.gen_string = chosen["genstring"]
        self.car_id = chosen["number"] or \
            os.path.splitext(os.path.basename(chosen["path"]))[0].split("_")[0]
        self.car_extras = dict(chosen["extras"])
        self.wc_tex_map = dict(chosen["wc_tex"])
        self.result.driver_info = {
            "gtl_number": chosen["number"] or "-",
            "gtl_team": chosen["team"] or "-",
            "gtl_driver": chosen["driver"] or "-",
            "gtl_genstring": chosen["genstring"] or "-",
            "gtl_driver_file": os.path.basename(chosen["path"]),
        }

        labels = ", ".join(
            "%s (%s)" % (d["number"] or "?", d["team"] or d["driver"] or "?")
            for d in drivers[:6])
        self._log("Driver file: %s (Number=%s, Team=%s, GenString=%s)%s"
                  % (os.path.basename(chosen["path"]),
                     chosen["number"] or "?", chosen["team"] or "?",
                     chosen["genstring"] or "?",
                     (" | available: " + labels) if len(drivers) > 1 else ""))

        # SteerLock aus .hdc (im Fahrzeugordner)
        hdc_path = None
        try:
            for entry in os.scandir(scene_dir):
                if entry.is_file() and entry.name.lower().endswith(".hdc"):
                    hdc_path = hdc_path or entry.path
        except OSError:
            pass
        if hdc_path:
            try:
                with open(hdc_path, "r", encoding="latin-1",
                          errors="replace") as fh:
                    hdc = fh.read()
                m = re.search(r"SteerLockRange\s*=\s*\(([^)]*)\)", hdc,
                              re.IGNORECASE)
                if m:
                    parts = [p.strip() for p in m.group(1).split(",")]
                    try:
                        self.car_extras["SteerLockRange"] = \
                            [float(p) for p in parts]
                    except ValueError:
                        pass
                m = re.search(r"SteerLockSetting\s*=\s*([-\d.eE+]+)", hdc,
                              re.IGNORECASE)
                if m:
                    try:
                        self.car_extras["SteerLockSetting"] = \
                            float(m.group(1))
                    except ValueError:
                        pass
            except OSError:
                pass

    def _build_mesh_object(self, gmt, path, instance_name, tags):
        obj = MeshObjectData(gmt.instance_name or
                             os.path.splitext(os.path.basename(path))[0],
                             instance_name, tags)
        subs = [sb for sb in gmt.submeshes if len(sb.positions)]
        any_stored = False
        any_light = False
        vbase = 0
        V, N, C, T, Bn, UV, UL, F, FM = [], [], [], [], [], [], [], [], []
        for sb in subs:
            V.append(sb.positions)
            N.append(sb.normals)
            C.append(sb.colors)
            T.append(sb.tangents)
            Bn.append(sb.binormals)
            UV.append(sb.uvs)
            any_stored = any_stored or sb.tangents_stored
            if (sb.uvs_light is not None
                    and np.abs(sb.uvs - sb.uvs_light).max() > 1e-5):
                # real lightmap UVs (differ from the diffuse set)
                UL.append(sb.uvs_light)
                any_light = True
            else:
                UL.append(sb.uvs)
            if len(sb.faces):
                F.append(sb.faces + np.int32(vbase))
                FM.append(np.full(len(sb.faces), sb.material_index, np.int32))
            vbase += len(sb.positions)

        def cat(parts, cols, dtype):
            return np.concatenate(parts) if parts else np.zeros((0, cols), dtype)

        obj.verts = iso_to_blender_array(cat(V, 3, np.float32).astype(np.float64))
        obj.normals = iso_to_blender_array(cat(N, 3, np.float32))
        obj.tangents = iso_to_blender_array(cat(T, 3, np.float32))
        obj.binormals = iso_to_blender_array(cat(Bn, 3, np.float32))
        obj.colors = cat(C, 4, np.float32)
        obj.uvs = cat(UV, 2, np.float32)
        obj.uvs_light = cat(UL, 2, np.float32) if any_light else None
        obj.faces = cat(F, 3, np.int32)
        obj.face_mats = (np.concatenate(FM) if FM
                         else np.zeros(0, np.int32))
        obj.tangents_stored = any_stored

        if gmt.pivot is not None:
            # local geometry -> translate into place, keep origin on the pivot
            px, py, pz = iso_to_blender(gmt.pivot)
            obj.verts = obj.verts + np.array([px, py, pz])
            obj.pivot = (px, py, pz)
            obj.custom_props["gtl_pivot"] = "%.4f, %.4f, %.4f" % (px, py, pz)

        obj.source = path
        obj.custom_props["gtl_tangents"] = \
            "game" if obj.tangents_stored else "analytic"
        number = self.car_id if (self.car_id or "").isdigit() else ""
        for mi, mat in enumerate(gmt.materials):
            tex = mat.diffuse_texture or ""
            name_low = (mat.name or "").lower()
            if tex and WC_MATERIAL_RE.match(name_low):
                # isi wildcard paint material (wccarbody, wcwing, ...):
                # 1. explicit assignment from the driver file
                override = self.wc_tex_map.get(name_low)
                if override:
                    tex = override
                else:
                    # 2. driver-numbered version of the GMT texture
                    #    (GENERIC placeholders are stripped first)
                    stem, ext = os.path.splitext(tex)
                    if stem.lower().startswith("generic"):
                        tex = stem[7:] + ext
                    hit = self.store.materialize("%s_%s" % (number, tex)) \
                        if number else None
                    if hit:
                        tex = "%s_%s" % (number, tex)
            spec = MatSpec(mat.name or "%s_mat%d" % (obj.name, mi), tex,
                           mat.has_alpha)
            # CMAPADDALPHAREFLECT & co use the alpha as reflection amount
            spec.alpha_mask = any("ALPHAREFLECT" in l.upper()
                                  for l in mat.layers)
            if tex:
                tex_path = self._resolve_texture(tex)
                if tex_path:
                    spec.found_path = tex_path
                    spec.tex_alpha, spec.alpha_max = self._tex_alpha(tex_path)
                else:
                    self.result.missing_textures.add(tex)
            obj.materials.append(spec)
        return obj

    def _tex_alpha(self, path):
        """(has_alpha, max_alpha) of a texture file, cached per file."""
        cache = self.__dict__.setdefault("_alpha_cache", {})
        info = cache.get(path)
        if info is None:
            info = cache[path] = assets.file_alpha_info(path)
        return info

    def _resolve_texture(self, tex):
        """Memoised texture lookup (the same names come back for every
        mesh of a track; the fallback chain is only walked once)."""
        cache = self.__dict__.setdefault("_tex_cache", {})
        if tex not in cache:
            cache[tex] = self._resolve_texture_uncached(tex)
        return cache[tex]

    def _resolve_texture_uncached(self, tex):
        """Texture lookup with the game's fallback rules:
        1. exact name
        2. GenString placeholder expansion ('NAME<1>.TGA')
        3. driver paint upgrade: 'X.TGA' -> '<Number>_X.DDS' and
           'GENERICX.DDS' -> '<Number>_X.DDS' (team liveries from the
           driver folder, e.g. 859_AC_COBRA_BODY1.DDS)
        4. same basename with another texture extension (GMT says .TGA,
           archive ships .DDS)
        5. without vehicle-number prefix ('859_AC_X.BMP' -> 'AC_X.DDS')
        6. basename without animation frame digits ('LED_RED1.BMP' -> 'LED_RED.DDS')"""
        path = self.store.materialize(tex)
        stem, ext = os.path.splitext(tex)
        exts = (".dds", ".tga", ".bmp", ".png", ".jpg")
        number = self.car_id if (self.car_id or "").isdigit() else ""

        # GENERIC* textures are placeholders for the driver's own paint:
        # try '<Number>_<Name>' FIRST (840_FORD_GT40_BODY1 instead of
        # the 64x64 GENERICFORD_GT40_BODY1 placeholder).
        if number and stem.lower().startswith("generic"):
            alt = stem[7:]
            cands = ([ext] if ext.lower() in exts else []) + list(exts)
            for e in cands:
                path = self.store.materialize("%s_%s%s" % (number, alt, e))
                if path:
                    return path

        if path:
            return path

        # GenString placeholder expansion for texture names
        if "<" in stem:
            m = MESH_NAME_RE.match(os.path.basename(tex))
            if m:
                digits = m.group(3)
                for cand in asftext.placeholder_candidates(self.gen_string,
                                                          digits):
                    expanded = re.sub(r"<\d+>", cand, stem)
                    cands = ([ext] if ext.lower() in exts else []) + list(exts)
                    for e in cands:
                        path = self.store.materialize(expanded + e)
                        if path:
                            return path

        # driver paint upgrade (GENERICX -> <Number>_X, X -> <Number>_X)
        if number:
            alt = stem[7:] if stem.lower().startswith("generic") else stem
            cands = ([ext] if ext.lower() in exts else []) + list(exts)
            for e in cands:
                path = self.store.materialize("%s_%s%s" % (number, alt, e))
                if path:
                    return path

        for e in exts:
            path = self.store.materialize(stem + e)
            if path:
                return path
        m = re.match(r"^(\d+)_(.*)$", stem)
        if m:
            path = self._resolve_texture(m.group(2) + ext)
            if path:
                return path
        m2 = re.match(r"^(.*?)(\d+)$", stem)
        if m2 and m2.group(1):
            base, digits = m2.group(1), m2.group(2)
            # animation frames: LED_RED1 -> LED_RED01 / LED_RED001
            n = int(digits)
            for width in (2, 3, 4):
                if len(digits) < width:
                    path = self._resolve_texture(
                        "%s%0*d%s" % (base, width, n, ext))
                    if path:
                        return path
            path = self._resolve_texture(base + ext)
            if path:
                return path
            # any frame of the same family (BFA0 -> BFA1 ...)
            path = self.store.find_family(base)
            if path:
                return path
        return None

    # ------------------------------------------------------------- car flow
    def import_car(self, cas_path):
        self.result = ImportResult()
        scene_dir = os.path.dirname(os.path.abspath(cas_path))
        cas_name = os.path.splitext(os.path.basename(cas_path))[0]
        self.result.collection_name = cas_name

        doc = asftext.parse_asf_file(cas_path)
        self._register_scene_assets(doc, scene_dir)
        self._read_car_extras(scene_dir)

        slot_id = self.car_id or "0"

        # --- 1. collect every mesh line of the CAS ---------------------------
        # The game loads ONE detail level and then switches between several
        # meshes by camera distance (LODIn/LODOut).  Per instance and tag
        # only the nearest mesh (smallest LODIn - the full detail one) is a
        # model; the others are its distance LODs and are not imported.
        cands = []                      # (inst_name, ref, group, lod_in, order)
        for blk in doc.blocks_by_kind("INSTANCE"):
            inst_name = blk.name.replace("<ID>", slot_id)
            for ref in blk.mesh_refs:
                if ref.is_shadow:
                    continue
                base_no_ext = ref.mesh_file.split("<")[0]
                stem_l = (base_no_ext.rsplit(".", 1)[0] if "." in base_no_ext else base_no_ext)
                if stem_l.upper().endswith("_EXPO") and not any(
                        t in stem_l.upper() for t in ("WHEEL", "TIRE", "SPINDLE")):
                    continue  # the game's reflection demo quad, not a car part
                cands.append((inst_name, ref, classify_tags(ref.tags),
                              _lod_in(ref), len(cands)))
        nearest = {}
        for inst_name, ref, group, lod_in, _o in cands:
            k = (inst_name, group)
            nearest[k] = min(nearest.get(k, lod_in), lod_in)
        cands = [c for c in cands if c[3] <= nearest[(c[0], c[2])]]

        # --- 2. resolve, then keep each model exactly ONCE --------------------
        # priority: untagged, Max, High, Med, Low, then the type tags
        def rank(group):
            return GROUP_ORDER.index(group) if group in GROUP_ORDER \
                else len(GROUP_ORDER)
        chosen = {}                     # resolved mesh -> candidate
        for cand in sorted(cands, key=lambda c: (rank(c[2]), c[4])):
            inst_name, ref, group, _l, _o = cand
            resolved = self._resolve_mesh_name(ref.mesh_file)
            if not resolved:
                miss = "%s: %s" % (inst_name, ref.mesh_file)
                if miss not in self.result.missing_meshes:
                    self.result.missing_meshes.append(miss)
                continue
            chosen.setdefault(resolved.lower(), (resolved, cand))

        # --- 3. build in CAS order ---------------------------------------------
        insts = {}
        for resolved, (inst_name, ref, group, _l, order) in sorted(
                chosen.values(), key=lambda v: v[1][4]):
            gmt, path = self._load_gmt(resolved)
            if not gmt:
                continue
            mesh_obj = self._build_mesh_object(gmt, path, inst_name, ref.tags)
            mesh_obj.group = group
            mesh_obj.hidden = ref.render_false
            mesh_obj.custom_props["gtl_tags"] = \
                ",".join(t for t in ref.tags if not t.isdigit()) or "-"
            mesh_obj.custom_props["gtl_group"] = group
            lodout = ref.params.get("LODOUT", "")
            if lodout:
                mesh_obj.custom_props["gtl_lodout"] = lodout
            self._place_driver_mesh(inst_name, mesh_obj)
            inst = insts.get(inst_name)
            if inst is None:
                inst = insts[inst_name] = InstanceData(inst_name)
                self.result.instances.append(inst)
            inst.meshes.append(mesh_obj)

        self.result.parsed_files = len(chosen)
        groups = set(m.group for i in self.result.instances for m in i.meshes)
        self.result.groups = [g for g in GROUP_ORDER if g in groups] + \
            sorted(g for g in groups if g not in GROUP_ORDER)
        self._add_car_empties()
        return self.result

    def _place_driver_mesh(self, inst_name, mesh_obj):
        """Move the driver meshes (EXTDRIVERBODY/HEAD, FPDRIVER) to the
        position from the driver's *.CAR file."""
        key = DRIVER_POS_KEYS.get(inst_name.upper())
        if not key:
            return
        pos = self.car_extras.get(key)
        if not pos or len(pos) != 3:
            self._log("No %s in the driver file - %s stays at the origin"
                      % (key, inst_name))
            return
        px, py, pz = iso_to_blender(tuple(pos))
        ox, oy, oz = mesh_obj.pivot or (0.0, 0.0, 0.0)
        mesh_obj.verts = mesh_obj.verts - np.array([ox, oy, oz]) \
            + np.array([px, py, pz])
        mesh_obj.pivot = (px, py, pz)
        mesh_obj.custom_props["gtl_pivot"] = "%.4f, %.4f, %.4f" % (px, py, pz)
        mesh_obj.custom_props["gtl_placed_by"] = key

    def _add_car_empties(self):
        """Attach car level info (steering wheel axis etc.) to the result."""
        sw_axis = self.car_extras.get("SteeringWheelAxis")
        if sw_axis and len(sw_axis) == 3:
            bl_axis = iso_to_blender(tuple(sw_axis))
            self.result.add_info(
                "SteeringWheelAxis (Blender): %.3f, %.3f, %.3f" % bl_axis)
            for inst in self.result.instances:
                if inst.name.upper().startswith("WHEEL"):
                    for mesh in inst.meshes:
                        mesh.custom_props["gtl_steering_axis"] = "%.4f, %.4f, %.4f" % bl_axis
                        lock = self.car_extras.get("SteerLockRange")
                        if lock:
                            mesh.custom_props["gtl_steerlock"] = \
                                "min=%.1f step=%.1f max=%.1f" % tuple(lock[:3])

    # ------------------------------------------------------------ track flow
    def import_track(self, scene_path):
        self.result = ImportResult()
        self.result.is_track = True
        scene_dir = os.path.dirname(os.path.abspath(scene_path))
        name = os.path.splitext(os.path.basename(scene_path))[0]
        self.result.collection_name = name

        doc = asftext.parse_asf_file(scene_path)
        self._register_scene_assets(doc, scene_dir)
        self._register_track_shared_assets(scene_dir, doc)

        seen = {}
        for blk in doc.blocks_by_kind("INSTANCE"):
            refs = [r for r in blk.mesh_refs if not r.is_shadow]
            inst_hidden = str(blk.params.get("RENDER", "")).split(" ")[0] \
                .lower() == "false"
            if not refs:
                continue
            if not self.opts.import_sky and blk.name.lower() in (
                    "skyboxi", "day", "night", "sunrise", "sunset", "sun", "moon",
                    "sky", "xsky", "skyi"):
                continue
            inst = InstanceData(blk.name)
            vis = blk.params.get("VISGROUPS", "")
            for ref in refs:
                resolved = self._resolve_mesh_name(ref.mesh_file)
                if not resolved:
                    miss = "%s: %s" % (inst.name, ref.mesh_file)
                    if miss not in self.result.missing_meshes:
                        self.result.missing_meshes.append(miss)
                    continue
                gmt, path = self._load_gmt(resolved)
                if not gmt:
                    continue
                # a mesh is imported once; the .trk may list it twice
                key = resolved.lower()
                hidden = inst_hidden or ref.render_false
                if key in seen:
                    if not hidden:
                        seen[key].hidden = False     # a visible listing wins
                    continue
                mesh_obj = self._build_mesh_object(gmt, path, inst.name, ref.tags)
                seen[key] = mesh_obj
                mesh_obj.hidden = hidden
                if vis:
                    mesh_obj.custom_props["gtl_visgroups"] = vis.strip("()")
                inst.meshes.append(mesh_obj)
            if inst.meshes:
                self.result.instances.append(inst)

        if self.opts.import_lights:
            for blk in doc.blocks_by_kind("LIGHT"):
                pos = asftext.parse_vector(blk.params.get("POS"))
                if not pos:
                    continue
                color = asftext.parse_vector(blk.params.get("COLOR")) or (255, 255, 255)
                rng = asftext.parse_vector(blk.params.get("RANGE")) or (0.0, 40.0)
                inst = InstanceData(blk.name)
                inst.light = {
                    "pos": iso_to_blender(tuple(pos)),
                    "color": tuple(c / 255.0 for c in color[:3]),
                    "range": rng[1] if len(rng) > 1 else 40.0,
                    "type": str(blk.params.get("TYPE", "Omni")),
                    "intensity": float(asftext.parse_vector(blk.params.get("INTENSITY"))[0])
                    if asftext.parse_vector(blk.params.get("INTENSITY")) else 0.85,
                }
                self.result.instances.append(inst)

        self.result.parsed_files = len(seen)
        return self.result

    # ------------------------------------------------------------- assets
    def _register_scene_assets(self, doc, scene_dir):
        game_root = assets.find_game_root(scene_dir)
        loose = []
        for sp in doc.search_paths:
            loose.extend(assets.expand_search_path(sp, scene_dir, game_root))
        loose.extend([scene_dir, os.path.dirname(scene_dir)])
        for d in loose:
            self.store.add_loose_dir(d, recursive=False)

        for mf in doc.mas_files:
            path = self._find_masfile(mf, scene_dir, game_root, doc)
            if path:
                self._log("MASFile %s -> %s" % (mf, path))
                if not self.store.add_archive(path):
                    self._log("MASFile could not be read: %s" % path)
            else:
                self._log("MASFile NOT FOUND (textures will be missing): %s "
                          "- searched %s and %d parent folders"
                          % (mf, scene_dir, len(self._ancestors(scene_dir)) - 1))

        if not self.result.is_track:
            # driver folders carry the team liveries (<Number>_*.GTL)
            self._register_driver_archives(scene_dir)

        # Recursive fallback inside the mod folder so "everything available"
        # really means everything.
        self.store.add_loose_dir(scene_dir, recursive=True, max_depth=3)

    def _register_driver_archives(self, scene_dir):
        """Index .GTL/.MAS archives from the mod tree (driver folders) so the
        team textures (<Number>_...) resolve.  ULTRA variants first so the
        highest resolution textures win."""
        found = []
        base = os.path.abspath(scene_dir)
        for root, _dirs, files in os.walk(base):
            if root[len(base):].count(os.sep) > 2:
                continue
            for f in files:
                if f.lower().endswith((".gtl", ".mas")):
                    p = os.path.join(root, f)
                    if p not in self.store.archive_by_name:
                        found.append(p)
        found.sort(key=lambda p: (
            0 if "ultra" in os.path.basename(p).lower() else 1, p))
        for p in found:
            self.store.add_archive(p)

    def _register_track_shared_assets(self, scene_dir, doc):
        # Tracks: also index sibling archives of the scene folder once.
        try:
            for entry in os.scandir(scene_dir):
                if entry.is_file() and entry.name.lower().endswith((".gtl", ".mas")):
                    self.store.add_archive(entry.path)
        except OSError:
            pass

    def _ancestors(self, scene_dir, levels=8):
        out = []
        cur = os.path.abspath(scene_dir)
        for _ in range(levels):
            out.append(cur)
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return out

    def _search_dirs(self, scene_dir, doc):
        """Ordered folder candidates for MASFile=: SearchPath entries, the
        scene folder and EVERY ancestor folder (the shared STEX/TALENT
        archives live in 'Teams' above the car folder, but the exact nesting
        differs between installs).  Computed once per import."""
        if self._search_cache is None:
            game_root = assets.find_game_root(scene_dir)
            cand = []
            for sp in doc.search_paths:
                cand.extend(assets.expand_search_path(sp, scene_dir, game_root))
            cand.extend(self._ancestors(scene_dir))
            out = []
            for d in cand:
                d = os.path.abspath(d)
                if d not in out and os.path.isdir(d):
                    out.append(d)
            self._search_cache = out
        return self._search_cache

    def _scan(self, d):
        """(files {lower name: real name}, sub folders [path]) of a folder -
        ONE scandir per folder for the whole import."""
        hit = self._scan_cache.get(d)
        if hit is None:
            files, subs = {}, []
            try:
                with os.scandir(d) as it:
                    for e in it:
                        try:
                            if "." not in e.name and e.is_dir():
                                subs.append(e.path)
                            else:
                                files[e.name.lower()] = e.name
                        except OSError:
                            pass
            except OSError:
                pass
            hit = self._scan_cache[d] = (files, subs)
        return hit

    def _find_masfile(self, name, scene_dir, game_root, doc):
        low = name.lower()
        for d in self._search_dirs(scene_dir, doc):
            real = self._scan(d)[0].get(low)
            if real:
                return os.path.join(d, real)
        # sibling folders of the scene folder and of its parent (shared
        # archive folders such as Locations\Shared next to the track)
        cur = os.path.abspath(scene_dir)
        for _ in range(2):
            cur = os.path.dirname(cur)
            for sd in self._scan(cur)[1][:200]:
                cand = os.path.join(sd, name)       # one stat, no listing
                if os.path.isfile(cand):
                    return cand
        # recursive last resort under the scene dir - ONE walk for all lookups
        if self._mas_walk_cache is None:
            idx = {}
            for root, _dirs, files in os.walk(scene_dir):
                for f in files:
                    fl = f.lower()
                    if fl.endswith((".gtl", ".mas")) and fl not in idx:
                        idx[fl] = os.path.join(root, f)
            self._mas_walk_cache = idx
        return self._mas_walk_cache.get(low)


def read_first_gtl_or_scene(path):
    """If the user picked a .GTL container, look for the matching scene file
    (.cas / .trk / .scn) next to it; otherwise hand the path back.
    For cars the Graphics= entry of the selected driver file decides which
    .cas variant is used (e.g. AC_Cobra.cas vs AC_Cobra_2DW.cas)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".car":
        # driver file: Graphics= names the .cas, found next to the .car or in
        # a parent folder
        d = read_driver_file(path)
        want = (d["graphics"] or "").strip().lower()
        cur = os.path.dirname(os.path.abspath(path))
        for _ in range(6):
            try:
                for f in os.listdir(cur):
                    if f.lower().endswith(".cas") and (not want or f.lower() == want):
                        return os.path.join(cur, f), "car"
            except OSError:
                pass
            nxt = os.path.dirname(cur)
            if nxt == cur:
                break
            cur = nxt
        return path, "unknown"
    if ext in (".cas", ".trk", ".scn"):
        return path, ("car" if ext == ".cas" else "track")
    if ext == ".gtl" or ext == ".mas":
        directory = os.path.dirname(os.path.abspath(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        cars, tracks = assets.find_scene_sources(directory)
        # driver preference: Graphics= of the chosen driver file
        graphics = ""
        base = directory
        for root, _dirs, files in os.walk(base):
            if root[len(base):].count(os.sep) > 2:
                continue
            for f in files:
                if f.lower().endswith(".car"):
                    d = read_driver_file(os.path.join(root, f))
                    if d["graphics"]:
                        graphics = d["graphics"]
                        break
            if graphics:
                break
        if graphics:
            for c in cars:
                if os.path.basename(c).lower() == graphics.lower():
                    return c, "car"
        if tracks:
            return tracks[0], "track"
        if cars:
            return cars[0], "car"
        # no scene file: import the archive contents directly
        return path, "archive"
    return path, "unknown"
