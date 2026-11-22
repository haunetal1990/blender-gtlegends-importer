# GT Legends Importer for Blender 4.0+
# Author: haunetal1990
#
# Imports GT Legends (SimBin gMotor2) cars and tracks from .CAR / .CAS /
# .TRK / .SCN / .GTL / .GMT files.  See README_GT-Legends-Importer.md.

bl_info = {
    "name": "GT Legends Importer (.GTL)",
    "author": "haunetal1990",
    "version": (0, 8, 0),
    "blender": (4, 0, 0),
    "location": "File > Import > GT Legends (.GTL)",
    "description": "Import GT Legends cars and tracks (.GTL / .CAS / .TRK / .SCN / .GMT)",
    "category": "Import",
}

import os

import bpy
from bpy.props import (BoolProperty, EnumProperty, StringProperty,
                       CollectionProperty)
from bpy.types import Operator, AddonPreferences
from bpy_extras.io_utils import ImportHelper

from .gtl_core import importer as gtl_importer
from .gtl_bpy import assemble


def _blender_temp_dir():
    # bpy.app.tempdir respects the user preference "Temporary Files" path and
    # works the same in Blender 4.0 and 4.2+.
    try:
        return bpy.app.tempdir
    except Exception:                                        # noqa: BLE001
        return None


def _prefs(context):
    try:
        return context.preferences.addons[__package__].preferences
    except Exception:                                        # noqa: BLE001
        return None


class GTL_PT_preferences(AddonPreferences):
    bl_idname = __package__

    cache_mode: EnumProperty(
        name="Archive cache",
        description="Where extracted .GTL archive contents are stored",
        items=(
            ("TEMP", "Blender temp folder",
             "Uses Blender's own temporary directory (bpy.app.tempdir, "
             "respects the 'Temporary Files' preference). Same mechanism "
             "Blender uses for unpacked files. In 4.2+ extensions this "
             "stays session based."),
            ("BESIDE", "Folder next to the original files",
             "Creates '_gtl_extracted/<archive name>' next to the original "
             ".GTL files (persistent, survives Blender restarts)."),
            ("CUSTOM", "Custom folder",
             "Use the path below as cache root."),
        ),
        default="TEMP",
    )
    cache_dir: StringProperty(
        name="Cache folder",
        description="Used when 'Archive cache' is set to 'Custom folder'",
        subtype="DIR_PATH",
        default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "cache_mode")
        if self.cache_mode == "CUSTOM":
            layout.prop(self, "cache_dir")
        layout.label(text="Blender temp: %s" % (_blender_temp_dir() or "?"))


class IMPORT_OT_gtl(Operator, ImportHelper):
    """Import GT Legends cars or tracks (.CAR driver files, .GTL archives, .CAS, .TRK, .SCN)"""

    bl_idname = "import_scene.gtl"
    bl_label = "GT Legends (.GTL)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".gtl"
    filter_glob: StringProperty(default="*.car;*.gtl;*.cas;*.trk;*.scn;*.gmt", options={"HIDDEN"})

    files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                              options={"HIDDEN"})
    directory: StringProperty(subtype="FILE_PATH", options={"HIDDEN"})

    import_sky: BoolProperty(
        name="Import sky/skybox",
        description="Import the sky dome, sun and moon objects of tracks",
        default=False,
    )
    import_lights: BoolProperty(
        name="Import lights",
        description="Import the night lights of tracks as Blender point lights",
        default=True,
    )
    use_texture_alpha: BoolProperty(
        name="Use texture alpha",
        description="Textures that really contain an alpha channel make the "
                    "material transparent (cut-out or blended). Masks used "
                    "for reflections are ignored",
        default=True,
    )
    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Options")
        box.prop(self, "import_sky")
        box.prop(self, "import_lights")
        box.prop(self, "use_texture_alpha")

    def _cache_mode(self):
        prefs = _prefs(self)
        mode = prefs.cache_mode if prefs else "TEMP"
        if mode == "CUSTOM":
            return "CUSTOM", (prefs.cache_dir if prefs else "")
        if mode == "BESIDE":
            return "BESIDE", ""
        return "TEMP", ""

    def execute(self, context):
        log_lines = []

        def log(msg):
            log_lines.append(msg)
            print("[GTL] " + msg)

        paths = []
        if self.files:
            for f in self.files:
                paths.append(os.path.join(self.directory, f.name))
        else:
            paths.append(self.filepath)

        opts = gtl_importer.ImportOptions()
        opts.import_sky = self.import_sky
        opts.import_lights = self.import_lights
        opts.cache_mode, opts.cache_dir = self._cache_mode()
        opts.use_texture_alpha = self.use_texture_alpha

        results = []

        for path in paths:
            kind_path, kind = gtl_importer.read_first_gtl_or_scene(path)
            opts.driver_file = path if path.lower().endswith(".car") else ""
            imp = gtl_importer.GtlImporter(opts, temp_dir_fn=_blender_temp_dir,
                                           log=log)
            try:
                if kind == "car":
                    result = imp.import_car(kind_path)
                elif kind == "track":
                    result = imp.import_track(kind_path)
                elif kind == "archive":
                    result = self._import_archive_only(path, imp, log)
                else:
                    self.report({"WARNING"},
                                "Unsupported file: %s" % path)
                    continue
            except Exception as exc:                        # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.report({"ERROR"},
                            "Import failed: %s (%s)" % (path, exc))
                continue

            results.append(result)
            assemble.build_scene(result, opts)
            log("'%s': %d instances, %d GMTs, %d missing textures, "
                "%d missing meshes"
                % (result.collection_name, len(result.instances),
                   result.parsed_files, len(result.missing_textures),
                   len(result.missing_meshes)))
            # missing data is reported in the system console only - no
            # on-screen pop-up
            if result.missing_textures:
                log("Missing textures (shown pink): %s"
                    % ", ".join(sorted(result.missing_textures)))
            if result.missing_meshes:
                log("Missing meshes: %s" % "; ".join(result.missing_meshes))

        if not results:
            self.report({"ERROR"}, "No importable data found.")
            return {"CANCELLED"}
        return {"FINISHED"}

    def _import_archive_only(self, path, imp, log):
        """Fallback when neither a .CAS nor a .TRK/.SCN is next to the
        archive: import every GMT found inside."""
        from .gtl_core import masfile
        result = gtl_importer.ImportResult()
        imp.result = result            # _build_mesh_object records missing textures here
        result.is_track = False
        name = os.path.splitext(os.path.basename(path))[0]
        result.collection_name = name + "_raw"
        entries = masfile.read_archive(path)
        gmts = [e for e in entries if e.name.lower().endswith(".gmt")]
        for e in gmts:
            try:
                from .gtl_core.gmtmesh import parse_gmt
                gmt = parse_gmt(e.data)
            except Exception as exc:                    # noqa: BLE001
                log("GMT parse error %s: %s" % (e.name, exc))
                continue
            mesh_obj = imp._build_mesh_object(gmt, path, name, [])
            inst = gtl_importer.InstanceData(gmt.instance_name or name)
            inst.meshes.append(mesh_obj)
            result.instances.append(inst)
        result.parsed_files = len(gmts)
        return result


def menu_func(self, context):
    self.layout.operator(IMPORT_OT_gtl.bl_idname, text="GT Legends (.GTL)")


classes = (
    GTL_PT_preferences,
    IMPORT_OT_gtl,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
