# Asset store: resolves mesh/texture names against
#  - GMOTORMAS archives (.GTL/.MAS containers referenced by MASFile=)
#  - loose directories (SearchPath= entries, mod folder)
# and extracts archives into a configurable cache directory
# (Blender temp folder  OR  next to the original files  OR  custom path).

import os
import struct

import numpy as np

from . import masfile

TEX_EXTS = (".dds", ".tga", ".bmp", ".png", ".jpg")
MESH_EXTS = (".gmt",)
SCENE_EXTS = (".cas", ".trk", ".scn")


class ArchiveRef:
    __slots__ = ("path", "name", "entries", "index")

    def __init__(self, path, name, entries):
        self.path = path
        self.name = name
        self.entries = entries
        self.index = {e.lower_name: e for e in entries}


class AssetStore:
    def __init__(self, cache_dir_provider, log=None):
        # cache_dir_provider(archive_path, archive_name) -> extraction dir
        self.cache_dir_provider = cache_dir_provider
        self.log = log or (lambda msg: None)
        self.archives = []                 # ordered ArchiveRefs (priority order)
        self.archive_by_name = {}
        self.loose_dirs = []               # ordered search dirs
        self._dir_index = {}               # lower basename -> full path

    # ------------------------------------------------------------- archives
    def add_archive(self, path):
        apath = os.path.abspath(path)
        if apath in self.archive_by_name:
            return self.archive_by_name[apath]
        try:
            entries = masfile.read_archive(apath)
        except Exception as exc:                       # noqa: BLE001
            self.log("Archive not readable: %s (%s)" % (apath, exc))
            return None
        ref = ArchiveRef(apath, os.path.basename(apath), entries)
        self.archives.append(ref)
        self.archive_by_name[apath] = ref
        self.log("Archive indexed: %s (%d entries)" % (os.path.basename(apath), len(entries)))
        return ref

    def add_loose_dir(self, path, recursive=True, max_depth=3):
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            return
        if path not in self.loose_dirs:
            self.loose_dirs.append(path)
        if recursive:
            depth = 0
            scan = [path]
            while scan and depth < max_depth:
                nxt = []
                for d in scan:
                    try:
                        with os.scandir(d) as it:
                            for entry in it:
                                if entry.is_dir():
                                    nxt.append(entry.path)
                                elif entry.is_file():
                                    low = entry.name.lower()
                                    if low.endswith(MESH_EXTS + SCENE_EXTS) \
                                            and low not in self._dir_index:
                                        self._dir_index[low] = entry.path
                    except OSError:
                        continue
                scan = nxt
                depth += 1
        else:
            try:
                for entry in os.scandir(path):
                    if entry.is_file():
                        low = entry.name.lower()
                        if low not in self._dir_index:
                            self._dir_index[low] = entry.path
            except OSError:
                pass

    # ------------------------------------------------------------ resolving
    def has(self, name):
        low = name.lower()
        for ref in self.archives:
            if low in ref.index:
                return True
        if low in self._dir_index:
            return True
        return False

    def read_bytes(self, name):
        """(data, source description) of a mesh/scene file straight from the
        archive in memory (no extraction to disk), or from a loose file."""
        low = name.lower()
        ref, entry = self.archive_entry(low)
        if ref is not None:
            return entry.data, "%s/%s" % (ref.name, entry.name)
        path = self._dir_index.get(low)
        if path:
            try:
                with open(path, "rb") as fh:
                    return fh.read(), path
            except OSError:
                pass
        return None, None

    def find_family(self, base):
        """First texture whose name is BASE + digits (+ extension)."""
        import re
        pat = re.compile(r"^%s\d*\.(dds|tga|bmp|png|jpg)$" % re.escape(base.lower()))
        for ref in self.archives:
            for low in sorted(ref.index):
                if pat.match(low):
                    return self.materialize(low)
        for low in sorted(self._dir_index):
            if pat.match(low):
                return self._dir_index[low]
        return None

    def archive_entry(self, name):
        low = name.lower()
        for ref in self.archives:
            if low in ref.index:
                return ref, ref.index[low]
        return None, None

    def materialize(self, name):
        """Return a real file path for 'name' (extracts from an archive into
        the cache if needed) or None when the asset does not exist.
        Textures get an extension fallback: mods often reference
        'NAME.TGA' while the archive ships 'NAME.DDS' (game build step)."""
        low = name.lower()
        ref, entry = self.archive_entry(low)
        if ref is None and low.endswith(TEX_EXTS):
            stem = os.path.splitext(low)[0]
            for ext in (".dds", ".tga", ".bmp", ".png", ".jpg"):
                ref, entry = self.archive_entry(stem + ext)
                if ref is not None:
                    break
        if ref is not None:
            dest_dir = self.cache_dir_provider(ref.path, ref.name)
            if dest_dir is None:
                return None
            out_path = os.path.join(dest_dir, entry.name)
            if not os.path.exists(out_path) or os.path.getsize(out_path) != entry.size:
                os.makedirs(dest_dir, exist_ok=True)
                with open(out_path, "wb") as fh:
                    fh.write(entry.data)
            return out_path
        if low in self._dir_index:
            return self._dir_index[low]
        if low.endswith(TEX_EXTS):
            stem = os.path.splitext(low)[0]
            for ext in (".dds", ".tga", ".bmp", ".png", ".jpg"):
                if stem + ext in self._dir_index:
                    return self._dir_index[stem + ext]
        return None


def file_alpha_info(path):
    """(has_alpha, max_alpha 0..1) of a texture FILE.

    has_alpha: DXT1 with 1-bit transparent texels, DXT3/DXT5 with
    non-opaque alpha, 32-bit TGA/BMP/DDS.  max_alpha is the highest alpha the
    file can produce (only meaningful for DXT3/DXT5; 1.0 otherwise) - used to
    pick the clip threshold for very transparent textures such as glass.
    Pure numpy on the top mip level - never touches Blender's image buffers,
    so no pixel decoding is needed at import time."""
    try:
        with open(path, "rb") as fh:
            d = fh.read()
    except OSError:
        return False, 1.0
    try:
        if d[:4] == b"DDS ":
            fcc = d[84:88]
            if fcc in (b"DXT1", b"DXT3", b"DXT5"):
                h, w = struct.unpack_from("<II", d, 12)
                bsize = 8 if fcc == b"DXT1" else 16
                nb = min(((w + 3) // 4) * ((h + 3) // 4),
                         (len(d) - 128) // bsize)
                if nb <= 0:
                    return False, 1.0
                if fcc == b"DXT1":
                    wd = np.frombuffer(d, "<u4", nb * 2, 128).reshape(nb, 2)
                    c0 = wd[:, 0] & 0xFFFF
                    c1 = wd[:, 0] >> 16
                    idx = wd[:, 1]
                    has = bool(((c0 <= c1)
                                & (((idx & (idx >> 1)) & 0x55555555) != 0)).any())
                    return has, 1.0
                blk = np.frombuffer(d, np.uint8, nb * 16, 128).reshape(nb, 16)
                if fcc == b"DXT3":
                    a = blk[:, :8]
                    if not (a != 255).any():
                        return False, 1.0
                    return True, max(int((a >> 4).max()),
                                     int((a & 15).max())) / 15.0
                a0, a1 = blk[:, 0], blk[:, 1]
                if int(np.minimum(a0, a1).min()) >= 250:
                    return False, 1.0
                return True, int(np.maximum(a0, a1).max()) / 255.0
            flags = struct.unpack_from("<I", d, 80)[0]
            return bool(flags & 0x1), 1.0                # DDPF_ALPHAPIXELS
        if path.lower().endswith(".tga") and len(d) > 18:
            return (d[16] == 32 and (d[17] & 0x0F) > 0), 1.0
        if d[:2] == b"BM" and len(d) > 30:
            return struct.unpack_from("<H", d, 28)[0] == 32, 1.0
    except (struct.error, IndexError, ValueError):
        return False, 1.0
    return False, 1.0


def default_cache_provider(mode, custom_dir, temp_dir_fn):
    """mode: 'TEMP' | 'BESIDE' | 'CUSTOM'.  temp_dir_fn returns Blender's
    temporary directory (bpy.app.tempdir).  Returns a callable
    (archive_path, archive_name) -> directory or None."""

    def provider(archive_path, archive_name):
        stem = os.path.splitext(archive_name)[0]
        if mode == "BESIDE":
            return os.path.join(os.path.dirname(archive_path),
                                "_gtl_extracted", stem)
        if mode == "CUSTOM" and custom_dir:
            return os.path.join(custom_dir, stem)
        base = temp_dir_fn() if temp_dir_fn else None
        if not base:
            return None
        return os.path.join(base, "gtlegends_importer_cache", stem)

    return provider


def find_game_root(start_dir):
    """Walk upwards looking for a GAMEDATA directory (isi game layout)."""
    cur = os.path.abspath(start_dir)
    for _ in range(8):
        for entry in ("gamedata", "GAMEDATA"):
            candidate = os.path.join(cur, entry)
            if os.path.isdir(candidate):
                return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def expand_search_path(search_path, scene_dir, game_root):
    """Turn a SearchPath= entry into an absolute directory candidate list."""
    sp = search_path.replace("\\", "/").strip()
    out = []
    candidates = []
    if sp.startswith("<"):
        # <TEAMDIR>/<VEHDIR> style placeholders: fall back to the scene dir.
        candidates.append(scene_dir)
        candidates.append(os.path.dirname(scene_dir))
    else:
        if game_root:
            candidates.append(os.path.join(game_root, sp))
        candidates.append(os.path.join(scene_dir, sp))
        tail = os.path.basename(sp.rstrip("/"))
        if tail:
            candidates.append(os.path.join(os.path.dirname(scene_dir), tail))
        candidates.append(scene_dir)
    for cand in candidates:
        if cand and os.path.isdir(cand) and cand not in out:
            out.append(cand)
    return out


def find_scene_sources(start_dir):
    """Find .cas (car) / .trk|.scn (track) files near start_dir."""
    cars, tracks = [], []
    cur = os.path.abspath(start_dir)
    for _ in range(3):
        try:
            for entry in os.scandir(cur):
                if not entry.is_file():
                    continue
                low = entry.name.lower()
                if low.endswith(".cas"):
                    cars.append(entry.path)
                elif low.endswith(".trk") or low.endswith(".scn"):
                    tracks.append(entry.path)
        except OSError:
            pass
        cur = os.path.dirname(cur)
    return cars, tracks
