# SimBin binary GMT mesh parser (GT Legends / GTR2 / Race07).
#
# Reverse engineered layout (verified against original GT Legends data):
#
#   0x00  u32   marker -> material section starts at this value + 4
#   0x04  u32   flags (bit 0x04000000 seen on meshes with 4 UV channels)
#   0x08  16B   constant GUID/hash (c8 cf d2 d8 ce d8 e6 c3 d6 de d8 be bb a7 bf 90)
#   0x18  508B  preamble: bounding box corners, transform leftovers,
#               and for animated parts (steering wheel, driver) an instance
#               pivot stored as (px, py, pz, 1.0, 0.0, px, py, pz)
#   0x1FC ..     submesh headers, 108 bytes each:
#          u32   magic   low word 0x4011/0x4015/0x4035,
#                        high word 0x0000 = triangle strip, 0x2000 = list
#          +0x04 u32 submesh id (== material index)
#          +0x08 u32 vertex count
#          +0x0C u32 vertex buffer offset (absolute)
#          +0x14 u32 uv stream offset (absolute)
#          +0x24 u32 triangle count
#          +0x28 u32 index buffer offset (absolute)
#   vertex buffer: 32 bytes per vertex
#          +0  float pad (0.0/NaN)      +4  pos X   +8  pos Y  +12 pos Z
#          +16 normal X  +20 normal Y  +24 normal Z   +28 DIFFUSE COLOR BGRA
#          (verified: streetlight glows carry yellow, marshal posts red and
#           headlight lenses warm-orange vertex colors -> DirectX BGRA order;
#           alpha is 0xFF on every vertex found so far)
#   uv stream (right after the vertex buffer):
#          u32 header (observed 0 on every file so far), then 32 bytes per
#          vertex = four float2 UV sets:  set0 @ +0, set1 @ +8, set2 @ +16,
#          set3 @ +24.  SET 0 IS THE DIFFUSE MAP (materials use stage T0);
#          set3 always duplicates set0.
#          V CONVENTION (verified against the 3dsimed 2 / game logic and by
#          rendering the original Cobra data):
#          The GMT stores UVs in DirectX convention - v = 0 is the TOP row of
#          the texture, exactly like the game samples them.  Many files keep
#          V in [-1, 0] (e.g. the whole Cobra body/cockpit); with REPEAT
#          addressing this samples the same texels as v + 1.  Blender's image
#          space is bottom-up, so EVERY uv set of EVERY mesh is converted with
#          v' = 1 - v on import (no per-material exceptions: body, cockpit,
#          wheels, suspension, windows, track objects all use the same rule).
#          Verified: the Cobra body UV islands (hood number, side roundels,
#          wheel arches) and the cockpit dial islands only land on the matching
#          texture regions with this rule; the old "bottom-up unless WINDOW"
#          rule put dashboard/instrument texels on the hood.
#          NOTE: the total uv-stream area is only v_cnt*32 bytes, so the
#          header "overlaps" the last vertex' set3 - harmless, set3 unused.
#          0x4015/0x4035 meshes append a tangent stream between the uv data
#          and the indices; 0x4035 additionally a second uv block:
#              uv block A : header u32 + v_cnt * 32  (sets 0..3)
#              [0x4035]   uv block B : v_cnt * 32    (sets 4..7, set4 =
#                        lightmap/duplicate of set0, sets 5..7 zero)
#              tangent    : v_cnt * 24   tangent(3f) + binormal(3f) per vertex
#                        (the last 4 bytes overlap the index buffer)
#              indices
#          Tangent convention (verified on original data): T is the direction
#          of increasing U, binormal B = cross(N, T); T/B/N form an
#          orthonormal right-handed frame in isi space.
#   index buffer: (triangle_count + 2) u16 indices forming a triangle strip
#   materials section: u32 numMaterials, then material blocks
#
# Material blocks are variable length; they are recovered by scanning for
# the layer tokens (L0DIFFUSET0, L1CMAPADDALPHAREFLECTT0, ...): each material
# owns exactly one L0 token.  Names / textures / colors are picked from the
# surrounding bytes.

import re
import struct

import numpy as np


TAN_LOWORDS = (0x4015, 0x4035, 0x6015, 0x6035)   # meshes with tangent stream
UV8_LOWORDS = (0x4035, 0x6035)                   # meshes with second uv block
HEADER_STRIDE = 108
PREAMBLE_SIZE = 508
NAME_OFFSET_IN_PREAMBLE = 404  # absolute 24 + 380

_LAYER_RE = re.compile(rb"(?<![A-Za-z0-9_.\-])L[0-9][A-Za-z0-9_]{2,40}\x00")
_NAME_RE = re.compile(rb"[A-Za-z0-9_\.\-' #()&+,]{3,60}\x00")
_LAYER_TOKEN_RE = re.compile(r"L[0-9][A-Z0-9_]{2,40}$")
_TEX_EXT_RE = re.compile(r"\.(DDS|TGA|BMP|PNG|JPG)$", re.IGNORECASE)


class Submesh:
    """One decoded submesh.  All vertex data are numpy arrays (one row per
    vertex), positions/normals/tangents in isi coordinates, UVs already
    converted to Blender's V direction."""
    __slots__ = ("id", "positions", "normals", "uvs", "faces", "material_index",
                 "colors", "uvs_light", "tangents", "binormals",
                 "tangents_stored", "has_tangent_stream", "has_uv8")

    def __init__(self, sid):
        self.id = sid
        self.positions = None    # (n, 3) float32
        self.normals = None      # (n, 3) float32
        self.uvs = None          # (n, 2) float32, V = 1 - file V
        self.faces = None        # (m, 3) int32, winding exactly as in the file
        self.material_index = sid
        self.colors = None       # (n, 4) float32 RGBA (file: BGRA bytes)
        self.uvs_light = None    # (n, 2) float32 (uv set 4) or None
        self.tangents = None     # (n, 3) float32 (stored or analytic)
        self.binormals = None    # (n, 3) float32
        self.tangents_stored = False   # True when game tangents were usable
        self.has_tangent_stream = False
        self.has_uv8 = False


class GmtMaterial:
    __slots__ = ("name", "layers", "textures", "start", "end")

    def __init__(self, name, start, end):
        self.name = name
        self.layers = []
        self.textures = []
        self.start = start
        self.end = end

    @property
    def diffuse_texture(self):
        return self.textures[0] if self.textures else None

    @property
    def has_alpha(self):
        return any("ALPHA" in layer.upper() for layer in self.layers)


class GmtFile:
    def __init__(self):
        self.instance_name = ""
        self.flags = 0
        self.pivot = None            # (x, y, z) in isi coordinates or None
        self.submeshes = []
        self.materials = []


def _read_string(buf, offset, maxlen=240):
    out = bytearray()
    for i in range(offset, min(offset + maxlen, len(buf))):
        b = buf[i]
        if b == 0:
            break
        out.append(b)
    return out.decode("ascii", errors="replace")


def _find_pivot(data):
    """Animated parts store their pivot twice: (px,py,pz, 1.0, 0.0, px,py,pz)."""
    limit = min(2048, len(data) - 32)
    if limit <= 24:
        return None, -1
    n_off = (limit - 24 + 3) // 4              # offsets 24, 28, ... < limit
    fl = np.frombuffer(data, dtype="<f4", count=n_off + 7, offset=24)
    f = np.lib.stride_tricks.sliding_window_view(fl, 8).astype(np.float64)
    with np.errstate(invalid="ignore"):
        ok = ((np.abs(f[:, 3] - 1.0) < 1e-6) & (np.abs(f[:, 4]) < 1e-6)
              & (f[:, 0] == f[:, 5]) & (f[:, 1] == f[:, 6])
              & (f[:, 2] == f[:, 7])
              & ((f[:, 0] != 0.0) | (f[:, 1] != 0.0) | (f[:, 2] != 0.0))
              & (np.abs(f).max(axis=1) < 1e6))
    if not ok.any():
        return None, -1
    i = int(np.argmax(ok))
    return (float(f[i, 0]), float(f[i, 1]), float(f[i, 2])), 24 + 4 * i


def _count_records(limit, base, stride, size, cap):
    """Number of k in [0, cap) with base + k*stride + size <= limit."""
    if base < 0 or base + size > limit:
        return 0
    return min(cap, (limit - base - size) // stride + 1)


def _strided(data, count, cols, dtype, offset, row_stride, col_stride):
    return np.ndarray((count, cols), dtype, data, offset,
                      (row_stride, col_stride))


def _read_u16(data, start, count):
    """Up to `count` little-endian u16 starting at byte `start` (truncated at
    the end of the file)."""
    dlen = len(data)
    if start < 0 or start > dlen:
        return np.empty(0, np.int64)
    avail = min(count, (dlen - start) // 2)
    if avail <= 0:
        return np.empty(0, np.int64)
    return np.frombuffer(data, dtype="<u2", count=avail,
                         offset=start).astype(np.int64)


def _normalize_rows(v):
    """Rows scaled to unit length where |v|^2 > 1e-12 (others untouched)."""
    l2 = (v * v).sum(axis=1)
    big = l2 > 1e-12
    inv = np.ones_like(l2)
    inv[big] = 1.0 / np.sqrt(l2[big])
    return v * inv[:, None]


def _fill_tangents(pos, nrm, uvs, faces, T, B, tan_ok, bin_ok):
    """Lengyel per-vertex tangents from uv set 0: T = direction of
    increasing U (independent of the V flip), B = cross(N, T) - the game
    convention.  Stored tangents (tan_ok) are kept, missing/degenerate ones
    are replaced (0x4011 meshes get full analytic frames); a missing stored
    binormal is rebuilt from cross(N, T)."""
    n = len(pos)
    P = pos.astype(np.float64)
    N = nrm.astype(np.float64)
    UV = uvs.astype(np.float64)
    acc = np.zeros((n, 3))
    if len(faces):
        a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
        e1 = P[b] - P[a]
        e2 = P[c] - P[a]
        d1 = UV[b, 0] - UV[a, 0]
        d2 = UV[b, 1] - UV[a, 1]
        d3 = UV[c, 0] - UV[a, 0]
        d4 = UV[c, 1] - UV[a, 1]
        det = d1 * d4 - d3 * d2
        good = np.abs(det) >= 1e-12
        r = np.zeros_like(det)
        r[good] = 1.0 / det[good]
        s = (d4[:, None] * e1 - d2[:, None] * e2) * r[:, None]
        for idx in (a, b, c):
            for k in range(3):
                acc[:, k] += np.bincount(idx, weights=s[:, k], minlength=n)
    need = ~tan_ok
    zero_s = need & ((acc * acc).sum(axis=1) < 1e-16)
    calc = need & ~zero_s
    # orthogonalise against the normal, normalise
    t = acc - N * (acc * N).sum(axis=1)[:, None]
    tl2 = (t * t).sum(axis=1)
    inv = np.zeros_like(tl2)
    nz = tl2 >= 1e-16
    inv[nz] = tl2[nz] ** -0.5
    t = t * inv[:, None]
    T_out = np.where(tan_ok[:, None], np.nan_to_num(T), np.where(calc[:, None], t, 0.0))
    B_from_T = _normalize_rows(np.cross(N, T_out))
    B_out = np.where(bin_ok[:, None], np.nan_to_num(B),
                     np.where(zero_s[:, None], 0.0, B_from_T))
    return T_out.astype(np.float32), B_out.astype(np.float32)


def _decode_submesh(data, hdr_pos):
    (magic, sid, v_cnt, v_off, _h4, uv_off, _h6, _h7, _h8, t_cnt,
     i_off) = struct.unpack_from("<11I", data, hdr_pos)
    dlen = len(data)
    sub = Submesh(sid)
    lo = magic & 0xFFFF
    sub.has_uv8 = lo in UV8_LOWORDS
    sub.has_tangent_stream = lo in TAN_LOWORDS

    # --- vertex buffer: 32 bytes per vertex --------------------------------
    n = max(0, min(v_cnt, (dlen - v_off) // 32)) if v_off <= dlen else 0
    if n:
        sub.positions = np.ascontiguousarray(
            _strided(data, n, 3, "<f4", v_off + 4, 32, 4), dtype=np.float32)
        sub.normals = np.ascontiguousarray(
            _strided(data, n, 3, "<f4", v_off + 16, 32, 4), dtype=np.float32)
        # vertex diffuse color, DirectX BGRA byte order -> RGBA floats
        col = _strided(data, n, 4, "u1", v_off + 28, 32, 1)
        sub.colors = (col[:, [2, 1, 0, 3]].astype(np.float64)
                      / 255.0).astype(np.float32)
    else:
        sub.positions = np.zeros((0, 3), np.float32)
        sub.normals = np.zeros((0, 3), np.float32)
        sub.colors = np.zeros((0, 4), np.float32)

    # --- UV stream: u32 header (0) + 32 bytes per vertex -----------------
    # set0 (= diffuse) lives at stream_base + k*32.  Verified against the
    # original game data by render comparison (Cobra stripe, GT40 steering
    # wheel hub, building facades, grandstand roof, Q8 logo orientation).
    stream_base = uv_off
    if uv_off + 4 <= dlen:
        potential = struct.unpack_from("<I", data, uv_off)[0]
        if potential == 0 or potential == v_cnt:
            stream_base = uv_off + 4
    uv_d = np.zeros((n, 2), np.float64)
    m = _count_records(dlen, stream_base, 32, 8, n)
    if m:
        uv_d[:m] = _strided(data, m, 2, "<f4", stream_base, 32, 4)
    # DirectX V (0 = top row) -> Blender V (0 = bottom row), ALL meshes
    uv_d[:, 1] = 1.0 - uv_d[:, 1]
    sub.uvs = uv_d.astype(np.float32)
    if sub.has_uv8 and n:
        # second uv block: set4 = lightmap (usually == set0), 5..7 zero
        base4 = stream_base + v_cnt * 32
        if _count_records(dlen, base4, 32, 8, n) == n:
            lm = np.array(_strided(data, n, 2, "<f4", base4, 32, 4),
                          dtype=np.float64)
            lm[:, 1] = 1.0 - lm[:, 1]
            sub.uvs_light = lm.astype(np.float32)

    # --- tangent stream (0x4015 / 0x4035): 24 bytes per vertex ------------
    # tangent(3f) + binormal(3f); the last 4 bytes overlap the index buffer,
    # so the final vertex' binormal is rebuilt from cross(normal, tangent).
    T = np.zeros((n, 3))
    B = np.zeros((n, 3))
    tan_ok = np.zeros(n, bool)
    bin_ok = np.zeros(n, bool)
    if sub.has_tangent_stream and n:
        tan_base = stream_base + v_cnt * 32
        if sub.has_uv8:
            tan_base += v_cnt * 32
        lim = min(dlen, i_off)
        n_t = _count_records(lim, tan_base, 24, 12, n)
        n_b = _count_records(lim, tan_base, 24, 24, n)
        T = np.full((n, 3), np.nan)
        B = np.full((n, 3), np.nan)
        if n_t:
            T[:n_t] = _strided(data, n_t, 3, "<f4", tan_base, 24, 4)
        if n_b:
            B[:n_b] = _strided(data, n_b, 3, "<f4", tan_base + 12, 24, 4)
        # validity: unit length; orthogonality is only enforced against
        # obvious garbage (the original data keeps slightly skewed frames
        # on a few meshes - preserve them as the engine received them)
        with np.errstate(invalid="ignore"):
            tl2 = (T * T).sum(axis=1)
            tan_ok = (tl2 > 0.5) & (tl2 < 1.7)
            bl2 = (B * B).sum(axis=1)
            dnb = np.abs((B * sub.normals.astype(np.float64)).sum(axis=1))
            bin_ok = tan_ok & (bl2 > 0.5) & (bl2 < 1.7) & ~(dnb > 0.9)
        sub.tangents_stored = int(tan_ok.sum()) > v_cnt // 2

    # --- indices ------------------------------------------------------------
    if (magic >> 16) == 0x2000:
        # LIST: 4 bytes extra value (float) precede t_cnt u16 indices
        raw = _read_u16(data, i_off + 4, t_cnt)
        tri = raw[:(len(raw) // 3) * 3].reshape(-1, 3)
    else:
        # STRIP: (t_cnt + 2) u16; odd triangles swap their last two indices
        raw = _read_u16(data, i_off, t_cnt + 2)
        if len(raw) >= 3:
            A = raw[:-2]
            Bc = raw[1:-1].copy()
            Cc = raw[2:].copy()
            odd = (np.arange(len(A)) & 1).astype(bool)
            Bc[odd], Cc[odd] = Cc[odd].copy(), Bc[odd].copy()
            tri = np.stack([A, Bc, Cc], axis=1)
        else:
            tri = np.empty((0, 3), np.int64)
    if len(tri):
        keep = ((tri.max(axis=1) < n) & (tri[:, 0] != tri[:, 1])
                & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2]))
        tri = tri[keep]
    sub.faces = tri.astype(np.int32)

    if n:
        sub.tangents, sub.binormals = _fill_tangents(
            sub.positions, sub.normals, uv_d, sub.faces,
            T, B, tan_ok, bin_ok)
    else:
        sub.tangents = np.zeros((0, 3), np.float32)
        sub.binormals = np.zeros((0, 3), np.float32)
    return sub


def _find_headers(data, marker):
    """Submesh header positions: structural validation (independent of the
    magic low word, which encodes the vertex format: 0x4011/0x6011 =
    pos+normal+uv, 0x4015/0x6015 = additionally a 24-byte stream (tangents)
    between the uv stream and the index buffer), evaluated for every 4-byte
    aligned position at once; then the monotonic header chain is walked."""
    dlen = len(data)
    data_limit = min(marker + 4, dlen)
    max_pos = min(marker, dlen - 52)
    first = 24 + PREAMBLE_SIZE
    nwords = dlen // 4
    i0 = first // 4
    i1 = min((max_pos + 3) // 4, nwords - 10)
    if i1 <= i0:
        return []
    w = np.frombuffer(data, dtype="<u4", count=nwords)
    idx = np.arange(i0, i1)
    pos = idx * 4
    hi = (w[idx] >> 16).astype(np.int64)
    v_cnt = w[idx + 2].astype(np.int64)
    v_off = w[idx + 3].astype(np.int64)
    uv_off = w[idx + 5].astype(np.int64)
    t_cnt = w[idx + 9].astype(np.int64)
    i_off = w[idx + 10].astype(np.int64)
    ok = (hi == 0) | (hi == 0x2000)
    ok &= (v_cnt > 0) & (v_cnt < 400000) & (t_cnt < 900000)
    ok &= (v_off > pos) & (i_off > pos)
    ok &= uv_off == v_off + v_cnt * 32
    ok &= (i_off >= uv_off) & (i_off <= uv_off + v_cnt * 128)
    end = i_off + (t_cnt + 2) * 2
    ok &= (end <= data_limit) & (end <= dlen)
    headers = []
    nxt = first
    prev_v_off = -1
    for p, vo in zip(pos[ok].tolist(), v_off[ok].tolist()):
        if p < nxt:
            continue
        if vo > prev_v_off:            # keep monotonic chain, no overlaps
            headers.append(p)
            prev_v_off = vo
            nxt = p + HEADER_STRIDE
    return headers


def parse_gmt(data: bytes) -> GmtFile:
    if len(data) < 64:
        raise ValueError("GMT too small")
    g = GmtFile()
    marker, g.flags = struct.unpack_from("<2I", data, 0)

    # Instance name lives at a fixed slot inside the preamble.
    g.instance_name = _read_string(data, NAME_OFFSET_IN_PREAMBLE, 64).strip()
    if not g.instance_name:
        g.instance_name = "GMT"

    g.pivot, _pivot_off = _find_pivot(data)

    # u32@0 = materials section start - 4, so geometry may run to marker+4.
    headers = _find_headers(data, marker)
    if not headers:
        raise ValueError("no submesh headers found")
    for hdr in headers:
        g.submeshes.append(_decode_submesh(data, hdr))

    # --- materials ---------------------------------------------------------
    mat_start = marker + 4
    if mat_start >= len(data):
        mat_start = marker
    g.materials, _declared = _decode_materials(data, mat_start)

    n_sub = len(g.submeshes)
    n_mat = len(g.materials)
    if n_sub and n_mat and n_sub != n_mat:
        for i, sub in enumerate(g.submeshes):
            sub.material_index = min(i, n_mat - 1)
    elif n_mat:
        for sub in g.submeshes:
            sub.material_index = min(sub.id, n_mat - 1) if sub.id < n_mat else 0
    return g


def _decode_materials(data, section_start):
    """Scan the material section; each material owns one L0 layer token."""
    section = data[section_start:]
    if not section:
        return [], 0

    declared = 0
    if len(section) >= 8:
        declared = struct.unpack_from("<I", section, 4)[0]
        if not (0 < declared <= 512):
            declared = 0

    layers = [(m.start(), m.group()[:-1].decode("ascii", errors="replace"))
              for m in _LAYER_RE.finditer(section)]
    l0_positions = [pos for pos, tok in layers if tok.startswith("L0")]
    if not l0_positions:
        return [], declared

    names = [(m.start(), m.group()[:-1].decode("ascii", errors="replace").strip())
             for m in _NAME_RE.finditer(section)]
    tokens = set(tok for _p, tok in layers)
    for pos, tok in layers:
        tokens.add(tok)
    candidates = []
    for pos, tok in layers:
        candidates.append((pos, tok, "layer"))
    for pos, nm in names:
        if nm in tokens or _TEX_EXT_RE.search(nm):
            continue
        candidates.append((pos, nm, "name"))

    bounds = l0_positions + [len(section)]
    materials = []
    for i, start in enumerate(l0_positions):
        end = bounds[i + 1]
        mat = GmtMaterial("", start, end)
        best_name = ""
        best_pos = -1
        for pos, tok, kind in candidates:
            if kind == "name" and start - 1200 <= pos < start:
                if pos > best_pos:
                    best_pos = pos
                    best_name = tok
        mat.name = best_name or "GMT_Material_%d" % i
        for pos, tok in layers:
            if start <= pos < end:
                mat.layers.append(tok)
        joined = _collect_textures(section, start, end)
        mat.textures = joined
        materials.append(mat)
    return materials, declared


def _collect_textures(section, start, end):
    """Collect texture file names; re-join animated sequences split as
    'BASENAME\\0' + '12.BMP\\0' (frame number + extension)."""
    raw = section[start:min(end, start + 100000)]
    found = []
    pieces = [(m.start(), m.group()[:-1].decode("ascii", errors="replace").strip())
              for m in _NAME_RE.finditer(raw)]
    i = 0
    while i < len(pieces):
        pos, tok = pieces[i]
        nxt = pieces[i + 1] if i + 1 < len(pieces) else None
        if nxt and nxt[0] - (pos + len(tok) + 1) <= 2 \
                and re.fullmatch(r"(\d*\.)?(DDS|TGA|BMP|PNG|JPG)", nxt[1], re.IGNORECASE) \
                and not _TEX_EXT_RE.search(tok) and not _LAYER_TOKEN_RE.match(tok):
            sep = "" if "." in nxt[1] else "."
            joined = tok + sep + nxt[1]
            if _TEX_EXT_RE.search(joined):
                found.append(joined)
                i += 2
                continue
        if _TEX_EXT_RE.search(tok):
            found.append(tok)
        i += 1
    out = []
    for t in found:
        if t not in out:
            out.append(t)
    return out
