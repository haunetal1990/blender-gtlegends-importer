# Parser for the "CUBEASF" style text formats used by the isi/gMotor2 engine:
# GT Legends .TRK / .SCN scene files and .CAS car structure files.
# (Nearly identical to rFactor 1 / rFactor 2 / F1 Challenge 99-02 .SCN/.GEN.)
#
# Grammar (line based):
#   Key=Value               simple parameter
#   Key=(a, b, c)           vector parameter
#   Name=Type               block header (Instance=xxx, Light=xxx, View=xxx)
#   { ... }                 block body, blocks nest
#   // comment              comments
#
# Inside Instance blocks, mesh references look like:
#   <LOW><SPIN> MeshFile=CAR_BODY_STYLE<1>.gmt CollTarget=False LODOut=(50.0)
# The <TAG> prefixes are view/LOD conditions, <n> are "generation" slots
# resolved from GenString in the .CAR file.

import re

_BLOCK_KINDS = ("INSTANCE", "LIGHT", "VIEW", "PROJECTOR", "ACTOR", "SOUND")
_BLOCK_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S+)\s*$')
_KEYVAL_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_<>]*)\s*=\s*(.*?)\s*$')
_MESH_RE = re.compile(r'MeshFile\s*=\s*(\S+)', re.IGNORECASE)
_TAG_RE = re.compile(r'<([A-Za-z0-9_]+)>')
_TOKEN_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\([^)]*\)|[^\s]+)')


class MeshRef:
    __slots__ = ("raw", "mesh_file", "tags", "params")

    def __init__(self, raw, mesh_file, tags, params):
        self.raw = raw
        self.mesh_file = mesh_file
        self.tags = tags            # e.g. ['SPIN', 'MAX'] or ['NOTSPIN','DASHHIGH']
        self.params = params        # CollTarget, LODIn, LODOut, Render, ...

    @property
    def is_shadow(self):
        return "SHADOWOBJECT" in self.params

    @property
    def render_false(self):
        val = str(self.params.get("RENDER", "")).lower()
        return val == "false"


class AsfBlock:
    __slots__ = ("kind", "name", "params", "children", "mesh_refs")

    def __init__(self, kind, name):
        self.kind = kind
        self.name = name
        self.params = {}
        self.children = []
        self.mesh_refs = []

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


class AsfDoc:
    def __init__(self):
        self.search_paths = []
        self.mas_files = []
        self.blocks = []              # top level blocks

    def blocks_by_kind(self, kind):
        out = []
        for blk in self.blocks:
            out.extend(b for b in blk.walk() if b.kind == kind)
        return out


def _strip_comment(line):
    idx = line.find("//")
    if idx >= 0:
        line = line[:idx]
    return line.strip()


def parse_asf_text(text):
    doc = AsfDoc()
    stack = []

    def add_param(key, value, block):
        key = key.upper()
        if block is None:
            if key == "SEARCHPATH":
                doc.search_paths.append(value)
            elif key == "MASFILE":
                doc.mas_files.append(value)
        else:
            block.params.setdefault(key, value)

    lines = [_strip_comment(l.replace("\t", " ")) for l in text.splitlines()]
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        raw_line = lines[i]
        line = raw_line
        if not line:
            i += 1
            continue

        current = stack[-1] if stack else None

        # Mesh lines may be prefixed with <TAG> conditions and carry several
        # key/value tokens on one line.
        if current is not None and "meshfile=" in line.lower():
            m = _MESH_RE.search(line)
            if m:
                tags = [t.upper() for t in _TAG_RE.findall(line)]
                mesh_file = m.group(1).strip()
                params = {}
                for tm in _TOKEN_RE.finditer(line):
                    k = tm.group(1).upper()
                    if k == "MESHFILE":
                        continue
                    params[k] = tm.group(2).strip("()\"")
                current.mesh_refs.append(MeshRef(line.strip(), mesh_file,
                                                 tags, params))
                i += 1
                continue

        if line.startswith("{"):
            i += 1
            continue
        if line.startswith("}"):
            if stack:
                stack.pop()
            i += 1
            continue

        blk_match = _BLOCK_RE.match(line)
        if blk_match and blk_match.group(1).upper() in _BLOCK_KINDS:
            opens = line.endswith("{")
            if not opens:
                j = i + 1
                while j < n_lines and not lines[j]:
                    j += 1
                opens = j < n_lines and lines[j].startswith("{")
            if opens:
                blk = AsfBlock(blk_match.group(1).upper(), blk_match.group(2))
                if stack:
                    stack[-1].children.append(blk)
                else:
                    doc.blocks.append(blk)
                stack.append(blk)
                i += 1
                continue

        if current is not None and len(_TOKEN_RE.findall(line)) > 1:
            # several Key=Value tokens on one line (Light: Type=Omni Pos=(..) ...)
            for tm in _TOKEN_RE.finditer(line):
                add_param(tm.group(1), tm.group(2), current)
            i += 1
            continue

        keyval = _KEYVAL_RE.match(line)
        if keyval:
            add_param(keyval.group(1), keyval.group(2).strip(), current)
        i += 1

    return doc


def parse_asf_file(path):
    with open(path, "r", encoding="latin-1", errors="replace") as fh:
        return parse_asf_text(fh.read())


def _vec3(value):
    if isinstance(value, str):
        parts = value.strip("()").replace(" ", "").split(",")
        try:
            return [float(p) for p in parts[:3]]
        except (ValueError, IndexError):
            return None
    return None


def parse_vector(value):
    return _vec3(value)


def placeholder_candidates(gen_string, token_digits):
    """Possible values for a <n> generation slot, best guess first."""
    candidates = []
    if gen_string:
        try:
            pos = int(token_digits)
        except ValueError:
            pos = 0
        if 1 <= pos <= len(gen_string):
            candidates.append(gen_string[pos - 1])
    if token_digits.isdigit():
        candidates.append(token_digits)
    candidates.extend("123456789")
    seen = set()
    out = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out
