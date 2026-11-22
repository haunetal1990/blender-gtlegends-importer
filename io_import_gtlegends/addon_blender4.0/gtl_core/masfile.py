# GT Legends / SimBin (gMotor2) container reader.
#
# The .GTL files used by GT Legends (and .MAS in GTR2/rFactor) are
# "GMOTORMAS10" containers:
#
#   0x00  char[12]  magic "GMOTORMAS10" (11 chars + NUL)
#   0x10  u32       entry count
#   0x14  u32       total compressed data size (informational)
#   0x18  entry[ n ]  (256 bytes each)
#           u32  type          (observed 17 for GMT entries, 23 for DDS)
#           u32  offset        (relative to end of the entry table)
#           u32  uncompressed size
#           u32  compressed size
#           char[240] entry name (NUL padded, upper case)
#   ...    zlib-compressed blobs, back to back
#
# Blob N is located at  table_end + offset  with length compressed_size,
# decompressed with zlib.

import struct
import zlib

ENTRY_SIZE = 256
HEADER_SIZE = 24


def is_archive(data: bytes) -> bool:
    return data is not None and len(data) >= 16 and data[:9] == b"GMOTORMAS"


class ArchiveEntry:
    """One container entry - decompressed LAZILY.

    Indexing an archive only reads the entry table and slices the
    compressed blobs; zlib runs only when a file is actually requested
    (entry.data).  This is the main track-import speedup: a track
    references hundreds of textures, but only a fraction is really used."""

    __slots__ = ("name", "size", "_blob", "_data")

    def __init__(self, name: str, size: int, blob: bytes = None,
                 data: bytes = None):
        self.name = name
        self.size = size
        self._blob = blob
        self._data = data

    @property
    def lower_name(self) -> str:
        return self.name.lower()

    @property
    def data(self) -> bytes:
        if self._data is None:
            blob = self._blob or b""
            try:
                self._data = zlib.decompress(blob)
            except zlib.error:
                # Some mods store uncompressed blobs; fall back to raw.
                self._data = blob[:self.size] if len(blob) >= self.size \
                    else blob
            self._blob = None
        return self._data


def read_archive(path: str, lazy: bool = True):
    """Read a GMOTORMAS container.

    lazy=True (default): parse only the entry table, decompress each entry
    on first access to entry.data.  lazy=False decompresses everything
    immediately (used when the archive will be fully extracted anyway)."""
    with open(path, "rb") as fh:
        data = fh.read()
    return parse_archive_bytes(data, lazy=lazy)


def parse_archive_bytes(data: bytes, lazy: bool = True):
    if not is_archive(data):
        raise ValueError("not a GMOTORMAS archive")
    (count,) = struct.unpack_from("<I", data, 16)
    table_end = HEADER_SIZE + count * ENTRY_SIZE
    if table_end > len(data):
        raise ValueError("corrupt archive: entry table out of range")
    entries = []
    for i in range(count):
        base = HEADER_SIZE + i * ENTRY_SIZE
        _type, off, usize, csize = struct.unpack_from("<4I", data, base)
        raw_name = data[base + 16: base + ENTRY_SIZE]
        name = raw_name.split(b"\x00")[0].decode("ascii", errors="replace")
        name = name.replace("\\", "/").rsplit("/", 1)[-1]   # drop any folder
        start = table_end + off
        end = start + csize
        if end > len(data) and off + csize <= len(data):
            start, end = off, off + csize     # offset counted from file start
        if csize == 0 or end > len(data):
            blob = b""
        else:
            blob = data[start:end]
        if lazy:
            entries.append(ArchiveEntry(name, usize, blob=blob))
        else:
            try:
                content = zlib.decompress(blob)
            except zlib.error:
                content = blob[:usize] if len(blob) >= usize else blob
            entries.append(ArchiveEntry(name, usize, data=content))
    return entries


