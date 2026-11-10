#!/usr/bin/env python3
"""SBB1: a small self-describing binary blob format.

Layout (all little-endian):
  SbbHeader  (64 bytes, packed)
    char     magic[4]      = "SBB1"
    uint32   version       = 1
    uint32   n_sections
    uint64   dir_offset    # offset of the SbbSection directory
    uint64   names_offset  # offset of the packed string blob
    uint64   names_len
    uint64   total_size
    uint64   reserved
  SbbSection (64 bytes each)
    uint32   name_off      # offset into the names blob
    uint32   dtype         # 0 f64, 1 f32, 2 i64, 3 i32, 4 i8, 5 u8, 6 u32, 7 str
    uint32   ndim
    uint32   reserved
    uint64   shape[4]      # element counts; product == element count
    uint64   data_offset
  names blob  (names_len bytes, NUL-separated ASCII)
  data area   (raw arrays, in section order)

`str` sections are a single byte buffer; callers store offsets separately.
This module is imported by the baker (write) and the L0 checker (read), and
mirrors what csim/sb_baked.h parses in C.
"""
from __future__ import annotations

import struct
from typing import Iterable

import numpy as np

MAGIC = b"SBB1"
VERSION = 1
_HDR = struct.Struct("<4sII5Q")      # 4+4+4+8*5 = 52 bytes, exact
_SEC = struct.Struct("<IIII5Q")      # 4*4 + 5*8 = 56 bytes, exact

DTYPES = {
    "f64": (0, np.float64),
    "f32": (1, np.float32),
    "i64": (2, np.int64),
    "i32": (3, np.int32),
    "i8": (4, np.int8),
    "u8": (5, np.uint8),
    "u32": (6, np.uint32),
    "str": (7, np.uint8),
}
_DTYPE_BY_CODE = {code: (name, np.dtype(np_ty)) for name, (code, np_ty) in DTYPES.items()}

assert _HDR.size == 52, _HDR.size
assert _SEC.size == 56, _SEC.size


class BlobWriter:
    """Accumulates typed sections and writes one .sbb file."""

    def __init__(self) -> None:
        self._names = bytearray()
        self._name_off: dict[str, int] = {}
        self._sections: list[tuple[str, int, tuple[int, ...], bytes]] = []

    def _intern(self, name: str) -> int:
        if name in self._name_off:
            return self._name_off[name]
        off = len(self._names)
        self._names += name.encode("ascii") + b"\x00"
        self._name_off[name] = off
        return off

    def add(self, name: str, arr: np.ndarray | Iterable, dtype: str | None = None) -> None:
        """Adds an array section. dtype defaults to the array's own dtype."""
        if dtype is None:
            a = np.asarray(arr)
            if a.dtype == np.bool_:
                a = a.astype(np.uint8)
            arr = a
            dname = np.dtype(arr.dtype).name
            for k, (code, np_ty) in DTYPES.items():
                if np.dtype(np_ty) == np.dtype(arr.dtype):
                    dname = k
                    break
            else:
                raise TypeError(f"unsupported dtype {arr.dtype} for {name}")
            dtype = dname
        else:
            arr = np.ascontiguousarray(arr, dtype=DTYPES[dtype][1])
        arr = np.ascontiguousarray(arr)
        if dtype == "str":
            arr = arr.view(np.uint8)
        self._intern(name)
        self._sections.append((name, DTYPES[dtype][0], tuple(arr.shape), arr.tobytes()))

    def add_str(self, name: str, value: str) -> None:
        self.add(name, np.frombuffer(value.encode("utf-8"), dtype=np.uint8), dtype="str")

    def add_matrix_csc(self, prefix: str, m) -> None:
        self.add(f"{prefix}_shape", np.asarray(m.shape, dtype=np.int64), "i64")
        self.add(f"{prefix}_indptr", m.indptr, "i32")
        self.add(f"{prefix}_indices", m.indices, "i32")
        self.add(f"{prefix}_data", m.data, "f64")

    def add_matrix_csr(self, prefix: str, m) -> None:
        csr = m.tocsr()
        self.add(f"{prefix}_shape", np.asarray(csr.shape, dtype=np.int64), "i64")
        self.add(f"{prefix}_indptr", csr.indptr, "i32")
        self.add(f"{prefix}_indices", csr.indices, "i32")
        self.add(f"{prefix}_data", csr.data, "f64")

    def write(self, path) -> int:
        names = bytes(self._names)
        # Directory first (fixed size), then names, then data.
        dir_size = _SEC.size * len(self._sections)
        dir_off = _HDR.size
        names_off = dir_off + dir_size
        data_off = names_off + len(names)
        # align data start to 64
        pad = (-data_off) % 64
        data_off += pad

        sec_blob = bytearray()
        cur = data_off
        for name, dtype_code, shape, raw in self._sections:
            shape4 = list(shape) + [1] * (4 - len(shape))
            sec_blob += _SEC.pack(self._name_off[name], dtype_code, len(shape),
                                  0, *shape4, cur)
            cur += len(raw)
        total = cur

        hdr = _HDR.pack(MAGIC, VERSION, len(self._sections), dir_off, names_off,
                        len(names), total, 0)
        with open(path, "wb") as f:
            f.write(hdr)
            f.write(sec_blob)
            f.write(names)
            f.write(b"\x00" * pad)
            for _, _, _, raw in self._sections:
                f.write(raw)
        return total


class BlobReader:
    """Reads a .sbb file produced by BlobWriter."""

    def __init__(self, path) -> None:
        self.path = path
        self._sections: dict[str, tuple[int, tuple[int, ...], int, int]] = {}
        with open(path, "rb") as f:
            raw = f.read()
        self._raw = raw
        magic, version, n_sec, dir_off, names_off, names_len, total, _ = _HDR.unpack_from(raw, 0)
        assert magic == MAGIC, magic
        assert version == VERSION, version
        self.names = raw[names_off:names_off + names_len]
        for i in range(n_sec):
            off = dir_off + i * _SEC.size
            name_off, dtype_code, ndim, _res, s0, s1, s2, s3, data_off = _SEC.unpack_from(raw, off)
            name = self._name_bytes(name_off).decode("ascii")
            shape = tuple(x for x in (s0, s1, s2, s3)[:ndim])
            n = int(np.prod(shape)) if shape else 1
            self._sections[name] = (dtype_code, shape, data_off, n)

    def _name_bytes(self, off: int) -> bytes:
        end = self.names.index(b"\x00", off)
        return self.names[off:end]

    def names_list(self) -> list[str]:
        out = []
        off = 0
        while off < len(self.names):
            end = self.names.index(b"\x00", off)
            out.append(self.names[off:end].decode("ascii"))
            off = end + 1
        return out

    def has(self, name: str) -> bool:
        return name in self._sections

    def keys(self) -> list[str]:
        return list(self._sections.keys())

    def get(self, name: str) -> np.ndarray:
        dtype_code, shape, data_off, n = self._sections[name]
        dt_name, np_dt = _DTYPE_BY_CODE[dtype_code]
        if dt_name == "str":
            buf = self._raw[data_off:data_off + n]
            return np.frombuffer(buf, dtype=np.uint8).copy()
        arr = np.frombuffer(self._raw, dtype=np_dt, count=n, offset=data_off).copy()
        return arr.reshape(shape) if shape else arr.reshape(())

    def get_str(self, name: str) -> str:
        b = self.get(name).tobytes()
        return b.decode("utf-8")

    def get_matrix_csc(self, prefix: str):
        import scipy.sparse as sp
        shape = tuple(int(x) for x in self.get(f"{prefix}_shape"))
        return sp.csc_matrix(
            (self.get(f"{prefix}_data"), self.get(f"{prefix}_indices"),
             self.get(f"{prefix}_indptr")), shape=shape)

    def get_matrix_csr(self, prefix: str):
        import scipy.sparse as sp
        shape = tuple(int(x) for x in self.get(f"{prefix}_shape"))
        return sp.csr_matrix(
            (self.get(f"{prefix}_data"), self.get(f"{prefix}_indices"),
             self.get(f"{prefix}_indptr")), shape=shape)


def _self_test() -> None:
    import io
    import tempfile
    w = BlobWriter()
    w.add("a", np.arange(6, dtype=np.float64).reshape(2, 3))
    w.add("b", np.array([1, 2, 3], dtype=np.int32))
    w.add_str("note", "hello world")
    w.add("c", np.array([True, False, True]), dtype="u8")
    with tempfile.NamedTemporaryFile(suffix=".sbb", delete=False) as f:
        path = f.name
    w.write(path)
    r = BlobReader(path)
    assert r.keys() == ["a", "b", "note", "c"], r.keys()
    assert np.array_equal(r.get("a"), np.arange(6).reshape(2, 3))
    assert np.array_equal(r.get("b"), np.array([1, 2, 3], dtype=np.int32))
    assert r.get_str("note") == "hello world"
    assert np.array_equal(r.get("c"), np.array([1, 0, 1], dtype=np.uint8))
    print("sbb self-test OK")


if __name__ == "__main__":
    _self_test()
