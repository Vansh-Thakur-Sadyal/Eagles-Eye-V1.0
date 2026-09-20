#!/usr/bin/env python
"""Recover the intact leading entries of a truncated ZIP archive.

A ZIP's index (the central directory) lives at the END of the file, so a
download cut off part-way leaves an archive every normal tool refuses to open -
even though every entry before the cut is perfectly intact. This walks the
local file headers from the start and inflates each entry until the data runs
out, handling entries written with trailing data descriptors (flag bit 3).

    python tools/recover_partial_zip.py <archive.zip> <out_dir> [--only-small MB]
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

LOCAL = b"PK\x03\x04"
DESCRIPTOR = b"PK\x07\x08"


def _zip64_sizes(extra: bytes, csize: int, usize: int) -> tuple:
    """Real sizes from the ZIP64 extra field (id 0x0001) when the 32-bit ones overflow."""
    i = 0
    while i + 4 <= len(extra):
        tag, size = struct.unpack("<HH", extra[i:i + 4])
        body = extra[i + 4:i + 4 + size]
        if tag == 0x0001:
            j = 0
            if usize == 0xFFFFFFFF and j + 8 <= len(body):
                usize = struct.unpack("<Q", body[j:j + 8])[0]
                j += 8
            if csize == 0xFFFFFFFF and j + 8 <= len(body):
                csize = struct.unpack("<Q", body[j:j + 8])[0]
            break
        i += 4 + size
    return csize, usize


def _stream_entry(f, target: Path, method: int, csize: int, crc: int) -> bool:
    """Inflate one entry to disk in 4 MB blocks and verify its CRC-32."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    inflater = zlib.decompressobj(-15) if method == 8 else None
    remaining, running = csize, 0
    with tmp.open("wb") as out:
        while remaining:
            block = f.read(min(remaining, 4 << 20))
            if not block:
                break
            remaining -= len(block)
            piece = inflater.decompress(block) if inflater else block
            running = zlib.crc32(piece, running)
            out.write(piece)
        if inflater:
            tail = inflater.flush()
            running = zlib.crc32(tail, running)
            out.write(tail)
    if remaining or (running & 0xFFFFFFFF) != crc:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(target)
    return True


def recover(archive: Path, out: Path, max_bytes: int | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    stats = {"recovered": 0, "skipped_large": 0, "already_present": 0,
             "crc_failed": [], "truncated_at": None, "files": []}
    archive_size = archive.stat().st_size
    with archive.open("rb") as f:
        while True:
            start = f.tell()
            header = f.read(30)
            if len(header) < 30 or header[:4] != LOCAL:
                break
            (_sig, _ver, flags, method, _t, _d, crc, csize, usize,
             name_len, extra_len) = struct.unpack("<IHHHHHIIIHH", header)
            name = f.read(name_len).decode("utf-8", "replace")
            extra = f.read(extra_len)
            csize, usize = _zip64_sizes(extra, csize, usize)
            uses_descriptor = bool(flags & 0x08)
            is_dir = name.endswith("/")
            target = out / name

            if is_dir:
                target.mkdir(parents=True, exist_ok=True)
                continue

            if method not in (0, 8):
                stats["truncated_at"] = f"{name}: unsupported compression {method}"
                break

            if not uses_descriptor:
                data_start = f.tell()
                if data_start + csize > archive_size:
                    stats["truncated_at"] = name
                    break
                if max_bytes is not None and usize > max_bytes:
                    stats["skipped_large"] += 1
                elif target.exists() and target.stat().st_size == usize:
                    stats["already_present"] += 1
                else:
                    ok = _stream_entry(f, target, method, csize, crc)
                    if ok:
                        stats["recovered"] += 1
                        stats["files"].append(name)
                    else:
                        stats["crc_failed"].append(name)
                f.seek(data_start + csize)
                continue

            # Stored entry, size unknown: scan for a descriptor whose recorded
            # compressed size equals the distance travelled. The size check is
            # what stops a "PK\x07\x08" that happens to occur inside the data
            # from being mistaken for the end of the entry.
            if method == 0:
                data_start = f.tell()
                window = f.read(64 * 1024 * 1024)
                found = None
                idx = window.find(DESCRIPTOR)
                while idx != -1:
                    if idx + 16 <= len(window):
                        _crc, csz, _usz = struct.unpack("<III", window[idx + 4: idx + 16])
                        if csz == idx:
                            found = idx
                            break
                    idx = window.find(DESCRIPTOR, idx + 1)
                if found is None:
                    stats["truncated_at"] = f"{name}: stored entry larger than scan window"
                    break
                payload = window[:found]
                f.seek(data_start + found + 16)
                if max_bytes is None or len(payload) <= max_bytes:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                    stats["recovered"] += 1
                    stats["files"].append(name)
                else:
                    stats["skipped_large"] += 1
                continue

            # Deflated, size unknown: stream-inflate until the stream ends.
            inflater = zlib.decompressobj(-15)
            chunks, total, complete = [], 0, False
            while True:
                block = f.read(1 << 20)
                if not block:
                    break
                piece = inflater.decompress(block)
                total += len(piece)
                if max_bytes is None or total <= max_bytes:
                    chunks.append(piece)
                if inflater.eof:
                    # rewind past the unused tail, then skip the descriptor
                    f.seek(f.tell() - len(inflater.unused_data))
                    sig = f.read(4)
                    f.read(12 if sig == DESCRIPTOR else 8)
                    if sig not in (DESCRIPTOR,):
                        f.seek(f.tell() - 4)
                    complete = True
                    break
            if not complete:
                stats["truncated_at"] = name
                break
            if max_bytes is None or total <= max_bytes:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"".join(chunks))
                stats["recovered"] += 1
                stats["files"].append(name)
            else:
                stats["skipped_large"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--only-small", type=float, default=None,
                        help="skip entries larger than this many MB")
    args = parser.parse_args()
    limit = int(args.only_small * 1024 * 1024) if args.only_small else None
    stats = recover(args.archive, args.out, limit)
    print(f"recovered {stats['recovered']} file(s), skipped {stats['skipped_large']} large, "
          f"{stats['already_present']} already present, {len(stats['crc_failed'])} failed CRC")
    for name in stats["crc_failed"]:
        print("   CRC FAILED:", name)
    print(f"archive ends inside: {stats['truncated_at']}")
    for name in stats["files"][:40]:
        print("  ", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
