"""
Materialize the dataset from the committed recipe. THIS is the script your
teammates run.

    bash scripts/download_data.sh          # or: python3 scripts/fetch_dataset.py

Reads manifest/data_spec.csv, pulls only the listed members out of the remote
WildFake archives via HTTP range requests, normalizes each image identically,
verifies it against the recorded sha256, and writes manifest/manifest.csv.

Transfer is well under 1 GB rather than the ~89 GB a full-archive download
would cost, and no ModelScope account is needed.

Properties worth knowing:
  * Resumable — an already-present file with a matching hash is skipped, so
    an interrupted run costs nothing to restart.
  * Verified — a truncated or corrupted fetch fails loudly here instead of
    silently giving one teammate different pixels from another.
  * Parallel — members are fetched by direct byte-range reads of each local
    file header, so workers don't contend over a single zipfile handle.
"""
import argparse
import csv
import hashlib
import struct
import sys
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wildfake_remote import HttpFile, normalize, resolve  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def range_get(session, url, start, end, retries=4):
    headers = {"Range": f"bytes={start}-{end}"}
    last = None
    for _ in range(retries):
        try:
            r = session.get(url, headers=headers, timeout=180)
            if r.status_code in (200, 206):
                return r.content
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = repr(e)
    raise IOError(f"range {start}-{end} failed: {last}")


def fetch_member(session, url, zi):
    """Byte-range read one zip member, parsing its local header ourselves.

    zipfile.ZipFile can't be shared across threads, and giving each worker its
    own handle would mean re-reading a 493k-entry central directory per
    worker. Reading the local header directly sidesteps both problems.
    """
    pad = 256  # local header extra field can differ from the central one
    start = zi.header_offset
    data = range_get(session, url, start,
                     start + 30 + len(zi.filename.encode()) + pad + zi.compress_size)
    name_len, extra_len = struct.unpack("<HH", data[26:30])
    off = 30 + name_len + extra_len
    payload = data[off:off + zi.compress_size]
    if len(payload) < zi.compress_size:  # pad guess was short
        data = range_get(session, url, start + off, start + off + zi.compress_size)
        payload = data[:zi.compress_size]
    if zi.compress_type == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(payload, -15)
    return payload


def needs_fetch(row, write_hashes: bool) -> bool:
    """Re-fetch unless a correct file is already on disk.

    Existence alone is not enough: a half-written file from an interrupted run
    would be skipped forever and quietly leave one teammate training on
    different pixels than everyone else. Hashing what's already there is cheap
    and makes the byte-identical guarantee real rather than assumed.
    """
    if write_hashes:
        return True
    dest = REPO / row["dest_path"]
    if not dest.exists():
        return True
    if not row["sha256"]:
        return False
    return hashlib.sha256(dest.read_bytes()).hexdigest() != row["sha256"]


def fetch_sid(rows, write_hashes, workers):
    """Fetch SID_Set images by parquet row group.

    Parquet stores columns in per-row-group chunks, so projecting to
    img_id/image/label skips the `mask` column's bytes entirely — dead weight
    for us, and about a third of these rows carry one.
    """
    done = failed = 0
    groups = defaultdict(list)
    for r in rows:
        groups[(r["archive"], int(r["rowgroup"]))].append(r)

    for (url, rg), items in sorted(groups.items()):
        head = requests.head(url, allow_redirects=True, timeout=60)
        size = int(head.headers["content-length"])
        pf = pq.ParquetFile(pa.PythonFile(HttpFile(head.url, size)))
        table = pf.read_row_group(rg, columns=["img_id", "image"])
        blobs = dict(zip(table.column("img_id").to_pylist(),
                         table.column("image").to_pylist()))
        for r in items:
            try:
                raw = blobs[r["member"]]["bytes"]
                img = normalize(raw)
                digest = hashlib.sha256(img).hexdigest()
                if write_hashes:
                    r["sha256"] = digest
                elif r["sha256"] and r["sha256"] != digest:
                    raise ValueError(f"hash mismatch for {r['member']}")
                dest = REPO / r["dest_path"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(img)
                done += 1
            except Exception as e:
                failed += 1
                print(f"  FAILED {r['member']}: {type(e).__name__}: {e}")
        print(f"  row group {rg}: {len(items)} images", flush=True)
    return done, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="manifest/data_spec.csv")
    ap.add_argument("--manifest-out", default="manifest/manifest.csv")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--write-hashes", action="store_true",
                    help="record sha256 of each fetched image back into the "
                         "spec (run once by whoever builds the spec)")
    args = ap.parse_args()

    spec_path = REPO / args.spec
    with open(spec_path) as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} images in {args.spec}")

    by_archive = defaultdict(list)
    for r in rows:
        if r["source"] == "wildfake":
            by_archive[r["archive"]].append(r)

    done = failed = skipped = 0
    for archive, items in by_archive.items():
        pending = [r for r in items if needs_fetch(r, args.write_hashes)]
        skipped += len(items) - len(pending)
        if not pending:
            continue
        print(f"\n{archive}: {len(pending)} to fetch")
        cdn, size = resolve(archive)
        with zipfile.ZipFile(HttpFile(cdn, size)) as zf:
            infos = {r["member"]: zf.getinfo(r["member"]) for r in pending}

        def work(r):
            session = requests.Session()
            raw = fetch_member(session, cdn, infos[r["member"]])
            img = normalize(raw)
            digest = hashlib.sha256(img).hexdigest()
            if args.write_hashes:
                r["sha256"] = digest
            elif r["sha256"] and r["sha256"] != digest:
                raise ValueError(f"hash mismatch for {r['member']}")
            dest = REPO / r["dest_path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(img)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (r, fut) in enumerate(
                    [(r, pool.submit(work, r)) for r in pending]):
                try:
                    fut.result()
                    done += 1
                except Exception as e:
                    failed += 1
                    print(f"  FAILED {r['member']}: {type(e).__name__}: {e}")
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(pending)}", flush=True)

    sid_pending = [r for r in rows
                   if r["source"] == "sid_set" and needs_fetch(r, args.write_hashes)]
    if sid_pending:
        print(f"\nSID_Set: {len(sid_pending)} to fetch (held-out, never trained on)")
        d, f_ = fetch_sid(sid_pending, args.write_hashes, args.workers)
        done += d
        failed += f_
    skipped += sum(1 for r in rows
                   if r["source"] == "sid_set" and r not in sid_pending)

    if args.write_hashes:
        cols = list(rows[0].keys())
        with open(spec_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nrecorded sha256 for {done} images into {args.spec}")

    # manifest.csv is what train.py/evaluate.py actually read
    man = REPO / args.manifest_out
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "generator", "domain", "split"])
        for r in rows:
            if (REPO / r["dest_path"]).exists():
                w.writerow([r["dest_path"], r["label"], r["generator"],
                            r["domain"], r["split"]])

    print(f"\nfetched {done}, skipped {skipped}, failed {failed}")
    print(f"wrote {args.manifest_out}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
