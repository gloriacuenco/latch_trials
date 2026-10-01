"""Download ABIDE II T1-weighted scans from the public INDI bucket (no login), keeping the BIDS layout.

Source: s3://fcp-indi/data/Projects/ABIDE2/RawData/ over HTTPS. Downloads every ``anat/*_T1w.nii.gz``
plus each site's ``participants.tsv``, ``T1w.json`` and ``dataset_description.json``; files already
present with the right size are skipped, so the script can be rerun to resume. Writes ``manifest.csv``
(one row per T1 scan). ABIDE data-use terms apply (non-commercial use, acknowledgment).
Set ``https_proxy`` to an ``http://`` proxy where needed (vpulab).
"""

from __future__ import annotations

import argparse
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BUCKET = "https://s3.amazonaws.com/fcp-indi"
PREFIX = "data/Projects/ABIDE2/RawData/"
SITE_FILES = ("participants.tsv", "T1w.json", "dataset_description.json")


def list_keys(prefix: str) -> list[tuple[str, int]]:
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        body = urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(q)}", timeout=120).read().decode()
        keys += [(k, int(s)) for k, s in re.findall(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", body, re.S)]
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", body)
        if not m:
            return keys
        token = m.group(1)


def fetch(job: tuple[str, int, Path]) -> str:
    key, size, dest = job
    if dest.exists() and dest.stat().st_size == size:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(f"{BUCKET}/{urllib.parse.quote(key)}", timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if tmp.stat().st_size != size:
        tmp.unlink()
        raise IOError(f"size mismatch for {key}")
    tmp.rename(dest)
    return "ok"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    out = Path(args.out_dir)

    keys = list_keys(PREFIX)
    wanted = [(k, s) for k, s in keys
              if ("/anat/" in k and k.endswith("_T1w.nii.gz"))
              or (k[len(PREFIX):].count("/") == 1 and k.rsplit("/", 1)[1] in SITE_FILES)]
    print(f"{len(wanted)} files, {sum(s for _, s in wanted) / 1e9:.1f} GB", flush=True)
    jobs = [(k, s, out / k[len(PREFIX):]) for k, s in wanted]
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for status in pool.map(fetch, jobs):
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} ({status})", flush=True)

    rows = []
    for k, s in wanted:
        if not k.endswith("_T1w.nii.gz"):
            continue
        rel = k[len(PREFIX):]
        site, subject = rel.split("/")[:2]
        name = rel.rsplit("/", 1)[1]
        ses = re.search(r"_ses-([^_]+)", name)
        run = re.search(r"_run-([^_]+)", name)
        rows.append({"site": site, "subject": subject, "session": ses.group(1) if ses else "",
                     "run": run.group(1) if run else "", "path": rel, "bytes": s})
    pd.DataFrame(rows).sort_values(["site", "subject", "session", "run"]).to_csv(out / "manifest.csv", index=False)
    print(f"manifest: {len(rows)} T1 scans, {len({(r['site'], r['subject']) for r in rows})} subjects", flush=True)


if __name__ == "__main__":
    main()
