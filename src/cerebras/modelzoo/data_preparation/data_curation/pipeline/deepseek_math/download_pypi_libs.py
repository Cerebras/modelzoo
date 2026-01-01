# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""
Installable-latest extractor (code + metadata, no wheel/archives kept)

What it guarantees
- Only projects whose LATEST STABLE release has at least one non-yanked WHEEL
  (i.e., pip-installable somewhere).
- For such projects, it downloads the LATEST version's artifact:
    - Prefer sdist (tar.gz/zip) to get source layout.
    - If no sdist exists, use the wheel (zip) and extract source/metadata.
- Extracts ONLY selected files (.py/.pyi, README, LICENSE, CHANGELOG, PKG-INFO,
  pyproject.toml, setup.py, setup.cfg, MANIFEST.in, requirements.txt, tox.ini, Makefile).
- Writes metadata.json from /pypi/<name>/json.
- Structured as outdir/<project>/<version>/...
- Parallel, retries, polite delay. No archives left on disk.

Usage examples
  pip install requests packaging tqdm
  python pypi_installable_latest_src.py -o ./pypi_src -w 24

"""

import argparse
import concurrent.futures as cf
import io
import json
import os
import re
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

BASE_SIMPLE_URL = "https://pypi.org/simple/"
JSON_URL = "https://pypi.org/pypi/{name}/json"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
POOL_SIZE = 64
DEFAULT_OUTDIR = "pypi_installable_latest_src"
DEFAULT_WORKERS = 24


# -------- HTTP session --------
def make_session():
    s = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    ad = HTTPAdapter(
        max_retries=retry, pool_connections=POOL_SIZE, pool_maxsize=POOL_SIZE
    )
    s.mount("http://", ad)
    s.mount("https://", ad)
    s.headers.update({"User-Agent": "PyPI-installable-latest-src/1.0"})
    return s


# -------- utilities --------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def parse_iso8601(dt: str):
    try:
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    except Exception:
        return None


def list_all_projects(session: requests.Session):
    r = session.get(BASE_SIMPLE_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    # Simple index anchor text = canonical lowercase names
    return re.findall(r'>([^<]+)</a>', r.text)


def latest_stable(releases: dict):
    """Return (version_str, files[]) for newest non-prerelease w/ files."""
    versions = []
    for vs in releases.keys():
        try:
            v = parse_version(vs)
            versions.append((v, vs))
        except InvalidVersion:
            continue
    versions.sort(key=lambda t: t[0], reverse=True)
    for vobj, vstr in versions:
        if vobj.is_prerelease or getattr(vobj, "is_devrelease", False):
            continue
        files = releases.get(vstr) or []
        if files:
            return vstr, files
    return None


def qualifies_installable(files: list) -> bool:
    wheels = [
        f
        for f in files
        if f.get("packagetype") == "bdist_wheel" and not f.get("yanked")
    ]

    return bool(wheels)


def pick_artifact_to_extract(files: list):
    """
    Prefer sdist (source layout). If none, fall back to a wheel.
    Returns file record dict or None.
    """
    sdists = [
        f
        for f in files
        if f.get("packagetype") == "sdist" and not f.get("yanked")
    ]
    if sdists:
        sdists.sort(
            key=lambda f: parse_iso8601(f.get("upload_time_iso_8601") or "")
            or datetime(1970, 1, 1, tzinfo=timezone.utc),
            reverse=True,
        )
        return sdists[0]
    wheels = [
        f
        for f in files
        if f.get("packagetype") == "bdist_wheel" and not f.get("yanked")
    ]
    if wheels:
        wheels.sort(
            key=lambda f: parse_iso8601(f.get("upload_time_iso_8601") or "")
            or datetime(1970, 1, 1, tzinfo=timezone.utc),
            reverse=True,
        )
        return wheels[0]

    return None


def strip_topdir(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[1:]) if len(parts) > 1 else parts[0]


def safe_join(root: str, rel: str) -> str:
    dest = os.path.abspath(os.path.join(root, rel))
    root_abs = os.path.abspath(root)
    if not dest.startswith(root_abs + os.sep) and dest != root_abs:
        raise ValueError("Path traversal detected")
    return dest


def write_marker(prj_dir: str):
    with open(os.path.join(prj_dir, ".extracted.ok"), "w") as f:
        f.write("ok\n")


def already_done(prj_dir: str) -> bool:
    return os.path.isfile(os.path.join(prj_dir, ".extracted.ok"))


# -------- extraction --------
def extract_from_tar(buf: bytes, out_root: str):
    with tarfile.open(fileobj=io.BytesIO(buf), mode="r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            rel = strip_topdir(m.name)
            dest = safe_join(out_root, rel)
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                continue
            ensure_dir(os.path.dirname(dest))
            with tf.extractfile(m) as src, open(dest, "wb") as dst:
                if src:
                    dst.write(src.read())


def extract_from_zip(buf: bytes, out_root: str):
    with zipfile.ZipFile(io.BytesIO(buf)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            rel = strip_topdir(name)
            dest = safe_join(out_root, rel)
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                continue
            ensure_dir(os.path.dirname(dest))
            with zf.open(name, "r") as src, open(dest, "wb") as dst:
                dst.write(src.read())


# -------- worker --------
def process_project(
    session_factory, project: str, outdir: str, polite_delay: float
):
    s = session_factory()
    try:
        r = s.get(JSON_URL.format(name=project), timeout=REQUEST_TIMEOUT)
        if r.status_code == 404:
            return project, "not-found"
        r.raise_for_status()
        j = r.json()

        info = j.get("info") or {}
        releases = j.get("releases") or {}
        classifiers = info.get("classifiers") or []

        chosen = latest_stable(releases)
        if not chosen:
            return project, "no-stable-release"
        ver, files = chosen

        # enforce installability (wheel existence)
        if not qualifies_installable(files):
            return project, "no-installable-wheel"

        # choose artifact to extract (prefer sdist; else wheel). We won't save the archive.
        file_rec = pick_artifact_to_extract(files)
        if not file_rec:
            return project, "no-artifact"

        url = file_rec["url"]
        filename = file_rec.get("filename") or os.path.basename(
            urlparse(url).path
        )

        prj_dir = os.path.join(outdir, project, ver)
        ensure_dir(prj_dir)
        if already_done(prj_dir):
            return project, "exists"

        # fetch artifact into memory
        resp = s.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        buf = resp.content

        # extract selected files
        if filename.endswith((".tar.gz", ".tar", ".tar.bz2", ".tar.xz")):
            extract_from_tar(buf, prj_dir)
        elif filename.endswith(".zip") or filename.endswith(".whl"):
            extract_from_zip(buf, prj_dir)
        else:
            return project, "unknown-archive"

        # write metadata.json last (don't let it short-circuit extraction)
        meta_path = os.path.join(prj_dir, "metadata.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(j, f, ensure_ascii=False, indent=2)

        # mark success
        write_marker(prj_dir)

        if polite_delay:
            time.sleep(polite_delay)

        return project, "ok"
    except requests.HTTPError as e:
        return (
            project,
            f"http-{e.response.status_code if e.response else 'err'}",
        )
    except Exception as e:
        return project, f"err-{type(e).__name__}"
    finally:
        s.close()


# -------- driver --------
def main():
    ap = argparse.ArgumentParser(
        description="Download latest source files (.py + metadata & build files) only for installable projects."
    )
    ap.add_argument("-o", "--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument(
        "--polite-delay",
        type=float,
        default=0.05,
        help="sleep per job to reduce load",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit number of projects (testing)",
    )

    args = ap.parse_args()

    ensure_dir(args.outdir)

    idx = make_session()
    projects = list_all_projects(idx)
    idx.close()

    if args.limit:
        projects = projects[: args.limit]
    stats = {}
    pbar = tqdm(
        total=len(projects),
        desc="Extracting installable latest sources",
        unit="pkg",
        smoothing=0.05,
    )

    def _done(f):
        proj, status = f.result()
        stats[status] = stats.get(status, 0) + 1
        pbar.set_postfix_str(
            f"ok:{stats.get('ok',0)} exist:{stats.get('exists',0)} "
            f"skip:{sum(stats.get(k,0) for k in ['no-stable-release','no-installable-wheel','no-artifact','no-python-classifier','too-old','not-found'])}"
        )
        pbar.update(1)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        session_factory = make_session
        futs = []
        for p in projects:
            fut = ex.submit(
                process_project,
                session_factory,
                p,
                args.outdir,
                args.polite_delay,
            )
            fut.add_done_callback(_done)
            futs.append(fut)
        for fut in cf.as_completed(futs):
            _ = fut.result()

    pbar.close()
    total = sum(stats.values())
    print(
        "Summary:",
        total,
        "processed |",
        " ".join(f"{k}={v}" for k, v in sorted(stats.items())),
    )


if __name__ == "__main__":
    main()
