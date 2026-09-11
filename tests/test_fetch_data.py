#!/usr/bin/env python3
"""Offline tests for scripts/fetch_data.py.

Serves a miniature Zenodo API over the loopback interface so the real code
path -- record lookup, checksum verification, download, extraction -- is
exercised without touching the network.

Run with:  python3 tests/test_fetch_data.py
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import socketserver
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FETCH = REPO_ROOT / "scripts" / "fetch_data.py"

RECORD_ID = "18391920"
ARCHIVE = "J2_data.zip"
MEMBERS = {
    "data/newiterwall_offset10cm.h5": b"fake wall geometry",
    "data/part_out_eta_10x_fo.h5": b"fake particle output",
    "data/3deg7T.mat": b"fake deposition maps",
}

failures = 0


def check(condition: bool, message: str) -> None:
    global failures
    if condition:
        print(f"  PASS  {message}")
    else:
        failures += 1
        print(f"  FAIL  {message}")


def build_fixture(root: Path, origin: str) -> Path:
    """Lay out a directory tree that mimics the Zenodo REST API."""
    archive_path = root / ARCHIVE
    with zipfile.ZipFile(archive_path, "w") as zf:
        for name, content in MEMBERS.items():
            zf.writestr(name, content)
    digest = hashlib.md5(archive_path.read_bytes()).hexdigest()

    # /api/records/<id>/           -> index.html holding the record JSON
    # /api/records/<id>/files/<k>/content -> the bytes
    record_dir = root / "www" / "api" / "records" / RECORD_ID
    files_dir = record_dir / "files" / ARCHIVE
    files_dir.mkdir(parents=True)
    (files_dir / "content").write_bytes(archive_path.read_bytes())

    record = {
        "id": int(RECORD_ID),
        "doi": "10.5281/zenodo.0000000",
        "files": [
            {
                "key": ARCHIVE,
                "size": archive_path.stat().st_size,
                "checksum": f"md5:{digest}",
                "links": {"self": f"{origin}/api/records/{RECORD_ID}/files/{ARCHIVE}"},
            }
        ],
    }
    (record_dir / "index.html").write_text(json.dumps(record))
    return root / "www"


def write_manifest(path: Path, sha256_ok: bool = True) -> None:
    """A manifest describing the fixture archive.

    The production manifest pins digests of the real datasets, so the tests
    carry their own pointing at the fixture's contents. With sha256_ok False
    the wall digest is deliberately wrong, to exercise the rejection path.
    """
    def digest(member: str) -> str:
        return hashlib.sha256(MEMBERS[member]).hexdigest()

    wall_digest = digest("data/newiterwall_offset10cm.h5")
    if not sha256_ok:
        wall_digest = "0" * 64

    manifest = {
        "resources": [
            {
                "name": "iter-wall",
                "description": "fixture wall geometry",
                "zenodo_record": RECORD_ID,
                "archive": ARCHIVE,
                "members": {"data/newiterwall_offset10cm.h5": "newiterwall_offset10cm.h5"},
                "sha256": {"newiterwall_offset10cm.h5": wall_digest},
                "rights_holder": "ITER Organization",
                "license": "CC-BY-4.0",
            },
            {
                "name": "jorek-particles",
                "description": "fixture particle output",
                "zenodo_record": RECORD_ID,
                "archive": ARCHIVE,
                "members": {"data/part_out_eta_10x_fo.h5": "part_out_eta_10x_fo.h5"},
                "sha256": {"part_out_eta_10x_fo.h5": digest("data/part_out_eta_10x_fo.h5")},
            },
            {
                "name": "deposition-maps",
                "description": "fixture deposition maps",
                "zenodo_record": RECORD_ID,
                "archive": ARCHIVE,
                "members": {"data/3deg7T.mat": "3deg7T.mat"},
                "sha256": {"3deg7T.mat": digest("data/3deg7T.mat")},
            },
        ]
    }
    path.write_text(json.dumps(manifest, indent=2))


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102 - silence the access log
        pass


def serve(directory: Path) -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(QuietHandler, directory=str(directory))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(FETCH)] + args,
                          capture_output=True, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        www = root / "www"
        www.mkdir()
        httpd, port = serve(www)
        origin = f"http://127.0.0.1:{port}"
        build_fixture(root, origin)
        api = f"{origin}/api"
        dest = root / "data"
        manifest = root / "manifest.json"
        write_manifest(manifest)
        common = ["--api-base", api, "--dest", str(dest), "--manifest", str(manifest)]

        try:
            print("fetching all resources")
            result = run(common)
            check(result.returncode == 0, f"exit 0 (got {result.returncode})")
            if result.returncode != 0:
                print(result.stdout, result.stderr)
            for target in ("newiterwall_offset10cm.h5",
                           "part_out_eta_10x_fo.h5", "3deg7T.mat"):
                check((dest / target).exists(), f"{target} written")
            wall = dest / "newiterwall_offset10cm.h5"
            check(wall.exists() and wall.read_bytes()
                  == MEMBERS["data/newiterwall_offset10cm.h5"],
                  "wall geometry content matches")

            print("second run is a no-op")
            result = run(common)
            check("already present" in result.stdout, "skips files already present")

            print("--only fetches a single resource")
            dest2 = root / "wall-only"
            result = run(["--api-base", api, "--dest", str(dest2),
                          "--manifest", str(manifest), "--only", "iter-wall"])
            check(result.returncode == 0, "exit 0")
            check((dest2 / "newiterwall_offset10cm.h5").exists(), "wall geometry fetched")
            check(not (dest2 / "3deg7T.mat").exists(), "other datasets not fetched")

            print("--list reports provenance without downloading")
            result = run(["--list", "--manifest", str(manifest)])
            check(result.returncode == 0, "exit 0")
            check("ITER Organization" in result.stdout, "names the rights holder")

            print("the production manifest is valid and pins every digest")
            result = run(["--list"])
            check(result.returncode == 0, "exit 0")
            real = json.loads((REPO_ROOT / "tests" / "data_sources.json").read_text())
            names = [r["name"] for r in real["resources"]]
            check(names == ["iter-wall", "jorek-particles", "deposition-maps"],
                  f"declares the expected resources ({names})")
            pins = [v for r in real["resources"] for v in r["sha256"].values()]
            check(all(isinstance(v, str) and len(v) == 64 for v in pins),
                  "every dataset has a pinned sha256")

            print("--print-checksums emits pinnable digests")
            result = run(["--print-checksums", "--dest", str(dest),
                          "--manifest", str(manifest)])
            expected = hashlib.sha256(
                MEMBERS["data/newiterwall_offset10cm.h5"]).hexdigest()
            check(expected in result.stdout, "sha256 of the wall geometry printed")

            print("a wrong pinned digest is rejected")
            bad_manifest = root / "bad_manifest.json"
            write_manifest(bad_manifest, sha256_ok=False)
            dest4 = root / "badpin"
            result = run(["--api-base", api, "--dest", str(dest4),
                          "--manifest", str(bad_manifest),
                          "--cache-dir", str(root / "cache-badpin"),
                          "--only", "iter-wall"])
            check(result.returncode != 0, "non-zero exit")
            check("Checksum mismatch" in result.stderr, "reports the mismatch")

            print("a corrupted download is rejected")
            bad = www / "api" / "records" / RECORD_ID / "files" / ARCHIVE / "content"
            bad.write_bytes(b"corrupted")
            dest3 = root / "corrupt"
            result = run(["--api-base", api, "--dest", str(dest3),
                          "--manifest", str(manifest),
                          "--cache-dir", str(root / "cache-corrupt")])
            check(result.returncode != 0, "non-zero exit")
            check("Checksum mismatch" in result.stderr, "reports a checksum mismatch")
            check(not (dest3 / "newiterwall_offset10cm.h5").exists(),
                  "nothing written on mismatch")

            print("an unknown resource name is rejected")
            result = run(["--only", "nope", "--list", "--manifest", str(manifest)])
            check(result.returncode != 0, "non-zero exit")
            check("Unknown resource" in result.stderr, "explains the problem")
        finally:
            httpd.shutdown()
            httpd.server_close()

    print()
    if failures:
        print(f"{failures} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
