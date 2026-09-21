#!/usr/bin/env python3
"""Fetch the FIREWALL input datasets from Zenodo.

Every dataset the test pipeline needs is described in a manifest
(``tests/data_sources.json``) as a named resource carrying its own Zenodo
record, its own provenance and its own checksums. Nothing about where a
dataset lives is hard-coded here, so a dataset can be repointed at a
different record -- for example moving the ITER wall geometry to a record
published by the ITER Organization -- by editing the manifest alone.

Files are resolved through the Zenodo REST API rather than by guessing a
download URL, which means the record's own checksum and size are known
before anything is written to disk.

Uses only the Python standard library.

Examples
--------
Fetch everything the pipeline needs::

    scripts/fetch_data.py

Fetch only the wall geometry::

    scripts/fetch_data.py --only iter-wall

Show where each dataset comes from, without downloading::

    scripts/fetch_data.py --list

Compute the digests to pin in the manifest::

    scripts/fetch_data.py --print-checksums
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_API_BASE = "https://zenodo.org/api"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "data_sources.json"
DEFAULT_DEST = REPO_ROOT / "tests" / "data"

CHUNK = 1 << 20  # 1 MiB


class FetchError(RuntimeError):
    """Raised for any unrecoverable problem while fetching a dataset."""


def display_path(path: Path) -> str:
    """Path relative to the repository when it is inside it, else absolute."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    try:
        with path.open() as handle:
            document = json.load(handle)
    except FileNotFoundError:
        raise FetchError(f"Manifest not found: {path}")
    except json.JSONDecodeError as exc:
        raise FetchError(f"Manifest {path} is not valid JSON: {exc}")

    resources = document.get("resources")
    if not isinstance(resources, list) or not resources:
        raise FetchError(f"Manifest {path} declares no resources.")

    for resource in resources:
        for key in ("name", "zenodo_record", "members"):
            if key not in resource:
                raise FetchError(
                    f"Resource {resource.get('name', '<unnamed>')!r} in {path} "
                    f"is missing the required key {key!r}."
                )
    return resources


def outputs_of(resource: dict) -> list[str]:
    """Filenames this resource produces in the destination directory."""
    return list(resource["members"].values())


# --------------------------------------------------------------------------
# Zenodo
# --------------------------------------------------------------------------

def fetch_record(api_base: str, record_id: str, cache: dict) -> dict:
    """Return the Zenodo record metadata, fetching it at most once."""
    if record_id in cache:
        return cache[record_id]

    url = f"{api_base.rstrip('/')}/records/{record_id}"
    try:
        with urllib.request.urlopen(url) as response:
            record = json.load(response)
    except urllib.error.HTTPError as exc:
        raise FetchError(f"Zenodo returned HTTP {exc.code} for record {record_id} ({url}).")
    except urllib.error.URLError as exc:
        raise FetchError(f"Could not reach Zenodo at {url}: {exc.reason}")
    except ValueError as exc:
        raise FetchError(f"Malformed record URL {url!r}: {exc}")

    cache[record_id] = record
    return record


def find_file(record: dict, filename: str) -> dict:
    """Locate one file entry inside a Zenodo record."""
    files = record.get("files") or []
    for entry in files:
        if entry.get("key") == filename or entry.get("filename") == filename:
            return entry
    available = ", ".join(sorted(e.get("key") or e.get("filename", "?") for e in files)) or "none"
    raise FetchError(
        f"Record {record.get('id', '?')} does not contain {filename!r}. Available files: {available}"
    )


def download_url_for(record: dict, entry: dict, api_base: str) -> str:
    """Best available download URL for a file entry."""
    links = entry.get("links") or {}
    for key in ("content", "self", "download"):
        url = links.get(key)
        if not url:
            continue
        # The 'self' link of the modern API is the file's metadata document,
        # not its bytes; the bytes live one level down at /content.
        if key == "self" and not url.endswith("/content"):
            url = url.rstrip("/") + "/content"
        # Zenodo returns absolute links, but tolerate a relative one rather
        # than handing urllib something it cannot parse.
        return urllib.parse.urljoin(api_base.rstrip("/") + "/", url)
    name = entry.get("key") or entry.get("filename")
    return f"{api_base.rstrip('/')}/records/{record['id']}/files/{name}/content"


# --------------------------------------------------------------------------
# download and verification
# --------------------------------------------------------------------------

def digest_file(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download(url: str, destination: Path, expected: str | None) -> None:
    """Download ``url`` to ``destination``, verifying Zenodo's own checksum.

    ``expected`` is the record's checksum in Zenodo's ``"<algo>:<hex>"`` form.
    The write is staged through a temporary file so an interrupted transfer
    can never leave a half-written dataset behind.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    algorithm, _, expected_hex = (expected or "").partition(":")
    hasher = hashlib.new(algorithm) if expected_hex else None

    print(f"    downloading {url}")
    tmp_fd, tmp_name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with urllib.request.urlopen(url) as response, open(tmp_fd, "wb") as out:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
                if hasher is not None:
                    hasher.update(block)
    except urllib.error.HTTPError as exc:
        tmp_path.unlink(missing_ok=True)
        raise FetchError(f"Zenodo returned HTTP {exc.code} for {url}.")
    except urllib.error.URLError as exc:
        tmp_path.unlink(missing_ok=True)
        raise FetchError(f"Could not download {url}: {exc.reason}")
    except ValueError as exc:
        tmp_path.unlink(missing_ok=True)
        raise FetchError(f"Malformed download URL {url!r}: {exc}")
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    if hasher is not None and hasher.hexdigest() != expected_hex:
        tmp_path.unlink(missing_ok=True)
        raise FetchError(
            f"Checksum mismatch for {url}\n"
            f"  record says  {algorithm}:{expected_hex}\n"
            f"  downloaded   {algorithm}:{hasher.hexdigest()}"
        )
    if hasher is None:
        print(f"    warning: record declares no checksum for {destination.name}")

    tmp_path.replace(destination)


def verify_pinned(resource: dict, dest: Path) -> None:
    """Check extracted files against the digests pinned in the manifest."""
    pinned = resource.get("sha256") or {}
    for filename, expected in pinned.items():
        if not expected:
            continue
        actual = digest_file(dest / filename)
        if actual != expected:
            raise FetchError(
                f"Checksum mismatch for {filename}\n"
                f"  manifest says sha256:{expected}\n"
                f"  on disk       sha256:{actual}"
            )
        print(f"    verified {filename} against pinned sha256")


def extract(archive: Path, members: dict[str, str], dest: Path) -> None:
    """Extract selected members of a zip archive into ``dest``, flattened."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        available = set(zf.namelist())
        for member, target in members.items():
            if member not in available:
                raise FetchError(
                    f"{archive.name} does not contain {member!r}.\n"
                    f"  Archive contains: {', '.join(sorted(available))}"
                )
            # Never trust a path from the archive: write only to dest/<basename>.
            out_path = dest / Path(target).name
            with zf.open(member) as src, out_path.open("wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            print(f"    extracted {member} -> {display_path(out_path)}")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def process(resource: dict, dest: Path, api_base: str, cache: dict,
            cache_dir: Path, force: bool) -> None:
    name = resource["name"]
    produced = [dest / filename for filename in outputs_of(resource)]

    if not force and all(path.exists() for path in produced):
        print(f"  {name}: already present, skipping "
              f"({', '.join(p.name for p in produced)})")
        return

    print(f"  {name}: {resource.get('description', '')}")
    record = fetch_record(api_base, str(resource["zenodo_record"]), cache)
    doi = record.get("doi") or record.get("metadata", {}).get("doi")
    print(f"    record {resource['zenodo_record']}"
          + (f"  doi:{doi}" if doi else ""))

    archive_name = resource.get("archive")
    if archive_name:
        entry = find_file(record, archive_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive_path = cache_dir / archive_name
        if force or not archive_path.exists():
            download(download_url_for(record, entry, api_base), archive_path,
                     entry.get("checksum"))
        else:
            print(f"    using cached {display_path(archive_path)}")
        extract(archive_path, resource["members"], dest)
    else:
        # No archive: each member key is a file in the record itself.
        for member, target in resource["members"].items():
            entry = find_file(record, member)
            download(download_url_for(record, entry, api_base),
                     dest / Path(target).name, entry.get("checksum"))

    verify_pinned(resource, dest)


def list_resources(resources: list[dict]) -> None:
    for resource in resources:
        print(f"{resource['name']}")
        print(f"  description   : {resource.get('description', '-')}")
        print(f"  zenodo record : {resource['zenodo_record']}"
              + (f" ({resource['archive']})" if resource.get("archive") else ""))
        print(f"  provides      : {', '.join(outputs_of(resource))}")
        print(f"  rights holder : {resource.get('rights_holder', '-')}")
        print(f"  license       : {resource.get('license', '-')}")
        print(f"  provenance    : {resource.get('provenance', '-')}")
        print()


def print_checksums(resources: list[dict], dest: Path) -> int:
    """Print sha256 digests of the files on disk, ready to paste into the manifest."""
    missing = 0
    for resource in resources:
        print(f"{resource['name']}:")
        print('  "sha256": {')
        entries = []
        for filename in outputs_of(resource):
            path = dest / filename
            if path.exists():
                entries.append(f'    "{filename}": "{digest_file(path)}"')
            else:
                entries.append(f'    "{filename}": null')
                missing += 1
        print(",\n".join(entries))
        print("  }")
    if missing:
        print(f"\n{missing} file(s) not present in {dest}; fetch them first.",
              file=sys.stderr)
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the FIREWALL input datasets from Zenodo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"dataset manifest (default: {display_path(DEFAULT_MANIFEST)})")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"where to place the datasets (default: {display_path(DEFAULT_DEST)})")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where to keep downloaded archives (default: <dest>/../.cache)")
    parser.add_argument("--only", action="append", metavar="NAME", default=None,
                        help="fetch only this resource; may be repeated")
    parser.add_argument("--force", action="store_true",
                        help="re-download and re-extract even if the files are present")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE,
                        help=f"Zenodo API base URL (default: {DEFAULT_API_BASE})")
    parser.add_argument("--list", action="store_true",
                        help="show the provenance of each dataset and exit")
    parser.add_argument("--print-checksums", action="store_true",
                        help="print sha256 digests of the files on disk and exit")
    args = parser.parse_args(argv)

    try:
        resources = load_manifest(args.manifest)

        if args.only:
            known = {r["name"] for r in resources}
            unknown = set(args.only) - known
            if unknown:
                raise FetchError(
                    f"Unknown resource(s): {', '.join(sorted(unknown))}. "
                    f"Known: {', '.join(sorted(known))}"
                )
            resources = [r for r in resources if r["name"] in args.only]

        if args.list:
            list_resources(resources)
            return 0

        if args.print_checksums:
            return print_checksums(resources, args.dest)

        cache_dir = args.cache_dir or (args.dest.parent / ".cache")
        record_cache: dict = {}
        print(f"Fetching {len(resources)} dataset(s) into {args.dest}")
        for resource in resources:
            process(resource, args.dest, args.api_base, record_cache,
                    cache_dir, args.force)
        print("Done.")
        return 0

    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
