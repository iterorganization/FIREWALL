# Input datasets

FIREWALL's test pipeline needs three input datasets. None of them are stored in
this repository; they are fetched from Zenodo on demand.

| Dataset | File | Size | Purpose |
| --- | --- | --- | --- |
| ITER wall geometry | `newiterwall_offset10cm.h5` | 69 MB | Triangular surface mesh of the ITER first wall, offset 10 cm |
| JOREK particle output | `part_out_eta_10x_fo.h5` | 84 MB | Runaway-electron markers: strike element, energies, incidence angles |
| Energy-deposition maps | `3deg7T.mat` | 7.8 MB | Monte-Carlo deposition profiles in tungsten versus energy and angle |

All three currently come from one Zenodo record,
[10.5281/zenodo.18391920](https://doi.org/10.5281/zenodo.18391920)
("FIREWALL J2 data", CC-BY-4.0), as members of a single 88.5 MB `J2_data.zip`.

## Fetching them

```bash
# everything the pipeline needs
scripts/fetch_data.py

# just the wall geometry
scripts/fetch_data.py --only iter-wall

# where each dataset comes from, without downloading anything
scripts/fetch_data.py --list
```

`tests/run_pipeline.sh` calls the fetcher itself, so running the pipeline is
still a single command.

The fetcher uses only the Python standard library. It resolves each dataset
through the Zenodo REST API rather than guessing a download URL, checks the
downloaded bytes against the checksum the record itself declares, writes
through a temporary file so an interrupted transfer cannot leave a truncated
dataset behind, caches the downloaded archive under `tests/.cache/`, and skips
anything already present unless `--force` is given.

## How datasets are described

Everything about where a dataset lives is in `tests/data_sources.json`. Each
entry is one logical dataset with its own Zenodo record, its own provenance and
its own checksums:

```json
{
  "name": "iter-wall",
  "description": "ITER first-wall triangular surface mesh, offset 10 cm.",
  "zenodo_record": "18391920",
  "archive": "J2_data.zip",
  "members": { "data/newiterwall_offset10cm.h5": "newiterwall_offset10cm.h5" },
  "sha256": { "newiterwall_offset10cm.h5": "df077588e722a7ce17be..." },
  "zenodo_doi": "10.5281/zenodo.18391920",
  "rights_holder": "ITER Organization (geometry); record published by V. J. Svensson",
  "license": "CC-BY-4.0"
}
```

A dataset can also be a file published directly in a record rather than a
member of an archive. In that case drop `archive` and let each key in `members`
name a file in the record:

```json
{
  "name": "iter-wall",
  "zenodo_record": "1234567",
  "members": { "newiterwall_offset10cm.h5": "newiterwall_offset10cm.h5" }
}
```

No code changes are needed to switch between the two forms.

## Checksums

Two independent checks are applied. The downloaded archive is verified against
the md5 that the Zenodo record itself declares, and each extracted file is
verified against the sha256 pinned in the manifest. The second matters because
the first only proves the download matched the record — it would not notice the
record being replaced.

The pinned digests are:

| File | sha256 |
| --- | --- |
| `newiterwall_offset10cm.h5` | `df077588e722a7ce17bedafc6d4a5b86f0373b1d89b0bb0ef0444f99b3b67270` |
| `part_out_eta_10x_fo.h5` | `9ea2509f3c98abb862626cea7104fe92ba4a0f4866b619808c862423d5fb8d94` |
| `3deg7T.mat` | `922d743a4b68a8a34e044fcca8e7338e414214db3daa3b0d51921687b232d94d` |

To re-derive them from a trusted copy:

```bash
scripts/fetch_data.py --print-checksums
```

## The wall geometry and the particle data are coupled

The wall file is a full-torus triangular surface mesh: 962,939 triangles,
R 3.877–8.584 m, Z −3.305–4.827 m, stored as a flat `nodes` array of 9 doubles
per triangle (three vertices, Cartesian x, y, z) plus `ntriangle`.

FIREWALL uses it for exactly two quantities per wall element — the unit surface
normal, which gives the incidence angle, and the element area, taken as half the
cross-product magnitude.

It is important to understand that **the wall mesh cannot be replaced
independently of the particle file.** The particle output identifies where each
marker struck by element index (`i_elm`, negated to give FIREWALL's `wall_id`)
and by the barycentric position within that element (`st`). Those indices refer
to positions in this exact triangulation — `wall_id` is used directly as an
offset of `wall_id * 9` into the `nodes` array. Substituting any other wall
mesh, however physically reasonable, silently repoints every index at an
unrelated triangle. The post-processing scripts index the same mesh.

So a different wall geometry implies regenerating the particle data against it,
not just swapping a file. The manifest keeps `iter-wall` as a separate entry so
that a future mesh *and* its matching particle output can be repointed
independently, but the two must move together.

## Open question: separate citability

The wall geometry is derived from the ITER first-wall CAD description, and the
Zenodo record that carries it is published under CC-BY-4.0. What is still worth
settling is that the geometry is not separately citable: someone who wants only
the wall mesh must download the whole 88.5 MB bundle, and no DOI refers to the
geometry alone.

Publishing it as its own Zenodo record would address that. The manifest is
already structured for it — `iter-wall` is a separate entry, so the move is a
two-field edit: point `zenodo_record` at the new record and replace
`archive`/`members` with the direct-file form shown above. No code changes, and
`scripts/fetch_data.py --only iter-wall` keeps working.

## Reference result

`tests/results/pipeline_results.h5` is committed to the repository as the
reference the pipeline compares against. It contains `surf_temp` and `wall_ids`
for 15421 wall elements. The code version and configuration that produced it are
not currently recorded; they should be, so that a future mismatch can be
attributed to a change in the code rather than a change in the reference.
