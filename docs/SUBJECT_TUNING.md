# Subject-Specific Reconstruction Tuning

Human workflow for tuning the 3D reconstruction pipeline for a single
subject, from first `nextflow.config` to a passing reconstruction.

See also:
- {doc}`Reconstruction Tuning Quick Reference <RECONSTRUCTION_TUNING_QUICKREF>` — symptom → parameter lookup
- {doc}`Reconstruction Parameter Tuning Guide <RECONSTRUCTION_TUNING>` — parameter deep-dives
- `workflows/reconst_3d/subject.config.template` — the config template referenced below

---

## 1. Start from the template

Copy `workflows/reconst_3d/subject.config.template` to
`/scratch/workspace/sub-XX/nextflow.config` on the reconstruction server and
replace every `sub-XX` placeholder with the real subject ID.

```bash
scp workflows/reconst_3d/subject.config.template \
    132.207.157.41:/scratch/workspace/sub-18/nextflow.config
```

## 2. Run `linum-suggest-params`

`linum-suggest-params` reads the subject's acquisition metadata
(`state.json` / `metadata.json`) and emits a config snippet with the
measured tile overlap and other acquisition-derived defaults.

```bash
linum-suggest-params /scratch/workspace/sub-18/mosaic-grids/
```

Copy the emitted `stitch_overlap_fraction` (3D) / `initial_overlap` (2.5D)
value into the subject's `nextflow.config`, replacing the template default.

## 3. Run the pipeline

Run the reconstruction on the server (interactive SSH session — the
pipeline takes hours and must never be started by an agent):

```bash
nextflow run /home/frans/code/linumpy/workflows/reconst_3d/soct_3d_reconst.nf \
  -resume \
  -c /scratch/workspace/sub-18/nextflow.config \
  --input /scratch/workspace/sub-18/mosaic-grids/ \
  --output /scratch/workspace/sub-18/output/
```

## 4. Upstream-first tuning: gates 1-5 before downstream sections

**Do not tune downstream parameters (`bias_*`, `align_to_ras_*`) until gates
1-5 below all PASS.** Tuning downstream first hides upstream problems and
wastes iteration time — a bad stack cannot be fixed by atlas registration.

Download the subject outputs (see [Downloading results](#6-download-and-inspect-results) below),
then walk through gates 1-5 in order:

| Gate | Artifact | PASS condition |
|------|----------|-----------------|
| 1 | `mosaic-grids/shifts_xy.csv` | No large `x_shift_mm`/`y_shift_mm` step without a tile-FOV explanation |
| 2 | `output/detect_rehoming_events/shifts_xy_clean.csv` | `reliable=0` rows have a plan (`common_space_refine_unreliable`) |
| 3 | `output/common_space_previews/*.png` | No obvious slice-to-slice XY jumps |
| 4 | `output/register_pairwise/*_metrics.json` | No excessive `mag` without explanation |
| 5 | `output/stack/stacking_decisions.csv` | No unexplained `transform_loaded=False` gaps |

Fix upstream (gates 1-5) using the `stitch_*`, `detect_rehoming`/`tile_fov_mm`,
`common_space_*`, `registration_*`, and `stack_*` parameters in the template —
see the Quick Reference for symptom-specific fixes.

## 5. Downstream tuning: bias field and atlas (only after gates 1-5 PASS)

Once gates 1-5 pass, tune the downstream sections of the template:

- `correct_bias_field`, `bias_mode`, `bias_strength` — N4 bias field correction
- `align_to_ras_enabled`, `allen_resolution`, `ras_input_orientation`,
  `ras_initial_rotation` — Allen atlas registration

## 6. Download and inspect results

Use the repo-parameterized download script instead of ad hoc `scp`:

```bash
scripts/subject/update_files.sh sub-18
```

The script reads the subject's remote `nextflow.config` to decide which
output categories to download (it will not fetch outputs for disabled
stages, e.g. `output/align_to_ras/` when `align_to_ras_enabled = false`).
Run with `--dry-run` to preview which categories would be downloaded
without contacting the server.

## Re-running after a config change

`nextflow run ... -resume` reuses cached outputs whose task hash is
unchanged. See the diagnostic playbook's re-run table for which stage to
invalidate after each type of config or code change.
