# soct_3d_reconst — operator quick-start

## a6000 production profile

Launch the 3D reconstruction pipeline with the Phase 7 production profile on the 2× A6000 server:

```bash
nextflow run workflows/reconst_3d/soct_3d_reconst.nf -profile a6000 \
  --input /path/to/subject/input \
  --output /path/to/subject/output
```

Run from the linumpy repository root (`/home/frans/code/linumpy` on the A6000 server).

### Fast-path environment (set globally by `-profile a6000`)

| Variable | Value |
|----------|-------|
| `LINUM_BASIC_DCT_KERNEL` | `tuned` |
| `TORCHINDUCTOR_CACHE_DIR` | `~/.cache/linum-basic/inductor/{torch_version}-{linum_basic_short_sha}` (e.g. `2.12.1+cu130-1f2cfd5`) |
| `TORCHINDUCTOR_FX_GRAPH_CACHE` | `1` |

These are injected via the profile `env` block before any child Python process imports PyTorch.

### Concurrency (fix_illumination_basic)

`maxForks = params.gpu_count`; **gpuPinBlock** when `maxForks > 1` (one GPU per concurrent slice task); **gpuExposeAllBlock** when `maxForks == 1` (intra-slice multi-GPU fan-out). The linum-basic wrapper uses `strategy=auto` (no explicit `--strategy` override).

### Detailed evidence workflow

For harness parity checks, regression triage, and the `phase7_integration_check.sh` entry point, see **OPTIMIZATION_RUNBOOK.md** in the linum-basic repository:

`.planning/phases/07-nextflow-pipeline-integration/OPTIMIZATION_RUNBOOK.md` (branch `modernisation`, repo `/home/frans/code/linum-basic`).
