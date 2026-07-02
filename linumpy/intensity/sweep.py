"""Illumination-correction parameter sweep core logic.

Extracted from ``scripts/illumination/linum_sweep_illumination.py`` (D-84 #6,
D-86): pure fitting/scoring and parameter-grid helpers that drive the
sweep tool, kept independent of matplotlib visualization and zarr I/O so
they can be exercised directly on small synthetic volumes.
"""

import itertools

import numpy as np


def split_into_tiles(plane: np.ndarray, tile_shape: tuple[int, int]) -> np.ndarray:
    """Split a 2-D plane into a stack of non-overlapping tiles (row-major order)."""
    ty, tx = tile_shape
    ny, nx = plane.shape[0] // ty, plane.shape[1] // tx
    tiles = np.empty((ny * nx, ty, tx), dtype=plane.dtype)
    for i in range(ny):
        for j in range(nx):
            tiles[i * nx + j] = plane[i * ty : (i + 1) * ty, j * tx : (j + 1) * tx]
    return tiles


def assemble_from_tiles(tiles: np.ndarray, plane_shape: tuple[int, int], tile_shape: tuple[int, int]) -> np.ndarray:
    """Reassemble a plane from tiles produced by :func:`split_into_tiles`."""
    ty, tx = tile_shape
    ny, nx = plane_shape[0] // ty, plane_shape[1] // tx
    out = np.zeros(plane_shape, dtype=tiles.dtype)
    for i in range(ny):
        for j in range(nx):
            out[i * ty : (i + 1) * ty, j * tx : (j + 1) * tx] = tiles[i * nx + j]
    return out


def run_one_config(
    vol: np.ndarray,
    tile_shape: tuple[int, int],
    *,
    percentile_max: float | None,
    use_darkfield: bool,
    darkfield_percentile: float,
    fit_max_samples: int,
    max_iterations: int,
    smoothness_flatfield: float | None,
    working_size: int | None,
    apply_z: list[int],
    preview_z: int,
    per_z_fit: bool = False,
    darkfield_smooth_sigma: float = 0.0,
    darkfield_z_window: int = 0,
    flatfield_smooth_sigma: float = 0.0,
    use_gpu: bool = False,
    n_workers: int = 1,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray | None, dict]:
    """Fit linum-basic and apply the model to ``apply_z``.

    When ``per_z_fit=False`` (default) fits a global field on pooled Z planes.
    When ``per_z_fit=True`` fits a separate model per axial plane.

    Returns
    -------
    corrected : dict mapping z-index -> corrected (float32, clipped >= 0)
    flatfield : (ty, tx) float32 array — the model applied to ``preview_z``
    darkfield : (ty, tx) float32 array or None — the model applied to ``preview_z``
    stats     : dict with fit diagnostics for ``preview_z``
    """
    from linum_basic.fit import apply_fit, fit_mosaic
    from linum_basic.mosaic import MosaicGrid

    n_axial = vol.shape[0]
    plane_shape: tuple[int, int] = (vol.shape[1], vol.shape[2])
    th, tw = tile_shape
    h, w = plane_shape
    h_crop = (h // th) * th
    w_crop = (w // tw) * tw
    if h_crop != h or w_crop != w:
        print(f"Cropping mosaic from ({h},{w}) to ({h_crop},{w_crop}) to match tile grid (tile={tile_shape}).")
    vol_crop = vol[:, :h_crop, :w_crop]
    crop_shape: tuple[int, int] = (h_crop, w_crop)

    tiles_per_plane = (crop_shape[0] // tile_shape[0]) * (crop_shape[1] // tile_shape[1])
    if percentile_max is not None:
        print("Note: percentile_max is ignored by linum-basic backend.")
    if darkfield_smooth_sigma > 0:
        print("Note: darkfield_smooth_sigma is ignored by linum-basic backend.")
    if darkfield_z_window != 0:
        print("Note: darkfield_z_window is ignored by linum-basic backend.")
    if flatfield_smooth_sigma > 0:
        print("Note: flatfield_smooth_sigma is ignored by linum-basic backend.")
    if working_size is not None:
        print("Note: working_size is ignored by linum-basic backend.")
    if darkfield_percentile != 5.0:
        print("Note: darkfield_percentile is accepted for compatibility but ignored by linum-basic backend.")

    fit_max_eff = max(fit_max_samples, tiles_per_plane)
    n_planes = min(n_axial, max(1, fit_max_eff // tiles_per_plane))
    z_indices = list(range(n_axial)) if n_planes >= n_axial else np.linspace(0, n_axial - 1, n_planes, dtype=int).tolist()

    field_mode = "per-z" if per_z_fit else "global"
    backend = "numpy"
    device: str | None = None
    if use_gpu:
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                backend = "torch"
                device = "cuda"
                print(f"GPU acceleration enabled: {_torch.cuda.get_device_name(0)}")
            elif _torch.backends.mps.is_available():
                backend = "torch"
                device = "mps"
                print("GPU acceleration enabled: Apple MPS")
            else:
                print("--use_gpu requested but no CUDA/MPS device found; using NumPy CPU.")
        except ImportError:
            print("--use_gpu requested but PyTorch is not installed; using NumPy CPU.")

    basic_kwargs: dict[str, object] = {
        "estimate_darkfield": use_darkfield,
        "max_reweighting_iterations": max_iterations,
        "backend": backend,
        "verbose": False,
    }
    if device is not None:
        basic_kwargs["device"] = device
    if smoothness_flatfield is not None:
        basic_kwargs["l_s"] = smoothness_flatfield

    mosaic = MosaicGrid(array=vol_crop, tile_shape=tile_shape)
    fit = fit_mosaic(
        mosaic,
        z_indices=z_indices,
        field_mode=field_mode,
        basic_kwargs=basic_kwargs,
        n_extra_rows=0,
        n_workers=n_workers,
        verbose=True,
    )

    corrected_crop = np.asarray(apply_fit(mosaic, fit, n_extra_rows=0), dtype=np.float32)
    corrected_full = vol.astype(np.float32, copy=True)
    corrected_full[:, :h_crop, :w_crop] = corrected_crop
    corrected = {z: np.clip(corrected_full[z], 0.0, None) for z in apply_z}

    z_fit_indices = list(getattr(fit, "z_indices", z_indices))
    if fit.flatfields.ndim == 2:
        flatfield = np.asarray(fit.flatfields, dtype=np.float32)
        darkfield = np.asarray(fit.darkfields, dtype=np.float32) if fit.darkfields is not None else None
    else:
        if preview_z in z_fit_indices:
            model_idx = z_fit_indices.index(preview_z)
        else:
            nearest_z = min(z_fit_indices, key=lambda z: abs(z - preview_z))
            model_idx = z_fit_indices.index(nearest_z)
        flatfield = np.asarray(fit.flatfields[model_idx], dtype=np.float32)
        darkfield = np.asarray(fit.darkfields[model_idx], dtype=np.float32) if fit.darkfields is not None else None

    n_fit = len(z_indices) * tiles_per_plane
    df = darkfield
    stats: dict = {
        "n_fit_tiles": n_fit,
        "ff_min": float(flatfield.min()),
        "ff_max": float(flatfield.max()),
        "ff_mean": float(flatfield.mean()),
        "df_min": float(df.min()) if df is not None else None,
        "df_max": float(df.max()) if df is not None else None,
        "df_mean": float(df.mean()) if df is not None else None,
        "smoothness_flatfield": smoothness_flatfield,
        "working_size": working_size,
    }
    return corrected, flatfield, darkfield, stats


def parse_float_none_list(s: str) -> list[float | None]:
    """Parse a comma-separated list of floats, allowing ``'none'`` tokens."""
    result: list[float | None] = []
    for tok in s.split(","):
        tok = tok.strip()
        result.append(None if tok.lower() == "none" else float(tok))
    return result


def parse_bool_list(s: str) -> list[bool]:
    """Parse a comma-separated list of booleans (``true/false/1/0/yes/no``)."""
    result = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok in ("true", "1", "yes"):
            result.append(True)
        elif tok in ("false", "0", "no"):
            result.append(False)
        else:
            msg = f"Cannot parse bool value: {tok!r}. Use true/false."
            raise ValueError(msg)
    return result


def build_sweep_grid(
    p_maxes: list[float | None],
    use_darks: list[bool],
    df_percs: list[float],
    fit_samps: list[int],
    max_iters: list[int],
    smooth_ffs: list[float | None],
    working_sizes: list[float | None],
    per_z_fits: list[bool],
    df_smooth_sigmas: list[float | None],
    df_z_windows: list[int],
    ff_smooth_sigmas: list[float | None],
) -> list[tuple]:
    """Build the de-duplicated cartesian-product sweep grid.

    When ``use_darkfield=False``, darkfield-only params are irrelevant;
    when ``per_z_fit=False``, ``df_z_window`` is irrelevant. Configs that
    collapse to the same effective parameter set after masking those
    irrelevant axes are de-duplicated, keeping the first occurrence.
    """
    raw_grid = list(
        itertools.product(
            p_maxes,
            use_darks,
            df_percs,
            fit_samps,
            max_iters,
            smooth_ffs,
            working_sizes,
            per_z_fits,
            df_smooth_sigmas,
            df_z_windows,
            ff_smooth_sigmas,
        )
    )

    seen: set[tuple] = set()
    configs: list[tuple] = []
    for c in raw_grid:
        pmax, use_dark, df_p, samp, iters, smooth_ff, ws, per_z, df_sig, df_zw, ff_sig = c
        key = (
            pmax,
            use_dark,
            df_p if use_dark else "N/A",
            samp,
            iters,
            smooth_ff,
            ws,
            per_z,
            df_sig if use_dark else "N/A",
            df_zw if (use_dark and per_z) else "N/A",
            ff_sig,
        )
        if key not in seen:
            seen.add(key)
            configs.append(c)

    return configs
