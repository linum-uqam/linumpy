"""Spectral-domain OCT data loader for ThorLabs microscopes."""

import warnings
from pathlib import Path

import numpy as np

from linumpy.geometry import galvo as xyzcorr

# TODO: consider the 'n_repeat' parameter when loading the data
# TODO: reorder the dimension, position, etc to be n_depths, n_alines and n_bscans

_AXIAL_RES_KEYS = ("axial_resolution", "z_resolution", "axial_res", "dz")


class OCT:
    """
    Spectral-domain OCT class to reconstruct the data.

    Parameters
    ----------
    directory: string
        Path to the directory containing the raw tiles and info file.
    axial_res: float, optional
        Axial resolution of the data in microns.
    """

    def __init__(self, directory: Path, axial_res: float = 3.5) -> None:
        self.directory = Path(directory)
        self.info_filename = self.directory / "info.txt"
        self.info = {}
        self.rz = axial_res

        # Read the scan info
        self.read_scan_info(self.info_filename)

    def read_scan_info(self, filename: Path) -> None:
        """Read the scan information file.

        Parameters
        ----------
        filename
            Path to the scan_file written by the OCT (.txt)
        """
        foo = Path(filename).read_text()

        # Process the file input
        foo = foo.split("\n")
        for elem in foo:
            hello = elem.split(": ")
            if len(hello) == 1:
                continue
            key, val = hello
            if val.isnumeric() or val.strip("-").isnumeric():
                val = int(val)
            self.info[key] = val

        self._apply_axial_res_from_info()

    def _apply_axial_res_from_info(self) -> None:
        """Set ``self.rz`` from info.txt when an axial-resolution key is present."""
        for key in _AXIAL_RES_KEYS:
            if key not in self.info:
                continue
            raw = self.info[key]
            try:
                self.rz = float(raw)
            except TypeError, ValueError:
                warnings.warn(f"Ignoring invalid axial resolution {key}={raw!r}", stacklevel=2)
            return

    def load_image(
        self, crop: bool = True, fix_galvo_shift: bool | int | None = True, fix_camera_shift: bool = False
    ) -> np.ndarray:
        """Load an image dataset.

        Parameters
        ----------
        crop
            If crop is True, the galvo returns will be cropped from the volume
        fix_galvo_shift
            If True, the shift caused by the galvo mirror return will be evaluated from the data. If an integer value
            is given, this value will be used to fix the shift. The fix is only applied if detection confidence >= 0.3.
        fix_camera_shift
            If True, the camera shift will be evaluated and compensated from the data. This will detect
            the first pixel of the scan that is always overexposed and shift the data to compensate for this.

        Notes
        -----
        * The returned volume is in ``(Z, Y, X)`` order: depth (z), b-scan (y), a-line (x).
        * When ``info['n_repeat'] > 1``, repeated frames are averaged before galvo/camera correction.
        """
        n_alines = self.info["nx"]
        n_bscans = self.info["ny"]
        n_extra = self.info["n_extra"]
        n_alines_per_bscan = n_alines + n_extra
        n_z = self.info["bottom_z"] - self.info["top_z"] + 1
        n_repeat = self.info.get("n_repeat", 1)

        # Load the fringe
        files = list(self.directory.glob("image_*.bin"))
        files.sort()
        chunks = []
        for file in files:
            with Path(file).open("rb") as f:
                foo = np.fromfile(f, dtype=np.float32)
            n_frames = int(len(foo) / (n_alines_per_bscan * n_z))
            foo = np.reshape(foo, (n_z, n_alines_per_bscan, n_frames), order="F")
            chunks.append(foo)
        vol = np.concatenate(chunks, axis=2) if len(chunks) > 1 else chunks[0]

        if n_repeat > 1:
            n_frames = vol.shape[2]
            if n_frames % n_repeat != 0:
                warnings.warn(
                    f"n_repeat={n_repeat} does not divide frame count {n_frames}; skipping averaging.",
                    stacklevel=2,
                )
            else:
                n_bscans_raw = n_frames // n_repeat
                vol = vol.reshape(n_z, n_alines_per_bscan, n_bscans_raw, n_repeat).mean(axis=3)

        # Compensate camera shift (required for old acquisitions on polymtl server)
        aip = None  # cache for vol.mean(axis=0)
        if fix_camera_shift:
            aip = vol.mean(axis=0)
            pix_max = np.where(aip == aip.max())
            cam_shift = pix_max[0][0]
            vol = np.roll(vol, -cam_shift, axis=1)
            aip = None  # vol was modified; cache is stale

            # Replace the saturated pixel value by its neighbor
            vol[:, 0, 0] = vol[:, 1, 0]

        # Estimate the galvo shift
        if isinstance(fix_galvo_shift, bool) and fix_galvo_shift is True:
            if n_extra == 0:
                warnings.warn("Cannot estimate the shift correction as there are no extra a-lines in the file.", stacklevel=2)
            else:
                if aip is None:
                    aip = vol.mean(axis=0)
                shift, confidence = xyzcorr.detect_galvo_shift(aip, n_pixel_return=n_extra)
                # Only apply fix if confidence is high enough (galvo shift is likely present)
                if confidence >= 0.5:
                    vol = xyzcorr.fix_galvo_shift(vol, shift=shift)
        elif isinstance(fix_galvo_shift, (int, np.integer)) and fix_galvo_shift != 0:
            vol = xyzcorr.fix_galvo_shift(vol, shift=int(fix_galvo_shift))

        # Crop the volume
        # After galvo fix, the galvo return region is shifted to positions n_alines:n_alines+n_extra
        # (i.e., at the END), so we crop [0:n_alines] to remove it
        # Without galvo fix, we also crop [0:n_alines] since galvo return could be anywhere
        if crop:
            vol = vol[:, 0:n_alines, 0:n_bscans]

        # (Z, X, Y) from reshape/crop -> (Z, Y, X) lab convention (D-75)
        return vol.transpose(0, 2, 1)

    @property
    def position_available(self) -> bool:
        """Return True if the position is available in the info.txt file."""
        return "stage_x_pos_mm" in self.info

    @property
    def dimension(self) -> tuple[float, float, float]:
        """Return the OCT physical dimension in mm as ``(X, Y, Z)`` stage extent."""
        try:
            nz = self.shape[2]
            rz = self.rz / 1000.0
            return self.info["width"] / 1000.0, self.info["height"] / 1000.0, nz * rz
        except KeyError:
            return 1, 1, 1

    @property
    def position(self) -> tuple[float, float, float]:
        """Return the OCT physical position in mm. Will be (0, 0, 0) if not found."""
        try:
            x = float(self.info["stage_x_pos_mm"])
            y = float(self.info["stage_y_pos_mm"])
            z = float(self.info["stage_z_pos_mm"])
            return x, y, z
        except KeyError:
            return 0, 0, 0

    @property
    def resolution(self) -> tuple[float, float, float]:
        """Return the OCT physical resolution in mm as ``(rz, ry, rx)`` matching ``(Z, Y, X)``.

        Will be ``(1, 1, 1)`` if lateral keys are missing.
        """
        try:
            rx = self.info["width"] / self.info["nx"] / 1000.0
            ry = self.info["height"] / self.info["ny"] / 1000.0
            rz = self.rz / 1000.0
            return rz, ry, rx
        except KeyError:
            return 1, 1, 1

    @property
    def shape(self) -> tuple[float, float, float]:
        """Return the OCT shape in pixels from the info.txt file. Returns (nx, ny, nz)."""
        nx = self.info["nx"]
        ny = self.info["ny"]
        if "bottom_z" in self.info and "top_z" in self.info:
            nz = self.info["bottom_z"] - self.info["top_z"] + 1
        else:
            nz = self.info.get("n_samples", 0) // 2
        return nx, ny, nz
