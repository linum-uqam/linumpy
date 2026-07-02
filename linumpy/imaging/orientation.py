"""
Utilities for handling 3D volume orientation codes and transformations.

The target convention matches the Allen Atlas RAS+ template as returned by
:func:`linumpy.reference.allen.download_template_ras_aligned`. After
:func:`apply_orientation_transform`, the numpy array obeys:

  - numpy dim 0 → SITK Z → Allen S (Superior)
  - numpy dim 1 → SITK Y → Allen A (Anterior)
  - numpy dim 2 → SITK X → Allen R (Right)

When this volume is handed to
:func:`linumpy.reference.allen.numpy_to_sitk_image` it becomes a SITK image
with axis order ``(X=R, Y=A, Z=S)``, which directly matches the
RAS-aligned Allen template.  Any other target ordering would force the
rigid registration to recover an extra 90° axis swap from gradient steps,
which routinely fails to converge.

The RAS target orientation maps:
  - output dim 0  ←→  Superior (S)
  - output dim 1  ←→  Anterior (A)
  - output dim 2  ←→  Right (R)
"""

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def parse_orientation_code(orientation: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Parse an orientation code and return axis permutation and flips for RAS alignment.

    Parameters
    ----------
    orientation : str
        3-letter code (R/L, A/P, S/I) describing what each *source* axis points to.
        Example: 'AIR' means dim0→Anterior, dim1→Inferior, dim2→Right.

    Returns
    -------
    axis_permutation : tuple of int
        Source indices for each target dimension, such that
        ``np.transpose(volume, axis_permutation)`` produces a volume whose axes are
        ordered (S, A, R) -- matching the Allen RAS+ template convention where:

          - numpy dim 0 → SITK Z → Allen S (Superior)
          - numpy dim 1 → SITK Y → Allen A (Anterior)
          - numpy dim 2 → SITK X → Allen R (Right)
    axis_flips : tuple of int
        Sign for each axis **after** permutation: -1 means flip that axis, +1 means keep.

    Raises
    ------
    ValueError
        If the orientation code is not exactly 3 letters, contains invalid letters,
        or has duplicate axis directions.

    Examples
    --------
    >>> parse_orientation_code('SAR')  # source already in (S, A, R) order -- identity
    ((0, 1, 2), (1, 1, 1))
    >>> parse_orientation_code('PIR')  # common OCT orientation
    ((1, 0, 2), (-1, -1, 1))
    """
    if len(orientation) != 3:
        raise ValueError(f"Orientation code must be 3 letters, got '{orientation}'")

    orientation = orientation.upper()

    # Map each letter to the TARGET numpy dimension and the sign for that direction.
    # Target dimensions (after permutation) match the Allen RAS+ template's numpy
    # ordering ((S, A, R) → SITK (R, A, S)):
    #   dim 0 → S (Superior)    letter 'S' → same direction, 'I' → flipped
    #   dim 1 → A (Anterior)    letter 'A' → same direction, 'P' → flipped
    #   dim 2 → R (Right)       letter 'R' → same direction, 'L' → flipped
    letter_map = {
        "S": (0, 1),
        "I": (0, -1),  # target dim 0 (Superior)
        "A": (1, 1),
        "P": (1, -1),  # target dim 1 (Anterior)
        "R": (2, 1),
        "L": (2, -1),  # target dim 2 (Right)
    }

    source_to_target = {}
    axes_used = set()

    for source_dim, letter in enumerate(orientation):
        if letter not in letter_map:
            raise ValueError(f"Invalid orientation letter '{letter}'. Use R/L, A/P, or S/I.")
        target_dim, sign = letter_map[letter]
        if target_dim in axes_used:
            raise ValueError(
                f"Duplicate axis direction in orientation code '{orientation}': "
                f"letter '{letter}' maps to an already-used target axis."
            )
        axes_used.add(target_dim)
        source_to_target[source_dim] = (target_dim, sign)

    if axes_used != {0, 1, 2}:
        raise ValueError(f"Orientation code '{orientation}' must specify all three axes (S/I, R/L, A/P).")

    # Build target_dim -> (source_dim, sign)
    target_to_source = {v[0]: (k, v[1]) for k, v in source_to_target.items()}

    axis_permutation = tuple(target_to_source[i][0] for i in range(3))
    axis_flips = tuple(target_to_source[i][1] for i in range(3))

    return axis_permutation, axis_flips


def apply_orientation_transform(
    volume: np.ndarray, permutation: tuple[int, ...], flips: tuple[int, ...] | None = None
) -> np.ndarray:
    """
    Reorient a 3D volume by applying an axis permutation followed by axis flips.

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D volume (any shape).
    permutation : tuple of int
        Axis permutation as returned by :func:`parse_orientation_code`.
        ``np.transpose(volume, permutation)`` is applied first.
    flips : tuple of int
        Sign for each axis after permutation.  A value of -1 means that axis is
        flipped (``np.flip``); +1 means the axis is kept as-is.

    Returns
    -------
    np.ndarray
        Reoriented volume.  The returned array may share memory with *volume*
        for the non-contiguous transpose, but ``np.flip`` produces a view, so
        callers should copy if in-place modification is needed.
    """
    result = np.transpose(volume, permutation)
    if flips is not None:
        for axis, flip in enumerate(flips):
            if flip < 0:
                result = np.flip(result, axis=axis)
    return result


def reorder_resolution(resolution: tuple[float, ...], permutation: tuple[int, ...]) -> tuple[float, ...]:
    """
    Reorder a per-axis resolution tuple to match the axis permutation.

    Parameters
    ----------
    resolution : tuple of float
        Per-axis resolution values, one per spatial dimension.
    permutation : tuple of int
        Axis permutation as returned by :func:`parse_orientation_code`.

    Returns
    -------
    tuple of float
        Resolution values reordered so that ``reordered[i] == resolution[permutation[i]]``,
        i.e. the resolution now corresponds to the target axis ordering.
    """
    return tuple(resolution[permutation[i]] for i in range(len(permutation)))


def sitk_transform_to_affine_matrix(transform: sitk.Transform) -> np.ndarray:
    """
    Convert a SimpleITK transform to a 4x4 affine matrix.

    Parameters
    ----------
    transform : sitk.Transform
        SimpleITK Euler3DTransform or AffineTransform.

    Returns
    -------
    np.ndarray
        4x4 affine matrix in (Z, Y, X) coordinate ordering, matching the
        OME-NGFF axis declaration used by the pipeline.
    """
    if isinstance(transform, sitk.Euler3DTransform):
        center = np.array(transform.GetCenter())
        params = transform.GetParameters()
        rx, ry, rz = params[:3]
        translation = np.array(params[3:6])

        cx, cy, cz = np.cos([rx, ry, rz])
        sx, sy, sz = np.sin([rx, ry, rz])

        rotation = np.array(
            [
                [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                [-sy, cy * sx, cy * cx],
            ]
        )

        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation + center - rotation @ center
    elif isinstance(transform, sitk.AffineTransform):
        rotation = np.array(transform.GetMatrix()).reshape(3, 3)
        translation = np.array(transform.GetTranslation())
        center = np.array(transform.GetCenter())

        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation + center - rotation @ center
    else:
        raise ValueError(f"Unsupported transform type: {type(transform)}")

    permute = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    return permute @ matrix @ permute.T


def store_transform_in_metadata(zarr_path: Path, transform: sitk.Transform) -> None:
    """Store a transform in OME-Zarr metadata as an affine transform."""
    affine_matrix = sitk_transform_to_affine_matrix(transform)
    zattrs_path = Path(zarr_path) / ".zattrs"

    if not zattrs_path.exists():
        raise FileNotFoundError(f".zattrs not found: {zarr_path}")

    with Path(zattrs_path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    affine_transform = {"type": "affine", "affine": affine_matrix.flatten().tolist()}

    multiscales = metadata.get("multiscales", [])
    if not multiscales:
        raise ValueError("No multiscales entry found in metadata")

    for dataset in multiscales[0].get("datasets", []):
        existing = dataset.get("coordinateTransformations", [])
        dataset["coordinateTransformations"] = [affine_transform, *existing]

    with Path(zattrs_path).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def compute_centered_reference_and_transform(
    moving_sitk: sitk.Image, transform: sitk.Transform, output_spacing: tuple[float, float, float] | None = None
) -> tuple[sitk.Image, sitk.Transform]:
    """
    Compute a centered reference image and the composite transform for resampling.

    Parameters
    ----------
    moving_sitk : sitk.Image
        The input moving image.
    transform : sitk.Transform
        Transform to apply (moving -> fixed/RAS space).
    output_spacing : tuple of float, optional
        Output voxel spacing. If None, uses the moving image spacing.

    Returns
    -------
    tuple[sitk.Image, sitk.Transform]
        Reference image with origin at 0 and the composite transform that maps
        output coordinates back into the moving image.
    """
    if output_spacing is None:
        output_spacing = moving_sitk.GetSpacing()

    size = moving_sitk.GetSize()
    corners = [
        (0, 0, 0),
        (size[0] - 1, 0, 0),
        (0, size[1] - 1, 0),
        (0, 0, size[2] - 1),
        (size[0] - 1, size[1] - 1, 0),
        (size[0] - 1, 0, size[2] - 1),
        (0, size[1] - 1, size[2] - 1),
        (size[0] - 1, size[1] - 1, size[2] - 1),
    ]

    inverse_transform = transform.GetInverse()
    transformed_points = []
    for idx in corners:
        point = moving_sitk.TransformContinuousIndexToPhysicalPoint(idx)
        transformed_points.append(inverse_transform.TransformPoint(point))

    points = np.array(transformed_points)
    points_min = points.min(axis=0)
    points_max = points.max(axis=0)

    spacing = np.array(output_spacing)
    extent = points_max - points_min
    new_size = np.ceil(extent / spacing).astype(int)

    reference = sitk.Image([int(axis_size) for axis_size in new_size], moving_sitk.GetPixelIDValue())
    reference.SetSpacing(tuple(spacing))
    reference.SetOrigin((0.0, 0.0, 0.0))
    reference.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    shift_transform = sitk.TranslationTransform(3)
    shift_transform.SetOffset(tuple(points_min))

    composite = sitk.CompositeTransform(3)
    composite.AddTransform(transform)
    composite.AddTransform(shift_transform)

    return reference, composite
