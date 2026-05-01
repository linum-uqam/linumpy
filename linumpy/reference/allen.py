"""Methods to download data from the Allen Institute."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import requests
import SimpleITK as sitk
from tqdm import tqdm

AVAILABLE_RESOLUTIONS = [10, 25, 50, 100]


def numpy_to_sitk_image(volume: np.ndarray, spacing: tuple | Sequence, cast_dtype: type | None = None) -> sitk.Image:
    """Convert numpy array (Z, Y, X) to SimpleITK image format.

    Parameters
    ----------
    volume : np.ndarray
        3D volume with shape (Z, Y, X) matching the project-wide convention
        (axis 0 = Z/depth, axis 1 = Y/row, axis 2 = X/column).
    spacing : tuple
        Voxel spacing in mm as (res_z, res_y, res_x).
    cast_dtype : numpy dtype or None
        If provided, cast the volume to this dtype before creating the SITK image
        (useful for registration where float32 is expected). If None, preserve
        the input numpy dtype.

    Returns
    -------
    sitk.Image
        SimpleITK image with proper spacing and orientation
    """
    # sitk.GetImageFromArray interprets a numpy array with shape (Z, Y, X) as a
    # SITK image with size (X, Y, Z), so no transpose is needed.  The SITK call
    # copies the buffer into its own storage, so we only allocate an extra
    # numpy array when an explicit dtype cast is requested.
    vol_for_sitk = volume.astype(cast_dtype, copy=False) if cast_dtype is not None else volume
    vol_sitk = sitk.GetImageFromArray(vol_for_sitk)
    # Spacing: SimpleITK uses (X, Y, Z) = (width, height, depth).
    # Our spacing is (res_z, res_y, res_x), so SITK spacing is (res_x, res_y, res_z).
    vol_sitk.SetSpacing([spacing[2], spacing[1], spacing[0]])
    vol_sitk.SetOrigin([0, 0, 0])
    vol_sitk.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    return vol_sitk


def download_template(resolution: int, cache: bool = True, cache_dir: str = ".data/") -> sitk.Image:
    """Download a 3D average mouse brain.

    Parameters
    ----------
    resolution
        Allen template resolution in micron. Must be 10, 25, 50 or 100.
    cache
        Keep the downloaded volume in cache
    cache_dir
        Cache directory

    Returns
    -------
    Allen average mouse brain.
    """
    assert resolution in AVAILABLE_RESOLUTIONS

    # Preparing the cache directory
    output = Path(cache_dir)
    output.mkdir(exist_ok=True, parents=True)

    # Preparing the filenames
    nrrd_file = output / f"allen_template_{resolution}um.nrrd"

    # Preparing the request
    url = f"http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/average_template/average_template_{int(resolution)}.nrrd"

    # Check that the data is in cache
    if not (nrrd_file.is_file()):
        # Download the template
        response = requests.get(url, stream=True)
        with Path(nrrd_file).open("wb") as f:
            f.writelines(tqdm(response.iter_content()))

    # Loading the nrrd file
    vol = sitk.ReadImage(str(nrrd_file))

    # Remove the file from cache
    if not cache:
        nrrd_file.unlink()  # Removes the nrrd file

    return vol


def download_template_ras_aligned(resolution: int, cache: bool = True, cache_dir: str = ".data/") -> sitk.Image:
    """Download a 3D average mouse brain and align it to RAS+ orientation.

    The Allen CCF v3 template is stored in PIR orientation
    (SITK axes ``(X, Y, Z) = (AP, DV, ML)`` with ``+X = Posterior``,
    ``+Y = Inferior``, ``+Z = Right``).  Converting to RAS+
    (``+X = Right``, ``+Y = Anterior``, ``+Z = Superior``) requires
    ``PermuteAxes((2, 0, 1))`` followed by flipping **both** the Y and Z
    axes (I → S and P → A).

    Parameters
    ----------
    resolution
        Allen template resolution in micron. Must be 10, 25, 50 or 100.
    cache
        Keep the downloaded volume in cache
    cache_dir
        Cache directory

    Returns
    -------
    Allen average mouse brain in RAS+ orientation.
    """
    vol = download_template(resolution, cache, cache_dir)

    # Preparing the affine to align the template in the RAS+
    r_mm = resolution / 1e3  # Convert the resolution from micron to mm
    vol.SetSpacing([r_mm] * 3)  # Set the spacing in mm
    # Ensure origin/direction are standardized so physical coordinates are stable
    vol.SetOrigin([0.0, 0.0, 0.0])
    vol.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

    # Convert PIR → RAS:
    #   PermuteAxes((2, 0, 1)) maps (P, I, R) → (R, P, I)
    #   Flip Y (P → A) and Z (I → S) to reach (R, A, S).
    vol = sitk.PermuteAxes(vol, (2, 0, 1))
    vol = sitk.Flip(vol, (False, True, True))
    # After permuting/flipping, also ensure origin/direction are identity/zero
    vol.SetOrigin([0.0, 0.0, 0.0])
    vol.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

    return vol


def register_3d_rigid_to_allen(
    moving_image: np.ndarray,
    moving_spacing: tuple,
    allen_resolution: int = 100,
    metric: str = "MI",
    max_iterations: int = 1000,
    verbose: bool = False,
    progress_callback: Callable[[Any], None] | None = None,
    initial_rotation_deg: tuple = (0.0, 0.0, 0.0),
) -> tuple:
    """Perform 3D rigid registration of a brain volume to the Allen atlas.

    Parameters
    ----------
    moving_image : np.ndarray
        3D brain volume to register (shape: Z, Y, X)
    moving_spacing : tuple
        Voxel spacing in mm (res_z, res_y, res_x)
    allen_resolution : int
        Allen template resolution in micron (default: 100)
    metric : str
        Similarity metric: 'MI' (mutual information), 'MSE', 'CC' (correlation),
        or 'AntsCC' (ANTS correlation)
    max_iterations : int
        Maximum number of iterations
    verbose : bool
        Print registration progress
    progress_callback : callable, optional
        Callback function called on each iteration with the registration method.
        Function signature: callback(registration_method)
    initial_rotation_deg : tuple, optional
        Initial rotation in degrees (rx, ry, rz) applied before optimization.

    Returns
    -------
    transform : sitk.Euler3DTransform
        Rigid transform to align moving_image to Allen atlas
    stop_condition : str
        Optimizer stopping condition
    error : float
        Final registration metric value
    """
    # Download and prepare Allen atlas in RAS orientation
    allen_atlas = download_template_ras_aligned(allen_resolution, cache=True)

    # If the moving image is coarser than the Allen atlas along any axis,
    # downsample the atlas to match the moving resolution.  The registration
    # cost is dominated by the fixed (Allen) image size, so downsampling the
    # atlas up-front is much cheaper than upsampling moving to a finer grid
    # that carries no additional information.
    moving_min_spacing_mm = min(moving_spacing)
    allen_spacing_mm = allen_atlas.GetSpacing()
    allen_min_spacing_mm = min(allen_spacing_mm)
    if moving_min_spacing_mm > allen_min_spacing_mm * 1.2:
        target_spacing_mm = float(moving_min_spacing_mm)
        allen_size = allen_atlas.GetSize()
        new_size = [max(1, round(sz * sp / target_spacing_mm)) for sz, sp in zip(allen_size, allen_spacing_mm, strict=False)]
        ref = sitk.Image(new_size, allen_atlas.GetPixelIDValue())
        ref.SetOrigin(allen_atlas.GetOrigin())
        ref.SetDirection(allen_atlas.GetDirection())
        ref.SetSpacing((target_spacing_mm,) * 3)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        allen_atlas = resampler.Execute(allen_atlas)
        if verbose:
            print(
                f"Downsampled Allen atlas to match moving spacing: "
                f"{allen_spacing_mm} mm → {allen_atlas.GetSpacing()} mm, "
                f"size {allen_size} → {allen_atlas.GetSize()}"
            )

    # Crop moving image to tissue bounding box to reduce volume size.
    # Large motor drift during acquisition inflates the canvas with empty space,
    # causing the Allen-domain resampling to clip away brain tissue.  Cropping
    # first keeps the volume compact so most of the brain survives resampling,
    # giving the optimizer a much better cost-function landscape.
    margin_voxels = 10
    crop_origin_mm = (0.0, 0.0, 0.0)  # physical offset in (Z, Y, X) order
    nonzero_coords = np.nonzero(moving_image)
    if len(nonzero_coords[0]) > 0:
        bbox_slices = tuple(
            slice(
                max(0, int(dim.min()) - margin_voxels),
                min(moving_image.shape[ax], int(dim.max()) + margin_voxels + 1),
            )
            for ax, dim in enumerate(nonzero_coords)
        )
        crop_origin_mm = (
            bbox_slices[0].start * moving_spacing[0],
            bbox_slices[1].start * moving_spacing[1],
            bbox_slices[2].start * moving_spacing[2],
        )
        cropped = moving_image[bbox_slices]
        if verbose:
            print(f"Cropped tissue bounding box: {moving_image.shape} -> {cropped.shape}")
        moving_image = cropped

    # Convert moving image to SimpleITK format.
    # Origin stays at (0,0,0) so the compact brain sits at the start of physical
    # space and overlaps with the Allen atlas domain during resampling.  The crop
    # offset is added to the final transform's translation after registration so
    # the transform remains valid for the original (uncropped) full volume.
    moving_sitk = numpy_to_sitk_image(moving_image, moving_spacing)

    # Compute a preliminary brain centre BEFORE any resampling.
    # This is used as the fallback only when needs_resample=False (images already
    # share the same physical space).  When resampling IS needed, this value is
    # overwritten below with the centroid of the clipped brain within the Allen
    # domain, because the full-brain geometric centre can be tens of mm outside
    # the Allen atlas extent and would produce a translation that maps every
    # Allen voxel outside the resampled moving image buffer.
    original_moving_size = moving_sitk.GetSize()
    original_moving_center_idx = [s / 2.0 for s in original_moving_size]
    original_moving_center = np.array(moving_sitk.TransformContinuousIndexToPhysicalPoint(original_moving_center_idx))

    # Resample moving image to match Allen atlas spacing and size for better registration.
    # NOTE: we deliberately keep the original moving center computed above so that the
    # centre-aligned fallback initialisation is always correct even after resampling.
    allen_spacing = allen_atlas.GetSpacing()
    allen_size = allen_atlas.GetSize()
    moving_spacing_sitk = moving_sitk.GetSpacing()
    moving_size_sitk = moving_sitk.GetSize()

    # Check if resampling is needed (if spacing differs significantly or sizes are very different)
    spacing_ratio = np.array(allen_spacing) / np.array(moving_spacing_sitk)
    size_ratio = np.array(allen_size, dtype=float) / np.array(moving_size_sitk, dtype=float)

    # Resample if spacing differs by more than 10% or if volumes are very different sizes
    needs_resample = np.any(np.abs(spacing_ratio - 1.0) > 0.1) or np.any(size_ratio < 0.5) or np.any(size_ratio > 2.0)

    if needs_resample:
        if verbose:
            print(
                f"Resampling moving image from {moving_spacing_sitk} mm, size {moving_size_sitk} "
                f"to {allen_spacing} mm, size {allen_size}"
            )
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(allen_atlas)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        moving_sitk = resampler.Execute(moving_sitk)

        # Recompute the effective brain centre from the RESAMPLED image.
        # The pre-resampling centre can lie far outside the Allen domain (e.g. a
        # large 25 µm brain whose geometric centre is at ~37 mm, while the Allen
        # atlas only spans ~11 mm).  Using that centre directly gives a translation
        # of +31 mm, which maps every Allen voxel outside the moving image buffer.
        # Instead, use the centroid of the non-zero (brain-tissue) voxels that
        # survived the clipping into the Allen domain.
        moving_arr = sitk.GetArrayFromImage(moving_sitk)  # shape (Z, Y, X) in numpy
        nonzero_idx = np.argwhere(moving_arr > 0)  # rows are (z, y, x)
        if len(nonzero_idx) > 0:
            centroid_zyx = nonzero_idx.mean(axis=0)
            # SITK index order is (x, y, z), reverse of numpy (z, y, x)
            centroid_xyz = [float(centroid_zyx[2]), float(centroid_zyx[1]), float(centroid_zyx[0])]
            original_moving_center = np.array(moving_sitk.TransformContinuousIndexToPhysicalPoint(centroid_xyz))
            if verbose:
                print(f"Resampled brain centroid (physical): {original_moving_center} mm")
        # If all voxels are zero (brain entirely outside Allen domain), keep
        # the pre-resampling centre and accept a potentially poor initialization.

    # Normalize images for better registration
    fixed_image = sitk.Normalize(allen_atlas)
    moving_image_sitk = sitk.Normalize(moving_sitk)

    if verbose:
        print(f"Fixed (Allen) image: size={fixed_image.GetSize()}, spacing={fixed_image.GetSpacing()}")
        print(f"Moving (brain) image: size={moving_image_sitk.GetSize()}, spacing={moving_image_sitk.GetSpacing()}")

    # Initialize registration
    registration_method = sitk.ImageRegistrationMethod()

    # Set metric
    # Note: For correlation-based metrics, negative values are possible
    # The optimizer will maximize MI/CC and minimize MSE
    if metric.upper() == "MI":
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric.upper() == "MSE":
        registration_method.SetMetricAsMeanSquares()
    elif metric.upper() == "CC":
        registration_method.SetMetricAsCorrelation()
    elif metric.upper() == "ANTSCC":
        registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=20)
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose from: MI, MSE, CC, AntsCC")

    # Set metric sampling - use regular sampling for reproducibility and speed
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.25)  # 25% of pixels is usually sufficient

    # Set optimizer with conservative parameters
    # Use smaller learning rate and steps to prevent overshooting
    learning_rate = 0.5  # Smaller learning rate for stability
    min_step = 0.0001
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=max_iterations,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )

    # Use physical shift for scaling - more appropriate for physical coordinate registration
    # This computes scales based on how a 1mm shift affects the metric
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution approach - start coarse, refine progressively
    # More levels for robustness
    registration_method.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Initialize rigid transform with guaranteed overlap.
    # Use the ORIGINAL moving image centre (before any resampling) so that
    # the centre-aligned fallback always produces a meaningful initial translation
    # regardless of the resolution/size relationship between the two images.
    initial_transform = sitk.Euler3DTransform()

    # Calculate image centres in physical space
    fixed_size = fixed_image.GetSize()
    fixed_center_idx = [s / 2.0 for s in fixed_size]
    fixed_center = np.array(fixed_image.TransformContinuousIndexToPhysicalPoint(fixed_center_idx))

    # Translation to align brain centre with Allen centre (ensures initial overlap).
    # ITK transform maps fixed→moving: T(p) = R(p - c) + c + t
    # For identity rotation and c=fixed_center: T(fixed_center) = fixed_center + t
    # We need T(fixed_center) = original_moving_center, so t = moving_center - fixed_center.
    translation = tuple(original_moving_center - fixed_center)

    # Set center of rotation to fixed image center
    initial_transform.SetCenter(fixed_center)

    # Convert initial rotation from degrees to radians
    rx_rad = np.deg2rad(initial_rotation_deg[0])
    ry_rad = np.deg2rad(initial_rotation_deg[1])
    rz_rad = np.deg2rad(initial_rotation_deg[2])

    # Set translation to align centers and apply initial rotation
    initial_transform.SetTranslation(translation)
    initial_transform.SetRotation(rx_rad, ry_rad, rz_rad)

    if verbose:
        print(f"Initial center alignment: fixed={fixed_center}, moving (original)={original_moving_center}")
        print(f"Translation to align centers: {translation}")
        if any(r != 0 for r in initial_rotation_deg):
            print(f"Initial rotation (deg): {initial_rotation_deg}")

    # Only try MOMENTS initialization if no initial rotation was specified
    # (user-specified rotation takes precedence) and the image was NOT resampled
    # into the Allen domain.  After resampling, the brain occupies only a small
    # corner of the 640³ Allen image; sitk.Normalize then gives the large
    # zero-padded background a uniform negative value that dominates the
    # centre-of-mass computation, producing translation ≈ 0 which places every
    # sample point outside the brain buffer.
    if all(r == 0 for r in initial_rotation_deg) and not needs_resample:
        try:
            # Use MOMENTS initialization which is more robust
            init_transform = sitk.Euler3DTransform()
            init_transform = sitk.CenteredTransformInitializer(
                fixed_image, moving_image_sitk, init_transform, sitk.CenteredTransformInitializerFilter.MOMENTS
            )
            # Verify the initialized transform has reasonable translation
            init_params = init_transform.GetParameters()
            init_translation = np.array(init_params[3:6])

            # Check if the initialized transform is reasonable (translation not too large)
            # If translation is reasonable, use it; otherwise use our center-aligned one
            translation_magnitude = np.linalg.norm(init_translation)
            fixed_size_mm = np.array(fixed_image.GetSpacing()) * np.array(fixed_image.GetSize())
            max_reasonable_translation = np.linalg.norm(fixed_size_mm) * 0.5  # Half the image size

            if translation_magnitude < max_reasonable_translation:
                initial_transform = init_transform
                if verbose:
                    print(f"Using MOMENTS initialization (translation magnitude: {translation_magnitude:.2f} mm)")
            else:
                if verbose:
                    print(
                        f"MOMENTS initialization translation too large ({translation_magnitude:.2f} mm), using center-aligned"
                    )
        except Exception as e:
            if verbose:
                print(f"MOMENTS initialization failed: {e}, using center-aligned translation")

    if verbose:
        final_params = initial_transform.GetParameters()
        final_center = initial_transform.GetCenter()
        print(f"Final initial transform: rotation={final_params[:3]}, translation={final_params[3:]}")
        print(f"Transform center: {final_center}")

    registration_method.SetInitialTransform(initial_transform)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Set up iteration callback
    if verbose or progress_callback is not None:

        def command_iteration(method: Any) -> None:
            if verbose:
                if method.GetOptimizerIteration() == 0:
                    print(f"Estimated scales: {method.GetOptimizerScales()}")
                print(
                    f"Iteration {method.GetOptimizerIteration():3d} = "
                    f"{method.GetMetricValue():7.5f} : "
                    f"{method.GetOptimizerPosition()}"
                )
            if progress_callback is not None:
                progress_callback(method)

        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

    # Execute registration
    final_transform = registration_method.Execute(fixed_image, moving_image_sitk)

    stop_condition = registration_method.GetOptimizerStopConditionDescription()
    error = registration_method.GetMetricValue()

    if verbose:
        print(f"Registration complete: {stop_condition}")
        print(f"Final metric value: {error:.6f}")
        final_params = final_transform.GetParameters()
        print(f"Final transform: rotation={final_params[:3]}, translation={final_params[3:]}")
        print(f"Fixed image size: {fixed_image.GetSize()}, spacing: {fixed_image.GetSpacing()}")
        print(f"Moving image size: {moving_image_sitk.GetSize()}, spacing: {moving_image_sitk.GetSpacing()}")

    # Restore crop offset in the translation so the transform is valid for the
    # full original (uncropped) brain volume.  Derivation:
    #   T(p) = R(p-c)+c+t maps Allen coords to cropped-brain coords (origin=0).
    #   Same tissue in full brain is at (cropped_coord + crop_origin_mm).
    #   So t_full = t_crop + crop_origin_sitk  (center c cancels out).
    if any(v != 0.0 for v in crop_origin_mm):
        params = list(final_transform.GetParameters())
        # SITK Euler3D params: (rx, ry, rz, tx, ty, tz) in SITK XYZ order
        # numpy axis order (Z, Y, X)  ->  SITK (X, Y, Z):
        params[3] += crop_origin_mm[2]  # SITK X = numpy axis 2
        params[4] += crop_origin_mm[1]  # SITK Y = numpy axis 1
        params[5] += crop_origin_mm[0]  # SITK Z = numpy axis 0
        final_transform.SetParameters(params)
        if verbose:
            print(
                f"Adjusted translation for crop: +"
                f"[{crop_origin_mm[2]:.3f}, {crop_origin_mm[1]:.3f}, {crop_origin_mm[0]:.3f}] mm (SITK XYZ)"
            )

    return final_transform, stop_condition, error
