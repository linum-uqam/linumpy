import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt


def normalize(img, saturation=99.7):
    imin = img.min()
    imax = np.percentile(img, saturation)
    img = (img.astype(np.float32) - imin) / (imax - imin)
    img[img>1] = 1
    return img


def get_overlay(img1, img2):
    img1, img2 = match_shape(img1, img2)
    rgb = np.zeros((*img1.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (img1 * 255).astype(np.uint8)
    rgb[..., 1] = (img2 * 255).astype(np.uint8)
    return rgb


def match_shape(img1, img2):
    nr1, nc1 = img1.shape
    nr2, nc2 = img2.shape
    n_rows = max(nr1, nr2)
    n_cols = max(nc1, nc2)

    # Pad image 1
    pad_r_0 = max((n_rows - img1.shape[0]) // 2, 0)
    pad_r_1 = max((n_rows - img1.shape[0] - pad_r_0), 0)
    pad_c_0 = max((n_cols - img1.shape[1]) // 2, 0)
    pad_c_1 = max((n_cols - img1.shape[1] - pad_c_0), 0)
    img1_p = np.pad(img1, ((pad_r_0, pad_r_1), (pad_c_0, pad_c_1)))

    # Pad image 2
    pad_r_0 = max((n_rows - img2.shape[0]) // 2, 0)
    pad_r_1 = max((n_rows - img2.shape[0] - pad_r_0), 0)
    pad_c_0 = max((n_cols - img2.shape[1]) // 2, 0)
    pad_c_1 = max((n_cols - img2.shape[1] - pad_c_0), 0)
    img2_p = np.pad(img2, ((pad_r_0, pad_r_1), (pad_c_0, pad_c_1)))

    return img1_p, img2_p


def display_overlap(img1, img2, title=None, do_normalization=False):
    if do_normalization:
        img1 = normalize(img1)
        img2 = normalize(img2)
    img1, img2 = match_shape(img1, img2)
    plt.figure(figsize=(12, 12))
    plt.imshow(get_overlay(img1, img2))
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def apply_xy_shift(img, reference, dx, dy):
    fixed = sitk.GetImageFromArray(reference)
    moving = sitk.GetImageFromArray(img)

    # Set the transform
    transform = sitk.TranslationTransform(fixed.GetDimension())
    transform.SetParameters((dx, dy))

    """Apply a shift to the image in the xy plane."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    warped_moving_image = resampler.Execute(moving)
    img_warped = sitk.GetArrayFromImage(warped_moving_image)
    return img_warped
