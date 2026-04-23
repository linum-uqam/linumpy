"""RGB overlay generation and display utilities."""

import numpy as np
from matplotlib import pyplot as plt

from linumpy.imaging.transform import match_shape, normalize


def get_overlay_as_rgb(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Combine the two images into a single RGB image.

    Parameters
    ----------
    img1 : np.ndarray
        The first image.
    img2 : np.ndarray
        The second image.

    Returns
    -------
    np.ndarray
        The overlay image.
    """
    img1, img2 = match_shape(img1, img2)
    rgb = np.zeros((*img1.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (img1 * 255).astype(np.uint8)
    rgb[..., 1] = (img2 * 255).astype(np.uint8)
    return rgb


def display_overlap(img1: np.ndarray, img2: np.ndarray, title: str | None = None, do_normalization: bool = False) -> None:
    """Display two images as an RGB overlay for visual comparison."""
    if do_normalization:
        img1 = normalize(img1)
        img2 = normalize(img2)
    img1, img2 = match_shape(img1, img2)
    plt.figure(figsize=(12, 12))
    plt.imshow(get_overlay_as_rgb(img1, img2))
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()
