#!/usr/bin/env python3
"""Pixel-level image processing: posterisation and sharpening."""

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from curios_comic_crawler.models import PosteriseMethod

logger = logging.getLogger(__name__)

_MAX_CLUSTERS = 255  # uint8 has 256 possible values (0-255)
_VALUES = np.arange(256, dtype=np.float64)  # every possible uint8 value
_CONVERGENCE_EPSILON = 1.0  # stop when no gray level moves by a full step


def _initial_levels(low: int, high: int, nbr_cluster: int) -> "np.ndarray":
    """Assign each of the 256 values to a uniform bin between the data's extremes.

    Deterministic start shared by both methods (no random seeding).

    Args:
        low (int): Smallest value present.
        high (int): Largest value present.
        nbr_cluster (int): Number of bins.

    Returns:
        np.ndarray: The bin (cluster index) of each value 0-255, `int32`.
    """
    edges = np.linspace(low, high, nbr_cluster + 1, dtype=np.float32)
    return np.searchsorted(edges[1:-1], _VALUES, side="right").astype(np.int32)


def _histogram_levels(
    histogram: "np.ndarray", nbr_cluster: int, nbr_iterations: int
) -> "np.ndarray":
    """Weighted k-means over the 256 possible values, each weighted by its pixel count.

    Gives the same clustering problem as k-means on every pixel, but each step costs 256
    operations instead of one per pixel.

    Args:
        histogram (np.ndarray): Pixel count of each value 0-255, `float64`.
        nbr_cluster (int): Number of gray levels to find.
        nbr_iterations (int): Maximum number of refinement passes.

    Returns:
        np.ndarray: The gray level (cluster centre) assigned to each value 0-255, `float64`.
    """
    present = np.flatnonzero(histogram)
    labels = _initial_levels(int(present[0]), int(present[-1]), nbr_cluster)
    centers = np.zeros(nbr_cluster)
    for _ in range(nbr_iterations):
        weights = np.bincount(labels, weights=histogram, minlength=nbr_cluster)
        sums = np.bincount(labels, weights=histogram * _VALUES, minlength=nbr_cluster)
        new_centers = np.where(weights > 0, sums / np.maximum(weights, 1.0), centers)
        if not weights.all():
            new_centers = _reseed_empty(new_centers, weights, labels, histogram)
        shift = float(np.abs(new_centers - centers).max())
        centers = new_centers
        labels = np.abs(_VALUES[:, np.newaxis] - centers[np.newaxis, :]).argmin(axis=1)
        if shift < _CONVERGENCE_EPSILON:
            break
    return centers[labels]


def _reseed_empty(
    centers: "np.ndarray", weights: "np.ndarray", labels: "np.ndarray", histogram: "np.ndarray"
) -> "np.ndarray":
    """Move each empty cluster onto the present value farthest from its own gray level.

    A starting bin can fall in a gap of the histogram (line art: dark ink, white paper,
    nothing between); left empty, it would never catch a value, and the page would get fewer
    levels than asked. OpenCV's k-means re-seeds empty clusters the same way.

    Args:
        centers (np.ndarray): Gray level of each cluster, `float64`.
        weights (np.ndarray): Pixel count of each cluster (0 for an empty one).
        labels (np.ndarray): The cluster of each value 0-255.
        histogram (np.ndarray): Pixel count of each value 0-255.

    Returns:
        np.ndarray: The gray levels, the empty clusters moved.
    """
    centers = centers.copy()
    labels = labels.copy()
    present = histogram > 0
    for cluster in np.flatnonzero(weights == 0):
        distance = np.where(present, np.abs(_VALUES - centers[labels]), -1.0)
        value = int(distance.argmax())
        centers[cluster] = _VALUES[value]
        labels[value] = cluster  # now at distance 0: the next empty cluster picks another value
    return centers


def _kmeans_posterise(
    data: "np.ndarray", histogram: "np.ndarray", nbr_cluster: int, nbr_iterations: int
) -> "np.ndarray":
    """OpenCV k-means on every pixel, from the same start as `_histogram_levels`.

    Args:
        data (np.ndarray): The `uint8` pixels.
        histogram (np.ndarray): Pixel count of each value 0-255.
        nbr_cluster (int): Number of gray levels to find.
        nbr_iterations (int): Maximum number of refinement passes.

    Returns:
        np.ndarray: Each pixel replaced by its (rounded) gray level, same shape as `data`.
    """
    present = np.flatnonzero(histogram)
    initial = _initial_levels(int(present[0]), int(present[-1]), nbr_cluster)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        nbr_iterations,
        _CONVERGENCE_EPSILON,
    )
    _, labels, centers = cv2.kmeans(
        data=data.reshape(-1, 1).astype(np.float32),
        K=nbr_cluster,
        bestLabels=initial[data.reshape(-1)].reshape(-1, 1),
        criteria=criteria,
        attempts=1,
        flags=cv2.KMEANS_USE_INITIAL_LABELS,
    )
    gray_levels = _to_uint8(np.asarray(centers).reshape(-1))
    pixel_labels = np.asarray(labels, dtype=np.intp).reshape(-1)
    return gray_levels[pixel_labels].reshape(data.shape)


def _to_uint8(levels: "np.ndarray") -> "np.ndarray":
    """Round gray levels to `uint8` (a level of 254.9 is 255, not 254).

    Args:
        levels (np.ndarray): Float gray levels.

    Returns:
        np.ndarray: The rounded levels.
    """
    return np.clip(np.rint(levels), 0, 255).astype(np.uint8)


def area_posterise(
    data: "np.ndarray",
    nbr_cluster: int = 2,
    nbr_iterations: int = 10,
    method: "PosteriseMethod" = "histogram",
) -> "np.ndarray":
    """Reduce `uint8` data to `nbr_cluster` gray levels, chosen by k-means.

    Args:
        data (np.ndarray): `uint8` array of values to clusterize (any shape). Every caller
            feeds `cv2.imread`/upscaler output; only `uint8` input is accepted.
        nbr_cluster (int, optional): Number of gray levels, 2-254 (255 would leave the data
            virtually unchanged: it is returned as is, as are out-of-range values). Defaults
            to 2.
        nbr_iterations (int, optional): Maximum number of refinement passes. Defaults to 10.
        method (PosteriseMethod, optional): `"histogram"` runs k-means on the 256-value
            histogram (fast, little memory); `"kmeans"` runs OpenCV's k-means on every pixel.
            Both start from the same levels; they may settle on slightly different ones.
            Defaults to `"histogram"`.

    Returns:
        np.ndarray: The posterised data (same shape and dtype), or `data` itself when there is
        nothing to reduce.

    Raises:
        TypeError: If `data.dtype` isn't `uint8`.
    """
    if data.dtype != np.uint8:
        raise TypeError(f"area_posterise only supports uint8 data, got {data.dtype}")
    if not 1 < nbr_cluster <= _MAX_CLUSTERS:
        logger.warning(
            "Change the value of %s for 'nbr_cluster' should be (1 ~ %s)",
            nbr_cluster,
            _MAX_CLUSTERS,
        )
        return data
    if nbr_cluster == _MAX_CLUSTERS or data.size == 0:
        return data

    # calcHist counts in place; np.bincount would first widen every pixel to 8 bytes.
    histogram = cv2.calcHist([data.reshape(-1, 1)], [0], None, [256], [0, 256]).reshape(-1)
    unique_count = int(np.count_nonzero(histogram))
    if unique_count <= nbr_cluster:
        logger.warning(
            "Requested clusters %s can't be higher than the %s distinct values to organise",
            nbr_cluster,
            unique_count,
        )
        return data

    iterations = max(int(nbr_iterations), 1)
    if method == "kmeans":
        return _kmeans_posterise(data, histogram, nbr_cluster, iterations)
    levels = _histogram_levels(histogram.astype(np.float64), nbr_cluster, iterations)
    return cv2.LUT(data, _to_uint8(levels)).reshape(data.shape)


def _unsharp_pass(image: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    """Run one unsharp-mask pass: blur, then push the original away from the blur.

    Args:
        image (np.ndarray): The image.
        ksize (int): Gaussian kernel size (odd).
        sigma (float): Gaussian standard deviation.

    Returns:
        np.ndarray: The sharpened image.
    """
    blur = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return cv2.addWeighted(image, 1.5, blur, -0.5, 0)


def sharpen_image(
    image: np.ndarray,
    ksize_1: int = 7,
    ksize_2: int = 7,
    sigma_1: float = 1.0,
    sigma_2: float = 2.0,
) -> np.ndarray:
    """Sharpen an image with a two-pass unsharp mask.

    Args:
        image (np.ndarray): Input image (BGR).
        ksize_1 (int, optional): Kernel n°1 size (Odd number). Defaults to 7.
        ksize_2 (int, optional): Kernel n°2 size (Odd number). Defaults to 7.
        sigma_1 (float, optional): Blur value n°1. Defaults to 1.
        sigma_2 (float, optional): Blur value n°2. Defaults to 2.

    Returns:
        np.ndarray: Sharpened image.
    """
    temp_1 = _unsharp_pass(image, ksize_1, sigma_1)
    temp_2 = _unsharp_pass(image, ksize_2, sigma_2)

    return cv2.addWeighted(temp_1, 0.5, temp_2, 0.5, 0)
