#!/usr/bin/env python3
"""Pixel-level image processing: posterisation and sharpening."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MAX_CLUSTERS = 255  # uint8 has 256 possible values (0-255)


def area_posterise(data: np.ndarray, nbr_cluster: int = 2, nbr_iterations: int = 10) -> np.ndarray:
    """Apply the K-means clustering algorithm on a N-D array of `uint8` data.

    Args:
        data (np.ndarray): `uint8` array of values to clusterize (N-D array with values between
            0 and 255). Every caller in this codebase feeds `cv2.imread`/upscaler output, which
            is always `uint8`; the output is coerced to `uint8` regardless of input dtype, so
            only `uint8` input is accepted.
        nbr_cluster (int, optional): Number of expected clusters (bigger than 1 and smaller than
            256). Defaults to 2.
        nbr_iterations (int, optional): Number of loops. Defaults to 10.

    Returns:
        np.ndarray: Quantized data.

    Raises:
        TypeError: If `data.dtype` isn't `uint8`.
    """
    if data.dtype != np.uint8:
        raise TypeError(f'area_posterise only supports uint8 data, got {data.dtype}')

    if not 1 < nbr_cluster <= _MAX_CLUSTERS:
        logger.warning("Change the value of %s for 'nbr_cluster' should be (1 ~ %s)", nbr_cluster, _MAX_CLUSTERS)
        return data

    # 255 clusters over 256 possible uint8 values would leave the data virtually unchanged;
    # skip the expensive kmeans call in that case.
    if nbr_cluster == _MAX_CLUSTERS:
        return data

    if data.size == 0:
        return data

    data_1d = data.reshape(-1)
    unique_count = np.count_nonzero(np.bincount(data_1d, minlength=256))

    if unique_count <= nbr_cluster:
        logger.warning(
            "Requested clusters %s can't be higher than the number of unique elements %s to organise",
            nbr_cluster, unique_count,
        )
        return data

    # OpenCV kmeans expects samples as (N, features); we cluster scalar intensities, so features=1.
    area_to_posterise = np.ascontiguousarray(data_1d, dtype=np.float32).reshape(-1, 1)

    # Build deterministic initial labels from uniform bins to avoid random starts.
    data_min = float(area_to_posterise.min())
    data_max = float(area_to_posterise.max())
    if data_min == data_max:
        return data

    bin_edges = np.linspace(data_min, data_max, nbr_cluster + 1, dtype=np.float32)
    initial_bins = np.searchsorted(bin_edges[1:-1], area_to_posterise.ravel(), side='right')
    best_labels = initial_bins.astype(np.int32).reshape(-1, 1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max(int(nbr_iterations), 1), 1.0)
    flags = cv2.KMEANS_USE_INITIAL_LABELS
    _, labels, centers = cv2.kmeans(data=area_to_posterise,
                                     K=nbr_cluster,
                                     bestLabels=best_labels,
                                     criteria=criteria,
                                     attempts=1,
                                     flags=flags)
    centers = centers.astype(np.uint8, copy=False)
    labels_flat = labels.ravel().astype(np.intp, copy=False)
    return np.take(centers, labels_flat, axis=0).reshape(data.shape)


def _unsharp_pass(image: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    """Run one unsharp-mask pass: blur, then push the original away from the blur."""
    blur = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return cv2.addWeighted(image, 1.5, blur, -0.5, 0)


def sharpen_image(
    image: np.ndarray, ksize_1: int = 7, ksize_2: int = 7, sigma_1: float = 1., sigma_2: float = 2.,
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
