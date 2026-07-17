"""Tests for `curios_comic_crawler.image_ops`."""

import numpy as np
import pytest

from curios_comic_crawler.image_ops import area_posterise, sharpen_image


def _gradient_image(height: int = 16, width: int = 16) -> np.ndarray:
    """A 3-channel uint8 image with plenty of distinct values, for exercising real clustering."""
    row = np.linspace(0, 255, width, dtype=np.uint8)
    channel = np.tile(row, (height, 1))
    return np.stack([channel, channel, 255 - channel], axis=-1)


def test_area_posterise_rejects_non_uint8_dtype() -> None:
    data = _gradient_image().astype(np.float32)

    with pytest.raises(TypeError, match='uint8'):
        area_posterise(data, nbr_cluster=4)


def test_area_posterise_rejects_out_of_range_cluster_count() -> None:
    data = _gradient_image()

    assert area_posterise(data, nbr_cluster=1) is data
    assert area_posterise(data, nbr_cluster=0) is data
    assert area_posterise(data, nbr_cluster=256) is data


def test_area_posterise_skips_kmeans_at_255_clusters() -> None:
    data = _gradient_image()

    assert area_posterise(data, nbr_cluster=255) is data


def test_area_posterise_returns_empty_input_unchanged() -> None:
    data = np.empty((0,), dtype=np.uint8)

    assert area_posterise(data, nbr_cluster=4) is data


def test_area_posterise_skips_when_not_enough_unique_values() -> None:
    data = np.array([0, 0, 255, 255], dtype=np.uint8)

    assert area_posterise(data, nbr_cluster=8) is data


def test_area_posterise_skips_constant_data() -> None:
    data = np.full((8, 8), 42, dtype=np.uint8)

    assert area_posterise(data, nbr_cluster=4) is data


def test_area_posterise_reduces_to_requested_cluster_count() -> None:
    data = _gradient_image()

    result = area_posterise(data, nbr_cluster=4)

    assert result.shape == data.shape
    assert result.dtype == data.dtype
    assert np.unique(result).size <= 4


def test_sharpen_image_preserves_shape_and_dtype() -> None:
    image = _gradient_image()

    result = sharpen_image(image)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
