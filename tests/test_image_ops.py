"""Tests for `curios_comic_crawler.image_ops`."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from curios_comic_crawler.image_ops import area_posterise, sharpen_image

if TYPE_CHECKING:
    from curios_comic_crawler.models import PosteriseMethod


METHODS = ["histogram", "kmeans"]


def _gradient_image(height: int = 16, width: int = 16) -> np.ndarray:
    """Build a 3-channel uint8 image with plenty of distinct values, for real clustering.

    Args:
        height: Rows.
        width: Columns.

    Returns:
        The image.
    """
    row = np.linspace(0, 255, width, dtype=np.uint8)
    channel = np.tile(row, (height, 1))
    return np.stack([channel, channel, 255 - channel], axis=-1)


def test_area_posterise_rejects_non_uint8_dtype() -> None:
    """Only 8-bit images can be posterised."""
    data = _gradient_image().astype(np.float32)

    with pytest.raises(TypeError, match="uint8"):
        _ = area_posterise(data, nbr_cluster=4)


def test_area_posterise_rejects_out_of_range_cluster_count() -> None:
    """A cluster count outside 2-255 leaves the image as is."""
    data = _gradient_image()

    assert area_posterise(data, nbr_cluster=1) is data
    assert area_posterise(data, nbr_cluster=0) is data
    assert area_posterise(data, nbr_cluster=256) is data


def test_area_posterise_skips_kmeans_at_255_clusters() -> None:
    """255 clusters change nothing: k-means isn't run."""
    data = _gradient_image()

    assert area_posterise(data, nbr_cluster=255) is data


def test_area_posterise_returns_empty_input_unchanged() -> None:
    """An empty image is returned as is."""
    data = np.empty((0,), dtype=np.uint8)

    assert area_posterise(data, nbr_cluster=4) is data


def test_area_posterise_skips_when_not_enough_unique_values() -> None:
    """Fewer distinct values than clusters: nothing to reduce."""
    data = np.array([0, 0, 255, 255], dtype=np.uint8)

    assert area_posterise(data, nbr_cluster=8) is data


def test_area_posterise_skips_constant_data() -> None:
    """A single-valued image is returned as is."""
    data = np.full((8, 8), 42, dtype=np.uint8)

    assert area_posterise(data, nbr_cluster=4) is data


def test_area_posterise_reduces_to_requested_cluster_count() -> None:
    """The result keeps shape and dtype, with at most the requested gray levels."""
    data = _gradient_image()

    result = area_posterise(data, nbr_cluster=4)

    assert result.shape == data.shape
    assert result.dtype == data.dtype
    assert np.unique(result).size <= 4


def test_sharpen_image_preserves_shape_and_dtype() -> None:
    """Sharpening keeps the image's shape and dtype."""
    image = _gradient_image()

    result = sharpen_image(image)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


@pytest.mark.parametrize("method", METHODS)
def test_both_methods_reduce_to_the_requested_levels(method: "PosteriseMethod") -> None:
    """Each method keeps shape and dtype and uses at most the requested gray levels."""
    data = _gradient_image()

    result = area_posterise(data, nbr_cluster=4, method=method)

    assert result.shape == data.shape
    assert result.dtype == np.uint8
    assert np.unique(result).size <= 4


def test_methods_agree_on_a_clear_cut_image() -> None:
    """Two well-separated tones: both methods find the same two levels."""
    data = np.array([[10] * 50 + [200] * 50] * 4, dtype=np.uint8)
    data[0, 0], data[0, 99] = 12, 198  # a little noise so there is something to reduce

    histogram = area_posterise(data, nbr_cluster=2, method="histogram")
    kmeans = area_posterise(data, nbr_cluster=2, method="kmeans")

    assert np.array_equal(histogram, kmeans)


def test_levels_are_rounded_not_truncated() -> None:
    """A level averaging 254.5+ becomes 255 (casting used to truncate it to 254)."""
    data = np.array([0] * 10 + [254, 255, 255, 255], dtype=np.uint8)

    result = area_posterise(data, nbr_cluster=2, method="histogram")

    assert int(result.max()) == 255


@pytest.mark.parametrize("nbr_cluster", [2, 3, 8, 254])
@pytest.mark.parametrize("method", METHODS)
def test_posterise_maps_values_monotonically(method: "PosteriseMethod", nbr_cluster: int) -> None:
    """Each input value gets one gray level, and a brighter input never gets a darker level."""
    data = np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8)

    result = area_posterise(data, nbr_cluster=nbr_cluster, method=method)

    pairs = np.unique(np.stack([data.reshape(-1), result.reshape(-1)]), axis=1)
    assert pairs.shape[1] == np.unique(data).size  # a function of the input value
    assert np.all(np.diff(pairs[1].astype(np.int16)) >= 0)  # sorted by input: never decreases
    assert np.unique(result).size == nbr_cluster


@pytest.mark.parametrize("method", METHODS)
def test_as_many_distinct_values_as_clusters_returns_data_unchanged(
    method: "PosteriseMethod",
) -> None:
    """Exactly K distinct values: already posterised, returned as is."""
    data = np.array([[0, 50, 100, 200]] * 4, dtype=np.uint8)

    assert area_posterise(data, nbr_cluster=4, method=method) is data


@pytest.mark.parametrize("method", METHODS)
def test_one_value_more_than_clusters_is_reduced(method: "PosteriseMethod") -> None:
    """K + 1 distinct values are reduced to K levels."""
    data = np.array([[0, 50, 100, 200, 255]] * 4, dtype=np.uint8)

    result = area_posterise(data, nbr_cluster=4, method=method)

    assert np.unique(result).size == 4


@pytest.mark.parametrize("method", METHODS)
def test_zero_iterations_still_posterises(method: "PosteriseMethod") -> None:
    """`nbr_iterations=0` runs one pass instead of returning unposterised data."""
    data = _gradient_image()

    result = area_posterise(data, nbr_cluster=4, nbr_iterations=0, method=method)

    assert np.unique(result).size == 4


@pytest.mark.parametrize("method", METHODS)
def test_254_clusters_on_full_range_data(method: "PosteriseMethod") -> None:
    """The largest real reduction: all 256 values become exactly 254 levels."""
    data = np.tile(np.arange(256, dtype=np.uint8), (4, 1))

    result = area_posterise(data, nbr_cluster=254, method=method)

    assert result.shape == data.shape
    assert np.unique(result).size == 254


@pytest.mark.parametrize("nbr_cluster", [8, 16])
def test_histogram_gives_every_level_asked_despite_a_gap(nbr_cluster: int) -> None:
    """Line art (dark ink, white paper, nothing between) still gets every level asked for.

    Uniform starting bins fall in the gap; left empty, they used to stay empty.
    """
    values = np.concatenate([np.arange(0, 12), np.arange(244, 256)]).astype(np.uint8)
    data = np.random.default_rng(0).choice(values, size=(64, 64)).astype(np.uint8)

    result = area_posterise(data, nbr_cluster, method="histogram")

    assert len(np.unique(result)) == nbr_cluster
