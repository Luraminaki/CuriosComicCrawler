"""Tests for `curios_comic_crawler.upscaler`.

Actual image upscaling (`_process_one` / `_init_worker`) needs a real OpenCV model and a
worker process, and is covered by manual end-to-end runs instead -- these tests stick to the
pure orchestration logic around it.
"""

import pathlib

from curios_comic_crawler import upscaler


def test_select_remaining_without_force_skips_completed() -> None:
    originals = [pathlib.Path(f'p{i}.jpg') for i in range(5)]
    completed = originals[:3]

    assert upscaler._select_remaining(originals, completed, force=False) == originals[3:]


def test_select_remaining_with_force_reprocesses_everything() -> None:
    originals = [pathlib.Path(f'p{i}.jpg') for i in range(5)]
    completed = originals[:3]

    assert upscaler._select_remaining(originals, completed, force=True) == originals


def test_format_duration_formats_days_hours_minutes_seconds() -> None:
    assert upscaler._format_duration(0) == '0d - 0h:0m:0s'
    assert upscaler._format_duration(3661) == '0d - 1h:1m:1s'
    assert upscaler._format_duration(90000) == '1d - 1h:0m:0s'


def test_format_duration_clamps_negative_to_zero() -> None:
    assert upscaler._format_duration(-5) == '0d - 0h:0m:0s'


def test_run_returns_success_when_nothing_to_upscale(config) -> None:
    assert upscaler.run(config) == 0


def test_eta_divides_by_full_worker_count_while_a_queue_remains() -> None:
    # 10 pages left, 4 workers still have queued work waiting behind them.
    assert upscaler._estimate_eta_seconds(avg_per_page=2.0, remaining_count=10, worker_count=4) == 5.0


def test_eta_does_not_shrink_below_one_round_in_the_tail() -> None:
    # Every remaining page is already running concurrently -- no more queue to divide away.
    for remaining_count in (1, 2, 3, 4):
        eta = upscaler._estimate_eta_seconds(avg_per_page=2.0, remaining_count=remaining_count, worker_count=4)
        assert eta == 2.0


def test_eta_is_zero_when_nothing_remains() -> None:
    assert upscaler._estimate_eta_seconds(avg_per_page=2.0, remaining_count=0, worker_count=4) == 0.0
