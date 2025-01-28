from typing import Callable

import pytest

import polars as pl

pytestmark = pytest.mark.benchmark()


@pytest.fixture(scope="module")
def lists_and_values() -> pl.DataFrame:
    return pl.DataFrame(
        {"lists": [[0, 2, 1, 3, 5]] * 1_000, "values": [3, 1, 4, 5, 0] * 200}
    )


def test_list_contains(
    benchmark: Callable[[Callable[[], None]], object], lists_and_values: pl.DataFrame
) -> None:
    def go() -> None:
        for _ in range(10):
            lists_and_values.select(pl.col("lists").list.contains(pl.col("values")))

    benchmark(go)
