from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars.type_aliases import AvroCompression
    from tests.unit.conftest import MemoryUsage


COMPRESSIONS = ["uncompressed", "snappy", "deflate"]


@pytest.fixture()
def example_df() -> pl.DataFrame:
    return pl.DataFrame({"i64": [1, 2], "f64": [0.1, 0.2], "str": ["a", "b"]})


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_from_to_buffer(example_df: pl.DataFrame, compression: AvroCompression) -> None:
    buf = io.BytesIO()
    example_df.write_avro(buf, compression=compression)
    buf.seek(0)

    read_df = pl.read_avro(buf)
    assert_frame_equal(example_df, read_df)


@pytest.mark.write_disk()
@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_from_to_file(
    example_df: pl.DataFrame, compression: AvroCompression, tmp_path: Path
) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.avro"
    example_df.write_avro(file_path, compression=compression)
    df_read = pl.read_avro(file_path)

    assert_frame_equal(example_df, df_read)


def test_select_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_avro(f)
    f.seek(0)

    read_df = pl.read_avro(f, columns=["b", "c"])
    assert_frame_equal(expected, read_df)


def test_select_projection() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_avro(f)
    f.seek(0)

    read_df = pl.read_avro(f, columns=[1, 2])
    assert_frame_equal(expected, read_df)


def test_with_name() -> None:
    df = pl.DataFrame({"a": [1]})
    expected = pl.DataFrame(
        {
            "type": ["record"],
            "name": ["my_schema_name"],
            "fields": [[{"name": "a", "type": ["null", "long"]}]],
        }
    )

    f = io.BytesIO()
    df.write_avro(f, name="my_schema_name")

    f.seek(0)
    raw = f.read()

    read_df = pl.read_json(raw[raw.find(b"{") : raw.rfind(b"}") + 1])

    assert_frame_equal(expected, read_df)


@pytest.mark.slow()
@pytest.mark.write_disk()
def test_read_avro_only_loads_selected_columns(
    memory_usage_without_pyarrow: MemoryUsage,
    tmp_path: Path,
) -> None:
    """Only requested columns are loaded by ``read_avro()``."""
    tmp_path.mkdir(exist_ok=True)

    # Each column will be about 8MB of RAM
    series = pl.arange(0, 1_000_000, dtype=pl.Int64, eager=True)

    file_path = tmp_path / "multicolumn.avro"
    df = pl.DataFrame(
        {
            "a": series,
            "b": series,
        }
    )
    df.write_avro(file_path)
    del df, series

    memory_usage_without_pyarrow.reset_tracking()

    # Only load one column:
    df = pl.read_avro(str(file_path), columns=["b"])
    del df
    # Only one column's worth of memory should be used; 2 columns would be
    # 16_000_000 at least, but there's some overhead.
    assert 8_000_000 < memory_usage_without_pyarrow.get_peak() < 13_000_000
