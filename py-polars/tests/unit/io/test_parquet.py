from __future__ import annotations

import decimal
import functools
import io
import warnings
from datetime import date, datetime, time, timezone
from decimal import Decimal
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Literal, cast
from zoneinfo import ZoneInfo

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from hypothesis import given
from hypothesis import strategies as st

import polars as pl
from polars.exceptions import ComputeError
from polars.io.parquet import ParquetFieldOverwrites
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import column, dataframes
from polars.testing.parametric.strategies.core import series

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import (
        ParallelStrategy,
        ParquetCompression,
        ParquetMetadata,
        ParquetMetadataContext,
    )
    from tests.unit.conftest import MemoryUsage


@pytest.mark.may_fail_auto_streaming
def test_round_trip(df: pl.DataFrame) -> None:
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


@pytest.mark.may_fail_auto_streaming
def test_scan_round_trip(df: pl.DataFrame) -> None:
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)
    assert_frame_equal(pl.scan_parquet(f).collect(), df)
    f.seek(0)
    assert_frame_equal(pl.scan_parquet(f).head().collect(), df.head())


COMPRESSIONS = [
    "lz4",
    "uncompressed",
    "snappy",
    "gzip",
    # "lzo",  # LZO compression currently not supported by Arrow backend
    "brotli",
    "zstd",
]


@pytest.mark.write_disk
def test_write_parquet_using_pyarrow_9753(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"a": [1, 2, 3]})
    df.write_parquet(
        tmp_path / "test.parquet",
        compression="zstd",
        statistics=True,
        use_pyarrow=True,
        pyarrow_options={"coerce_timestamps": "us"},
    )


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_write_parquet_using_pyarrow_write_to_dataset_with_partitioning(
    tmp_path: Path,
    compression: ParquetCompression,
) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "partition_col": ["one", "two", "two"]})
    path_to_write = tmp_path / "test_folder"
    path_to_write.mkdir(exist_ok=True)
    df.write_parquet(
        file=path_to_write,
        statistics=True,
        use_pyarrow=True,
        row_group_size=128,
        pyarrow_options={
            "partition_cols": ["partition_col"],
            "compression": compression,
        },
    )

    # cast is necessary as pyarrow writes partitions as categorical type
    read_df = pl.read_parquet(path_to_write, use_pyarrow=True).with_columns(
        pl.col("partition_col").cast(pl.String)
    )
    assert_frame_equal(df, read_df)


@pytest.fixture
def small_parquet_path(io_files_path: Path) -> Path:
    return io_files_path / "small.parquet"


@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("use_pyarrow", [True, False])
def test_to_from_buffer(
    df: pl.DataFrame, compression: ParquetCompression, use_pyarrow: bool
) -> None:
    df = df[["list_str"]]
    buf = io.BytesIO()
    df.write_parquet(buf, compression=compression, use_pyarrow=use_pyarrow)
    buf.seek(0)
    read_df = pl.read_parquet(buf, use_pyarrow=use_pyarrow)
    assert_frame_equal(df, read_df, categorical_as_str=True)


@pytest.mark.parametrize("use_pyarrow", [True, False])
@pytest.mark.parametrize("rechunk_and_expected_chunks", [(True, 1), (False, 3)])
@pytest.mark.may_fail_auto_streaming
def test_read_parquet_respects_rechunk_16416(
    use_pyarrow: bool, rechunk_and_expected_chunks: tuple[bool, int]
) -> None:
    # Create a dataframe with 3 chunks:
    df = pl.DataFrame({"a": [1]})
    df = pl.concat([df, df, df])
    buf = io.BytesIO()
    df.write_parquet(buf, row_group_size=1)
    buf.seek(0)

    rechunk, expected_chunks = rechunk_and_expected_chunks
    result = pl.read_parquet(buf, use_pyarrow=use_pyarrow, rechunk=rechunk)
    assert result.n_chunks() == expected_chunks


def test_to_from_buffer_lzo(df: pl.DataFrame) -> None:
    print(df)
    buf = io.BytesIO()
    # Writing lzo compressed parquet files is not supported for now.
    with pytest.raises(ComputeError):
        df.write_parquet(buf, compression="lzo", use_pyarrow=False)
    buf.seek(0)

    buf = io.BytesIO()
    with pytest.raises(OSError):
        # Writing lzo compressed parquet files is not supported for now.
        df.write_parquet(buf, compression="lzo", use_pyarrow=True)
    buf.seek(0)
    # Invalid parquet file as writing failed.
    with pytest.raises(ComputeError):
        _ = pl.read_parquet(buf)


@pytest.mark.write_disk
@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_to_from_file(
    df: pl.DataFrame, compression: ParquetCompression, tmp_path: Path
) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.avro"
    df.write_parquet(file_path, compression=compression)
    read_df = pl.read_parquet(file_path)
    assert_frame_equal(df, read_df, categorical_as_str=True)


@pytest.mark.write_disk
def test_to_from_file_lzo(df: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.avro"

    # Writing lzo compressed parquet files is not supported for now.
    with pytest.raises(ComputeError):
        df.write_parquet(file_path, compression="lzo", use_pyarrow=False)
    # Invalid parquet file as writing failed.
    with pytest.raises(ComputeError):
        _ = pl.read_parquet(file_path)

    # Writing lzo compressed parquet files is not supported for now.
    with pytest.raises(OSError):
        df.write_parquet(file_path, compression="lzo", use_pyarrow=True)
    # Invalid parquet file as writing failed.
    with pytest.raises(FileNotFoundError):
        _ = pl.read_parquet(file_path)


def test_select_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})

    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)

    read_df = pl.read_parquet(f, columns=["b", "c"], use_pyarrow=False)
    assert_frame_equal(expected, read_df)


def test_select_projection() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [True, False, True], "c": ["a", "b", "c"]})
    expected = pl.DataFrame({"b": [True, False, True], "c": ["a", "b", "c"]})
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)

    read_df = pl.read_parquet(f, columns=[1, 2], use_pyarrow=False)
    assert_frame_equal(expected, read_df)


@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("use_pyarrow", [True, False])
def test_parquet_datetime(compression: ParquetCompression, use_pyarrow: bool) -> None:
    # This failed because parquet writers cast datetime to Date
    f = io.BytesIO()
    data = {
        "datetime": [  # unix timestamp in ms
            1618354800000,
            1618354740000,
            1618354680000,
            1618354620000,
            1618354560000,
        ],
        "value1": [73.1999969482, 71.0999984741, 74.5, 69.5999984741, 69.6999969482],
        "value2": [59.5999984741, 61.0, 62.2999992371, 56.9000015259, 60.0],
    }
    df = pl.DataFrame(data)
    df = df.with_columns(df["datetime"].cast(pl.Datetime))

    df.write_parquet(f, use_pyarrow=use_pyarrow, compression=compression)
    f.seek(0)
    read = pl.read_parquet(f)
    assert_frame_equal(read, df)


def test_nested_parquet() -> None:
    f = io.BytesIO()
    data = [
        {"a": [{"b": 0}]},
        {"a": [{"b": 1}, {"b": 2}]},
    ]
    df = pd.DataFrame(data)
    df.to_parquet(f)

    read = pl.read_parquet(f, use_pyarrow=True)
    assert read.columns == ["a"]
    assert isinstance(read.dtypes[0], pl.datatypes.List)
    assert isinstance(read.dtypes[0].inner, pl.datatypes.Struct)


@pytest.mark.write_disk
def test_glob_parquet(df: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "small.parquet"
    df.write_parquet(file_path)

    path_glob = tmp_path / "small*.parquet"
    assert pl.read_parquet(path_glob).shape == (3, df.width)
    assert pl.scan_parquet(path_glob).collect().shape == (3, df.width)


def test_chunked_round_trip() -> None:
    df1 = pl.DataFrame(
        {
            "a": [1] * 2,
            "l": [[1] for j in range(2)],
        }
    )
    df2 = pl.DataFrame(
        {
            "a": [2] * 3,
            "l": [[2] for j in range(3)],
        }
    )

    df = df1.vstack(df2)

    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


@pytest.mark.write_disk
def test_lazy_self_join_file_cache_prop_3979(df: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.parquet"
    df.write_parquet(file_path)

    a = pl.scan_parquet(file_path)
    b = pl.DataFrame({"a": [1]}).lazy()

    expected_shape = (3, df.width + b.collect_schema().len())
    assert a.join(b, how="cross").collect().shape == expected_shape
    assert b.join(a, how="cross").collect().shape == expected_shape


def test_recursive_logical_type() -> None:
    df = pl.DataFrame({"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]})
    df = df.with_columns(pl.col("str").cast(pl.Categorical))

    df_groups = df.group_by("group").agg([pl.col("str").alias("cat_list")])
    f = io.BytesIO()
    df_groups.write_parquet(f, use_pyarrow=True)
    f.seek(0)
    read = pl.read_parquet(f, use_pyarrow=True)
    assert read.dtypes == [pl.Int64, pl.List(pl.Categorical)]
    assert read.shape == (2, 2)


def test_nested_dictionary() -> None:
    df = (
        pl.DataFrame({"str": ["A", "B", "A", "B", "C"], "group": [1, 1, 2, 1, 2]})
        .with_columns(pl.col("str").cast(pl.Categorical))
        .group_by("group")
        .agg([pl.col("str").alias("cat_list")])
    )
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)

    read_df = pl.read_parquet(f)
    assert_frame_equal(df, read_df)


def test_row_group_size_saturation() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    f = io.BytesIO()

    # request larger chunk than rows in df
    df.write_parquet(f, row_group_size=1024)
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


def test_nested_sliced() -> None:
    for df in [
        pl.Series([[1, 2], [3, 4], [5, 6]]).slice(2, 2).to_frame(),
        pl.Series([[None, 2], [3, 4], [5, 6]]).to_frame(),
        pl.Series([[None, 2], [3, 4], [5, 6]]).slice(2, 2).to_frame(),
        pl.Series([["a", "a"], ["", "a"], ["c", "de"]]).slice(3, 2).to_frame(),
        pl.Series([[None, True], [False, False], [True, True]]).slice(2, 2).to_frame(),
    ]:
        f = io.BytesIO()
        df.write_parquet(f)
        f.seek(0)
        assert_frame_equal(pl.read_parquet(f), df)


def test_parquet_5795() -> None:
    df_pd = pd.DataFrame(
        {
            "a": [
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
                None,
                None,
                None,
                None,
                None,
                None,
            ]
        }
    )
    f = io.BytesIO()
    df_pd.to_parquet(f)
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), pl.from_pandas(df_pd))


def test_parquet_nesting_structs_list() -> None:
    f = io.BytesIO()
    df = pl.from_records(
        [
            {
                "id": 1,
                "list_of_structs_col": [
                    {"a": 10, "b": [10, 11, 12]},
                    {"a": 11, "b": [13, 14, 15]},
                ],
            },
            {
                "id": 2,
                "list_of_structs_col": [
                    {"a": 44, "b": [12]},
                ],
            },
        ]
    )

    df.write_parquet(f)
    f.seek(0)

    assert_frame_equal(pl.read_parquet(f), df)


def test_parquet_nested_dictionaries_6217() -> None:
    _type = pa.dictionary(pa.int64(), pa.string())

    fields = [("a_type", _type)]
    struct_type = pa.struct(fields)

    col1 = pa.StructArray.from_arrays(
        [pa.DictionaryArray.from_arrays([0, 0, 1], ["A", "B"])],
        fields=struct_type,
    )

    table = pa.table({"Col1": col1})

    df = pl.from_arrow(table)

    f = io.BytesIO()
    import pyarrow.parquet as pq

    pq.write_table(table, f, compression="snappy")
    f.seek(0)
    read = pl.read_parquet(f)
    assert_frame_equal(read, df)  # type: ignore[arg-type]


@pytest.mark.write_disk
def test_head_union(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df1 = pl.DataFrame({"a": [0, 1, 2], "b": [1, 2, 3]})
    df2 = pl.DataFrame({"a": [3, 4, 5], "b": [4, 5, 6]})

    file_path_1 = tmp_path / "df_fetch_1.parquet"
    file_path_2 = tmp_path / "df_fetch_2.parquet"
    file_path_glob = tmp_path / "df_fetch_*.parquet"

    df1.write_parquet(file_path_1)
    df2.write_parquet(file_path_2)

    result_one = pl.scan_parquet(file_path_1).head(1).collect()
    result_glob = pl.scan_parquet(file_path_glob).head(1).collect()

    expected = pl.DataFrame({"a": [0], "b": [1]})
    assert_frame_equal(result_one, expected)

    # Both fetch 1 per file or 1 per dataset would be ok, as we don't guarantee anything
    # currently we have one per dataset.
    expected = pl.DataFrame({"a": [0], "b": [1]})
    assert_frame_equal(result_glob, expected)


@pytest.mark.slow
def test_struct_pyarrow_dataset_5796(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    num_rows = 2**17 + 1

    df = pl.from_records([{"id": i, "nested": {"a": i}} for i in range(num_rows)])
    file_path = tmp_path / "out.parquet"
    df.write_parquet(file_path, use_pyarrow=True)
    tbl = ds.dataset(file_path).to_table()
    result = pl.from_arrow(tbl)

    assert_frame_equal(result, df)  # type: ignore[arg-type]


@pytest.mark.slow
@pytest.mark.parametrize("case", [1048576, 1048577])
def test_parquet_chunks_545(case: int) -> None:
    f = io.BytesIO()
    # repeat until it has case instances
    df = pd.DataFrame(
        np.tile([1.0, pd.to_datetime("2010-10-10")], [case, 1]),
        columns=["floats", "dates"],
    )

    # write as parquet
    df.to_parquet(f)
    f.seek(0)

    # read it with polars
    polars_df = pl.read_parquet(f)
    assert_frame_equal(pl.DataFrame(df), polars_df)


def test_nested_null_roundtrip() -> None:
    f = io.BytesIO()
    df = pl.DataFrame(
        {
            "experiences": [
                [
                    {"company": "Google", "years": None},
                    {"company": "Facebook", "years": None},
                ],
            ]
        }
    )

    df.write_parquet(f)
    f.seek(0)
    df_read = pl.read_parquet(f)
    assert_frame_equal(df_read, df)


def test_parquet_nested_list_pandas() -> None:
    # pandas/pyarrow writes as nested null dict
    df_pd = pd.DataFrame({"listcol": [[] * 10]})
    f = io.BytesIO()
    df_pd.to_parquet(f)
    f.seek(0)
    df = pl.read_parquet(f)
    assert df.dtypes == [pl.List(pl.Null)]
    assert df.to_dict(as_series=False) == {"listcol": [[]]}


def test_parquet_cat_roundtrip() -> None:
    f = io.BytesIO()

    df = pl.DataFrame({"a": ["a", "b", "c", "d"]}).with_columns(
        pl.col("a").cast(pl.Categorical)
    )

    df.write_parquet(f, row_group_size=2)
    f.seek(0)
    assert_series_equal(pl.read_parquet(f)["a"], df["a"])


def test_tz_aware_parquet_9586(io_files_path: Path) -> None:
    result = pl.read_parquet(io_files_path / "tz_aware.parquet")
    expected = pl.DataFrame(
        {"UTC_DATETIME_ID": [datetime(2023, 6, 26, 14, 15, 0, tzinfo=timezone.utc)]}
    ).select(pl.col("*").cast(pl.Datetime("ns", "UTC")))
    assert_frame_equal(result, expected)


def test_nested_list_page_reads_to_end_11548() -> None:
    df = pl.select(
        pl.repeat(pl.arange(0, 2048, dtype=pl.UInt64).implode(), 2).alias("x"),
    )

    f = io.BytesIO()

    pq.write_table(df.to_arrow(), f, data_page_size=1)

    f.seek(0)

    result = pl.read_parquet(f).select(pl.col("x").list.len())
    assert result.to_series().to_list() == [2048, 2048]


def test_parquet_nano_second_schema() -> None:
    value = time(9, 0, 0)
    f = io.BytesIO()
    df = pd.DataFrame({"Time": [value]})
    df.to_parquet(f)
    f.seek(0)
    assert pl.read_parquet(f).item() == value


def test_nested_struct_read_12610() -> None:
    n = 1_025
    expect = pl.select(a=pl.int_range(0, n), b=pl.repeat(1, n)).with_columns(
        struct=pl.struct(pl.all())
    )

    f = io.BytesIO()
    expect.write_parquet(
        f,
        use_pyarrow=True,
    )
    f.seek(0)

    actual = pl.read_parquet(f)
    assert_frame_equal(expect, actual)


@pytest.mark.write_disk
def test_decimal_parquet(tmp_path: Path) -> None:
    path = tmp_path / "foo.parquet"
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": ["6", "7", "8"],
        }
    )

    df = df.with_columns(pl.col("bar").cast(pl.Decimal))

    df.write_parquet(path, statistics=True)
    out = pl.scan_parquet(path).filter(foo=2).collect().to_dict(as_series=False)
    assert out == {"foo": [2], "bar": [Decimal("7")]}


@pytest.mark.write_disk
def test_enum_parquet(tmp_path: Path) -> None:
    path = tmp_path / "enum.parquet"
    df = pl.DataFrame(
        [pl.Series("e", ["foo", "bar", "ham"], dtype=pl.Enum(["foo", "bar", "ham"]))]
    )
    df.write_parquet(path)
    out = pl.read_parquet(path)
    assert_frame_equal(df, out)


def test_parquet_rle_non_nullable_12814() -> None:
    column = (
        pl.select(x=pl.arange(0, 1025, dtype=pl.Int64) // 10).to_series().to_arrow()
    )
    schema = pa.schema([pa.field("foo", pa.int64(), nullable=False)])
    table = pa.Table.from_arrays([column], schema=schema)

    f = io.BytesIO()
    pq.write_table(table, f, data_page_size=1)
    f.seek(0)

    print(pq.read_table(f))

    f.seek(0)

    expect = pl.DataFrame(table).tail(10)
    actual = pl.read_parquet(f).tail(10)

    assert_frame_equal(expect, actual)


@pytest.mark.slow
def test_parquet_12831() -> None:
    n = 70_000
    df = pl.DataFrame({"x": ["aaaaaa"] * n})
    f = io.BytesIO()
    df.write_parquet(f, row_group_size=int(1e8), data_page_size=512)
    f.seek(0)
    assert_frame_equal(pl.from_arrow(pq.read_table(f)), df)  # type: ignore[arg-type]


@pytest.mark.write_disk
def test_parquet_struct_categorical(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame(
        [
            pl.Series("a", ["bob"], pl.Categorical),
            pl.Series("b", ["foo"], pl.Categorical),
        ]
    )

    file_path = tmp_path / "categorical.parquet"
    df.write_parquet(file_path)

    out = pl.read_parquet(file_path).select(pl.col("b").value_counts())
    assert out.to_dict(as_series=False) == {"b": [{"b": "foo", "count": 1}]}


@pytest.mark.write_disk
def test_null_parquet(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame([pl.Series("foo", [], dtype=pl.Int8)])
    file_path = tmp_path / "null.parquet"
    df.write_parquet(file_path)
    out = pl.read_parquet(file_path)
    assert_frame_equal(out, df)


@pytest.mark.write_disk
def test_write_parquet_with_null_col(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df1 = pl.DataFrame({"nulls": [None] * 2, "ints": [1] * 2})
    df2 = pl.DataFrame({"nulls": [None] * 2, "ints": [1] * 2})
    df3 = pl.DataFrame({"nulls": [None] * 3, "ints": [1] * 3})
    df = df1.vstack(df2)
    df = df.vstack(df3)
    file_path = tmp_path / "with_null.parquet"
    df.write_parquet(file_path, row_group_size=3)
    out = pl.read_parquet(file_path)
    assert_frame_equal(out, df)


@pytest.mark.write_disk
def test_scan_parquet_binary_buffered_reader(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"a": [1, 2, 3]})
    file_path = tmp_path / "test.parquet"
    df.write_parquet(file_path)

    with file_path.open("rb") as f:
        out = pl.scan_parquet(f).collect()
    assert_frame_equal(out, df)


@pytest.mark.write_disk
def test_read_parquet_binary_buffered_reader(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"a": [1, 2, 3]})
    file_path = tmp_path / "test.parquet"
    df.write_parquet(file_path)

    with file_path.open("rb") as f:
        out = pl.read_parquet(f)
    assert_frame_equal(out, df)


@pytest.mark.write_disk
def test_read_parquet_binary_file_io(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"a": [1, 2, 3]})
    file_path = tmp_path / "test.parquet"
    df.write_parquet(file_path)

    with file_path.open("rb", buffering=0) as f:
        out = pl.read_parquet(f)
    assert_frame_equal(out, df)


# https://github.com/pola-rs/polars/issues/15760
@pytest.mark.write_disk
def test_read_parquet_binary_fsspec(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"a": [1, 2, 3]})
    file_path = tmp_path / "test.parquet"
    df.write_parquet(file_path)

    with fsspec.open(file_path) as f:
        out = pl.read_parquet(f)
    assert_frame_equal(out, df)


def test_read_parquet_binary_bytes_io() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)

    out = pl.read_parquet(f)
    assert_frame_equal(out, df)


def test_read_parquet_binary_bytes() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    f = io.BytesIO()
    df.write_parquet(f)
    bytes = f.getvalue()

    out = pl.read_parquet(bytes)
    assert_frame_equal(out, df)


def test_utc_timezone_normalization_13670(tmp_path: Path) -> None:
    """'+00:00' timezones becomes 'UTC' timezone."""
    utc_path = tmp_path / "utc.parquet"
    zero_path = tmp_path / "00_00.parquet"
    utc_lowercase_path = tmp_path / "utc_lowercase.parquet"
    for tz, path in [
        ("+00:00", zero_path),
        ("UTC", utc_path),
        ("utc", utc_lowercase_path),
    ]:
        pq.write_table(
            pa.table(
                {"c1": [1234567890123] * 10},
                schema=pa.schema([pa.field("c1", pa.timestamp("ms", tz=tz))]),
            ),
            path,
        )

    df = pl.scan_parquet([utc_path, zero_path]).head(5).collect()
    assert cast(pl.Datetime, df.schema["c1"]).time_zone == "UTC"
    df = pl.scan_parquet([zero_path, utc_path]).head(5).collect()
    assert cast(pl.Datetime, df.schema["c1"]).time_zone == "UTC"
    df = pl.scan_parquet([zero_path, utc_lowercase_path]).head(5).collect()
    assert cast(pl.Datetime, df.schema["c1"]).time_zone == "UTC"


def test_parquet_rle_14333() -> None:
    vals = [True, False, True, False, True, False, True, False, True, False]
    table = pa.table({"a": vals})

    f = io.BytesIO()
    pq.write_table(table, f, data_page_version="2.0")
    f.seek(0)
    assert pl.read_parquet(f)["a"].to_list() == vals


def test_parquet_rle_null_binary_read_14638() -> None:
    df = pl.DataFrame({"x": [None]}, schema={"x": pl.String})

    f = io.BytesIO()
    df.write_parquet(f, use_pyarrow=True)
    f.seek(0)
    assert "RLE_DICTIONARY" in pq.read_metadata(f).row_group(0).column(0).encodings
    f.seek(0)
    assert_frame_equal(df, pl.read_parquet(f))


def test_parquet_string_rle_encoding() -> None:
    n = 3
    data = {
        "id": ["abcdefgh"] * n,
    }

    df = pl.DataFrame(data)
    f = io.BytesIO()
    df.write_parquet(f, use_pyarrow=False)
    f.seek(0)

    assert (
        "RLE_DICTIONARY"
        in pq.ParquetFile(f).metadata.to_dict()["row_groups"][0]["columns"][0][
            "encodings"
        ]
    )


@pytest.mark.may_fail_auto_streaming
def test_sliced_dict_with_nulls_14904() -> None:
    df = (
        pl.DataFrame({"x": [None, None]})
        .cast(pl.Categorical)
        .with_columns(y=pl.concat_list("x"))
        .slice(0, 1)
    )
    test_round_trip(df)


@pytest.fixture
def empty_compressed_datapage_v2_path(io_files_path: Path) -> Path:
    return io_files_path / "empty_datapage_v2.snappy.parquet"


def test_read_empty_compressed_datapage_v2_22170(
    empty_compressed_datapage_v2_path: Path,
) -> None:
    df = pl.DataFrame({"value": [None]}, schema={"value": pl.Float32})
    assert_frame_equal(df, pl.read_parquet(empty_compressed_datapage_v2_path))


def test_parquet_array_dtype() -> None:
    df = pl.DataFrame({"x": []})
    df = df.cast({"x": pl.Array(pl.Int64, shape=3)})
    test_round_trip(df)


def test_parquet_array_dtype_nulls() -> None:
    df = pl.DataFrame({"x": [[1, 2], None, [None, 3]]})
    df = df.cast({"x": pl.Array(pl.Int64, shape=2)})
    test_round_trip(df)


@pytest.mark.parametrize(
    ("series", "dtype"),
    [
        ([[1, 2, 3]], pl.List(pl.Int64)),
        ([[1, None, 3]], pl.List(pl.Int64)),
        (
            [{"a": []}, {"a": [1]}, {"a": [1, 2, 3]}],
            pl.Struct({"a": pl.List(pl.Int64)}),
        ),
        ([{"a": None}, None, {"a": [1, 2, None]}], pl.Struct({"a": pl.List(pl.Int64)})),
        (
            [[{"a": []}, {"a": [1]}, {"a": [1, 2, 3]}], None, [{"a": []}, {"a": [42]}]],
            pl.List(pl.Struct({"a": pl.List(pl.Int64)})),
        ),
        (
            [
                [1, None, 3],
                None,
                [1, 3, 4],
                None,
                [9, None, 4],
                [None, 42, 13],
                [37, 511, None],
            ],
            pl.List(pl.Int64),
        ),
        ([[1, 2, 3]], pl.Array(pl.Int64, 3)),
        ([[1, None, 3], None, [1, 2, None]], pl.Array(pl.Int64, 3)),
        ([[1, 2], None, [None, 3]], pl.Array(pl.Int64, 2)),
        ([[], [], []], pl.Array(pl.Int64, 0)),
        ([[], None, []], pl.Array(pl.Int64, 0)),
        (
            [[[1, 5, 2], [42, 13, 37]], [[1, 2, 3], [5, 2, 3]], [[1, 2, 1], [3, 1, 3]]],
            pl.Array(pl.Array(pl.Int8, 3), 2),
        ),
        (
            [[[1, 5, 2], [42, 13, 37]], None, [None, [3, 1, 3]]],
            pl.Array(pl.Array(pl.Int8, 3), 2),
        ),
        (
            [
                [[[2, 1], None, [4, 1], None], []],
                None,
                [None, [[4, 4], None, [1, 2]]],
            ],
            pl.Array(pl.List(pl.Array(pl.Int8, 2)), 2),
        ),
        ([[[], []]], pl.Array(pl.List(pl.Array(pl.Int8, 2)), 2)),
        (
            [
                [
                    [[[42, 13, 37, 15, 9, 20, 0, 0, 5, 10], None]],
                    [None, [None, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], None],
                ]
            ],
            pl.Array(pl.List(pl.Array(pl.Array(pl.Int8, 10), 2)), 2),
        ),
        (
            [
                None,
                [None],
                [[None]],
                [[[None]]],
                [[[[None]]]],
                [[[[[None]]]]],
                [[[[[1]]]]],
            ],
            pl.Array(pl.Array(pl.Array(pl.Array(pl.Array(pl.Int8, 1), 1), 1), 1), 1),
        ),
        (
            [
                None,
                [None],
                [[]],
                [[None]],
                [[[None], None]],
                [[[None], [None]]],
                [[[[None]], [[[1]]]]],
                [[[[[None]]]]],
                [[[[[1]]]]],
            ],
            pl.Array(pl.List(pl.Array(pl.List(pl.Array(pl.Int8, 1)), 1)), 1),
        ),
    ],
)
def test_complex_types(series: list[Any], dtype: pl.DataType) -> None:
    xs = pl.Series(series, dtype=dtype)
    df = pl.DataFrame({"x": xs})

    test_round_trip(df)


@pytest.mark.write_disk
def test_parquet_array_statistics(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    df = pl.DataFrame({"a": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "b": [1, 2, 3]})
    file_path = tmp_path / "test.parquet"

    df.with_columns(a=pl.col("a").list.to_array(3)).lazy().filter(
        pl.col("a") != [1, 2, 3]
    ).collect()
    df.with_columns(a=pl.col("a").list.to_array(3)).lazy().sink_parquet(file_path)

    result = pl.scan_parquet(file_path).filter(pl.col("a") != [1, 2, 3]).collect()
    assert result.to_dict(as_series=False) == {"a": [[4, 5, 6], [7, 8, 9]], "b": [2, 3]}


@pytest.mark.slow
@pytest.mark.write_disk
def test_read_parquet_only_loads_selected_columns_15098(
    memory_usage_without_pyarrow: MemoryUsage, tmp_path: Path
) -> None:
    """Only requested columns are loaded by ``read_parquet()``."""
    tmp_path.mkdir(exist_ok=True)

    # Each column will be about 8MB of RAM
    series = pl.arange(0, 1_000_000, dtype=pl.Int64, eager=True)

    file_path = tmp_path / "multicolumn.parquet"
    df = pl.DataFrame(
        {
            "a": series,
            "b": series,
        }
    )
    df.write_parquet(file_path)
    del df, series

    memory_usage_without_pyarrow.reset_tracking()

    # Only load one column:
    df = pl.read_parquet([file_path], columns=["b"], rechunk=False)
    del df
    # Only one column's worth of memory should be used; 2 columns would be
    # 16_000_000 at least, but there's some overhead.
    # assert 8_000_000 < memory_usage_without_pyarrow.get_peak() < 13_000_000


@pytest.mark.release
def test_max_statistic_parquet_writer() -> None:
    # this hits the maximal page size
    # so the row group will be split into multiple pages
    # the page statistics need to be correctly reduced
    # for this query to make sense
    n = 150_000

    # int64 is important to hit the page size
    df = pl.int_range(0, n, eager=True, dtype=pl.Int64).alias("int").to_frame()
    f = io.BytesIO()
    df.write_parquet(f, statistics=True, use_pyarrow=False, row_group_size=n)
    f.seek(0)
    result = pl.scan_parquet(f).filter(pl.col("int") > n - 3).collect()
    expected = pl.DataFrame({"int": [149998, 149999]})
    assert_frame_equal(result, expected)


@pytest.mark.slow
def test_hybrid_rle() -> None:
    # 10_007 elements to test if not a nice multiple of 8
    n = 10_007
    literal_literal = []
    literal_rle = []
    for i in range(500):
        literal_literal.append(np.repeat(i, 5))
        literal_literal.append(np.repeat(i + 2, 11))
        literal_rle.append(np.repeat(i, 5))
        literal_rle.append(np.repeat(i + 2, 15))
    literal_literal.append(np.random.randint(0, 10, size=2007))
    literal_rle.append(np.random.randint(0, 10, size=7))
    literal_literal = np.concatenate(literal_literal)
    literal_rle = np.concatenate(literal_rle)
    df = pl.DataFrame(
        {
            # Primitive types
            "i64": pl.Series([1, 2], dtype=pl.Int64).sample(n, with_replacement=True),
            "u64": pl.Series([1, 2], dtype=pl.UInt64).sample(n, with_replacement=True),
            "i8": pl.Series([1, 2], dtype=pl.Int8).sample(n, with_replacement=True),
            "u8": pl.Series([1, 2], dtype=pl.UInt8).sample(n, with_replacement=True),
            "string": pl.Series(["abc", "def"], dtype=pl.String).sample(
                n, with_replacement=True
            ),
            "categorical": pl.Series(["aaa", "bbb"], dtype=pl.Categorical).sample(
                n, with_replacement=True
            ),
            # Fill up bit-packing buffer in middle of consecutive run
            "large_bit_pack": np.concatenate(
                [np.repeat(i, 5) for i in range(2000)]
                + [np.random.randint(0, 10, size=7)]
            ),
            # Literal run that is not a multiple of 8 followed by consecutive
            # run initially long enough to RLE but not after padding literal
            "literal_literal": literal_literal,
            # Literal run that is not a multiple of 8 followed by consecutive
            # run long enough to RLE even after padding literal
            "literal_rle": literal_rle,
            # Final run not long enough to RLE
            "final_literal": np.concatenate(
                [np.random.randint(0, 100, 10_000), np.repeat(-1, 7)]
            ),
            # Final run long enough to RLE
            "final_rle": np.concatenate(
                [np.random.randint(0, 100, 9_998), np.repeat(-1, 9)]
            ),
            # Test filling up bit-packing buffer for encode_bool,
            # which is only used to encode validities
            "large_bit_pack_validity": [0, None] * 4092
            + [0] * 9
            + [1] * 9
            + [2] * 10
            + [0] * 1795,
        }
    )
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)
    for col in pq.ParquetFile(f).metadata.to_dict()["row_groups"][0]["columns"]:
        assert "RLE_DICTIONARY" in col["encodings"]
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


@given(
    df=dataframes(
        allowed_dtypes=[
            pl.Null,
            pl.List,
            pl.Array,
            pl.Int8,
            pl.UInt8,
            pl.UInt32,
            pl.Int64,
            # pl.Date, # Turned off because of issue #17599
            # pl.Time, # Turned off because of issue #17599
            pl.Binary,
            pl.Float32,
            pl.Float64,
            pl.String,
            pl.Boolean,
        ],
        min_size=1,
        max_size=500,
    )
)
@pytest.mark.slow
def test_roundtrip_parametric(df: pl.DataFrame) -> None:
    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)
    result = pl.read_parquet(f)

    assert_frame_equal(df, result)


def test_parquet_statistics_uint64_16683() -> None:
    u64_max = (1 << 64) - 1
    df = pl.Series("a", [u64_max, 0], dtype=pl.UInt64).to_frame()
    file = io.BytesIO()
    df.write_parquet(file, statistics=True)
    file.seek(0)
    statistics = pq.read_metadata(file).row_group(0).column(0).statistics

    assert statistics.min == 0
    assert statistics.max == u64_max


@pytest.mark.slow
@pytest.mark.parametrize("nullable", [True, False])
def test_read_byte_stream_split(nullable: bool) -> None:
    rng = np.random.default_rng(123)
    num_rows = 1_000
    values = rng.uniform(-1.0e6, 1.0e6, num_rows)
    if nullable:
        validity_mask = rng.integers(0, 2, num_rows).astype(np.bool_)
    else:
        validity_mask = None

    schema = pa.schema(
        [
            pa.field("floats", type=pa.float32(), nullable=nullable),
            pa.field("doubles", type=pa.float64(), nullable=nullable),
        ]
    )
    arrays = [pa.array(values, type=field.type, mask=validity_mask) for field in schema]
    table = pa.Table.from_arrays(arrays, schema=schema)
    df = cast(pl.DataFrame, pl.from_arrow(table))

    f = io.BytesIO()
    pq.write_table(
        table, f, compression="snappy", use_dictionary=False, use_byte_stream_split=True
    )

    f.seek(0)
    read = pl.read_parquet(f)

    assert_frame_equal(read, df)


@pytest.mark.slow
@pytest.mark.parametrize("rows_nullable", [True, False])
@pytest.mark.parametrize("item_nullable", [True, False])
def test_read_byte_stream_split_arrays(
    item_nullable: bool, rows_nullable: bool
) -> None:
    rng = np.random.default_rng(123)
    num_rows = 1_000
    max_array_len = 10
    array_lengths = rng.integers(0, max_array_len + 1, num_rows)
    if rows_nullable:
        row_validity_mask = rng.integers(0, 2, num_rows).astype(np.bool_)
        array_lengths[row_validity_mask] = 0
        row_validity_mask = pa.array(row_validity_mask)
    else:
        row_validity_mask = None

    offsets = np.zeros(num_rows + 1, dtype=np.int64)
    np.cumsum(array_lengths, out=offsets[1:])
    num_values = offsets[-1]
    values = rng.uniform(-1.0e6, 1.0e6, num_values)

    if item_nullable:
        element_validity_mask = rng.integers(0, 2, num_values).astype(np.bool_)
    else:
        element_validity_mask = None

    schema = pa.schema(
        [
            pa.field(
                "floats",
                type=pa.list_(pa.field("", pa.float32(), nullable=item_nullable)),
                nullable=rows_nullable,
            ),
            pa.field(
                "doubles",
                type=pa.list_(pa.field("", pa.float64(), nullable=item_nullable)),
                nullable=rows_nullable,
            ),
        ]
    )
    arrays = [
        pa.ListArray.from_arrays(
            pa.array(offsets),
            pa.array(values, type=field.type.field(0).type, mask=element_validity_mask),
            mask=row_validity_mask,
        )
        for field in schema
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    df = cast(pl.DataFrame, pl.from_arrow(table))

    f = io.BytesIO()
    pq.write_table(
        table, f, compression="snappy", use_dictionary=False, use_byte_stream_split=True
    )

    f.seek(0)
    read = pl.read_parquet(f)

    assert_frame_equal(read, df)


def test_parquet_nested_null_array_17795() -> None:
    f = io.BytesIO()
    pl.DataFrame([{"struct": {"field": None}}]).write_parquet(f)
    f.seek(0)
    pq.read_table(f)


def test_parquet_record_batches_pyarrow_fixed_size_list_16614() -> None:
    # @NOTE:
    # The minimum that I could get it to crash which was ~132000, but let's
    # just do 150000 to be sure.
    n = 150000
    x = pl.DataFrame(
        {"x": np.linspace((1, 2), (2 * n, 2 * n * 1), n, dtype=np.float32)},
        schema={"x": pl.Array(pl.Float32, 2)},
    )

    f = io.BytesIO()
    x.write_parquet(f)
    f.seek(0)
    b = pl.read_parquet(f, use_pyarrow=True)

    assert b["x"].shape[0] == n
    assert_frame_equal(b, x)


def test_parquet_list_element_field_name() -> None:
    f = io.BytesIO()
    (
        pl.DataFrame(
            {
                "a": [[1, 2], [1, 1, 1]],
            },
            schema={"a": pl.List(pl.Int64)},
        ).write_parquet(f, use_pyarrow=False)
    )

    f.seek(0)
    schema_str = str(pq.read_schema(f))
    assert "<element: int64>" in schema_str
    assert "child 0, element: int64" in schema_str


def test_nested_decimal() -> None:
    df = pl.DataFrame(
        {
            "a": [
                {"f0": None},
                None,
            ]
        },
        schema={"a": pl.Struct({"f0": pl.Decimal(precision=38, scale=8)})},
    )
    test_round_trip(df)


def test_nested_non_uniform_primitive() -> None:
    df = pl.DataFrame(
        {"a": [{"x": 0, "y": None}]},
        schema={
            "a": pl.Struct(
                {
                    "x": pl.Int16,
                    "y": pl.Int64,
                }
            )
        },
    )
    test_round_trip(df)


def test_parquet_nested_struct_17933() -> None:
    df = pl.DataFrame(
        {"a": [{"x": {"u": None}, "y": True}]},
        schema={
            "a": pl.Struct(
                {
                    "x": pl.Struct({"u": pl.String}),
                    "y": pl.Boolean(),
                }
            )
        },
    )
    test_round_trip(df)


# This is fixed with POLARS_FORCE_MULTISCAN=1. Without it we have
# first_metadata.unwrap() on None.
@pytest.mark.may_fail_auto_streaming
def test_parquet_pyarrow_map() -> None:
    xs = [
        [
            (0, 5),
            (1, 10),
            (2, 19),
            (3, 96),
        ]
    ]

    table = pa.table(
        [xs],
        schema=pa.schema(
            [
                ("x", pa.map_(pa.int32(), pa.int32(), keys_sorted=True)),
            ]
        ),
    )

    f = io.BytesIO()
    pq.write_table(table, f)

    expected = pl.DataFrame(
        {
            "x": [
                {"key": 0, "value": 5},
                {"key": 1, "value": 10},
                {"key": 2, "value": 19},
                {"key": 3, "value": 96},
            ]
        },
        schema={"x": pl.Struct({"key": pl.Int32, "value": pl.Int32})},
    )
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f).explode(["x"]), expected)

    # Test for https://github.com/pola-rs/polars/issues/21317
    # Specifying schema/allow_missing_columns
    for missing_columns in ["insert", "raise"]:
        assert_frame_equal(
            pl.read_parquet(
                f,
                schema={"x": pl.List(pl.Struct({"key": pl.Int32, "value": pl.Int32}))},
                missing_columns=missing_columns,  # type: ignore[arg-type]
            ).explode(["x"]),
            expected,
        )


@pytest.mark.parametrize(
    ("s", "elem"),
    [
        (pl.Series(["", "hello", "hi", ""], dtype=pl.String), ""),
        (pl.Series([0, 1, 2, 0], dtype=pl.Int64), 0),
        (pl.Series([[0], [1], [2], [0]], dtype=pl.Array(pl.Int64, 1)), [0]),
        (
            pl.Series([[0, 1], [1, 2], [2, 3], [0, 1]], dtype=pl.Array(pl.Int64, 2)),
            [0, 1],
        ),
    ],
)
def test_parquet_high_nested_null_17805(
    s: pl.Series, elem: str | int | list[int]
) -> None:
    test_round_trip(
        pl.DataFrame({"a": s}).select(
            pl.when(pl.col("a") == elem)
            .then(pl.lit(None))
            .otherwise(pl.concat_list(pl.col("a").alias("b")))
            .alias("c")
        )
    )


def test_struct_plain_encoded_statistics() -> None:
    df = pl.DataFrame(
        {
            "a": [None, None, None, None, {"x": None, "y": 0}],
        },
        schema={"a": pl.Struct({"x": pl.Int8, "y": pl.Int8})},
    )

    test_scan_round_trip(df)


@given(
    df=dataframes(
        min_size=5,
        excluded_dtypes=[pl.Decimal, pl.Categorical],
        allow_masked_out=False,  # PyArrow does not support this
    )
)
def test_scan_round_trip_parametric(df: pl.DataFrame) -> None:
    test_scan_round_trip(df)


def test_empty_rg_no_dict_page_18146() -> None:
    df = pl.DataFrame(
        {
            "a": [],
        },
        schema={"a": pl.String},
    )

    f = io.BytesIO()
    pq.write_table(df.to_arrow(), f, compression="NONE", use_dictionary=False)
    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


def test_write_sliced_lists_18069() -> None:
    f = io.BytesIO()
    a = pl.Series(3 * [None, ["$"] * 3], dtype=pl.List(pl.String))

    before = pl.DataFrame({"a": a}).slice(4, 2)
    before.write_parquet(f)

    f.seek(0)
    after = pl.read_parquet(f)

    assert_frame_equal(before, after)


def test_null_array_dict_pages_18085() -> None:
    test = pd.DataFrame(
        [
            {"A": float("NaN"), "B": 3, "C": None},
            {"A": float("NaN"), "B": None, "C": None},
        ]
    )

    f = io.BytesIO()
    test.to_parquet(f)
    f.seek(0)
    pl.read_parquet(f)


@given(
    df=dataframes(
        min_size=1,
        max_size=1000,
        allowed_dtypes=[
            pl.List,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ],
        allow_masked_out=False,  # PyArrow does not support this
    ),
    row_group_size=st.integers(min_value=10, max_value=1000),
)
def test_delta_encoding_roundtrip(df: pl.DataFrame, row_group_size: int) -> None:
    f = io.BytesIO()
    pq.write_table(
        df.to_arrow(),
        f,
        compression="NONE",
        use_dictionary=False,
        column_encoding="DELTA_BINARY_PACKED",
        write_statistics=False,
        row_group_size=row_group_size,
    )

    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


@given(
    df=dataframes(min_size=1, max_size=1000, allowed_dtypes=[pl.String, pl.Binary]),
    row_group_size=st.integers(min_value=10, max_value=1000),
)
def test_delta_length_byte_array_encoding_roundtrip(
    df: pl.DataFrame, row_group_size: int
) -> None:
    f = io.BytesIO()
    pq.write_table(
        df.to_arrow(),
        f,
        compression="NONE",
        use_dictionary=False,
        column_encoding="DELTA_LENGTH_BYTE_ARRAY",
        write_statistics=False,
        row_group_size=row_group_size,
    )

    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


@given(
    df=dataframes(min_size=1, max_size=1000, allowed_dtypes=[pl.String, pl.Binary]),
    row_group_size=st.integers(min_value=10, max_value=1000),
)
def test_delta_strings_encoding_roundtrip(
    df: pl.DataFrame, row_group_size: int
) -> None:
    f = io.BytesIO()
    pq.write_table(
        df.to_arrow(),
        f,
        compression="NONE",
        use_dictionary=False,
        column_encoding="DELTA_BYTE_ARRAY",
        write_statistics=False,
        row_group_size=row_group_size,
    )

    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)


EQUALITY_OPERATORS = ["__eq__", "__lt__", "__le__", "__gt__", "__ge__"]
BOOLEAN_OPERATORS = ["__or__", "__and__"]


@given(
    df=dataframes(
        min_size=0, max_size=100, min_cols=2, max_cols=5, allowed_dtypes=[pl.Int32]
    ),
    first_op=st.sampled_from(EQUALITY_OPERATORS),
    second_op=st.sampled_from(
        [None]
        + [
            (booljoin, eq)
            for booljoin in BOOLEAN_OPERATORS
            for eq in EQUALITY_OPERATORS
        ]
    ),
    l1=st.integers(min_value=0, max_value=1000),
    l2=st.integers(min_value=0, max_value=1000),
    r1=st.integers(min_value=0, max_value=1000),
    r2=st.integers(min_value=0, max_value=1000),
)
@pytest.mark.parametrize("parallel_st", ["auto", "prefiltered"])
def test_predicate_filtering(
    df: pl.DataFrame,
    first_op: str,
    second_op: None | tuple[str, str],
    l1: int,
    l2: int,
    r1: int,
    r2: int,
    parallel_st: Literal["auto", "prefiltered"],
) -> None:
    f = io.BytesIO()
    df.write_parquet(f, row_group_size=5)

    cols = df.columns

    l1s = cols[l1 % len(cols)]
    l2s = cols[l2 % len(cols)]
    expr = (getattr(pl.col(l1s), first_op))(pl.col(l2s))

    if second_op is not None:
        r1s = cols[r1 % len(cols)]
        r2s = cols[r2 % len(cols)]
        expr = getattr(expr, second_op[0])(
            (getattr(pl.col(r1s), second_op[1]))(pl.col(r2s))
        )

    f.seek(0)
    result = pl.scan_parquet(f, parallel=parallel_st).filter(expr).collect()
    assert_frame_equal(result, df.filter(expr))


@pytest.mark.parametrize(
    "use_dictionary",
    [False, True],
)
@pytest.mark.parametrize(
    "data_page_size",
    [1, None],
)
@given(
    s=series(
        min_size=1,
        max_size=10,
        excluded_dtypes=[
            pl.Decimal,
            pl.Categorical,
            pl.Enum,
            pl.Struct,  # See #19612.
        ],
        allow_masked_out=False,  # PyArrow does not support this
    ),
    offset=st.integers(0, 10),
    length=st.integers(0, 10),
)
def test_pyarrow_slice_roundtrip(
    s: pl.Series,
    use_dictionary: bool,
    data_page_size: int | None,
    offset: int,
    length: int,
) -> None:
    offset %= len(s) + 1
    length %= len(s) - offset + 1

    f = io.BytesIO()
    df = s.to_frame()
    pq.write_table(
        df.to_arrow(),
        f,
        compression="NONE",
        use_dictionary=use_dictionary,
        data_page_size=data_page_size,
    )

    f.seek(0)
    scanned = pl.scan_parquet(f).slice(offset, length).collect()
    assert_frame_equal(scanned, df.slice(offset, length))


@given(
    df=dataframes(
        min_size=1,
        max_size=5,
        min_cols=1,
        max_cols=1,
        excluded_dtypes=[pl.Decimal, pl.Categorical, pl.Enum],
    ),
    offset=st.integers(0, 100),
    length=st.integers(0, 100),
)
def test_slice_roundtrip(df: pl.DataFrame, offset: int, length: int) -> None:
    offset %= df.height + 1
    length %= df.height - offset + 1

    f = io.BytesIO()
    df.write_parquet(f)

    f.seek(0)
    scanned = pl.scan_parquet(f).slice(offset, length).collect()
    assert_frame_equal(scanned, df.slice(offset, length))


def test_struct_prefiltered() -> None:
    df = pl.DataFrame({"a": {"x": 1, "y": 2}})
    f = io.BytesIO()
    df.write_parquet(f)

    f.seek(0)
    (
        pl.scan_parquet(f, parallel="prefiltered")
        .filter(pl.col("a").struct.field("x") == 1)
        .collect()
    )


@pytest.mark.parametrize(
    "data",
    [
        (
            [{"x": ""}, {"x": "0"}],
            pa.struct([pa.field("x", pa.string(), nullable=True)]),
        ),
        (
            [{"x": ""}, {"x": "0"}],
            pa.struct([pa.field("x", pa.string(), nullable=False)]),
        ),
        ([[""], ["0"]], pa.list_(pa.field("item", pa.string(), nullable=False))),
        ([[""], ["0"]], pa.list_(pa.field("item", pa.string(), nullable=True))),
        ([[""], ["0"]], pa.list_(pa.field("item", pa.string(), nullable=False), 1)),
        ([[""], ["0"]], pa.list_(pa.field("item", pa.string(), nullable=True), 1)),
        (
            [["", "1"], ["0", "2"]],
            pa.list_(pa.field("item", pa.string(), nullable=False), 2),
        ),
        (
            [["", "1"], ["0", "2"]],
            pa.list_(pa.field("item", pa.string(), nullable=True), 2),
        ),
    ],
)
@pytest.mark.parametrize("nullable", [False, True])
def test_nested_skip_18303(
    data: tuple[list[dict[str, str] | list[str]], pa.DataType],
    nullable: bool,
) -> None:
    schema = pa.schema([pa.field("a", data[1], nullable=nullable)])
    tb = pa.table({"a": data[0]}, schema=schema)

    f = io.BytesIO()
    pq.write_table(tb, f)

    f.seek(0)
    scanned = pl.scan_parquet(f).slice(1, 1).collect()

    assert_frame_equal(scanned, pl.DataFrame(tb).slice(1, 1))


def test_nested_span_multiple_pages_18400() -> None:
    width = 4100
    df = pl.DataFrame(
        [
            pl.Series(
                "a",
                [
                    list(range(width)),
                    list(range(width)),
                ],
                pl.Array(pl.Int64, width),
            ),
        ]
    )

    f = io.BytesIO()
    pq.write_table(
        df.to_arrow(),
        f,
        use_dictionary=False,
        data_page_size=1024,
        column_encoding={"a": "PLAIN"},
    )

    f.seek(0)
    assert_frame_equal(df.head(1), pl.read_parquet(f, n_rows=1))


@given(
    df=dataframes(
        min_size=0,
        max_size=1000,
        min_cols=2,
        max_cols=5,
        excluded_dtypes=[pl.Decimal, pl.Categorical, pl.Enum, pl.Array],
        include_cols=[column("filter_col", pl.Boolean, allow_null=False)],
    ),
)
def test_parametric_small_page_mask_filtering(df: pl.DataFrame) -> None:
    f = io.BytesIO()
    df.write_parquet(f, data_page_size=1024)

    expr = pl.col("filter_col")
    f.seek(0)
    result = pl.scan_parquet(f, parallel="prefiltered").filter(expr).collect()
    assert_frame_equal(result, df.filter(expr))


@pytest.mark.parametrize(
    "value",
    [
        "abcd",
        0,
        0.0,
        False,
    ],
)
def test_different_page_validity_across_pages(value: str | int | float | bool) -> None:
    df = pl.DataFrame(
        {
            "a": [None] + [value] * 4000,
        }
    )

    f = io.BytesIO()
    pq.write_table(
        df.to_arrow(),
        f,
        use_dictionary=False,
        data_page_size=1024,
        column_encoding={"a": "PLAIN"},
    )

    f.seek(0)
    assert_frame_equal(df, pl.read_parquet(f))


@given(
    df=dataframes(
        min_size=0,
        max_size=100,
        min_cols=2,
        max_cols=5,
        allowed_dtypes=[pl.String, pl.Binary],
        include_cols=[
            column("filter_col", pl.Int8, st.integers(0, 1), allow_null=False)
        ],
    ),
)
def test_delta_length_byte_array_prefiltering(df: pl.DataFrame) -> None:
    cols = df.columns

    encodings = dict.fromkeys(cols, "DELTA_LENGTH_BYTE_ARRAY")
    encodings["filter_col"] = "PLAIN"

    f = io.BytesIO()
    pq.write_table(
        df.to_arrow(),
        f,
        use_dictionary=False,
        column_encoding=encodings,
    )

    f.seek(0)
    expr = pl.col("filter_col") == 0
    result = pl.scan_parquet(f, parallel="prefiltered").filter(expr).collect()
    assert_frame_equal(result, df.filter(expr))


@given(
    df=dataframes(
        min_size=0,
        max_size=10,
        min_cols=1,
        max_cols=5,
        excluded_dtypes=[pl.Decimal, pl.Categorical, pl.Enum],
        include_cols=[
            column("filter_col", pl.Int8, st.integers(0, 1), allow_null=False)
        ],
    ),
)
def test_general_prefiltering(df: pl.DataFrame) -> None:
    f = io.BytesIO()
    df.write_parquet(f)

    expr = pl.col("filter_col") == 0

    f.seek(0)
    result = pl.scan_parquet(f, parallel="prefiltered").filter(expr).collect()
    assert_frame_equal(result, df.filter(expr))


@given(
    df=dataframes(
        min_size=0,
        max_size=10,
        min_cols=1,
        max_cols=5,
        excluded_dtypes=[pl.Decimal, pl.Categorical, pl.Enum],
        include_cols=[column("filter_col", pl.Boolean, allow_null=False)],
    ),
)
def test_row_index_prefiltering(df: pl.DataFrame) -> None:
    f = io.BytesIO()
    df.write_parquet(f)

    expr = pl.col("filter_col")

    f.seek(0)
    result = (
        pl.scan_parquet(
            f, row_index_name="ri", row_index_offset=42, parallel="prefiltered"
        )
        .filter(expr)
        .collect()
    )
    assert_frame_equal(result, df.with_row_index("ri", 42).filter(expr))


def test_empty_parquet() -> None:
    f_pd = io.BytesIO()
    f_pl = io.BytesIO()

    pd.DataFrame().to_parquet(f_pd)
    pl.DataFrame().write_parquet(f_pl)

    f_pd.seek(0)
    f_pl.seek(0)

    empty_from_pd = pl.read_parquet(f_pd)
    assert empty_from_pd.shape == (0, 0)

    empty_from_pl = pl.read_parquet(f_pl)
    assert empty_from_pl.shape == (0, 0)


@pytest.mark.parametrize(
    "strategy",
    ["columns", "row_groups", "prefiltered"],
)
@pytest.mark.write_disk
def test_row_index_projection_pushdown_18463(
    tmp_path: Path, strategy: pl.ParallelStrategy
) -> None:
    tmp_path.mkdir(exist_ok=True)
    f = tmp_path / "test.parquet"

    pl.DataFrame({"A": [1, 4], "B": [2, 5]}).write_parquet(f)

    df = pl.scan_parquet(f, parallel=strategy).with_row_index()

    assert_frame_equal(df.select("index").collect(), df.collect().select("index"))

    df = pl.scan_parquet(f, parallel=strategy).with_row_index("other_idx_name")

    assert_frame_equal(
        df.select("other_idx_name").collect(), df.collect().select("other_idx_name")
    )

    df = pl.scan_parquet(f, parallel=strategy).with_row_index(offset=42)

    assert_frame_equal(df.select("index").collect(), df.collect().select("index"))

    df = pl.scan_parquet(f, parallel=strategy).with_row_index()

    assert_frame_equal(
        df.select("index").slice(1, 1).collect(),
        df.collect().select("index").slice(1, 1),
    )


@pytest.mark.write_disk
def test_write_binary_open_file(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    path = tmp_path / "test.parquet"

    with path.open("wb") as f_write:
        df.write_parquet(f_write)

    out = pl.read_parquet(path)
    assert_frame_equal(out, df)


def test_prefilter_with_projection() -> None:
    f = io.BytesIO()
    pl.DataFrame({"a": [1], "b": [2]}).write_parquet(f)

    f.seek(0)
    (
        pl.scan_parquet(f, parallel="prefiltered")
        .filter(pl.col.a == 1)
        .select(pl.col.a)
        .collect()
    )


@pytest.mark.parametrize("parallel_strategy", ["prefiltered", "row_groups"])
@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame({"x": 1, "y": 1}),
        pl.DataFrame({"x": 1, "b": 1, "y": 1}),  # hive columns in file
    ],
)
@pytest.mark.write_disk
def test_prefilter_with_hive_19766(
    tmp_path: Path, df: pl.DataFrame, parallel_strategy: str
) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "a=1/b=1").mkdir(exist_ok=True, parents=True)

    df.write_parquet(tmp_path / "a=1/b=1/1")
    expect = df.with_columns(a=pl.lit(1, dtype=pl.Int64), b=pl.lit(1, dtype=pl.Int64))

    lf = pl.scan_parquet(tmp_path, parallel=parallel_strategy)  # type: ignore[arg-type]

    for predicate in [
        pl.col("a") == 1,
        pl.col("x") == 1,
        (pl.col("a") == 1) & (pl.col("x") == 1),
        pl.col("b") == 1,
        pl.col("y") == 1,
        (pl.col("a") == 1) & (pl.col("b") == 1),
    ]:
        assert_frame_equal(
            lf.filter(predicate).collect(),
            expect,
        )


@pytest.mark.parametrize("parallel", ["columns", "row_groups", "prefiltered", "none"])
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("projection", [pl.all(), pl.col("b")])
@pytest.mark.write_disk
def test_allow_missing_columns(
    tmp_path: Path,
    parallel: str,
    streaming: bool,
    projection: pl.Expr,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    dfs = [pl.DataFrame({"a": 1, "b": 1}), pl.DataFrame({"a": 2})]
    paths = [tmp_path / "1", tmp_path / "2"]

    for df, path in zip(dfs, paths):
        df.write_parquet(path)

    expected_full = pl.DataFrame({"a": [1, 2], "b": [1, None]})
    expected = expected_full.select(projection)

    with pytest.raises(
        pl.exceptions.ColumnNotFoundError,
        match="passing `missing_columns='insert'`",
    ):
        pl.read_parquet(paths, parallel=parallel)  # type: ignore[arg-type]

    with pytest.raises(
        pl.exceptions.ColumnNotFoundError,
        match="passing `missing_columns='insert'`",
    ):
        pl.scan_parquet(paths, parallel=parallel).select(projection).collect(  # type: ignore[arg-type]
            engine="streaming" if streaming else "in-memory"
        )

    assert_frame_equal(
        pl.read_parquet(
            paths,
            parallel=parallel,  # type: ignore[arg-type]
            missing_columns="insert",
        ).select(projection),
        expected,
    )

    assert_frame_equal(
        pl.scan_parquet(paths, parallel=parallel, missing_columns="insert")  # type: ignore[arg-type]
        .select(projection)
        .collect(engine="streaming" if streaming else "in-memory"),
        expected,
    )

    # Test deprecated parameter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        with pytest.raises(
            pl.exceptions.ColumnNotFoundError,
            match="passing `missing_columns='insert'`",
        ):
            assert_frame_equal(
                pl.scan_parquet(
                    paths,
                    parallel=parallel,  # type: ignore[arg-type]
                    allow_missing_columns=False,
                ).collect(engine="streaming" if streaming else "in-memory"),
                expected_full,
            )

        assert_frame_equal(
            pl.scan_parquet(
                paths,
                parallel=parallel,  # type: ignore[arg-type]
                allow_missing_columns=True,
            ).collect(engine="streaming" if streaming else "in-memory"),
            expected_full,
        )


def test_nested_nonnullable_19158() -> None:
    # Bug is based on the top-level struct being nullable and the inner list
    # not being nullable.
    tbl = pa.table(
        {
            "a": [{"x": [1]}, None, {"x": [1, 2]}, None],
        },
        schema=pa.schema(
            [
                pa.field(
                    "a",
                    pa.struct([pa.field("x", pa.list_(pa.int8()), nullable=False)]),
                    nullable=True,
                )
            ]
        ),
    )

    f = io.BytesIO()
    pq.write_table(tbl, f)

    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), pl.DataFrame(tbl))


D = Decimal


@pytest.mark.parametrize("precision", range(1, 37, 2))
@pytest.mark.parametrize(
    "nesting",
    [
        # Struct
        lambda t: ([{"x": None}, None], pl.Struct({"x": t})),
        lambda t: ([None, {"x": None}], pl.Struct({"x": t})),
        lambda t: ([{"x": D("1.5")}, None], pl.Struct({"x": t})),
        lambda t: ([{"x": D("1.5")}, {"x": D("4.8")}], pl.Struct({"x": t})),
        # Array
        lambda t: ([[None, None, D("8.2")], None], pl.Array(t, 3)),
        lambda t: ([None, [None, D("8.9"), None]], pl.Array(t, 3)),
        lambda t: ([[D("1.5"), D("3.7"), D("4.1")], None], pl.Array(t, 3)),
        lambda t: (
            [[D("1.5"), D("3.7"), D("4.1")], [D("2.8"), D("5.2"), D("8.9")]],
            pl.Array(t, 3),
        ),
        # List
        lambda t: ([[None, D("8.2")], None], pl.List(t)),
        lambda t: ([None, [D("8.9"), None]], pl.List(t)),
        lambda t: ([[D("1.5"), D("4.1")], None], pl.List(t)),
        lambda t: ([[D("1.5"), D("3.7"), D("4.1")], [D("2.8"), D("8.9")]], pl.List(t)),
    ],
)
def test_decimal_precision_nested_roundtrip(
    nesting: Callable[[pl.DataType], tuple[list[Any], pl.DataType]],
    precision: int,
) -> None:
    # Limit the context as to not disturb any other tests
    with decimal.localcontext() as ctx:
        ctx.prec = precision

        decimal_dtype = pl.Decimal(precision=precision)
        values, dtype = nesting(decimal_dtype)

    df = pl.Series("a", values, dtype).to_frame()

    test_round_trip(df)


@pytest.mark.parametrize("parallel", ["prefiltered", "columns", "row_groups", "auto"])
def test_conserve_sortedness(
    monkeypatch: Any, capfd: Any, parallel: pl.ParallelStrategy
) -> None:
    f = io.BytesIO()

    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, None],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, None],
            "c": [None, 5, 4, 3, 2, 1],
            "d": [None, 5.0, 4.0, 3.0, 2.0, 1.0],
            "a_nosort": [1, 2, 3, 4, 5, None],
            "f": range(6),
        }
    )

    for col, descending, nulls_last in [("a", False, False), ("c", True, True)]:
        col_idx = df.get_column_index(col)
        f.seek(0)
        pq.write_table(
            df.to_arrow(),
            f,
            sorting_columns=[
                pq.SortingColumn(col_idx, descending, nulls_last),
            ],
        )
        f.truncate()
        f.seek(0)

        monkeypatch.setenv("POLARS_VERBOSE", "1")

        df = pl.scan_parquet(f, parallel=parallel).filter(pl.col.f > 1).collect()

        captured = capfd.readouterr().err

        # @NOTE: We don't conserve sortedness for anything except integers at the
        # moment.
        assert (
            captured.count("Parquet conserved SortingColumn for column chunk of") == 1
        )
        assert (
            f"Parquet conserved SortingColumn for column chunk of '{col}' to {'Descending' if descending else 'Ascending'}"
            in captured
        )


@pytest.mark.parametrize("use_dictionary", [True, False])
@pytest.mark.parametrize(
    "values",
    [
        (size, x)
        for size in [1, 2, 3, 4, 8, 12, 15, 16, 32]
        for x in [
            [list(range(size)), list(range(7, 7 + size))],
            [list(range(size)), None],
            [list(range(i, i + size)) for i in range(13)],
            [list(range(i, i + size)) if i % 3 < 2 else None for i in range(13)],
        ]
    ],
)
@pytest.mark.parametrize(
    "filt",
    [
        lambda _: None,
        lambda _: pl.col.f > 0,
        lambda _: pl.col.f > 1,
        lambda _: pl.col.f < 5,
        lambda _: pl.col.f % 2 == 0,
        lambda _: pl.col.f % 5 < 4,
        lambda values: (0, min(1, len(values))),
        lambda _: (1, 1),
        lambda _: (-2, 1),
        lambda values: (1, len(values) - 2),
    ],
)
def test_fixed_size_binary(
    use_dictionary: bool,
    values: tuple[int, list[None | list[int]]],
    filt: Callable[[list[None | list[int]]], None | pl.Expr | tuple[int, int]],
) -> None:
    size, elems = values
    bs = [bytes(v) if v is not None else None for v in elems]

    tbl = pa.table(
        {
            "a": bs,
            "f": range(len(bs)),
        },
        schema=pa.schema(
            [
                pa.field("a", pa.binary(length=size), nullable=True),
                pa.field("f", pa.int32(), nullable=True),
            ]
        ),
    )

    df = pl.DataFrame(tbl)

    f = io.BytesIO()
    pq.write_table(tbl, f, use_dictionary=use_dictionary)

    f.seek(0)

    loaded: pl.DataFrame
    if isinstance(filt, pl.Expr):
        loaded = pl.scan_parquet(f).filter(filt).collect()
        df = df.filter(filt)
    elif isinstance(filt, tuple):
        loaded = pl.scan_parquet(f).slice(filt[0], filt[1]).collect()
        df = df.slice(filt[0], filt[1])
    else:
        loaded = pl.read_parquet(f)

    assert_frame_equal(loaded, df)


def test_decode_f16() -> None:
    values = [float("nan"), 0.0, 0.5, 1.0, 1.5]

    table = pa.Table.from_pydict(
        {
            "x": pa.array(np.array(values, dtype=np.float16), type=pa.float16()),
        }
    )

    df = pl.Series("x", values, pl.Float32).to_frame()

    f = io.BytesIO()
    pq.write_table(table, f)

    f.seek(0)
    assert_frame_equal(pl.read_parquet(f), df)

    f.seek(0)
    assert_frame_equal(
        pl.scan_parquet(f).filter(pl.col.x > 0.5).collect(),
        df.filter(pl.col.x > 0.5),
    )

    f.seek(0)
    assert_frame_equal(
        pl.scan_parquet(f).slice(1, 3).collect(),
        df.slice(1, 3),
    )


def test_invalid_utf8_binary() -> None:
    a = pl.Series("a", [b"\x80"], pl.Binary).to_frame()
    f = io.BytesIO()

    a.write_parquet(f)
    f.seek(0)
    out = pl.read_parquet(f)

    assert_frame_equal(a, out)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Null,
        pl.Int8,
        pl.Int32,
        pl.Datetime(),
        pl.String,
        pl.Binary,
        pl.Boolean,
        pl.Struct({"x": pl.Int32}),
        pl.List(pl.Int32),
        pl.Array(pl.Int32, 0),
        pl.Array(pl.Int32, 2),
    ],
)
@pytest.mark.parametrize(
    "filt",
    [
        pl.col.f == 0,
        pl.col.f != 0,
        pl.col.f == 1,
        pl.col.f != 1,
        pl.col.f == 2,
        pl.col.f != 2,
        pl.col.f == 3,
        pl.col.f != 3,
    ],
)
def test_filter_only_invalid(dtype: pl.DataType, filt: pl.Expr) -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", [None, None, None], dtype),
            pl.Series("f", range(3), pl.Int32),
        ]
    )

    f = io.BytesIO()

    df.write_parquet(f)
    f.seek(0)
    out = pl.scan_parquet(f, parallel="prefiltered").filter(filt).collect()

    assert_frame_equal(df.filter(filt), out)


def test_nested_nulls() -> None:
    df = pl.Series(
        "a",
        [
            [None, None],
            None,
            [None, 1],
            [None, None],
            [2, None],
        ],
        pl.Array(pl.Int32, 2),
    ).to_frame()

    f = io.BytesIO()
    df.write_parquet(f)

    f.seek(0)
    out = pl.read_parquet(f)
    assert_frame_equal(out, df)


@pytest.mark.parametrize("content", [[], [None], [None, 0.0]])
def test_nested_dicts(content: list[float | None]) -> None:
    df = pl.Series("a", [content], pl.List(pl.Float64)).to_frame()

    f = io.BytesIO()
    df.write_parquet(f, use_pyarrow=True)
    f.seek(0)
    assert_frame_equal(df, pl.read_parquet(f))


@pytest.mark.parametrize(
    "leading_nulls",
    [
        [],
        [None] * 7,
    ],
)
@pytest.mark.parametrize(
    "trailing_nulls",
    [
        [],
        [None] * 7,
    ],
)
@pytest.mark.parametrize(
    "first_chunk",
    # Create both RLE and Bitpacked chunks
    [
        [1] * 57,
        [1 if i % 7 < 3 and i % 5 > 3 else None for i in range(57)],
        list(range(57)),
        [i if i % 7 < 3 and i % 5 > 3 else None for i in range(57)],
    ],
)
@pytest.mark.parametrize(
    "second_chunk",
    # Create both RLE and Bitpacked chunks
    [
        [2] * 57,
        [2 if i % 7 < 3 and i % 5 > 3 else None for i in range(57)],
        list(range(57)),
        [i if i % 7 < 3 and i % 5 > 3 else None for i in range(57)],
    ],
)
@pytest.mark.slow
def test_dict_slices(
    leading_nulls: list[None],
    trailing_nulls: list[None],
    first_chunk: list[None | int],
    second_chunk: list[None | int],
) -> None:
    df = pl.Series(
        "a", leading_nulls + first_chunk + second_chunk + trailing_nulls, pl.Int64
    ).to_frame()

    f = io.BytesIO()
    df.write_parquet(f)

    for offset in chain([0, 1, 2], range(3, df.height, 3)):
        for length in chain([df.height, 1, 2], range(3, df.height - offset, 3)):
            f.seek(0)
            assert_frame_equal(
                pl.scan_parquet(f).slice(offset, length).collect(),
                df.slice(offset, length),
            )


@pytest.mark.parametrize(
    "mask",
    [
        [i % 13 < 3 and i % 17 > 3 for i in range(57 * 2)],
        [False] * 23 + [True] * 68 + [False] * 23,
        [False] * 23 + [True] * 24 + [False] * 20 + [True] * 24 + [False] * 23,
        [True] + [False] * 22 + [True] * 24 + [False] * 20 + [True] * 24 + [False] * 23,
        [False] * 23 + [True] * 24 + [False] * 20 + [True] * 24 + [False] * 22 + [True],
        [True]
        + [False] * 22
        + [True] * 24
        + [False] * 20
        + [True] * 24
        + [False] * 22
        + [True],
        [False] * 56 + [True] * 58,
        [False] * 57 + [True] * 57,
        [False] * 58 + [True] * 56,
        [True] * 56 + [False] * 58,
        [True] * 57 + [False] * 57,
        [True] * 58 + [False] * 56,
    ],
)
@pytest.mark.parametrize(
    "first_chunk",
    # Create both RLE and Bitpacked chunks
    [
        [1] * 57,
        [1 if i % 7 < 3 and i % 5 > 3 else None for i in range(57)],
        list(range(57)),
        [i if i % 7 < 3 and i % 5 > 3 else None for i in range(57)],
    ],
)
@pytest.mark.parametrize(
    "second_chunk",
    # Create both RLE and Bitpacked chunks
    [
        [2] * 57,
        [2 if i % 7 < 3 and i % 5 > 3 else None for i in range(57)],
        list(range(57)),
        [i if i % 7 < 3 and i % 5 > 3 else None for i in range(57)],
    ],
)
def test_dict_masked(
    mask: list[bool],
    first_chunk: list[None | int],
    second_chunk: list[None | int],
) -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", first_chunk + second_chunk, pl.Int64),
            pl.Series("f", mask, pl.Boolean),
        ]
    )

    f = io.BytesIO()
    df.write_parquet(f)

    f.seek(0)
    assert_frame_equal(
        pl.scan_parquet(f, parallel="prefiltered").filter(pl.col.f).collect(),
        df.filter(pl.col.f),
    )


@pytest.mark.may_fail_auto_streaming
def test_categorical_sliced_20017() -> None:
    f = io.BytesIO()
    df = (
        pl.Series("a", ["a", None])
        .to_frame()
        .with_columns(pl.col.a.cast(pl.Categorical))
    )
    df.write_parquet(f)

    f.seek(0)
    assert_frame_equal(
        pl.read_parquet(f, n_rows=1),
        df.head(1),
    )


@given(
    s=series(name="a", dtype=pl.String, min_size=7, max_size=7),
    mask=series(
        name="mask", dtype=pl.Boolean, min_size=7, max_size=7, allow_null=False
    ),
)
def test_categorical_parametric_masked(s: pl.Series, mask: pl.Series) -> None:
    f = io.BytesIO()

    df = pl.DataFrame([s, mask]).with_columns(pl.col.a.cast(pl.Categorical))
    df.write_parquet(f)

    f.seek(0)
    assert_frame_equal(
        pl.scan_parquet(f, parallel="prefiltered").filter(pl.col.mask).collect(),
        df.filter(pl.col.mask),
    )


@given(
    s=series(name="a", dtype=pl.String, min_size=7, max_size=7),
    start=st.integers(0, 6),
    length=st.integers(1, 7),
)
def test_categorical_parametric_sliced(s: pl.Series, start: int, length: int) -> None:
    length = min(7 - start, length)

    f = io.BytesIO()

    df = s.to_frame().with_columns(pl.col.a.cast(pl.Categorical))
    df.write_parquet(f)

    f.seek(0)
    assert_frame_equal(
        pl.scan_parquet(f).slice(start, length).collect(),
        df.slice(start, length),
    )


@pytest.mark.write_disk
def test_prefilter_with_projection_column_order_20175(tmp_path: Path) -> None:
    path = tmp_path / "1"

    pl.DataFrame({"a": 1, "b": 1, "c": 1, "d": 1, "e": 1}).write_parquet(path)

    q = (
        pl.scan_parquet(path, parallel="prefiltered")
        .filter(pl.col("a") == 1)
        .select("a", "d", "c")
    )

    assert_frame_equal(q.collect(), pl.DataFrame({"a": 1, "d": 1, "c": 1}))

    f = io.BytesIO()

    pl.read_csv(b"""\
c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10
1,1,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,1
""").write_parquet(f)

    f.seek(0)

    q = (
        pl.scan_parquet(
            f,
            rechunk=True,
            parallel="prefiltered",
        )
        .filter(
            pl.col("c0") == 1,
        )
        .select("c0", "c9", "c3")
    )

    assert_frame_equal(
        q.collect(),
        pl.read_csv(b"""\
c0,c9,c3
1,1,1
1,1,1
"""),
    )


def test_utf8_verification_with_slice_20174() -> None:
    f = io.BytesIO()
    pq.write_table(
        pl.Series("s", ["a", "a" * 128]).to_frame().to_arrow(), f, use_dictionary=False
    )

    f.seek(0)
    assert_frame_equal(
        pl.scan_parquet(f).head(1).collect(),
        pl.Series("s", ["a"]).to_frame(),
    )


@pytest.mark.parametrize("parallel", ["prefiltered", "row_groups"])
@pytest.mark.parametrize(
    "projection",
    [
        {"a": pl.Int64(), "b": pl.Int64()},
        {"b": pl.Int64(), "a": pl.Int64()},
    ],
)
def test_parquet_prefiltered_unordered_projection_20175(
    parallel: str, projection: dict[str, pl.DataType]
) -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", [0], pl.Int64),
            pl.Series("b", [0], pl.Int64),
        ]
    )

    f = io.BytesIO()
    df.write_parquet(f)

    f.seek(0)
    out = (
        pl.scan_parquet(f, parallel=parallel)  # type: ignore[arg-type]
        .filter(pl.col.a >= 0)
        .select(*projection.keys())
        .collect()
    )

    assert out.schema == projection


def test_parquet_unsupported_dictionary_to_pl_17945() -> None:
    t = pa.table(
        {
            "col1": pa.DictionaryArray.from_arrays([0, 0, None, 1], [42, 1337]),
        },
        schema=pa.schema({"col1": pa.dictionary(pa.uint32(), pa.int64())}),
    )

    f = io.BytesIO()
    pq.write_table(t, f, use_dictionary=False)
    f.truncate()

    f.seek(0)
    assert_series_equal(
        pl.Series("col1", [42, 42, None, 1337], pl.Int64),
        pl.read_parquet(f).to_series(),
    )

    f.seek(0)
    pq.write_table(t, f)
    f.truncate()

    f.seek(0)
    assert_series_equal(
        pl.Series("col1", [42, 42, None, 1337], pl.Int64),
        pl.read_parquet(f).to_series(),
    )


@pytest.mark.may_fail_auto_streaming
def test_parquet_cast_to_cat() -> None:
    t = pa.table(
        {
            "col1": pa.DictionaryArray.from_arrays([0, 0, None, 1], ["A", "B"]),
        },
        schema=pa.schema({"col1": pa.dictionary(pa.uint32(), pa.string())}),
    )

    f = io.BytesIO()
    pq.write_table(t, f, use_dictionary=False)
    f.truncate()

    f.seek(0)
    assert_series_equal(
        pl.Series("col1", ["A", "A", None, "B"], pl.Categorical),
        pl.read_parquet(f).to_series(),
    )

    f.seek(0)
    pq.write_table(t, f)
    f.truncate()

    f.seek(0)
    assert_series_equal(
        pl.Series("col1", ["A", "A", None, "B"], pl.Categorical),
        pl.read_parquet(f).to_series(),
    )


def test_parquet_roundtrip_lex_cat_20288() -> None:
    f = io.BytesIO()
    df = pl.Series("a", ["A", "B"], pl.Categorical(ordering="lexical")).to_frame()
    df.write_parquet(f)
    f.seek(0)
    dt = pl.scan_parquet(f).collect_schema()["a"]
    assert isinstance(dt, pl.Categorical)
    assert dt.ordering == "lexical"


def test_from_parquet_20271() -> None:
    f = io.BytesIO()
    df = pl.Series("b", ["D", "E"], pl.Categorical).to_frame()
    df.write_parquet(f)
    del df
    f.seek(0)
    df = pl.read_parquet(f)
    assert_series_equal(df.to_series(), pl.Series("b", ["D", "E"], pl.Categorical))


def test_boolean_slice_pushdown_20314() -> None:
    s = pl.Series("a", [None, False, True])
    f = io.BytesIO()

    s.to_frame().write_parquet(f)

    f.seek(0)
    assert pl.scan_parquet(f).slice(2, 1).collect().item()


def test_load_pred_pushdown_fsl_19241() -> None:
    f = io.BytesIO()

    fsl = pl.Series("a", [[[1, 2]]], pl.Array(pl.Array(pl.Int8, 2), 1))
    filt = pl.Series("f", [1])

    pl.DataFrame([fsl, filt]).write_parquet(f)

    f.seek(0)
    q = pl.scan_parquet(f, parallel="prefiltered").filter(pl.col.f != 4)

    assert_frame_equal(q.collect(), pl.DataFrame([fsl, filt]))


def test_struct_list_statistics_20510() -> None:
    # Test PyArrow - Utf8ViewArray
    data = {
        "name": ["a", "b"],
        "data": [
            {"title": "Title", "data": [0, 1, 3]},
            {"title": "Title", "data": [0, 1, 3]},
        ],
    }
    df = pl.DataFrame(
        data,
        schema=pl.Schema(
            {
                "name": pl.String(),
                "data": pl.Struct(
                    {
                        "title": pl.String,
                        "data": pl.List(pl.Int64),
                    }
                ),
            }
        ),
    )

    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)
    result = pl.scan_parquet(f).filter(pl.col("name") == "b").collect()

    assert_frame_equal(result, df.filter(pl.col("name") == "b"))

    # Test PyArrow - Utf8Array
    tb = pa.table(
        data,
        schema=pa.schema(
            [
                ("name", pa.string()),
                (
                    "data",
                    pa.struct(
                        [
                            ("title", pa.string()),
                            ("data", pa.list_(pa.int64())),
                        ]
                    ),
                ),
            ]
        ),
    )

    f.seek(0)
    pq.write_table(tb, f)
    f.truncate()
    f.seek(0)
    result = pl.scan_parquet(f).filter(pl.col("name") == "b").collect()

    assert_frame_equal(result, df.filter(pl.col("name") == "b"))


def test_required_masked_skip_values_20809(monkeypatch: Any) -> None:
    df = pl.DataFrame(
        [pl.Series("a", list(range(20)) + [42] * 15), pl.Series("b", range(35))]
    )
    needle = [16, 33]

    f = io.BytesIO()
    df.write_parquet(f)

    f.seek(0)
    monkeypatch.setenv("POLARS_PQ_PREFILTERED_MASK", "pre")
    df1 = (
        pl.scan_parquet(f, parallel="prefiltered")
        .filter(pl.col.b.is_in(needle))
        .collect()
    )

    f.seek(0)
    df2 = pl.read_parquet(f, parallel="columns").filter(pl.col.b.is_in(needle))

    assert_frame_equal(df1, df2)


def get_tests_from_dtype(
    dtype: pl.DataType, f: Callable[[int], Any]
) -> list[tuple[pl.DataType, list[Any], list[Any]]]:
    return [
        (dtype, [f(i) for i in range(10)], [f(i) for i in range(11)]),
        (
            dtype,
            [f(i) for i in range(1337)],
            [f(i) for i in [0, 1, 5, 7, 101, 1023, 1336, 1337, 1338]],
        ),
        (
            dtype,
            list(
                functools.reduce(
                    lambda x, y: list(x) + y,
                    ([f(i)] * (i % 13) for i in range(1337)),
                    [],
                )
            ),
            [f(i) for i in [0, 1, 5, 7, 101, 1023, 1336, 1337, 1338]],
        ),
        (
            dtype,
            [f(5)] * 37 + [f(10)] * 61 + [f(1996)] * 21,
            [f(i) for i in [1, 5, 10, 1996]],
        ),
    ]


@pytest.mark.parametrize("strategy", ["columns", "prefiltered"])
@pytest.mark.parametrize(
    ("dtype", "values", "needles"),
    get_tests_from_dtype(pl.Int8(), lambda x: (x % 256) - 128)
    + get_tests_from_dtype(pl.Int32(), lambda x: x % 256)
    + get_tests_from_dtype(pl.Date(), lambda x: date(year=1 + x, month=10, day=5))
    + get_tests_from_dtype(pl.String(), lambda x: str(x))
    + get_tests_from_dtype(pl.String(), lambda x: "i" * x)
    + get_tests_from_dtype(pl.String(), lambda x: f"long_strings_with_the_number_{x}"),
)
def test_equality_filter(
    strategy: ParallelStrategy,
    dtype: pl.DataType,
    values: list[Any],
    needles: list[Any],
) -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", values, dtype),
        ]
    )

    f = io.BytesIO()
    df.write_parquet(f)

    for needle in needles:
        f.seek(0)
        scan = pl.scan_parquet(f, parallel=strategy)
        try:
            assert_frame_equal(
                df.filter(pl.col.a == pl.lit(needle, dtype)),
                scan.filter(pl.col.a == pl.lit(needle, dtype)).collect(),
            )
        except:
            import sys

            print(f"needle: {needle}", file=sys.stderr)
            raise

    pl.read_parquet(f)


def test_nested_string_slice_utf8_21202() -> None:
    s = pl.Series(
        "a",
        [
            ["A" * 128],
            ["A"],
        ],
        pl.List(pl.String()),
    )

    f = io.BytesIO()
    s.to_frame().write_parquet(f)

    f.seek(0)
    assert_series_equal(
        pl.scan_parquet(f).slice(1, 1).collect().to_series(),
        s.slice(1, 1),
    )


def test_filter_true_predicate_21204() -> None:
    f = io.BytesIO()

    df = pl.DataFrame({"a": [1]})
    df.write_parquet(f)
    f.seek(0)
    lf = pl.scan_parquet(f).filter(pl.lit(True))
    assert_frame_equal(lf.collect(), df)


def test_nested_deprecated_int96_timestamps_21332() -> None:
    f = io.BytesIO()

    df = pl.DataFrame({"a": [{"t": datetime(2025, 1, 1)}]})

    pq.write_table(
        df.to_arrow(),
        f,
        use_deprecated_int96_timestamps=True,
    )

    f.seek(0)
    assert_frame_equal(
        pl.read_parquet(f),
        df,
    )


def test_final_masked_optional_iteration_21378() -> None:
    # fmt: off
    values = [
        1, 0, 0, 0, 0, 1, 1, 1,
        1, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 0, 1, 0, 0,
        1, 1, 0, 0, 0, 1, 1, 1,
        0, 1, 0, 0, 1, 1, 1, 1,
        0, 1, 1, 1, 0, 1, 0, 1,
        0, 1, 1, 0, 1, 0, 1, 1,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 1, 1, 1, 0, 0, 1, 1,
        0, 0, 1, 1, 0, 0, 0, 1,
        1, 1, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 0, 1, 1, 0, 0,
        0, 1, 1, 0, 0, 1, 0, 0,
        1, 1, 1, 1, 0, 0, 1, 0,
        0, 1, 1, 0, 0, 1, 1, 1,
        1, 1, 1, 0, 1, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1,
        0, 0, 0, 1, 1, 0, 0, 0,
        1, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 1, 0, 0, 1, 1,
        0, 1, 0, 0, 0, 1, 1, 1,
        1, 0, 1, 0, 1, 0, 1, 1,
        1, 0, 1, 0, 0, 1, 0, 1,
        0, 1, 1, 1, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 0,
        1, 0, 1, 0, 0, 1, 0, 0,
        0, 1, 1, 1, 0, 0, 1, 1,
        1, 0, 1, 1, 0, 0, 0, 1,
        0, 0, 1, 1, 0, 1, 0, 1,
        0, 1, 1, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 1, 1,
        1, 0, 1, 1, 1, 1, 1, 0,
        1, 0, 1, 0, 0, 0, 1, 1,
        0, 0, 0, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 1, 0, 1, 1,
        1, 0, 0, 1, 0, 1, 1, 0,
        0, 1, 0, 1, 1, 0, 1, 0,
        0, 0, 0, 1, 1, 1, 0, 0,
        0, 1, 0, 1, 1, 0, 1, 1,
        1, 1, 0, 1, 0, 1, 0, 1,
        1, 1, 0, 1, 0, 0, 1, 0,
        1, 1, 0, 1, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 1, 1, 1, 1,
        1, 0, 1, 1, 1, 0, 1, 1,
        1, 1, 0, 0, 0, 0, 1, 1,
    ]

    df = pl.DataFrame(
        [
            pl.Series("x", [None if x == 1 else 0.0 for x in values], pl.Float32),
            pl.Series(
                "f",
                [False] * 164 +
                [True] * 10 +
                [False] * 264 +
                [True] * 10,
                pl.Boolean(),
            ),
        ]
    )

    f = io.BytesIO()
    df.write_parquet(f)
    f.seek(0)

    output = pl.scan_parquet(f, parallel="prefiltered").filter(pl.col.f).collect()
    assert_frame_equal(df.filter(pl.col.f), output)


def test_predicate_empty_is_in_21450() -> None:
    f = io.BytesIO()
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.write_parquet(f)

    f.seek(0)
    assert_frame_equal(
        df.clear(),
        pl.scan_parquet(f).filter(pl.col("a").is_in([])).collect(),
    )


@pytest.mark.write_disk
def test_scan_parquet_filter_statistics_load_missing_column_21391(
    tmp_path: Path,
) -> None:
    root = tmp_path
    dfs = [pl.DataFrame({"x": 1, "y": 1}), pl.DataFrame({"x": 2})]

    for i, df in enumerate(dfs):
        df.write_parquet(root / f"{i}.parquet")

    assert_frame_equal(
        (
            pl.scan_parquet(root, missing_columns="insert")
            .filter(pl.col("y") == 1)
            .collect()
        ),
        pl.DataFrame({"x": 1, "y": 1}),
    )


@pytest.mark.parametrize(
    "ty",
    [
        (lambda i: i, pl.Int8, True),
        (lambda i: datetime(year=2025, month=9, day=i), pl.Datetime, True),
        (lambda i: float(i), pl.Float32, True),
        (lambda i: str(i), pl.String, True),
        (lambda i: str(i) + "make it a bit longer", pl.String, True),
        (lambda i: [i, i + 7] * (i % 3), pl.List(pl.Int32), True),
        (lambda i: {"x": i}, pl.Struct({"x": pl.Int32}), True),
        (lambda i: [i, i + 3, i + 7], pl.Array(pl.Int32, 3), False),
    ],
)
def test_filter_nulls_21538(ty: tuple[Callable[[int], Any], pl.DataType, bool]) -> None:
    i_to_value, dtype, do_no_dicts = ty

    patterns: list[list[int | None]] = [
        [None, None, None, None, None],
        [1, None, None, 2, None],
        [None, 1, 2, 3, 4],
        [1, 2, 3, 4, None],
        [None, 1, 2, 3, None],
        [None, 1, None, 3, None],
        [1, 2, 3, 4, 5],
    ]

    df = pl.DataFrame(
        [
            pl.Series(
                f"p{i}", [None if v is None else i_to_value(v) for v in pattern], dtype
            )
            for i, pattern in enumerate(patterns)
        ]
    )

    fs = []

    dicts_f = io.BytesIO()
    df.write_parquet(dicts_f)
    fs += [dicts_f]

    if do_no_dicts:
        no_dicts_f = io.BytesIO()
        pq.write_table(df.to_arrow(), no_dicts_f, use_dictionary=False)
        fs += [no_dicts_f]

    for f in fs:
        for i in range(len(patterns)):
            f.seek(0)
            assert_frame_equal(
                pl.scan_parquet(f).filter(pl.col(f"p{i}").is_null()).collect(),
                df.filter(pl.col(f"p{i}").is_null()),
            )

            f.seek(0)
            assert_frame_equal(
                pl.scan_parquet(f).filter(pl.col(f"p{i}").is_not_null()).collect(),
                df.filter(pl.col(f"p{i}").is_not_null()),
            )


def test_unspecialized_decoding_prefiltering() -> None:
    df = pl.DataFrame(
        {
            "a": [None, None, None, "abc"],
            "b": [False, True, False, True],
        }
    )

    cols = df.columns

    encodings = dict.fromkeys(cols, "DELTA_LENGTH_BYTE_ARRAY")
    encodings["b"] = "PLAIN"

    f = io.BytesIO()
    pq.write_table(
        df.to_arrow(),
        f,
        use_dictionary=False,
        column_encoding=encodings,
    )

    f.seek(0)
    expr = pl.col("b")
    result = (
        pl.scan_parquet(f, parallel="prefiltered")
        .filter(expr)
        .collect(engine="streaming")
    )
    assert_frame_equal(result, df.filter(expr))


@pytest.mark.parametrize("parallel", ["columns", "row_groups"])
def test_filtering_on_other_parallel_modes_with_statistics(
    parallel: ParallelStrategy,
) -> None:
    f = io.BytesIO()

    pl.DataFrame(
        {
            "a": [1, 4, 9, 2, 4, 8, 3, 4, 7],
        }
    ).write_parquet(f, row_group_size=3)

    f.seek(0)
    assert_series_equal(
        pl.scan_parquet(f, parallel=parallel)
        .filter(pl.col.a == 4)
        .collect()
        .to_series(),
        pl.Series("a", [4, 4, 4]),
    )


def test_filter_on_logical_dtype_22252() -> None:
    f = io.BytesIO()
    pl.Series("a", [datetime(1996, 10, 5)]).to_frame().write_parquet(f)
    f.seek(0)
    pl.scan_parquet(f).filter(pl.col.a.dt.weekday() == 6).collect()


def test_filter_nan_22289() -> None:
    f = io.BytesIO()
    pl.DataFrame(
        {"a": [1, 2, float("nan")], "b": [float("nan"), 5, 6]}, strict=False
    ).write_parquet(f)

    f.seek(0)
    lf = pl.scan_parquet(f)

    assert_frame_equal(
        lf.collect().filter(pl.col.a.is_not_nan()),
        lf.filter(pl.col.a.is_not_nan()).collect(),
    )

    assert_frame_equal(
        lf.collect().filter(pl.col.a.is_nan()),
        lf.filter(pl.col.a.is_nan()).collect(),
    )


def test_encode_utf8_check_22467() -> None:
    f = io.BytesIO()
    values = ["😀" * 129, "😀"]

    pq.write_table(pl.Series(values).to_frame().to_arrow(), f, use_dictionary=False)

    f.seek(0)
    pl.scan_parquet(f).slice(1, 1).collect()


def test_reencode_categoricals_22385() -> None:
    tbl = pl.Series("a", ["abc"], pl.Categorical()).to_frame().to_arrow()
    tbl = tbl.cast(
        pa.schema(
            [
                pa.field(
                    "a",
                    pa.dictionary(pa.int32(), pa.large_string()),
                    metadata=tbl.schema[0].metadata,
                ),
            ]
        )
    )

    f = io.BytesIO()
    pq.write_table(tbl, f)

    f.seek(0)
    pl.scan_parquet(f).collect()


def test_parquet_read_timezone_22506() -> None:
    f = io.BytesIO()

    pd.DataFrame(
        {
            "a": [1, 2],
            "b": pd.to_datetime(
                ["2020-01-01T00:00:00+01:00", "2020-01-02T00:00:00+01:00"]
            ),
        }
    ).to_parquet(f)

    assert b'"metadata": {"timezone": "+01:00"}}' in f.getvalue()

    f.seek(0)

    assert_frame_equal(
        pl.read_parquet(f),
        pl.DataFrame(
            {
                "a": [1, 2],
                "b": [
                    datetime(2020, 1, 1, tzinfo=ZoneInfo("Etc/GMT-1")),
                    datetime(2020, 1, 2, tzinfo=ZoneInfo("Etc/GMT-1")),
                ],
            },
            schema={
                "a": pl.Int64,
                "b": pl.Datetime(time_unit="ns", time_zone="Etc/GMT-1"),
            },
        ),
    )


@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize("lazy", [True, False])
def test_read_write_metadata(static: bool, lazy: bool) -> None:
    metadata = {"hello": "world", "something": "else"}
    md: ParquetMetadata = metadata
    if not static:
        md = lambda ctx: metadata  # noqa: E731

    df = pl.DataFrame({"a": [1, 2, 3]})

    f = io.BytesIO()
    if lazy:
        df.lazy().sink_parquet(f, metadata=md)
    else:
        df.write_parquet(f, metadata=md)

    f.seek(0)
    actual = pl.read_parquet_metadata(f)
    assert "ARROW:schema" in actual
    assert metadata == {k: v for k, v in actual.items() if k != "ARROW:schema"}


@pytest.mark.write_disk
def test_metadata_callback_info(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    num_writes = 0

    def fn_metadata(ctx: ParquetMetadataContext) -> dict[str, str]:
        nonlocal num_writes
        num_writes += 1
        return {}

    df.write_parquet(tmp_path, partition_by="a", metadata=fn_metadata)

    assert num_writes == len(df)


def test_field_overwrites_metadata() -> None:
    f = io.BytesIO()
    lf = pl.LazyFrame(
        {
            "a": [None, 2, 3, 4],
            "b": [[1, 2, 3], [42], [13], [37]],
            "c": [
                {"x": "a", "y": 42},
                {"x": "b", "y": 13},
                {"x": "X", "y": 37},
                {"x": "Y", "y": 15},
            ],
        }
    )
    lf.sink_parquet(
        f,
        field_overwrites={
            "a": ParquetFieldOverwrites(metadata={"flat_from_polars": "yes"}),
            "b": ParquetFieldOverwrites(
                children=ParquetFieldOverwrites(metadata={"listitem": "yes"}),
                metadata={"list": "true"},
            ),
            "c": ParquetFieldOverwrites(
                children=[
                    ParquetFieldOverwrites(name="x", metadata={"md": "yes"}),
                    ParquetFieldOverwrites(name="y", metadata={"md2": "Yes!"}),
                ],
                metadata={"struct": "true"},
            ),
        },
    )

    f.seek(0)
    schema = pq.read_schema(f)
    assert schema[0].metadata[b"flat_from_polars"] == b"yes"
    assert schema[1].metadata[b"list"] == b"true"
    assert schema[1].type.value_field.metadata[b"listitem"] == b"yes"
    assert schema[2].metadata[b"struct"] == b"true"
    assert schema[2].type.fields[0].metadata[b"md"] == b"yes"
    assert schema[2].type.fields[1].metadata[b"md2"] == b"Yes!"


def multiple_test_sorting_columns() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [1, 2, 3, 1, 2, 3],
        }
    )

    f = io.BytesIO()
    pq.write_table(
        df.to_arrow(),
        f,
        sorting_columns=[pq.SortingColumn(0), pq.SortingColumn(1)],
    )

    f.seek(0)
    roundtrip = pl.read_parquet(f)
    assert roundtrip.get_column("a").is_sorted()
    assert not roundtrip.get_column("b").is_sorted()
    assert_frame_equal(roundtrip.sort("b"), df.sort("b"))


@pytest.mark.write_disk
def test_read_parquet_duplicate_range_start_fetch_23139(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "data.parquet"

    df = pl.DataFrame(
        schema={
            "a": pl.Boolean,
            "b": pl.Boolean,
        }
    )

    df.write_parquet(path, use_pyarrow=True)

    assert_frame_equal(pl.read_parquet(path), df)


@pytest.mark.parametrize(
    ("value", "scan_dtype", "filter_expr"),
    [
        (pl.lit(1, dtype=pl.Int8), pl.Int16, pl.col("x") > 1),
        (pl.lit(1.0, dtype=pl.Float64), pl.Float32, pl.col("x") > 1.0),
        (pl.lit(1.0, dtype=pl.Float32), pl.Float64, pl.col("x") > 1.0),
        (
            pl.lit(
                datetime(2025, 1, 1),
                dtype=pl.Datetime(time_unit="ns", time_zone="Europe/Amsterdam"),
            ),
            pl.Datetime(time_unit="ms", time_zone="Australia/Sydney"),
            pl.col("x")
            != pl.lit(
                datetime(2025, 1, 1, 10),
                dtype=pl.Datetime(time_unit="ms", time_zone="Australia/Sydney"),
            ),
        ),
        # Note: This is not implemented at all
        # (
        #     pl.lit({"a": 1}, dtype=pl.Struct({"a": pl.Int8})),
        #     pl.Struct({"a": pl.Int64}),
        #     pl.col("x").struct.field("a") > 1,
        # ),
    ],
)
def test_scan_parquet_filter_with_cast(
    value: Any,
    scan_dtype: pl.DataType,
    filter_expr: pl.Expr,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    f = io.BytesIO()

    df = pl.select(x=value)

    df.write_parquet(f)

    q = pl.scan_parquet(
        f,
        schema={"x": scan_dtype},
        cast_options=pl.ScanCastOptions(
            integer_cast="upcast",
            float_cast=["upcast", "downcast"],
            datetime_cast=["convert-timezone", "nanosecond-downcast"],
            missing_struct_fields="insert",
        ),
    ).filter(filter_expr)

    monkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect()
    assert "reading 0 / 1 row groups" in capfd.readouterr().err

    assert_frame_equal(out, pl.DataFrame(schema={"x": scan_dtype}))


@pytest.mark.parametrize(
    ("value", "scan_dtype", "filter_expr"),
    [
        (pl.lit(1, dtype=pl.Int8), pl.Int16, pl.col("x") == 1),
        (pl.lit(1.0, dtype=pl.Float64), pl.Float32, pl.col("x") == 1.0),
        (pl.lit(1.0, dtype=pl.Float32), pl.Float64, pl.col("x") == 1.0),
        (
            pl.lit(
                datetime(2025, 1, 1),
                dtype=pl.Datetime(time_unit="ns", time_zone="Europe/Amsterdam"),
            ),
            pl.Datetime(time_unit="ms", time_zone="Australia/Sydney"),
            pl.col("x")
            == pl.lit(
                datetime(2025, 1, 1, 10),
                dtype=pl.Datetime(time_unit="ms", time_zone="Australia/Sydney"),
            ),
        ),
        (
            pl.lit({"a": 1}, dtype=pl.Struct({"a": pl.Int8})),
            pl.Struct({"a": pl.Int64}),
            pl.col("x").struct.field("a") == 1,
        ),
    ],
)
def test_scan_parquet_filter_with_cast_inclusions(
    value: Any,
    scan_dtype: pl.DataType,
    filter_expr: pl.Expr,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    f = io.BytesIO()

    df = pl.select(x=value)

    df.write_parquet(f)

    q = pl.scan_parquet(
        f,
        schema={"x": scan_dtype},
        cast_options=pl.ScanCastOptions(
            integer_cast="upcast",
            float_cast=["upcast", "downcast"],
            datetime_cast=["convert-timezone", "nanosecond-downcast"],
            missing_struct_fields="insert",
        ),
    ).filter(filter_expr)

    monkeypatch.setenv("POLARS_VERBOSE", "1")
    capfd.readouterr()
    out = q.collect()
    assert "reading 1 / 1 row groups" in capfd.readouterr().err

    assert_frame_equal(out, pl.select(x=value).select(pl.first().cast(scan_dtype)))
