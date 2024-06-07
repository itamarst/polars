from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def is_sorted_any(s: pl.Series) -> bool:
    return s.flags["SORTED_ASC"] or s.flags["SORTED_DESC"]


def is_not_sorted(s: pl.Series) -> bool:
    return not is_sorted_any(s)


def test_sort_dates_multiples() -> None:
    df = pl.DataFrame(
        [
            pl.Series(
                "date",
                [
                    "2021-01-01 00:00:00",
                    "2021-01-01 00:00:00",
                    "2021-01-02 00:00:00",
                    "2021-01-02 00:00:00",
                    "2021-01-03 00:00:00",
                ],
            ).str.strptime(pl.Datetime, "%Y-%m-%d %T"),
            pl.Series("values", [5, 4, 3, 2, 1]),
        ]
    )

    expected = [4, 5, 2, 3, 1]

    # datetime
    out: pl.DataFrame = df.sort(["date", "values"])
    assert out["values"].to_list() == expected

    # Date
    out = df.with_columns(pl.col("date").cast(pl.Date)).sort(["date", "values"])
    assert out["values"].to_list() == expected


def test_sort_by() -> None:
    df = pl.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 1, 1, 2, 2], "c": [2, 3, 1, 2, 1]}
    )

    by: list[pl.Expr | str]
    for by in [["b", "c"], [pl.col("b"), "c"]]:  # type: ignore[assignment]
        out = df.select(pl.col("a").sort_by(by))
        assert out["a"].to_list() == [3, 1, 2, 5, 4]

    # Columns as positional arguments are also accepted
    out = df.select(pl.col("a").sort_by("b", "c"))
    assert out["a"].to_list() == [3, 1, 2, 5, 4]

    out = df.select(pl.col("a").sort_by(by, descending=False))
    assert out["a"].to_list() == [3, 1, 2, 5, 4]

    out = df.select(pl.col("a").sort_by(by, descending=True))
    assert out["a"].to_list() == [4, 5, 2, 1, 3]

    out = df.select(pl.col("a").sort_by(by, descending=[True, False]))
    assert out["a"].to_list() == [5, 4, 3, 1, 2]

    # by can also be a single column
    out = df.select(pl.col("a").sort_by("b", descending=[False]))
    assert out["a"].to_list() == [1, 2, 3, 4, 5]


def test_expr_sort_by_nulls_last() -> None:
    df = pl.DataFrame({"a": [1, 2, None, None, 5], "b": [None, 1, 1, 2, None]})

    # nulls last
    expected = pl.DataFrame({"a": [1, 2, 5, None, None], "b": [None, 1, None, 1, 2]})
    out = df.select(pl.all().sort_by("a", nulls_last=True))
    assert_frame_equal(out, expected)

    # nulls first (default)
    expected = pl.DataFrame({"a": [None, None, 1, 2, 5], "b": [1, 2, None, 1, None]})
    for out in (
        df.select(pl.all().sort_by("a", nulls_last=False)),
        df.select(pl.all().sort_by("a")),
    ):
        assert_frame_equal(out, expected)


def test_expr_sort_by_multi_nulls_last() -> None:
    df = pl.DataFrame({"x": [None, 1, None, 3], "y": [3, 2, None, 1]})

    res = df.sort("x", "y", nulls_last=[False, True])
    assert res.to_dict(as_series=False) == {
        "x": [None, None, 1, 3],
        "y": [3, None, 2, 1],
    }

    res = df.sort("x", "y", nulls_last=[True, False])
    assert res.to_dict(as_series=False) == {
        "x": [1, 3, None, None],
        "y": [2, 1, None, 3],
    }

    res = df.sort("x", "y", nulls_last=[True, False], descending=True)
    assert res.to_dict(as_series=False) == {
        "x": [3, 1, None, None],
        "y": [1, 2, None, 3],
    }

    res = df.sort("x", "y", nulls_last=[False, True], descending=True)
    assert res.to_dict(as_series=False) == {
        "x": [None, None, 3, 1],
        "y": [3, None, 1, 2],
    }

    res = df.sort("x", "y", nulls_last=[False, True], descending=[True, False])
    assert res.to_dict(as_series=False) == {
        "x": [None, None, 3, 1],
        "y": [3, None, 1, 2],
    }


def test_sort_by_exprs() -> None:
    # make sure that the expression does not overwrite columns in the dataframe
    df = pl.DataFrame({"a": [1, 2, -1, -2]})
    out = df.sort(pl.col("a").abs()).to_series()

    assert out.to_list() == [1, -1, 2, -2]


def test_arg_sort_nulls() -> None:
    a = pl.Series("a", [1.0, 2.0, 3.0, None, None])

    assert a.arg_sort(nulls_last=True).to_list() == [0, 1, 2, 3, 4]
    assert a.arg_sort(nulls_last=False).to_list() == [3, 4, 0, 1, 2]

    res = a.to_frame().sort(by="a", nulls_last=False).to_series().to_list()
    assert res == [None, None, 1.0, 2.0, 3.0]

    res = a.to_frame().sort(by="a", nulls_last=True).to_series().to_list()
    assert res == [1.0, 2.0, 3.0, None, None]


@pytest.mark.parametrize(
    ("nulls_last", "expected"),
    [
        (True, [0, 1, 4, 3, 2]),
        (False, [2, 3, 0, 1, 4]),
        ([True, False], [0, 1, 4, 2, 3]),
        ([False, True], [3, 2, 0, 1, 4]),
    ],
)
def test_expr_arg_sort_nulls_last(
    nulls_last: bool | list[bool], expected: list[int]
) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, None, None, 5],
            "b": [1, 2, None, 1, None],
            "c": [2, 3, 1, 2, 1],
        },
    )
    out = (
        df.select(pl.arg_sort_by("a", "b", nulls_last=nulls_last, maintain_order=True))
        .to_series()
        .to_list()
    )
    assert out == expected


def test_arg_sort_window_functions() -> None:
    df = pl.DataFrame({"Id": [1, 1, 2, 2, 3, 3], "Age": [1, 2, 3, 4, 5, 6]})
    out = df.select(
        pl.col("Age").arg_sort().over("Id").alias("arg_sort"),
        pl.arg_sort_by("Age").over("Id").alias("arg_sort_by"),
    )
    assert (
        out["arg_sort"].to_list() == out["arg_sort_by"].to_list() == [0, 1, 0, 1, 0, 1]
    )


def test_sort_nans_3740() -> None:
    df = pl.DataFrame(
        {
            "key": [1, 2, 3, 4, 5],
            "val": [0.0, None, float("nan"), float("-inf"), float("inf")],
        }
    )
    assert df.sort("val")["key"].to_list() == [2, 4, 1, 5, 3]


def test_sort_by_exps_nulls_last() -> None:
    df = pl.DataFrame({"a": [1, 3, -2, None, 1]}).with_row_index()

    assert df.sort(pl.col("a") ** 2, nulls_last=True).to_dict(as_series=False) == {
        "index": [0, 4, 2, 1, 3],
        "a": [1, 1, -2, 3, None],
    }


def test_sort_aggregation_fast_paths() -> None:
    df = pl.DataFrame(
        {
            "a": [None, 3, 2, 1],
            "b": [3, 2, 1, None],
            "c": [3, None, None, None],
            "e": [None, None, None, 1],
            "f": [1, 2, 5, 1],
        }
    )

    expected = df.select(
        pl.all().max().name.suffix("_max"),
        pl.all().min().name.suffix("_min"),
    )

    assert expected.to_dict(as_series=False) == {
        "a_max": [3],
        "b_max": [3],
        "c_max": [3],
        "e_max": [1],
        "f_max": [5],
        "a_min": [1],
        "b_min": [1],
        "c_min": [3],
        "e_min": [1],
        "f_min": [1],
    }

    for descending in [True, False]:
        for null_last in [True, False]:
            out = df.select(
                pl.all()
                .sort(descending=descending, nulls_last=null_last)
                .max()
                .name.suffix("_max"),
                pl.all()
                .sort(descending=descending, nulls_last=null_last)
                .min()
                .name.suffix("_min"),
            )
            assert_frame_equal(out, expected)


@pytest.mark.parametrize("dtype", [pl.Int8, pl.Int16, pl.Int32, pl.Int64])
def test_sorted_join_and_dtypes(dtype: pl.PolarsDataType) -> None:
    df_a = (
        pl.DataFrame({"a": [-5, -2, 3, 3, 9, 10]})
        .with_row_index()
        .with_columns(pl.col("a").cast(dtype).set_sorted())
    )

    df_b = pl.DataFrame({"a": [-2, -3, 3, 10]}).with_columns(
        pl.col("a").cast(dtype).set_sorted()
    )

    result_inner = df_a.join(df_b, on="a", how="inner")
    assert result_inner.to_dict(as_series=False) == {
        "index": [1, 2, 3, 5],
        "a": [-2, 3, 3, 10],
    }

    result_left = df_a.join(df_b, on="a", how="left", coalesce=True)
    assert result_left.to_dict(as_series=False) == {
        "index": [0, 1, 2, 3, 4, 5],
        "a": [-5, -2, 3, 3, 9, 10],
    }


def test_sorted_flag() -> None:
    s = pl.arange(0, 7, eager=True)
    assert s.flags["SORTED_ASC"]
    assert s.reverse().flags["SORTED_DESC"]
    assert pl.Series([b"a"]).set_sorted().flags["SORTED_ASC"]
    assert (
        pl.Series([date(2020, 1, 1), date(2020, 1, 2)])
        .set_sorted()
        .cast(pl.Datetime)
        .flags["SORTED_ASC"]
    )

    # empty
    q = pl.LazyFrame(
        schema={
            "store_id": pl.UInt16,
            "item_id": pl.UInt32,
            "timestamp": pl.Datetime,
        }
    ).sort("timestamp")

    assert q.collect()["timestamp"].flags["SORTED_ASC"]

    # ensure we don't panic for these types
    # struct
    pl.Series([{"a": 1}]).set_sorted(descending=True)
    # list
    pl.Series([[{"a": 1}]]).set_sorted(descending=True)
    # object
    pl.Series([{"a": 1}], dtype=pl.Object).set_sorted(descending=True)


def test_sorted_fast_paths() -> None:
    s = pl.Series([1, 2, 3]).sort()
    rev = s.sort(descending=True)

    assert rev.to_list() == [3, 2, 1]
    assert s.sort().to_list() == [1, 2, 3]

    s = pl.Series([None, 1, 2, 3]).sort()
    rev = s.sort(descending=True)
    assert rev.to_list() == [None, 3, 2, 1]
    assert rev.sort(descending=True).to_list() == [None, 3, 2, 1]
    assert rev.sort().to_list() == [None, 1, 2, 3]


def test_arg_sort_rank_nans() -> None:
    assert (
        pl.DataFrame(
            {
                "val": [1.0, float("nan")],
            }
        )
        .with_columns(
            pl.col("val").rank().alias("rank"),
            pl.col("val").arg_sort().alias("arg_sort"),
        )
        .select(["rank", "arg_sort"])
    ).to_dict(as_series=False) == {"rank": [1.0, 2.0], "arg_sort": [0, 1]}


def test_sorted_flag_unset_by_arithmetic_4937() -> None:
    df = pl.DataFrame(
        {
            "ts": [1, 1, 1, 0, 1],
            "price": [3.3, 3.0, 3.5, 3.6, 3.7],
            "mask": [1, 1, 1, 1, 0],
        }
    )

    assert df.sort("price").group_by("ts").agg(
        [
            (pl.col("price") * pl.col("mask")).max().alias("pmax"),
            (pl.col("price") * pl.col("mask")).min().alias("pmin"),
        ]
    ).sort("ts").to_dict(as_series=False) == {
        "ts": [0, 1],
        "pmax": [3.6, 3.5],
        "pmin": [3.6, 0.0],
    }


def test_unset_sorted_flag_after_extend() -> None:
    df1 = pl.DataFrame({"Add": [37, 41], "Batch": [48, 49]}).sort("Add")
    df2 = pl.DataFrame({"Add": [37], "Batch": [67]}).sort("Add")

    df1.extend(df2)
    assert not df1["Add"].flags["SORTED_ASC"]
    df = df1.group_by("Add").agg([pl.col("Batch").min()]).sort("Add")
    assert df["Add"].flags["SORTED_ASC"]
    assert df.to_dict(as_series=False) == {"Add": [37, 41], "Batch": [48, 49]}


def test_set_sorted_schema() -> None:
    assert (
        pl.DataFrame({"A": [0, 1]}).lazy().with_columns(pl.col("A").set_sorted()).schema
    ) == {"A": pl.Int64}


def test_sort_slice_fast_path_5245() -> None:
    df = pl.DataFrame(
        {
            "foo": ["f", "c", "b", "a"],
            "bar": [1, 2, 3, 4],
        }
    ).lazy()

    assert df.sort("foo").limit(1).select("foo").collect().to_dict(as_series=False) == {
        "foo": ["a"]
    }


def test_explicit_list_agg_sort_in_group_by() -> None:
    df = pl.DataFrame({"A": ["a", "a", "a", "b", "b", "a"], "B": [1, 2, 3, 4, 5, 6]})

    # this was col().implode().sort() before we changed the logic
    result = df.group_by("A").agg(pl.col("B").sort(descending=True)).sort("A")
    expected = df.group_by("A").agg(pl.col("B").sort(descending=True)).sort("A")
    assert_frame_equal(result, expected)


def test_sorted_join_query_5406() -> None:
    df = (
        pl.DataFrame(
            {
                "Datetime": [
                    "2022-11-02 08:00:00",
                    "2022-11-02 08:00:00",
                    "2022-11-02 08:01:00",
                    "2022-11-02 07:59:00",
                    "2022-11-02 08:02:00",
                    "2022-11-02 08:02:00",
                ],
                "Group": ["A", "A", "A", "B", "B", "B"],
                "Value": [1, 2, 1, 1, 2, 1],
            }
        )
        .with_columns(pl.col("Datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
        .with_row_index("RowId")
    )

    df1 = df.sort(by=["Datetime", "RowId"])

    filter1 = (
        df1.group_by(["Datetime", "Group"])
        .agg([pl.all().sort_by("Value", descending=True).first()])
        .sort(["Datetime", "RowId"])
    )

    out = df1.join(filter1, on="RowId", how="left").select(
        pl.exclude(["Datetime_right", "Group_right"])
    )
    assert out["Value_right"].to_list() == [1, None, 2, 1, 2, None]


def test_sort_by_in_over_5499() -> None:
    df = pl.DataFrame(
        {
            "group": [1, 1, 1, 2, 2, 2],
            "idx": pl.arange(0, 6, eager=True),
            "a": [1, 3, 2, 3, 1, 2],
        }
    )
    assert df.select(
        pl.col("idx").sort_by("a").over("group").alias("sorted_1"),
        pl.col("idx").shift(1).sort_by("a").over("group").alias("sorted_2"),
    ).to_dict(as_series=False) == {
        "sorted_1": [0, 2, 1, 4, 5, 3],
        "sorted_2": [None, 1, 0, 3, 4, None],
    }


def test_merge_sorted() -> None:
    df_a = (
        pl.datetime_range(
            datetime(2022, 1, 1), datetime(2022, 12, 1), "1mo", eager=True
        )
        .to_frame("range")
        .with_row_index()
    )

    df_b = (
        pl.datetime_range(
            datetime(2022, 1, 1), datetime(2022, 12, 1), "2mo", eager=True
        )
        .to_frame("range")
        .with_row_index()
        .with_columns(pl.col("index") * 10)
    )
    out = df_a.merge_sorted(df_b, key="range")
    assert out["range"].is_sorted()
    assert out.to_dict(as_series=False) == {
        "index": [0, 0, 1, 2, 10, 3, 4, 20, 5, 6, 30, 7, 8, 40, 9, 10, 50, 11],
        "range": [
            datetime(2022, 1, 1, 0, 0),
            datetime(2022, 1, 1, 0, 0),
            datetime(2022, 2, 1, 0, 0),
            datetime(2022, 3, 1, 0, 0),
            datetime(2022, 3, 1, 0, 0),
            datetime(2022, 4, 1, 0, 0),
            datetime(2022, 5, 1, 0, 0),
            datetime(2022, 5, 1, 0, 0),
            datetime(2022, 6, 1, 0, 0),
            datetime(2022, 7, 1, 0, 0),
            datetime(2022, 7, 1, 0, 0),
            datetime(2022, 8, 1, 0, 0),
            datetime(2022, 9, 1, 0, 0),
            datetime(2022, 9, 1, 0, 0),
            datetime(2022, 10, 1, 0, 0),
            datetime(2022, 11, 1, 0, 0),
            datetime(2022, 11, 1, 0, 0),
            datetime(2022, 12, 1, 0, 0),
        ],
    }


def test_merge_sorted_one_empty() -> None:
    df1 = pl.DataFrame({"key": [1, 2, 3], "a": [1, 2, 3]})
    df2 = pl.DataFrame([], schema=df1.schema)
    out = df1.merge_sorted(df2, key="a")
    assert_frame_equal(out, df1)
    out = df2.merge_sorted(df1, key="a")
    assert_frame_equal(out, df1)


def test_sort_args() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, None],
            "b": [6.0, 5.0, 4.0],
            "c": ["a", "c", "b"],
        }
    )
    expected = pl.DataFrame(
        {
            "a": [None, 1, 2],
            "b": [4.0, 6.0, 5.0],
            "c": ["b", "a", "c"],
        }
    )

    # Single column name
    result = df.sort("a")
    assert_frame_equal(result, expected)

    # Column names as list
    result = df.sort(["a", "b"])
    assert_frame_equal(result, expected)

    # Column names as positional arguments
    result = df.sort("a", "b")
    assert_frame_equal(result, expected)

    # nulls_last
    result = df.sort("a", nulls_last=True)
    assert_frame_equal(result, df)


def test_sort_type_coercion_6892() -> None:
    df = pl.DataFrame({"a": [2, 1], "b": [2, 3]})
    assert df.lazy().sort(pl.col("a") // 2).collect().to_dict(as_series=False) == {
        "a": [1, 2],
        "b": [3, 2],
    }


@pytest.mark.slow()
def test_sort_row_fmt(str_ints_df: pl.DataFrame) -> None:
    # we sort nulls_last as this will always dispatch
    # to row_fmt and is the default in pandas

    df = str_ints_df
    df_pd = df.to_pandas()

    for descending in [True, False]:
        assert_frame_equal(
            df.sort(["strs", "vals"], nulls_last=True, descending=descending),
            pl.from_pandas(
                df_pd.sort_values(["strs", "vals"], ascending=not descending)
            ),
        )


def test_sort_by_logical() -> None:
    test = pl.DataFrame(
        {
            "start": [date(2020, 5, 6), date(2020, 5, 13), date(2020, 5, 10)],
            "end": [date(2020, 12, 31), date(2020, 12, 31), date(2021, 1, 1)],
            "num": [0, 1, 2],
        }
    )
    assert test.select([pl.col("num").sort_by(["start", "end"]).alias("n1")])[
        "n1"
    ].to_list() == [0, 2, 1]
    df = pl.DataFrame(
        {
            "dt1": [date(2022, 2, 1), date(2022, 3, 1), date(2022, 4, 1)],
            "dt2": [date(2022, 2, 2), date(2022, 3, 2), date(2022, 4, 2)],
            "name": ["a", "b", "a"],
            "num": [3, 4, 1],
        }
    )
    assert df.group_by("name").agg([pl.col("num").sort_by(["dt1", "dt2"])]).sort(
        "name"
    ).to_dict(as_series=False) == {"name": ["a", "b"], "num": [[3, 1], [4]]}


def test_limit_larger_than_sort() -> None:
    assert pl.LazyFrame({"a": [1]}).sort("a").limit(30).collect().to_dict(
        as_series=False
    ) == {"a": [1]}


def test_sort_by_struct() -> None:
    df = pl.Series([{"a": 300}, {"a": 20}, {"a": 55}]).to_frame("st").with_row_index()
    assert df.sort("st").to_dict(as_series=False) == {
        "index": [1, 2, 0],
        "st": [{"a": 20}, {"a": 55}, {"a": 300}],
    }


def test_sort_descending() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.sort(["a", "b"], descending=True)
    expected = pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4]})
    assert_frame_equal(result, expected)
    result = df.sort(["a", "b"], descending=[True, True])
    assert_frame_equal(result, expected)
    with pytest.raises(
        ValueError,
        match=r"the length of `descending` \(1\) does not match the length of `by` \(2\)",
    ):
        df.sort(["a", "b"], descending=[True])


def test_sort_by_descending() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.select(pl.col("a").sort_by(["a", "b"], descending=True))
    expected = pl.DataFrame({"a": [3, 2, 1]})
    assert_frame_equal(result, expected)
    result = df.select(pl.col("a").sort_by(["a", "b"], descending=[True, True]))
    assert_frame_equal(result, expected)
    with pytest.raises(
        ValueError,
        match=r"the length of `descending` \(1\) does not match the length of `by` \(2\)",
    ):
        df.select(pl.col("a").sort_by(["a", "b"], descending=[True]))


def test_arg_sort_by_descending() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.select(pl.arg_sort_by(["a", "b"], descending=True))
    expected = pl.DataFrame({"a": [2, 1, 0]}).select(pl.col("a").cast(pl.UInt32))
    assert_frame_equal(result, expected)
    result = df.select(pl.arg_sort_by(["a", "b"], descending=[True, True]))
    assert_frame_equal(result, expected)
    with pytest.raises(
        ValueError,
        match=r"the length of `descending` \(1\) does not match the length of `exprs` \(2\)",
    ):
        df.select(pl.arg_sort_by(["a", "b"], descending=[True]))


def test_arg_sort_struct() -> None:
    df = pl.DataFrame(
        {
            "a": [100, 300, 100, 200, 200, 100, 300, 200, 400, 400],
            "b": [5, 5, 6, 7, 8, 1, 1, 2, 2, 3],
        }
    )
    assert df.select(pl.struct("a", "b").arg_sort()).to_series().to_list() == [
        5,
        0,
        2,
        7,
        3,
        4,
        6,
        1,
        8,
        9,
    ]


def test_sort_top_k_fast_path() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, None],
            "b": [6.0, 5.0, 4.0],
            "c": ["a", "c", "b"],
        }
    )
    # this triggers fast path as head is equal to n-rows
    assert df.lazy().sort("b").head(3).collect().to_dict(as_series=False) == {
        "a": [None, 2, 1],
        "b": [4.0, 5.0, 6.0],
        "c": ["b", "c", "a"],
    }


def test_sorted_flag_partition_by() -> None:
    assert (
        pl.DataFrame({"one": [1, 2, 3], "two": ["a", "a", "b"]})
        .set_sorted("one")
        .partition_by("two", maintain_order=True)[0]["one"]
        .flags["SORTED_ASC"]
    )


@pytest.mark.parametrize("value", [1, "a", True])
def test_sorted_flag_singletons(value: Any) -> None:
    assert pl.DataFrame({"x": [value]})["x"].flags["SORTED_ASC"] is True


def test_sorted_flag_null() -> None:
    assert pl.DataFrame({"x": [None] * 2})["x"].flags["SORTED_ASC"] is False


def test_sorted_update_flags_10327() -> None:
    assert pl.concat(
        [
            pl.Series("a", [1], dtype=pl.Int64).to_frame(),
            pl.Series("a", [], dtype=pl.Int64).to_frame(),
            pl.Series("a", [2], dtype=pl.Int64).to_frame(),
            pl.Series("a", [], dtype=pl.Int64).to_frame(),
        ]
    )["a"].to_list() == [1, 2]


def test_sort_by_11653() -> None:
    df = pl.DataFrame(
        {
            "id": [0, 0, 0, 0, 0, 1],
            "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "other": [0.8, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )

    assert df.group_by("id").agg(
        (pl.col("weights") / pl.col("weights").sum())
        .sort_by("other")
        .sum()
        .alias("sort_by"),
    ).sort("id").to_dict(as_series=False) == {"id": [0, 1], "sort_by": [1.0, 1.0]}


def test_sort_with_null_12139() -> None:
    df = pl.DataFrame(
        {
            "bool": [True, False, None, True, False],
            "float": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    assert df.sort("bool", descending=False, nulls_last=False).to_dict(
        as_series=False
    ) == {
        "bool": [None, False, False, True, True],
        "float": [3.0, 2.0, 5.0, 1.0, 4.0],
    }

    assert df.sort("bool", descending=False, nulls_last=True).to_dict(
        as_series=False
    ) == {
        "bool": [False, False, True, True, None],
        "float": [2.0, 5.0, 1.0, 4.0, 3.0],
    }

    assert df.sort("bool", descending=True, nulls_last=True).to_dict(
        as_series=False
    ) == {
        "bool": [True, True, False, False, None],
        "float": [1.0, 4.0, 2.0, 5.0, 3.0],
    }

    assert df.sort("bool", descending=True, nulls_last=False).to_dict(
        as_series=False
    ) == {
        "bool": [None, True, True, False, False],
        "float": [3.0, 1.0, 4.0, 2.0, 5.0],
    }


def test_sort_with_null_12272() -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 1.0, 1.0],
            "b": [2.0, -1.0, None],
        }
    )
    out = df.select((pl.col("a") * pl.col("b")).alias("product"))

    assert out.sort("product").to_dict(as_series=False) == {
        "product": [None, -1.0, 2.0]
    }


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        ([1, None, 3], [1, 3, None]),
        (
            [date(2024, 1, 1), None, date(2024, 1, 3)],
            [date(2024, 1, 1), date(2024, 1, 3), None],
        ),
        (["a", None, "c"], ["a", "c", None]),
    ],
)
def test_sort_series_nulls_last(input: list[Any], expected: list[Any]) -> None:
    assert pl.Series(input).sort(nulls_last=True).to_list() == expected


def test_sorted_flag_14552() -> None:
    a = pl.DataFrame({"a": [2, 1, 3]})

    a = pl.concat([a, a], rechunk=False)
    assert not a.join(a, on="a", how="left")["a"].flags["SORTED_ASC"]


def test_sorted_flag_concat_15072() -> None:
    # Both all-null
    a = pl.Series("x", [None, None], dtype=pl.Int8)
    b = pl.Series("x", [None, None], dtype=pl.Int8)
    assert pl.concat((a, b)).flags["SORTED_ASC"]

    # left all-null, right 0 < null_count < len
    a = pl.Series("x", [None, None], dtype=pl.Int8)
    b = pl.Series("x", [1, 2, 1, None], dtype=pl.Int8)

    out = pl.concat((a, b.sort()))
    assert out.to_list() == [None, None, None, 1, 1, 2]
    assert out.flags["SORTED_ASC"]

    out = pl.concat((a, b.sort(descending=True)))
    assert out.to_list() == [None, None, None, 2, 1, 1]
    assert out.flags["SORTED_DESC"]

    out = pl.concat((a, b.sort(nulls_last=True)))
    assert out.to_list() == [None, None, 1, 1, 2, None]
    assert is_not_sorted(out)

    out = pl.concat((a, b.sort(nulls_last=True, descending=True)))
    assert out.to_list() == [None, None, 2, 1, 1, None]
    assert is_not_sorted(out)

    # left 0 < null_count < len, right all-null
    a = pl.Series("x", [1, 2, 1, None], dtype=pl.Int8)
    b = pl.Series("x", [None, None], dtype=pl.Int8)

    out = pl.concat((a.sort(), b))
    assert out.to_list() == [None, 1, 1, 2, None, None]
    assert is_not_sorted(out)

    out = pl.concat((a.sort(descending=True), b))
    assert out.to_list() == [None, 2, 1, 1, None, None]
    assert is_not_sorted(out)

    out = pl.concat((a.sort(nulls_last=True), b))
    assert out.to_list() == [1, 1, 2, None, None, None]
    assert out.flags["SORTED_ASC"]

    out = pl.concat((a.sort(nulls_last=True, descending=True), b))
    assert out.to_list() == [2, 1, 1, None, None, None]
    assert out.flags["SORTED_DESC"]

    # both 0 < null_count < len
    assert pl.concat(
        (
            pl.Series([None, 1]).set_sorted(),
            pl.Series([2]).set_sorted(),
        )
    ).flags["SORTED_ASC"]

    assert is_not_sorted(
        pl.concat(
            (
                pl.Series([None, 1]).set_sorted(),
                pl.Series([2, None]).set_sorted(),
            )
        )
    )

    assert pl.concat(
        (
            pl.Series([None, 2]).set_sorted(descending=True),
            pl.Series([1]).set_sorted(descending=True),
        )
    ).flags["SORTED_DESC"]

    assert is_not_sorted(
        pl.concat(
            (
                pl.Series([None, 2]).set_sorted(descending=True),
                pl.Series([1, None]).set_sorted(descending=True),
            )
        )
    )

    # Concat with empty series
    s = pl.Series([None, 1]).set_sorted()

    out = pl.concat((s.clear(), s))
    assert_series_equal(out, s)
    assert out.flags["SORTED_ASC"]

    out = pl.concat((s, s.clear()))
    assert_series_equal(out, s)
    assert out.flags["SORTED_ASC"]

    s = pl.Series([1, None]).set_sorted()

    out = pl.concat((s.clear(), s))
    assert_series_equal(out, s)
    assert out.flags["SORTED_ASC"]

    out = pl.concat((s, s.clear()))
    assert_series_equal(out, s)
    assert out.flags["SORTED_ASC"]


@pytest.mark.parametrize("unit_descending", [True, False])
def test_sorted_flag_concat_unit(unit_descending: bool) -> None:
    unit = pl.Series([1]).set_sorted(descending=unit_descending)

    a = unit
    b = pl.Series([2, 3]).set_sorted()

    out = pl.concat((a, b))
    assert out.to_list() == [1, 2, 3]
    assert out.flags["SORTED_ASC"]

    out = pl.concat((b, a))
    assert out.to_list() == [2, 3, 1]
    assert is_not_sorted(out)

    a = unit
    b = pl.Series([3, 2]).set_sorted(descending=True)

    out = pl.concat((a, b))
    assert out.to_list() == [1, 3, 2]
    assert is_not_sorted(out)

    out = pl.concat((b, a))
    assert out.to_list() == [3, 2, 1]
    assert out.flags["SORTED_DESC"]

    # unit with nulls first
    unit = pl.Series([None, 1]).set_sorted(descending=unit_descending)

    a = unit
    b = pl.Series([2, 3]).set_sorted()

    out = pl.concat((a, b))
    assert out.to_list() == [None, 1, 2, 3]
    assert out.flags["SORTED_ASC"]

    out = pl.concat((b, a))
    assert out.to_list() == [2, 3, None, 1]
    assert is_not_sorted(out)

    a = unit
    b = pl.Series([3, 2]).set_sorted(descending=True)

    out = pl.concat((a, b))
    assert out.to_list() == [None, 1, 3, 2]
    assert is_not_sorted(out)

    out = pl.concat((b, a))
    assert out.to_list() == [3, 2, None, 1]
    assert is_not_sorted(out)

    # unit with nulls last
    unit = pl.Series([1, None]).set_sorted(descending=unit_descending)

    a = unit
    b = pl.Series([2, 3]).set_sorted()

    out = pl.concat((a, b))
    assert out.to_list() == [1, None, 2, 3]
    assert is_not_sorted(out)

    out = pl.concat((b, a))
    assert out.to_list() == [2, 3, 1, None]
    assert is_not_sorted(out)

    a = unit
    b = pl.Series([3, 2]).set_sorted(descending=True)

    out = pl.concat((a, b))
    assert out.to_list() == [1, None, 3, 2]
    assert is_not_sorted(out)

    out = pl.concat((b, a))
    assert out.to_list() == [3, 2, 1, None]
    assert out.flags["SORTED_DESC"]


@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("nulls_last", [True, False])
def test_sort_descending_nulls_last(descending: bool, nulls_last: bool) -> None:
    df = pl.DataFrame({"x": [1, 3, None, 2, None], "y": [1, 3, 0, 2, 0]})

    null_sentinel = 100 if descending ^ nulls_last else -100
    ref_x = [1, 3, None, 2, None]
    ref_x.sort(key=lambda k: null_sentinel if k is None else k, reverse=descending)
    ref_y = [1, 3, 0, 2, 0]
    ref_y.sort(key=lambda k: null_sentinel if k == 0 else k, reverse=descending)

    assert_frame_equal(
        df.sort("x", descending=descending, nulls_last=nulls_last),
        pl.DataFrame({"x": ref_x, "y": ref_y}),
    )

    assert_frame_equal(
        df.sort(["x", "y"], descending=descending, nulls_last=nulls_last),
        pl.DataFrame({"x": ref_x, "y": ref_y}),
    )


@pytest.mark.release()
def test_sort_nan_1942() -> None:
    # https://github.com/pola-rs/polars/issues/1942
    import time

    start = time.time()
    pl.repeat(float("nan"), 2**13, eager=True).sort()
    end = time.time()

    assert (end - start) < 1.0
