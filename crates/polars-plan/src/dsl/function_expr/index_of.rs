use polars_ops::series::index_of as index_of_op;

use super::*;

/// Given two columns, find the index of a value (the second column) within the
/// first column. Will use binary search if possible, as an optimization.
pub(super) fn index_of(s: &mut [Column]) -> PolarsResult<Option<Column>> {
    let series = if let Column::Scalar(ref sc) = s[0] {
        // We only care about the first value:
        &sc.as_single_value_series()
    } else {
        s[0].as_materialized_series()
    };
    let Some(value) = s[1].as_scalar_column().map(|sc| sc.scalar().value()) else {
        return Ok(None);
    };
    let is_sorted_flag = series.is_sorted_flag();
    let result = match is_sorted_flag {
        // If the Series is sorted, we can use an optimized binary search to
        // find the value.
        IsSorted::Ascending | IsSorted::Descending
            if !value.is_null() &&
            // search_sorted() doesn't support decimals at the moment.
            !series.dtype().is_decimal() =>
        {
            let Ok(value_as_series) = s[1].as_materialized_series().strict_cast(series.dtype())
            else {
                // If we can't cast, means we couldn't find the value.
                return Ok(None);
            };
            search_sorted(
                series,
                &value_as_series,
                SearchSortedSide::Left,
                IsSorted::Descending == is_sorted_flag,
            )?
            .get(0)
            .and_then(|idx| {
                // search_sorted() gives an index even if it's not an exact
                // match! So we want to make sure it actually found the value.
                if series.get(idx as usize).ok()? == *value {
                    Some(idx as usize)
                } else {
                    None
                }
            })
        },
        _ => index_of_op(series, value)?,
    };
    Ok(result.map(|r| Column::new(series.name().clone(), [r as IdxSize])))
}
