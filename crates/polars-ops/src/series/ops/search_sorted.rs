use polars_core::chunked_array::ops::search_sorted::{SearchSortedSide, binary_search_ca};
use polars_core::prelude::row_encode::_get_rows_encoded_ca;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use search_sorted::binary_search_ca_with_overrides;

use crate::series::SeriesMethods;

pub fn search_sorted(
    s: &Series,
    search_values: &Series,
    side: SearchSortedSide,
    descending: bool,
) -> PolarsResult<IdxCa> {
    let original_dtype = s.dtype();

    if s.dtype().is_categorical() {
        // See https://github.com/pola-rs/polars/issues/20171
        polars_bail!(InvalidOperation: "'search_sorted' is not supported on dtype: {}", s.dtype())
    }

    let s = s.to_physical_repr();
    let phys_dtype = s.dtype();

    match phys_dtype {
        DataType::String => {
            let ca = s.str().unwrap();
            let ca = ca.as_binary();
            let search_values = search_values.str()?;
            let search_values = search_values.as_binary();
            let idx = binary_search_ca(&ca, search_values.iter(), side, descending);
            Ok(IdxCa::new_vec(s.name().clone(), idx))
        },
        DataType::Boolean => {
            let ca = s.bool().unwrap();
            let search_values = search_values.bool()?;

            let mut none_pos = None;
            let mut false_pos = None;
            let mut true_pos = None;
            let idxs = search_values
                .iter()
                .map(|v| {
                    let cache = match v {
                        None => &mut none_pos,
                        Some(false) => &mut false_pos,
                        Some(true) => &mut true_pos,
                    };
                    *cache.get_or_insert_with(|| {
                        binary_search_ca(ca, [v].into_iter(), side, descending)[0]
                    })
                })
                .collect();
            Ok(IdxCa::new_vec(s.name().clone(), idxs))
        },
        DataType::Binary => {
            let ca = s.binary().unwrap();

            let idx = match search_values.dtype() {
                DataType::BinaryOffset => {
                    let search_values = search_values.binary_offset().unwrap();
                    binary_search_ca(ca, search_values.iter(), side, descending)
                },
                DataType::Binary => {
                    let search_values = search_values.binary().unwrap();
                    binary_search_ca(ca, search_values.iter(), side, descending)
                },
                _ => unreachable!(),
            };

            Ok(IdxCa::new_vec(s.name().clone(), idx))
        },
        dt if dt.is_primitive_numeric() => {
            let search_values = search_values.to_physical_repr();

            let idx = with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let search_values: &ChunkedArray<$T> = search_values.as_ref().as_ref().as_ref();
                binary_search_ca(ca, search_values.iter(), side, descending)
            });
            Ok(IdxCa::new_vec(s.name().clone(), idx))
        },
        dt if dt.is_nested() => {
            // Unfortunately in some combinations of ascending sort and
            // nulls_last, the row encoding does not preserve sort order. So for
            // now we don't support it.
            polars_ensure!(
                !descending,
                InvalidOperation: "descending sort is not supported in nested dtypes"
            );

            // We want to preserve the sort order after row encoding, so we need
            // to pick a nulls_last value that will ensure that. Nesting means
            // the naive algorithm of checking first item isn't sufficient.
            let maybe_nulls_last = if s.len() < 2 {
                Some(true) // doesn't matter, really
            } else if s.first().is_null() {
                Some(false)
            } else if s.last().is_null() {
                Some(true)
            } else {
                // We'll just guess the likely value, and we'll validate (expensively) later.
                None
            };

            // This is O(N), whereas typically search_sorted would be O(logN).
            // Ideally the implementation would only row-encode values that are
            // actively being looked up, instead of all of them...
            let mut ca = _get_rows_encoded_ca(
                "".into(),
                &[s.as_ref().clone().into_column()],
                &[descending],
                &[maybe_nulls_last.unwrap_or(false)],
            )?;
            let nulls_last = match maybe_nulls_last {
                Some(nulls_last) => nulls_last,
                None => {
                    // Validate nulls_last value:
                    let mut nulls_last = false;
                    let sort_options = SortOptions::new()
                        .with_order_descending(descending)
                        .with_nulls_last(nulls_last);
                    if !ca
                        .arg_sort(sort_options)
                        .into_series()
                        .is_sorted(sort_options)?
                    {
                        nulls_last = true;
                        // Reencode the series. This is a mutating side-effect:
                        ca = _get_rows_encoded_ca(
                            "".into(),
                            &[s.as_ref().clone().into_column()],
                            &[descending],
                            &[nulls_last],
                        )?;
                    }
                    nulls_last
                },
            };

            let search_values = _get_rows_encoded_ca(
                "".into(),
                &[search_values.clone().into_column()],
                &[descending],
                &[nulls_last],
            )?;

            let idx = binary_search_ca_with_overrides(
                &ca,
                search_values.iter(),
                side,
                descending,
                nulls_last,
            );
            Ok(IdxCa::new_vec(s.name().clone(), idx))
        },
        _ => polars_bail!(opq = search_sorted, original_dtype),
    }
}
