use arrow::compute::utils::combine_validities_and_many;
use polars_core::chunked_array::ops::search_sorted::{SearchSortedSide, binary_search_ca};
use polars_core::prelude::row_encode::_get_rows_encoded_ca;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use row_encode::encode_rows_unordered;
use search_sorted::binary_search_ca_with_overrides;

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

    print!("Original series {} needle {}", s, search_values);

    // We need to preserve null count so that the the binary search algorithm
    // knows to skip them in that case where we use row encoding.
    let original_null_count = s.null_count();

    // Figuring out if nulls are last is complicated by cases like row-encoded
    // lists, where the null may be part of a list that has been encoded into
    // bytes. So we detect it using sorting invariants, early on.
    let first = s.first();
    let last = s.last();
    let nulls_last = match (first.is_null(), last.is_null()) {
        (true, _) => false,
        (_, true) => true,
        _ => {
            // TODO what about nan
            if descending {
                first.value().tot_le(last.value())
            } else  {
                last.value() <= first.value()
            }
        }
    };

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
        DataType::BinaryOffset => {
            let ca = s.binary_offset().unwrap();
            let search_values = search_values.binary_offset().unwrap();
            let idx = binary_search_ca(ca, search_values.iter(), side, descending);
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
            let ca = _get_rows_encoded_ca(
                "".into(),
                &[s.as_ref().clone().into_column()],
                &[descending],
                &[false],
            )?;
            let search_values = _get_rows_encoded_ca(
                "".into(),
                &[search_values.clone().into_column()],
                &[descending],
                &[false],
            )?;
            let idx = binary_search_ca_with_overrides(&ca, search_values.iter(), side, false, original_null_count, nulls_last);
            Ok(IdxCa::new_vec(s.name().clone(), idx))
        },
        _ => polars_bail!(opq = search_sorted, original_dtype),
    }
}
