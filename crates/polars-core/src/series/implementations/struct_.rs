use std::ops::Not;

use arrow::bitmap::Bitmap;

use super::*;
use crate::chunked_array::StructChunked;
use crate::prelude::*;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};

impl PrivateSeriesNumeric for SeriesWrap<StructChunked> {
    fn bit_repr(&self) -> Option<BitRepr> {
        None
    }
}

impl PrivateSeries for SeriesWrap<StructChunked> {
    fn _field(&self) -> Cow<'_, Field> {
        Cow::Borrowed(self.0.ref_field())
    }

    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn compute_len(&mut self) {
        self.0.compute_len()
    }

    fn _get_flags(&self) -> StatisticsFlags {
        self.0.get_flags()
    }

    fn _set_flags(&mut self, flags: StatisticsFlags) {
        self.0.set_flags(flags);
    }

    // TODO! remove this. Very slow. Asof join should use row-encoding.
    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        let other = other.struct_().unwrap();
        self.0
            .fields_as_series()
            .iter()
            .zip(other.fields_as_series())
            .all(|(s, other)| s.equal_element(idx_self, idx_other, &other))
    }

    #[cfg(feature = "algorithm_group_by")]
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsType> {
        let ca = self.0.get_row_encoded(Default::default())?;
        ca.group_tuples(multithreaded, sorted)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        self.0
            .zip_with(mask, other.struct_()?)
            .map(|ca| ca.into_series())
    }

    fn into_total_eq_inner<'a>(&'a self) -> Box<dyn TotalEqInner + 'a> {
        invalid_operation_panic!(into_total_eq_inner, self)
    }
    fn into_total_ord_inner<'a>(&'a self) -> Box<dyn TotalOrdInner + 'a> {
        invalid_operation_panic!(into_total_ord_inner, self)
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_list(&self, groups: &GroupsType) -> Series {
        self.0.agg_list(groups)
    }

    fn vec_hash(
        &self,
        build_hasher: PlSeedableRandomStateQuality,
        buf: &mut Vec<u64>,
    ) -> PolarsResult<()> {
        let mut fields = self.0.fields_as_series().into_iter();

        if let Some(s) = fields.next() {
            s.vec_hash(build_hasher, buf)?
        };
        for s in fields {
            s.vec_hash_combine(build_hasher, buf)?
        }
        Ok(())
    }
}

impl SeriesTrait for SeriesWrap<StructChunked> {
    fn rename(&mut self, name: PlSmallStr) {
        self.0.rename(name)
    }

    fn chunk_lengths(&self) -> ChunkLenIter<'_> {
        self.0.chunk_lengths()
    }

    fn name(&self) -> &PlSmallStr {
        self.0.name()
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        &self.0.chunks
    }

    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        self.0.chunks_mut()
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.0.slice(offset, length).into_series()
    }

    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (l, r) = self.0.split_at(offset);
        (l.into_series(), r.into_series())
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        self.0.append(other.as_ref().as_ref())
    }
    fn append_owned(&mut self, other: Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        self.0.append_owned(other.take_inner())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), extend);
        self.0.extend(other.as_ref().as_ref())
    }

    fn filter(&self, _filter: &BooleanChunked) -> PolarsResult<Series> {
        ChunkFilter::filter(&self.0, _filter).map(|ca| ca.into_series())
    }

    fn take(&self, _indices: &IdxCa) -> PolarsResult<Series> {
        self.0.take(_indices).map(|ca| ca.into_series())
    }

    unsafe fn take_unchecked(&self, _idx: &IdxCa) -> Series {
        self.0.take_unchecked(_idx).into_series()
    }

    fn take_slice(&self, _indices: &[IdxSize]) -> PolarsResult<Series> {
        self.0.take(_indices).map(|ca| ca.into_series())
    }

    unsafe fn take_slice_unchecked(&self, _idx: &[IdxSize]) -> Series {
        self.0.take_unchecked(_idx).into_series()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        self.0.rechunk().into_owned().into_series()
    }

    fn new_from_index(&self, _index: usize, _length: usize) -> Series {
        self.0.new_from_index(_index, _length).into_series()
    }

    fn trim_lists_to_normalized_offsets(&self) -> Option<Series> {
        self.0
            .trim_lists_to_normalized_offsets()
            .map(IntoSeries::into_series)
    }

    fn propagate_nulls(&self) -> Option<Series> {
        self.0.propagate_nulls().map(IntoSeries::into_series)
    }

    fn cast(&self, dtype: &DataType, cast_options: CastOptions) -> PolarsResult<Series> {
        self.0.cast_with_options(dtype, cast_options)
    }

    unsafe fn get_unchecked(&self, index: usize) -> AnyValue<'_> {
        self.0.get_any_value_unchecked(index)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    /// Get unique values in the Series.
    #[cfg(feature = "algorithm_group_by")]
    fn unique(&self) -> PolarsResult<Series> {
        // this can called in aggregation, so this fast path can be worth a lot
        if self.len() < 2 {
            return Ok(self.0.clone().into_series());
        }
        let main_thread = POOL.current_thread_index().is_none();
        let groups = self.group_tuples(main_thread, false);
        // SAFETY:
        // groups are in bounds
        Ok(unsafe { self.0.clone().into_series().agg_first(&groups?) })
    }

    /// Get unique values in the Series.
    #[cfg(feature = "algorithm_group_by")]
    fn n_unique(&self) -> PolarsResult<usize> {
        // this can called in aggregation, so this fast path can be worth a lot
        match self.len() {
            0 => Ok(0),
            1 => Ok(1),
            _ => {
                // TODO! try row encoding
                let main_thread = POOL.current_thread_index().is_none();
                let groups = self.group_tuples(main_thread, false)?;
                Ok(groups.len())
            },
        }
    }

    /// Get first indexes of unique values.
    #[cfg(feature = "algorithm_group_by")]
    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        // this can called in aggregation, so this fast path can be worth a lot
        if self.len() == 1 {
            return Ok(IdxCa::new_vec(self.name().clone(), vec![0 as IdxSize]));
        }
        let main_thread = POOL.current_thread_index().is_none();
        let groups = self.group_tuples(main_thread, true)?;
        let first = groups.take_group_firsts();
        Ok(IdxCa::from_vec(self.name().clone(), first))
    }

    fn has_nulls(&self) -> bool {
        self.0.has_nulls()
    }

    fn is_null(&self) -> BooleanChunked {
        let iter = self.downcast_iter().map(|arr| {
            let bitmap = match arr.validity() {
                Some(valid) => valid.not(),
                None => Bitmap::new_with_value(false, arr.len()),
            };
            BooleanArray::from_data_default(bitmap, None)
        });
        BooleanChunked::from_chunk_iter(self.name().clone(), iter)
    }

    fn is_not_null(&self) -> BooleanChunked {
        let iter = self.downcast_iter().map(|arr| {
            let bitmap = match arr.validity() {
                Some(valid) => valid.clone(),
                None => Bitmap::new_with_value(true, arr.len()),
            };
            BooleanArray::from_data_default(bitmap, None)
        });
        BooleanChunked::from_chunk_iter(self.name().clone(), iter)
    }

    fn reverse(&self) -> Series {
        let validity = self
            .rechunk_validity()
            .map(|x| x.into_iter().rev().collect::<Bitmap>());
        self.0
            ._apply_fields(|s| s.reverse())
            .unwrap()
            .with_outer_validity(validity)
            .into_series()
    }

    fn shift(&self, periods: i64) -> Series {
        self.0.shift(periods).into_series()
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn find_validity_mismatch(&self, other: &Series, idxs: &mut Vec<IdxSize>) {
        self.0.find_validity_mismatch(other, idxs)
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.0
    }

    fn as_phys_any(&self) -> &dyn Any {
        &self.0
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self as _
    }

    fn sort_with(&self, options: SortOptions) -> PolarsResult<Series> {
        Ok(self.0.sort_with(options).into_series())
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        self.0.arg_sort(options)
    }
}
