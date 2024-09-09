use std::fmt;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock};

use hive::HivePartitions;
use polars_core::prelude::*;
use recursive::recursive;

use crate::prelude::*;

pub(crate) mod aexpr;
pub(crate) mod anonymous_scan;
pub(crate) mod ir;

mod apply;
mod builder_dsl;
mod builder_ir;
pub(crate) mod conversion;
#[cfg(feature = "debugging")]
pub(crate) mod debug;
pub mod expr_ir;
mod file_scan;
mod format;
mod functions;
pub mod hive;
pub(crate) mod iterator;
mod lit;
pub(crate) mod optimizer;
pub(crate) mod options;
#[cfg(feature = "python")]
pub mod python;
mod schema;
pub mod visitor;

pub use aexpr::*;
pub use anonymous_scan::*;
pub use apply::*;
pub use builder_dsl::*;
pub use builder_ir::*;
pub use conversion::*;
pub(crate) use expr_ir::*;
pub use file_scan::*;
pub use functions::*;
pub use ir::*;
pub use iterator::*;
pub use lit::*;
pub use optimizer::*;
pub use schema::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

#[derive(Clone, Copy, Debug)]
pub enum Context {
    /// Any operation that is done on groups
    Aggregation,
    /// Any operation that is done while projection/ selection of data
    Default,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct DslScanSources {
    pub sources: ScanSources,
    pub is_expanded: bool,
}

// https://stackoverflow.com/questions/1031076/what-are-projection-and-selection
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DslPlan {
    #[cfg(feature = "python")]
    PythonScan { options: PythonOptions },
    /// Filter on a boolean mask
    Filter {
        input: Arc<DslPlan>,
        predicate: Expr,
    },
    /// Cache the input at this point in the LP
    Cache {
        input: Arc<DslPlan>,
        id: usize,
        cache_hits: u32,
    },
    Scan {
        sources: Arc<Mutex<DslScanSources>>,
        // Option as this is mostly materialized on the IR phase.
        // During conversion we update the value in the DSL as well
        // This is to cater to use cases where parts of a `LazyFrame`
        // are used as base of different queries in a loop. That way
        // the expensive schema resolving is cached.
        file_info: Arc<RwLock<Option<FileInfo>>>,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        predicate: Option<Expr>,
        file_options: FileScanOptions,
        scan_type: FileScan,
    },
    // we keep track of the projection and selection as it is cheaper to first project and then filter
    /// In memory DataFrame
    DataFrameScan {
        df: Arc<DataFrame>,
        schema: SchemaRef,
        // schema of the projected file
        output_schema: Option<SchemaRef>,
        filter: Option<Expr>,
    },
    /// Polars' `select` operation, this can mean projection, but also full data access.
    Select {
        expr: Vec<Expr>,
        input: Arc<DslPlan>,
        options: ProjectionOptions,
    },
    /// Groupby aggregation
    GroupBy {
        input: Arc<DslPlan>,
        keys: Vec<Expr>,
        aggs: Vec<Expr>,
        #[cfg_attr(feature = "serde", serde(skip))]
        apply: Option<(Arc<dyn DataFrameUdf>, SchemaRef)>,
        maintain_order: bool,
        options: Arc<GroupbyOptions>,
    },
    /// Join operation
    Join {
        input_left: Arc<DslPlan>,
        input_right: Arc<DslPlan>,
        // Invariant: left_on and right_on are equal length.
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        // Invariant: Either left_on/right_on or predicates is set (non-empty).
        predicates: Vec<Expr>,
        options: Arc<JoinOptions>,
    },
    /// Adding columns to the table without a Join
    HStack {
        input: Arc<DslPlan>,
        exprs: Vec<Expr>,
        options: ProjectionOptions,
    },
    /// Remove duplicates from the table
    Distinct {
        input: Arc<DslPlan>,
        options: DistinctOptionsDSL,
    },
    /// Sort the table
    Sort {
        input: Arc<DslPlan>,
        by_column: Vec<Expr>,
        slice: Option<(i64, usize)>,
        sort_options: SortMultipleOptions,
    },
    /// Slice the table
    Slice {
        input: Arc<DslPlan>,
        offset: i64,
        len: IdxSize,
    },
    /// A (User Defined) Function
    MapFunction {
        input: Arc<DslPlan>,
        function: DslFunction,
    },
    /// Vertical concatenation
    Union {
        inputs: Vec<DslPlan>,
        args: UnionArgs,
    },
    /// Horizontal concatenation of multiple plans
    HConcat {
        inputs: Vec<DslPlan>,
        options: HConcatOptions,
    },
    /// This allows expressions to access other tables
    ExtContext {
        input: Arc<DslPlan>,
        contexts: Vec<DslPlan>,
    },
    Sink {
        input: Arc<DslPlan>,
        payload: SinkType,
    },
    IR {
        #[cfg_attr(feature = "serde", serde(skip))]
        node: Option<Node>,
        version: u32,
        // Keep the original Dsl around as we need that for serialization.
        dsl: Arc<DslPlan>,
    },
}

impl Clone for DslPlan {
    // Autogenerated by rust-analyzer, don't care about it looking nice, it just
    // calls clone on every member of every enum variant.
    #[rustfmt::skip]
    #[allow(clippy::clone_on_copy)]
    #[recursive]
    fn clone(&self) -> Self {
        match self {
            #[cfg(feature = "python")]
            Self::PythonScan { options } => Self::PythonScan { options: options.clone() },
            Self::Filter { input, predicate } => Self::Filter { input: input.clone(), predicate: predicate.clone() },
            Self::Cache { input, id, cache_hits } => Self::Cache { input: input.clone(), id: id.clone(), cache_hits: cache_hits.clone() },
            Self::Scan { sources, file_info, hive_parts, predicate, file_options, scan_type } => Self::Scan { sources: sources.clone(), file_info: file_info.clone(), hive_parts: hive_parts.clone(), predicate: predicate.clone(), file_options: file_options.clone(), scan_type: scan_type.clone() },
            Self::DataFrameScan { df, schema, output_schema, filter: selection } => Self::DataFrameScan { df: df.clone(), schema: schema.clone(), output_schema: output_schema.clone(), filter: selection.clone() },
            Self::Select { expr, input, options } => Self::Select { expr: expr.clone(), input: input.clone(), options: options.clone() },
            Self::GroupBy { input, keys, aggs,  apply, maintain_order, options } => Self::GroupBy { input: input.clone(), keys: keys.clone(), aggs: aggs.clone(), apply: apply.clone(), maintain_order: maintain_order.clone(), options: options.clone() },
            Self::Join { input_left, input_right, left_on, right_on, predicates, options } => Self::Join { input_left: input_left.clone(), input_right: input_right.clone(), left_on: left_on.clone(), right_on: right_on.clone(), options: options.clone(), predicates: predicates.clone() },
            Self::HStack { input, exprs, options } => Self::HStack { input: input.clone(), exprs: exprs.clone(),  options: options.clone() },
            Self::Distinct { input, options } => Self::Distinct { input: input.clone(), options: options.clone() },
            Self::Sort {input,by_column, slice, sort_options } => Self::Sort { input: input.clone(), by_column: by_column.clone(), slice: slice.clone(), sort_options: sort_options.clone() },
            Self::Slice { input, offset, len } => Self::Slice { input: input.clone(), offset: offset.clone(), len: len.clone() },
            Self::MapFunction { input, function } => Self::MapFunction { input: input.clone(), function: function.clone() },
            Self::Union { inputs, args} => Self::Union { inputs: inputs.clone(), args: args.clone() },
            Self::HConcat { inputs, options } => Self::HConcat { inputs: inputs.clone(), options: options.clone() },
            Self::ExtContext { input, contexts, } => Self::ExtContext { input: input.clone(), contexts: contexts.clone() },
            Self::Sink { input, payload } => Self::Sink { input: input.clone(), payload: payload.clone() },
            Self::IR {node, dsl, version} => Self::IR {node: *node, dsl: dsl.clone(), version: *version}
        }
    }
}

impl Default for DslPlan {
    fn default() -> Self {
        let df = DataFrame::empty();
        let schema = df.schema();
        DslPlan::DataFrameScan {
            df: Arc::new(df),
            schema: Arc::new(schema),
            output_schema: None,
            filter: None,
        }
    }
}

impl DslPlan {
    pub fn describe(&self) -> PolarsResult<String> {
        Ok(self.clone().to_alp()?.describe())
    }

    pub fn describe_tree_format(&self) -> PolarsResult<String> {
        Ok(self.clone().to_alp()?.describe_tree_format())
    }

    pub fn display(&self) -> PolarsResult<impl fmt::Display> {
        struct DslPlanDisplay(IRPlan);
        impl fmt::Display for DslPlanDisplay {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.as_ref().display().fmt(f)
            }
        }
        Ok(DslPlanDisplay(self.clone().to_alp()?))
    }

    pub fn to_alp(self) -> PolarsResult<IRPlan> {
        let mut lp_arena = Arena::with_capacity(16);
        let mut expr_arena = Arena::with_capacity(16);

        let node = to_alp(
            self,
            &mut expr_arena,
            &mut lp_arena,
            &mut OptFlags::default(),
        )?;
        let plan = IRPlan::new(node, lp_arena, expr_arena);

        Ok(plan)
    }
}
