use super::*;
use crate::plans::optimizer::join_utils::remove_suffix;

// Information concerning individual sides of a join.
#[derive(PartialEq, Eq)]
struct LeftRight<T>(T, T);

fn should_block_join_specific(
    ae: &AExpr,
    how: &JoinType,
    on_names: &PlHashSet<PlSmallStr>,
    expr_arena: &Arena<AExpr>,
    schema_left: &Schema,
    schema_right: &Schema,
) -> LeftRight<bool> {
    use AExpr::*;
    match ae {
        // joins can produce null values
        Function {
            function:
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull)
                | IRFunctionExpr::Boolean(IRBooleanFunction::IsNull)
                | IRFunctionExpr::FillNull,
            ..
        } => join_produces_null(how),
        #[cfg(feature = "is_in")]
        Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { .. }),
            ..
        } => join_produces_null(how),
        // joins can produce duplicates
        #[cfg(feature = "is_unique")]
        Function {
            function:
                IRFunctionExpr::Boolean(IRBooleanFunction::IsUnique)
                | IRFunctionExpr::Boolean(IRBooleanFunction::IsDuplicated),
            ..
        } => LeftRight(true, true),
        #[cfg(feature = "is_first_distinct")]
        Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsFirstDistinct),
            ..
        } => LeftRight(true, true),
        // any operation that checks for equality or ordering can be wrong because
        // the join can produce null values
        // TODO! check if we can be less conservative here
        BinaryExpr {
            op: Operator::Eq | Operator::NotEq,
            left,
            right,
        } => {
            let LeftRight(bleft, bright) = join_produces_null(how);

            let l_name = aexpr_output_name(*left, expr_arena).unwrap();
            let r_name = aexpr_output_name(*right, expr_arena).unwrap();

            let is_in_on = on_names.contains(&l_name) || on_names.contains(&r_name);

            let block_left =
                is_in_on && (schema_left.contains(&l_name) || schema_left.contains(&r_name));
            let block_right =
                is_in_on && (schema_right.contains(&l_name) || schema_right.contains(&r_name));
            LeftRight(block_left | bleft, block_right | bright)
        },
        _ => join_produces_null(how),
    }
}

/// Returns a tuple indicating whether predicates should be blocked for either side based on the
/// join type.
///
/// * `true` indicates that predicates must not be pushed to that side
fn join_produces_null(how: &JoinType) -> LeftRight<bool> {
    match how {
        JoinType::Left => LeftRight(false, true),
        JoinType::Right => LeftRight(true, false),

        JoinType::Full => LeftRight(true, true),
        JoinType::Cross => LeftRight(true, true),
        #[cfg(feature = "asof_join")]
        JoinType::AsOf(_) => LeftRight(true, true),

        JoinType::Inner => LeftRight(false, false),
        #[cfg(feature = "semi_anti_join")]
        JoinType::Semi | JoinType::Anti => LeftRight(false, false),
        #[cfg(feature = "iejoin")]
        JoinType::IEJoin => LeftRight(false, false),
    }
}

fn all_pred_cols_in_left_on(
    predicate: &ExprIR,
    expr_arena: &mut Arena<AExpr>,
    left_on: &[ExprIR],
) -> bool {
    aexpr_to_leaf_names_iter(predicate.node(), expr_arena)
        .all(|pred_column_name| left_on.iter().any(|e| e.output_name() == &pred_column_name))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_join(
    opt: &mut PredicatePushDown,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    input_left: Node,
    input_right: Node,
    left_on: Vec<ExprIR>,
    right_on: Vec<ExprIR>,
    schema: SchemaRef,
    options: Arc<JoinOptionsIR>,
    acc_predicates: PlHashMap<PlSmallStr, ExprIR>,
) -> PolarsResult<IR> {
    use IR::*;
    let schema_left = lp_arena.get(input_left).schema(lp_arena);
    let schema_right = lp_arena.get(input_right).schema(lp_arena);

    let on_names = left_on
        .iter()
        .flat_map(|e| aexpr_to_leaf_names_iter(e.node(), expr_arena))
        .chain(
            right_on
                .iter()
                .flat_map(|e| aexpr_to_leaf_names_iter(e.node(), expr_arena)),
        )
        .collect::<PlHashSet<_>>();

    let mut pushdown_left = init_hashmap(Some(acc_predicates.len()));
    let mut pushdown_right = init_hashmap(Some(acc_predicates.len()));
    let mut local_predicates = Vec::with_capacity(acc_predicates.len());

    for (_, predicate) in acc_predicates {
        let column_origins = ExprOrigin::get_expr_origin(
            predicate.node(),
            expr_arena,
            &schema_left,
            &schema_right,
            options.args.suffix(),
        )
        .unwrap_or_else(|e| {
            if cfg!(debug_assertions) {
                panic!("{e:?}")
            } else {
                ExprOrigin::None
            }
        });

        // Cross joins produce a cartesian product, so if a predicate combines columns from both tables, we should not push down.
        // Inequality joins logically produce a cartesian product, so the same logic applies.
        if (options.args.how.is_cross() || options.args.how.is_ie())
            && column_origins == ExprOrigin::Both
        {
            local_predicates.push(predicate);
            continue;
        }

        // check if predicate can pass the joins node
        let allow_pushdown_left = !has_aexpr(predicate.node(), expr_arena, |ae| {
            should_block_join_specific(
                ae,
                &options.args.how,
                &on_names,
                expr_arena,
                &schema_left,
                &schema_right,
            )
            .0
        });

        let allow_pushdown_right = !has_aexpr(predicate.node(), expr_arena, |ae| {
            should_block_join_specific(
                ae,
                &options.args.how,
                &on_names,
                expr_arena,
                &schema_left,
                &schema_right,
            )
            .1
        });

        // these indicate to which tables we are going to push down the predicate
        let mut filter_left = false;
        let mut filter_right = false;

        if allow_pushdown_left && column_origins == ExprOrigin::Left {
            filter_left = true;

            insert_and_combine_predicate(&mut pushdown_left, &predicate, expr_arena);
            // If we push down to the left and all predicate columns are also
            // join columns, we also push down right for inner, left or semi join
            if all_pred_cols_in_left_on(&predicate, expr_arena, &left_on) {
                filter_right = match &options.args.how {
                    // TODO! if join_on right has a different name
                    // we can set this to `true` IFF we rename the predicate
                    JoinType::Inner | JoinType::Left => {
                        check_input_node(predicate.node(), &schema_right, expr_arena)
                    },
                    #[cfg(feature = "semi_anti_join")]
                    JoinType::Semi => check_input_node(predicate.node(), &schema_right, expr_arena),
                    _ => false,
                }
            }
        // this is `else if` because if the predicate is in the left hand side
        // the right hand side should be renamed with the suffix.
        // in that case we should not push down as the user wants to filter on `x`
        // not on `x_rhs`.
        } else if allow_pushdown_right && column_origins == ExprOrigin::Right {
            filter_right = true;

            let mut predicate = predicate.clone();
            remove_suffix(
                &mut predicate,
                expr_arena,
                &schema_right,
                options.args.suffix(),
            );

            insert_and_combine_predicate(&mut pushdown_right, &predicate, expr_arena);
        }

        match (filter_left, filter_right, &options.args.how) {
            // if not pushed down on one of the tables we have to do it locally.
            (false, false, _) |
            // if left join and predicate only available in right table,
            // 'we should not filter right, because that would lead to
            // invalid results.
            // see: #2057
            (false, true, JoinType::Left)
            => {
                local_predicates.push(predicate);
                continue;
            },
            // business as usual
            _ => {}
        }
    }

    opt.pushdown_and_assign(input_left, pushdown_left, lp_arena, expr_arena)?;
    opt.pushdown_and_assign(input_right, pushdown_right, lp_arena, expr_arena)?;

    let lp = Join {
        input_left,
        input_right,
        left_on,
        right_on,
        schema,
        options,
    };

    Ok(opt.optional_apply_predicate(lp, local_predicates, lp_arena, expr_arena))
}
