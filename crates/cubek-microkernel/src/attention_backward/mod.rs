//! The microkernels of attention's backward pass, one folder per problem: [`query_gradient`], the
//! gradient of a block of queries, walking the keys it sees; and [`key_gradient`], the gradients of
//! a block of keys and of their values, walking the queries that see them. What both read lies
//! beside them: the operands and the work item. The axes, and which keys a query sees, are the
//! forward's ([`attention`](crate::attention)).

mod operand;
mod work_item;

pub mod key_gradient;
pub mod query_gradient;

pub use operand::*;
pub use work_item::*;
