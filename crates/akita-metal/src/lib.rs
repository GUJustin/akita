//! Prover-only K16/D512 trace kernels, using the shared `jolt-metal`
//! runtime and field arithmetic. Unsupported shapes fail before dispatch.

#![deny(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::panic_in_result_fn,
    clippy::unreachable
)]

mod error;
mod library;
pub mod onehot;
pub mod onehot_fold;
pub mod onehot_ntt;
pub mod onehot_packing;

pub use error::AkitaMetalError;
pub use jolt_metal::{ErrorClass, MetalError};
pub use library::AkitaMetal;
