use jolt_metal::{ErrorClass, MetalError};

/// A failure of an Akita Metal operation.
///
/// Capability selection precedes execution. A caller must propagate failures
/// from a selected operation, including malformed source and key shapes.
#[derive(Debug, thiserror::Error)]
pub enum AkitaMetalError {
    /// The device runtime failed; see [`MetalError::class`].
    #[error(transparent)]
    Metal(#[from] MetalError),
    /// Invalid or unsupported input geometry, checked before dispatch.
    #[error("invalid or unsupported shape: {0}")]
    Shape(String),
}

impl AkitaMetalError {
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Metal(error) => error.class(),
            Self::Shape(_) => ErrorClass::Setup,
        }
    }
}
