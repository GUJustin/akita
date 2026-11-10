use std::fmt::Write;

use jolt_metal::runtime::{Device, LibrarySpec, Pipeline, ShaderLibrary};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use crate::AkitaMetalError;

pub(crate) struct Instance {
    pub(crate) template: &'static str,
    pub(crate) args: String,
    pub(crate) host_name: String,
}

/// An explicitly owned device and its precompiled trace pipelines.
pub struct AkitaMetal {
    device: Device,
    library: ShaderLibrary,
}

impl AkitaMetal {
    /// Compiles all supported kernels before any trace operation runs.
    pub fn new() -> Result<Self, AkitaMetalError> {
        let device = Device::system_default()?;
        let mut spec = FIELD_HEADERS
            .iter()
            .fold(LibrarySpec::new(), |spec, (name, text)| {
                spec.source(name, text)
            });
        for (name, source) in [
            ("akita/mont.h", include_str!("../shaders/akita/mont.h")),
            ("akita/ntt.h", include_str!("../shaders/akita/ntt.h")),
            (
                "akita/onehot_ntt.metal",
                include_str!("../shaders/akita/onehot_ntt.metal"),
            ),
            (
                "akita/onehot_shared_ntt.metal",
                include_str!("../shaders/akita/onehot_shared_ntt.metal"),
            ),
            (
                "akita/onehot_fold.metal",
                include_str!("../shaders/akita/onehot_fold.metal"),
            ),
            (
                "akita/onehot_packing.metal",
                include_str!("../shaders/akita/onehot_packing.metal"),
            ),
        ] {
            spec = spec.source(name, source);
        }
        spec = spec.kernel(crate::onehot_fold::FOLD);
        let mut instantiations = String::new();
        for instance in crate::onehot_ntt::instances()
            .into_iter()
            .chain(crate::onehot_packing::instances())
        {
            let _ = writeln!(instantiations,
                "template [[host_name(\"{host}\")]] [[kernel]] decltype(akita::{template}<{args}>) akita::{template}<{args}>;",
                host = instance.host_name, template = instance.template, args = instance.args);
            spec = spec.kernel(&instance.host_name);
        }
        spec = spec.source("akita instantiations", &instantiations);
        let library = ShaderLibrary::compile(&device, &spec)?;
        Ok(Self { device, library })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn pipeline(&self, name: &str) -> Result<&Pipeline, AkitaMetalError> {
        self.library.pipeline(name).map_err(|error| match error {
            MetalError::UnknownPipeline { name } => {
                AkitaMetalError::Shape(format!("no compiled kernel for {name}"))
            }
            other => other.into(),
        })
    }
}
