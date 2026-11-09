//! Exact limb-NTT inner commitments for Fp128, degree 512 and K16 sources.
//!
//! The prepared key is independent of witness length. Chunked integer
//! convolutions reconstruct the same field commitment as the direct sweep;
//! bounded batches avoid retaining one output per reconstruction chunk.

mod key;
pub use key::DeviceOneHotNttKey;

use std::time::Duration;

use akita_algebra::ntt::I32_LAZY_DOT_BATCH;
use akita_algebra::tables::Q128_RAW_PRIMES;
use akita_error::checked;
use bytemuck::{Pod, Zeroable};
use jolt_field::Prime128OffsetA7F7 as F;
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid};

use crate::library::Instance;
use crate::onehot::{to_u32, to_u64, DeviceK16Sources, Geometry, OneHotCommitShape};
use crate::{AkitaMetal, AkitaMetalError};
use akita_algebra::NttPrime;
use key::D;

const COMMIT: &str = "akita_onehot_limb_commit_fp128";
const COMMIT_COMPACT: &str = "akita_onehot_limb_commit_compact_fp128";
const COMBINE: &str = "akita_onehot_limb_combine_fp128";
const THREADS: usize = D / 2;
const TARGET_GROUPS: usize = 4096;
const MAX_BATCH_PARTS: usize = 16;

pub(crate) fn instances() -> Vec<Instance> {
    Q128_RAW_PRIMES
        .first()
        .into_iter()
        .flat_map(|prime| {
            ["ulong", "uint"].map(|index| Instance {
                template: "akita_onehot_limb_commit",
                args: format!("{}u, {I32_LAZY_DOT_BATCH}u, {prime}u, {index}", F::C),
                host_name: if index == "uint" {
                    COMMIT_COMPACT.into()
                } else {
                    COMMIT.into()
                },
            })
        })
        .chain(std::iter::once(Instance {
            template: "akita_onehot_limb_combine",
            args: format!("{}u", F::C),
            host_name: COMBINE.into(),
        }))
        .collect()
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DevicePrime {
    p: i32,
    pinv: i32,
    mont: i32,
    montsq: i32,
}

impl From<NttPrime<i32>> for DevicePrime {
    fn from(prime: NttPrime<i32>) -> Self {
        Self {
            p: prime.p,
            pinv: prime.pinv,
            mont: prime.mont,
            montsq: prime.montsq,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SourceParams {
    fields: u64,
    chunks: u64,
    hot_offset: u64,
    first_ring: u32,
    hot_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CommitParams {
    prime: DevicePrime,
    positions: u32,
    columns: u32,
    digits: u32,
    rows: u32,
    rings: u32,
    reconstruction_positions: u32,
    first_chunk: u32,
    parts: u32,
    direct_output: u32,
}

/// Reusable final-output and bounded partial-output buffers.
///
/// At most 16 reconstruction chunks are in flight. Large output sets use
/// one chunk per dispatch and add directly into the output, with no partial
/// buffer. Smaller output sets use fewer than 64 MiB of partial storage.
pub struct OneHotNttWorkspace<'a> {
    metal: &'a AkitaMetal,
    out: Option<DeviceBuffer<F>>,
    partials: Option<DeviceBuffer<F>>,
}

impl<'a> OneHotNttWorkspace<'a> {
    pub fn new(metal: &'a AkitaMetal) -> Self {
        Self {
            metal,
            out: None,
            partials: None,
        }
    }

    /// Commits sources in source/block/A-row/coefficient order.
    ///
    /// Validates the entire shape before changing retained buffers. Source
    /// bounds and key-prefix interpretation are the direct kernel's bounds.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "batch differences are bounded by checked chunk counts"
    )]
    pub fn commit(
        &mut self,
        matrix: &DeviceOneHotNttKey,
        sources: &DeviceK16Sources,
        shape: OneHotCommitShape,
    ) -> Result<(&mut DeviceBuffer<F>, Duration), AkitaMetalError> {
        let geometry = Geometry::new(shape)?;
        let matrix_len = checked::product([geometry.rows, geometry.columns, D])
            .ok_or_else(|| AkitaMetalError::Shape("A matrix size overflows usize".into()))?;
        if matrix_len > matrix.coefficient_count {
            return Err(AkitaMetalError::Shape(format!(
                "the A matrix view needs {matrix_len} coefficients; the key has {}",
                matrix.coefficient_count
            )));
        }
        let mut layouts = Vec::with_capacity(sources.sources.len());
        let mut counts = Vec::with_capacity(sources.sources.len());
        let mut rings = 0usize;
        for source in &sources.sources {
            let count = checked::product([geometry.blocks::<D>(source)?, geometry.rows])
                .ok_or_else(|| {
                    AkitaMetalError::Shape("source output size overflows usize".into())
                })?;
            layouts.push(SourceParams {
                fields: to_u64(source.fields)?,
                chunks: to_u64(source.chunks)?,
                hot_offset: to_u64(source.hot_offset)?,
                first_ring: to_u32(rings)?,
                hot_stride: to_u32(source.hot_stride)?,
            });
            rings = checked::sum([rings, count]).ok_or_else(|| {
                AkitaMetalError::Shape("output ring count overflows usize".into())
            })?;
            counts.push(count);
        }
        let len = checked::product([rings, D])
            .ok_or_else(|| AkitaMetalError::Shape("output size overflows usize".into()))?;
        checked::product([len, size_of::<F>()])
            .ok_or_else(|| AkitaMetalError::Shape("output bytes overflow usize".into()))?;
        let chunks = checked::div_ceil(geometry.positions, matrix.reconstruction_positions)
            .ok_or_else(|| AkitaMetalError::Shape("invalid reconstruction bound".into()))?;
        let parts = checked::div_ceil(TARGET_GROUPS, rings)
            .unwrap_or(1)
            .clamp(1, MAX_BATCH_PARTS)
            .min(chunks);
        let partial_len = (parts > 1)
            .then(|| checked::product([parts, len]))
            .map(|len| {
                len.ok_or_else(|| AkitaMetalError::Shape("partial size overflows usize".into()))
            })
            .transpose()?;
        if let Some(len) = partial_len {
            checked::product([len, size_of::<F>()])
                .ok_or_else(|| AkitaMetalError::Shape("partial bytes overflow usize".into()))?;
        }
        let threads = checked::product([parts, rings, THREADS])
            .ok_or_else(|| AkitaMetalError::Shape("dispatch size overflows usize".into()))?;
        to_u32(threads)?;
        to_u32(len)?;
        let params = CommitParams {
            prime: matrix.prime.into(),
            positions: to_u32(geometry.positions)?,
            columns: to_u32(geometry.columns)?,
            digits: to_u32(geometry.digits)?,
            rows: to_u32(geometry.rows)?,
            rings: to_u32(rings)?,
            reconstruction_positions: to_u32(matrix.reconstruction_positions)?,
            first_chunk: 0,
            parts: to_u32(parts)?,
            direct_output: u32::from(parts == 1),
        };
        // No dispatch parameters are needed for an empty output, even if
        // the caller supplies a large (zero-row) matrix view.
        let windows = (0..if rings == 0 { 0 } else { chunks })
            .step_by(parts)
            .map(|first| {
                Ok(CommitParams {
                    first_chunk: to_u32(first)?,
                    parts: to_u32(parts.min(chunks - first))?,
                    ..params
                })
            })
            .collect::<Result<Vec<_>, AkitaMetalError>>()?;
        // A 32-bit word index saves address arithmetic for ordinary keys.
        // Larger views retain the 64-bit path; key preparation has no limit
        // tied to the current witness length.
        let compact =
            checked::product([matrix_len, 8]).is_some_and(|words| u32::try_from(words).is_ok());
        let commit = self
            .metal
            .pipeline(if compact { COMMIT_COMPACT } else { COMMIT })?;
        let combine = self.metal.pipeline(COMBINE)?;
        if commit.max_total_threads_per_threadgroup() < THREADS {
            return Err(AkitaMetalError::Shape(
                "the limb kernel needs 256 threads per group".into(),
            ));
        }
        let mut source_map = Vec::new();
        source_map.try_reserve_exact(rings).map_err(|_| {
            AkitaMetalError::Shape(format!("cannot stage {rings} output-ring indices"))
        })?;
        for (id, count) in counts.into_iter().enumerate() {
            source_map.extend(std::iter::repeat_n(to_u32(id)?, count));
        }
        let device = self.metal.device();
        let layouts = DeviceBuffer::from_slice(device, &layouts)?;
        let source_map = DeviceBuffer::from_slice(device, &source_map)?;
        if self.out.as_ref().is_some_and(|out| out.len() != len) {
            self.out = None;
        }
        if self.partials.as_ref().map(DeviceBuffer::len) != partial_len {
            self.partials = None;
        }
        if self.out.is_none() {
            self.out = Some(DeviceBuffer::zeroed(device, len)?);
        }
        if let Some(len) = partial_len {
            if self.partials.is_none() {
                self.partials = Some(DeviceBuffer::zeroed(device, len)?);
            }
        }
        let out = self
            .out
            .as_mut()
            .ok_or_else(|| AkitaMetalError::Shape("output workspace unavailable".into()))?;
        if rings == 0 {
            return Ok((out, Duration::ZERO));
        }
        let target = self.partials.as_ref().unwrap_or(out);
        let mut batch = Batch::new(device)?;
        for params in &windows {
            let grid = checked::product([params.parts as usize, rings, THREADS])
                .ok_or_else(|| AkitaMetalError::Shape("dispatch size overflows usize".into()))?;
            // SAFETY: all geometry, widths and buffer lengths were checked
            // above. Each group owns one source/block/row/part; source loads
            // check both backing and logical lengths. Serial dispatches order
            // scratch reuse and updates to the retained output.
            unsafe {
                batch.dispatch_unchecked(
                    commit,
                    &[
                        Binding::buffer(&sources.hot),
                        Binding::buffer(&matrix.transformed),
                        Binding::buffer(&matrix.tables),
                        Binding::buffer(&layouts),
                        Binding::buffer(&source_map),
                        Binding::buffer(target),
                        Binding::value(params),
                    ],
                    Grid::linear(grid, THREADS),
                )?;
            }
            if partial_len.is_some() {
                // SAFETY: each thread combines its own coefficient from the
                // checked parts, overwriting on the first batch and adding later.
                unsafe {
                    batch.dispatch_unchecked(
                        combine,
                        &[
                            Binding::buffer(target),
                            Binding::buffer(out),
                            Binding::value(params),
                        ],
                        Grid::linear(len, THREADS),
                    )?;
                }
            }
        }
        let time = batch.commit_and_wait()?;
        Ok((out, time))
    }
}
