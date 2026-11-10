//! Bounded source spectra and shared-key dots at the existing K16 parameters.

use std::mem::MaybeUninit;
use std::time::Duration;

use super::{
    CommitLayout, CommitParams, DeviceOneHotNttKey, SourceParams, COMBINE, D, F,
    I32_LAZY_DOT_BATCH, MAX_BATCH_PARTS, Q128_RAW_PRIMES,
};
use crate::library::Instance;
use crate::onehot::{to_u32, DeviceK16Sources, OneHotCommitShape};
use crate::{AkitaMetal, AkitaMetalError};
use akita_error::checked;
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid};

const SOURCE: &str = "akita_onehot_shared_source_fp128";
const DOT: &str = "akita_onehot_shared_dot_fp128";
const INVERSE: &str = "akita_onehot_shared_inverse_fp128";
const WORKSPACE_BYTES: usize = 256 << 20;
const OUTPUT_TILE: usize = 16;
const FREQUENCY_TILE: usize = 64;
const DOT_THREADS: usize = 512;
const SOURCE_THREADS: usize = 128;
const INVERSE_THREADS: usize = 256;

pub(super) fn instances() -> Vec<Instance> {
    Q128_RAW_PRIMES
        .first()
        .into_iter()
        .flat_map(|prime| {
            [
                Instance {
                    template: "akita_onehot_shared_source",
                    args: format!("{prime}u"),
                    host_name: SOURCE.into(),
                },
                Instance {
                    template: "akita_onehot_shared_dot",
                    args: format!("{prime}u, {I32_LAZY_DOT_BATCH}u"),
                    host_name: DOT.into(),
                },
                Instance {
                    template: "akita_onehot_shared_inverse",
                    args: format!("{}u, {prime}u", F::C),
                    host_name: INVERSE.into(),
                },
            ]
        })
        .collect()
}

struct Buffers {
    // These never expose host reads. Each GPU consumer follows its full writer
    // in the same batch, including zero writes for omitted source positions.
    spectra: DeviceBuffer<MaybeUninit<u32>>,
    residues: DeviceBuffer<MaybeUninit<u32>>,
    partials: DeviceBuffer<MaybeUninit<F>>,
    out: DeviceBuffer<F>,
}

/// Rank-one, single-digit K16 commitments with a bounded source-NTT workspace.
///
/// Uses the same key, prime, limb widths and reconstruction span as
/// [`super::OneHotNttWorkspace`]. All dispatches share one batch and one wait.
pub struct SharedKeyNttWorkspace<'a> {
    metal: &'a AkitaMetal,
    buffers: Option<Buffers>,
}

impl<'a> SharedKeyNttWorkspace<'a> {
    pub fn new(metal: &'a AkitaMetal) -> Self {
        Self {
            metal,
            buffers: None,
        }
    }

    /// Commits in the existing source/block/A-row/coefficient order.
    /// Unsupported geometry or resource limits fail before GPU execution.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "checked geometry and allocation extents bound dispatch products and batch differences"
    )]
    pub fn commit(
        &mut self,
        matrix: &DeviceOneHotNttKey,
        sources: &DeviceK16Sources,
        shape: OneHotCommitShape,
    ) -> Result<(&mut DeviceBuffer<F>, Duration), AkitaMetalError> {
        let layout = CommitLayout::new(matrix, sources, shape)?;
        if layout.geometry.rows != 1 || layout.geometry.digits != 1 {
            return Err(AkitaMetalError::Shape(
                "shared-key NTT requires rank one and one inner digit".into(),
            ));
        }
        if I32_LAZY_DOT_BATCH != 6 {
            return Err(AkitaMetalError::Shape(
                "shared key tile requires six positions".into(),
            ));
        }
        let source = self.metal.pipeline(SOURCE)?;
        let dot = self.metal.pipeline(DOT)?;
        let inverse = self.metal.pipeline(INVERSE)?;
        let combine = self.metal.pipeline(COMBINE)?;
        let device = self.metal.device();
        for (pipeline, threads) in [
            (source, SOURCE_THREADS),
            (dot, DOT_THREADS),
            (inverse, INVERSE_THREADS),
            (combine, INVERSE_THREADS),
        ] {
            if pipeline.thread_execution_width() != 32
                || pipeline.max_total_threads_per_threadgroup() < threads
                || pipeline.static_threadgroup_memory_length()
                    > device.limits().max_threadgroup_memory_length
            {
                return Err(AkitaMetalError::Shape(
                    "shared-key NTT pipeline exceeds device thread or shared-memory limits".into(),
                ));
            }
        }
        let spectrum_per_part =
            checked::product([layout.rings, matrix.reconstruction_positions, D])
                .ok_or_else(|| AkitaMetalError::Shape("source spectrum extent overflows".into()))?;
        let residue_per_part = checked::product([layout.rings, 8, D])
            .ok_or_else(|| AkitaMetalError::Shape("residue extent overflows".into()))?;
        let per_part_bytes = [
            checked::product([spectrum_per_part, size_of::<u32>()]),
            checked::product([residue_per_part, size_of::<u32>()]),
            checked::product([layout.len, size_of::<F>()]),
        ]
        .into_iter()
        .try_fold(0usize, |sum, value| checked::sum([sum, value?]))
        .ok_or_else(|| AkitaMetalError::Shape("workspace part extent overflows".into()))?;
        let fixed_bytes = [
            checked::product([layout.len, size_of::<F>()]),
            checked::product([layout.layouts.len(), size_of::<SourceParams>()]),
            checked::product([layout.source_map.len(), size_of::<u32>()]),
        ]
        .into_iter()
        .try_fold(0usize, |sum, value| checked::sum([sum, value?]))
        .ok_or_else(|| AkitaMetalError::Shape("workspace metadata extent overflows".into()))?;
        let available = WORKSPACE_BYTES
            .checked_sub(fixed_bytes)
            .ok_or_else(|| AkitaMetalError::Shape("workspace metadata exceeds budget".into()))?;
        let parts = if layout.rings == 0 {
            1
        } else {
            available
                .checked_div(per_part_bytes)
                .unwrap_or(0)
                .min(MAX_BATCH_PARTS)
                .min(layout.chunks)
        };
        if parts == 0 {
            return Err(AkitaMetalError::Shape(
                "one source spectrum span exceeds workspace budget".into(),
            ));
        }
        let spectrum_len = checked::product([parts, spectrum_per_part])
            .ok_or_else(|| AkitaMetalError::Shape("source workspace extent overflows".into()))?;
        let residue_len = checked::product([parts, residue_per_part])
            .ok_or_else(|| AkitaMetalError::Shape("residue workspace extent overflows".into()))?;
        let partial_len = checked::product([parts, layout.len])
            .ok_or_else(|| AkitaMetalError::Shape("partial workspace extent overflows".into()))?;
        let source_threads =
            checked::product([parts, matrix.reconstruction_positions, layout.rings, 32])
                .ok_or_else(|| AkitaMetalError::Shape("source grid overflows".into()))?;
        let output_tiles = checked::div_ceil(layout.rings, OUTPUT_TILE)
            .ok_or_else(|| AkitaMetalError::Shape("dot tiles overflow".into()))?;
        let dot_threads = checked::product([parts, D / FREQUENCY_TILE, output_tiles, DOT_THREADS])
            .ok_or_else(|| AkitaMetalError::Shape("dot grid overflows".into()))?;
        let inverse_threads = checked::product([parts, layout.rings, INVERSE_THREADS])
            .ok_or_else(|| AkitaMetalError::Shape("inverse grid overflows".into()))?;
        for length in [
            spectrum_len,
            residue_len,
            partial_len,
            layout.len,
            source_threads,
            dot_threads,
            inverse_threads,
        ] {
            to_u32(length)?;
        }
        let params = CommitParams {
            prime: matrix.prime.into(),
            positions: to_u32(layout.geometry.positions)?,
            columns: to_u32(layout.geometry.columns)?,
            digits: 1,
            rows: 1,
            rings: to_u32(layout.rings)?,
            reconstruction_positions: to_u32(matrix.reconstruction_positions)?,
            first_chunk: 0,
            parts: to_u32(parts)?,
            direct_output: 0,
        };
        let windows = (0..if layout.rings == 0 { 0 } else { layout.chunks })
            .step_by(parts)
            .map(|first| {
                Ok(CommitParams {
                    first_chunk: to_u32(first)?,
                    parts: to_u32(parts.min(layout.chunks - first))?,
                    ..params
                })
            })
            .collect::<Result<Vec<_>, AkitaMetalError>>()?;
        if self.buffers.as_ref().is_none_or(|buffers| {
            buffers.spectra.len() != spectrum_len
                || buffers.residues.len() != residue_len
                || buffers.partials.len() != partial_len
                || buffers.out.len() != layout.len
        }) {
            self.buffers = None;
            self.buffers = Some(Buffers {
                spectra: DeviceBuffer::uninit(device, spectrum_len)?,
                residues: DeviceBuffer::uninit(device, residue_len)?,
                partials: DeviceBuffer::uninit(device, partial_len)?,
                out: DeviceBuffer::zeroed(device, layout.len)?,
            });
        }
        let buffers = self
            .buffers
            .as_mut()
            .ok_or_else(|| AkitaMetalError::Shape("shared-key workspace unavailable".into()))?;
        if layout.rings == 0 {
            return Ok((&mut buffers.out, Duration::ZERO));
        }
        let layouts = DeviceBuffer::from_slice(device, &layout.layouts)?;
        let source_map = DeviceBuffer::from_slice(device, &layout.source_map)?;
        let mut batch = Batch::new(device)?;
        for params in &windows {
            let active = params.parts as usize;
            // SAFETY: all source/key extents are validated by CommitLayout.
            // Producer writes every spectrum slot consumed by dot, including
            // absent/partial positions. Dot and inverse each write all active
            // outputs before the next consumer. Unused capacity is never read.
            // All 512 dot threads reach barriers; buffers/params live to wait.
            unsafe {
                batch.dispatch_unchecked(
                    source,
                    &[
                        Binding::buffer(&sources.hot),
                        Binding::buffer(&sources.ring_kinds),
                        Binding::buffer(&matrix.tables),
                        Binding::buffer(&layouts),
                        Binding::buffer(&source_map),
                        Binding::buffer(&buffers.spectra),
                        Binding::value(params),
                    ],
                    Grid::linear(
                        active * matrix.reconstruction_positions * layout.rings * 32,
                        SOURCE_THREADS,
                    ),
                )?;
                batch.dispatch_unchecked(
                    dot,
                    &[
                        Binding::buffer(&buffers.spectra),
                        Binding::buffer(&matrix.transformed),
                        Binding::buffer(&buffers.residues),
                        Binding::value(params),
                    ],
                    Grid::linear(
                        active * (D / FREQUENCY_TILE) * output_tiles * DOT_THREADS,
                        DOT_THREADS,
                    ),
                )?;
                batch.dispatch_unchecked(
                    inverse,
                    &[
                        Binding::buffer(&buffers.residues),
                        Binding::buffer(&matrix.tables),
                        Binding::buffer(&buffers.partials),
                        Binding::value(params),
                    ],
                    Grid::linear(active * layout.rings * INVERSE_THREADS, INVERSE_THREADS),
                )?;
                batch.dispatch_unchecked(
                    combine,
                    &[
                        Binding::buffer(&buffers.partials),
                        Binding::buffer(&buffers.out),
                        Binding::value(params),
                    ],
                    Grid::linear(layout.len, INVERSE_THREADS),
                )?;
            }
        }
        let elapsed = batch.commit_and_wait()?;
        Ok((&mut buffers.out, elapsed))
    }
}
