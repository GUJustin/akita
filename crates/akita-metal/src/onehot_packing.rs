//! Base-field coefficient packing over retained K16/D512 trace selectors.

use std::time::Duration;

use akita_error::checked;
use bytemuck::{Pod, Zeroable};
use jolt_field::Prime128OffsetA7F7 as F;
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid};

use crate::library::Instance;
use crate::onehot::{to_u32, to_u64, DeviceK16Sources};
use crate::{AkitaMetal, AkitaMetalError};

const D: usize = 512;
const TILE: usize = 1024;
const THREADS: usize = 256;
const MAX_BUFFER_BYTES: usize = 64 << 20;
const GATHER: &str = "akita_onehot_packing_gather_fp128";
const COMBINE: &str = "akita_onehot_packing_combine_fp128";

pub(crate) fn instances() -> [Instance; 2] {
    [
        ("akita_onehot_packing_gather", GATHER),
        ("akita_onehot_packing_combine", COMBINE),
    ]
    .map(|(template, host)| Instance {
        template,
        args: format!("{}u", F::C),
        host_name: host.into(),
    })
}

/// Output order is column / block / subring coefficient. The caller owns the
/// prepared opening point and supplies its canonical position/packing weights.
#[derive(Clone, Copy, Debug)]
pub struct OneHotPackingShape {
    pub num_positions: usize,
    pub blocks_per_column: usize,
    pub num_columns: usize,
    pub subring_dimension: usize,
}

impl OneHotPackingShape {
    /// Arithmetic and memory admission, without allocations or GPU work.
    pub fn is_supported(self) -> bool {
        self.num_positions.is_power_of_two()
            && u32::try_from(self.num_positions).is_ok()
            && self.blocks_per_column > 0
            && u32::try_from(self.blocks_per_column).is_ok()
            && (1..=64).contains(&self.num_columns)
            && self.subring_dimension.is_power_of_two()
            && self.subring_dimension <= D
            && [
                Some(self.num_positions),
                checked::product([self.num_columns, self.num_positions.div_ceil(TILE), D]),
                checked::product([
                    self.num_columns,
                    self.blocks_per_column,
                    self.subring_dimension,
                ]),
            ]
            .into_iter()
            .all(|len| {
                len.and_then(|len| len.checked_mul(size_of::<F>()))
                    .is_some_and(|bytes| bytes <= MAX_BUFFER_BYTES)
            })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackingParams {
    prefix_rows: u64,
    positions: u32,
    columns: u32,
    blocks_per_column: u32,
    subring: u32,
    tiles: u32,
    local_block: u32,
    output_block: u32,
    logical_rings: u32,
}

/// One bounded block of tile partials, reused across consecutive windows.
/// Source buffers remain borrowed so later opening operations can reuse them.
pub struct OneHotPackingWorkspace<'a> {
    metal: &'a AkitaMetal,
    shape: OneHotPackingShape,
    positions: DeviceBuffer<F>,
    packing: DeviceBuffer<F>,
    partials: DeviceBuffer<F>,
    output: DeviceBuffer<F>,
    tiles: usize,
    partial_len: usize,
    combine_len: usize,
    next_block: usize,
}

impl<'a> OneHotPackingWorkspace<'a> {
    pub fn new(
        metal: &'a AkitaMetal,
        shape: OneHotPackingShape,
        position_weights: &[F],
        packing_weights: &[F],
    ) -> Result<Self, AkitaMetalError> {
        if !shape.is_supported()
            || position_weights.len() != shape.num_positions
            || checked::product([packing_weights.len(), shape.subring_dimension]) != Some(D)
        {
            return Err(shape_error("unsupported coefficient-packing geometry"));
        }
        for name in [GATHER, COMBINE] {
            if metal.pipeline(name)?.max_total_threads_per_threadgroup() < THREADS {
                return Err(shape_error(
                    "coefficient packing requires 256 threads per group",
                ));
            }
        }
        let tiles = shape.num_positions.div_ceil(TILE);
        let partial_len = checked::product([shape.num_columns, tiles, D])
            .ok_or_else(|| shape_error("packing partial size overflows"))?;
        let combine_len = checked::product([shape.num_columns, shape.subring_dimension])
            .ok_or_else(|| shape_error("packing block size overflows"))?;
        let output_len = checked::product([combine_len, shape.blocks_per_column])
            .ok_or_else(|| shape_error("packing output size overflows"))?;
        Ok(Self {
            metal,
            shape,
            positions: DeviceBuffer::from_slice(metal.device(), position_weights)?,
            packing: DeviceBuffer::from_slice(metal.device(), packing_weights)?,
            partials: DeviceBuffer::zeroed(metal.device(), partial_len)?,
            output: DeviceBuffer::zeroed(metal.device(), output_len)?,
            tiles,
            partial_len,
            combine_len,
            next_block: 0,
        })
    }

    /// Writes the next window's block partials. Missing stored rows contribute
    /// zero, and unvisited trailing logical blocks keep their initialized zero.
    pub fn accumulate(
        &mut self,
        sources: &DeviceK16Sources,
        first_block: usize,
    ) -> Result<Duration, AkitaMetalError> {
        let source = sources
            .sources
            .first()
            .ok_or_else(|| shape_error("packing has no source"))?;
        let rings = checked::exact_div(source.fields, D)
            .ok_or_else(|| shape_error("packing sources require whole rings"))?;
        let blocks = rings.div_ceil(self.shape.num_positions);
        let end = first_block
            .checked_add(blocks)
            .ok_or_else(|| shape_error("packing window end overflows"))?;
        if first_block != self.next_block
            || blocks == 0
            || end > self.shape.blocks_per_column
            || sources.sources.len() != self.shape.num_columns
            || sources.sources.iter().enumerate().any(|(column, layout)| {
                layout.fields != source.fields
                    || layout.chunks != source.chunks
                    || layout.hot_offset != column
                    || layout.hot_stride != self.shape.num_columns
            })
            || (!rings.is_multiple_of(self.shape.num_positions)
                && end != self.shape.blocks_per_column)
        {
            return Err(shape_error("packing source window disagrees with geometry"));
        }
        let base_params = PackingParams {
            prefix_rows: to_u64(source.chunks)?,
            positions: to_u32(self.shape.num_positions)?,
            columns: to_u32(self.shape.num_columns)?,
            blocks_per_column: to_u32(self.shape.blocks_per_column)?,
            subring: to_u32(self.shape.subring_dimension)?,
            tiles: to_u32(self.tiles)?,
            local_block: 0,
            output_block: 0,
            logical_rings: to_u32(rings)?,
        };
        // Binding borrows include inline values until command completion.
        let params = (0..blocks)
            .map(|block| {
                Ok(PackingParams {
                    local_block: to_u32(block)?,
                    output_block: to_u32(
                        first_block
                            .checked_add(block)
                            .ok_or_else(|| shape_error("packing output block overflows"))?,
                    )?,
                    ..base_params
                })
            })
            .collect::<Result<Vec<_>, AkitaMetalError>>()?;
        let mut batch = Batch::new(self.metal.device())?;
        for params in &params {
            // SAFETY: complete gather writes precede each combine. A lane owns
            // one tile/coefficient or column/subring coefficient; both kernels
            // guard rounded grids. Checked row-major layouts and prefix/ring
            // bounds cover every selector load. Dispatch ordering permits reuse
            // of the one-block partial buffer; output blocks are disjoint.
            unsafe {
                batch.dispatch_unchecked(
                    self.metal.pipeline(GATHER)?,
                    &[
                        Binding::buffer(&sources.hot),
                        Binding::buffer(&self.positions),
                        Binding::buffer(&self.partials),
                        Binding::value(params),
                    ],
                    Grid::linear(self.partial_len, THREADS),
                )?;
                batch.dispatch_unchecked(
                    self.metal.pipeline(COMBINE)?,
                    &[
                        Binding::buffer(&self.partials),
                        Binding::buffer(&self.packing),
                        Binding::buffer(&self.output),
                        Binding::value(params),
                    ],
                    Grid::linear(self.combine_len, THREADS),
                )?;
            }
        }
        let duration = batch.commit_and_wait()?;
        self.next_block = end;
        Ok(duration)
    }

    pub fn read(&mut self) -> Result<Vec<F>, AkitaMetalError> {
        Ok(self.output.read()?.to_vec())
    }
}

fn shape_error(message: &str) -> AkitaMetalError {
    AkitaMetalError::Shape(message.into())
}
