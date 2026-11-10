//! Compact row-major K16 sources and their commitment geometry.

use std::mem::MaybeUninit;

use akita_error::checked;
use jolt_metal::runtime::DeviceBuffer;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{AkitaMetal, AkitaMetalError};

#[derive(Clone, Copy, Debug)]
pub(crate) struct SourceLayout {
    pub(crate) hot_offset: usize,
    pub(crate) hot_stride: usize,
    pub(crate) chunks: usize,
    pub(crate) fields: usize,
}

/// One byte per stored selector: 0..15 means present, 255 means absent.
/// Each column keeps the full logical domain even when its suffix is omitted.
pub struct DeviceK16Sources {
    pub(crate) hot: DeviceBuffer<u8>,
    /// Per ring and column: 0..15 uniform, 254 mixed, 255 absent.
    pub(crate) ring_kinds: DeviceBuffer<u8>,
    pub(crate) sources: Vec<SourceLayout>,
}

// Aligned to complete 32-row rings; at 64 columns each active worker needs
// at most 512 KiB selector scratch plus 64 KiB masks. No full-window host mirror.
const PRODUCER_ROWS: usize = 8192;

#[derive(Clone, Copy, Default)]
struct RingCounts {
    uniform: usize,
    mixed: usize,
    absent: usize,
}

impl RingCounts {
    fn merge(self, other: Self) -> Result<Self, AkitaMetalError> {
        let uniform = checked::sum([self.uniform, other.uniform]);
        let mixed = checked::sum([self.mixed, other.mixed]);
        let absent = checked::sum([self.absent, other.absent]);
        match (uniform, mixed, absent) {
            (Some(uniform), Some(mixed), Some(absent)) => Ok(Self {
                uniform,
                mixed,
                absent,
            }),
            _ => Err(shape_error("ring classification count overflows")),
        }
    }
}

struct ProducerScratch {
    selected: Vec<u8>,
    masks: Vec<u64>,
}

struct ProducerChunk<'a> {
    index: usize,
    hot: &'a mut [MaybeUninit<u8>],
    kinds: &'a mut [MaybeUninit<u8>],
}

/// Canonical zero-mask normalization and ring classification for both source
/// constructors. Every successful call initializes its complete destinations.
fn encode_ring(
    selected: &[u8],
    masks: &[u64],
    columns: usize,
    output: &mut [MaybeUninit<u8>],
    ring_kinds: &mut [MaybeUninit<u8>],
) -> Result<RingCounts, AkitaMetalError> {
    let mut kinds = [255u8; 64];
    let kinds = kinds
        .get_mut(..columns)
        .ok_or_else(|| shape_error("ring classification column count"))?;
    for (row, ((selectors, &mask), output)) in selected
        .chunks_exact(columns)
        .zip(masks)
        .zip(output.chunks_exact_mut(columns))
        .enumerate()
    {
        for (column, ((&selector, output), kind)) in selectors
            .iter()
            .zip(output)
            .zip(kinds.iter_mut())
            .enumerate()
        {
            if selector >= 16 {
                return Err(shape_error("a K16 selector is outside 0..16"));
            }
            let encoded = if selector == 0 && mask & (1u64 << column) == 0 {
                255
            } else {
                selector
            };
            let _ = output.write(encoded);
            if row == 0 {
                *kind = encoded;
            } else if *kind != encoded {
                *kind = 254;
            }
        }
    }
    // Missing physical rows are absent, not copies of a uniform live digit.
    if masks.len() < 32 {
        for kind in kinds.iter_mut().filter(|kind| **kind < 16) {
            *kind = 254;
        }
    }
    let counts = RingCounts {
        uniform: kinds.iter().filter(|&&kind| kind < 16).count(),
        mixed: kinds.iter().filter(|&&kind| kind == 254).count(),
        absent: kinds.iter().filter(|&&kind| kind == 255).count(),
    };
    for (slot, &kind) in ring_kinds.iter_mut().zip(kinds.iter()) {
        let _ = slot.write(kind);
    }
    Ok(counts)
}

impl DeviceK16Sources {
    /// Validates an existing row-major prefix. Positive selectors are present;
    /// zero is present only when its mask bit is set. The producer constructor
    /// is preferable when the caller can fill bounded chunks directly.
    pub fn from_row_major(
        metal: &AkitaMetal,
        selected_rows: Vec<u8>,
        masks: &[u64],
        columns: usize,
        logical_rows: usize,
    ) -> Result<Self, AkitaMetalError> {
        if checked::product([masks.len(), columns]) != Some(selected_rows.len()) {
            return Err(shape_error("selector prefix and masks disagree"));
        }
        Self::from_row_major_producer(
            metal,
            columns,
            masks.len(),
            logical_rows,
            |first, selected, output_masks| {
                let start = checked::product([first, columns])
                    .ok_or_else(|| shape_error("selector start overflows"))?;
                let end = checked::sum([start, selected.len()])
                    .ok_or_else(|| shape_error("selector end overflows"))?;
                let mask_end = checked::sum([first, output_masks.len()])
                    .ok_or_else(|| shape_error("mask end overflows"))?;
                selected.copy_from_slice(
                    selected_rows
                        .get(start..end)
                        .ok_or_else(|| shape_error("selector range exceeds prefix"))?,
                );
                output_masks.copy_from_slice(
                    masks
                        .get(first..mask_end)
                        .ok_or_else(|| shape_error("mask range exceeds prefix"))?,
                );
                Ok(())
            },
        )
    }

    /// Produces final shared-storage buffers through bounded host scratch.
    ///
    /// `fill` receives a local starting row and must fill every selector and
    /// mask in its slices, using raw selectors 0..15. Calls cover the stored
    /// prefix once, may run concurrently/out of order, and never exceed 8192
    /// rows. Logical suffix rows remain absent. An error joins all workers and
    /// drops partially initialized buffers without publishing or reading them.
    /// Requires jolt-metal's typed uninit/write_uninit/assume_init runtime API.
    pub fn from_row_major_producer(
        metal: &AkitaMetal,
        columns: usize,
        stored_rows: usize,
        logical_rows: usize,
        fill: impl Fn(usize, &mut [u8], &mut [u64]) -> Result<(), AkitaMetalError> + Send + Sync,
    ) -> Result<Self, AkitaMetalError> {
        let _span =
            tracing::info_span!("metal_onehot_source_prepare", stored_rows, columns).entered();
        if !(1..=64).contains(&columns) || stored_rows > logical_rows {
            return Err(shape_error(
                "row-major source prefix/columns disagree with geometry",
            ));
        }
        let fields = checked::product([logical_rows, 16])
            .ok_or_else(|| shape_error("one-hot logical domain overflows"))?;
        let hot_len = checked::product([stored_rows, columns])
            .ok_or_else(|| shape_error("selector size overflows"))?;
        let ring_len = checked::product([stored_rows.div_ceil(32), columns])
            .ok_or_else(|| shape_error("ring classification size overflows"))?;
        let tile_bytes = checked::product([PRODUCER_ROWS, columns])
            .ok_or_else(|| shape_error("producer tile size overflows"))?;
        let ring_tile_bytes = checked::product([PRODUCER_ROWS / 32, columns])
            .ok_or_else(|| shape_error("producer ring tile size overflows"))?;
        let one_ring_bytes =
            checked::product([32, columns]).ok_or_else(|| shape_error("ring size overflows"))?;
        let allocation =
            tracing::info_span!("metal_onehot_source_allocate", hot_len, ring_len).entered();
        let mut hot = DeviceBuffer::<MaybeUninit<u8>>::uninit(metal.device(), hot_len)?;
        let mut ring_kinds = DeviceBuffer::<MaybeUninit<u8>>::uninit(metal.device(), ring_len)?;
        drop(allocation);
        let filling = tracing::info_span!("metal_onehot_source_fill").entered();
        let hot_slots = hot.write_uninit()?;
        let kind_slots = ring_kinds.write_uninit()?;
        let scratch = || ProducerScratch {
            selected: vec![0u8; tile_bytes],
            masks: vec![0u64; PRODUCER_ROWS],
        };
        let produce = |scratch: &mut ProducerScratch, task: ProducerChunk<'_>| {
            let ProducerChunk {
                index: chunk,
                hot,
                kinds,
            } = task;
            let rows = checked::exact_div(hot.len(), columns)
                .ok_or_else(|| shape_error("producer selector rows are not aligned"))?;
            let first = checked::product([chunk, PRODUCER_ROWS])
                .ok_or_else(|| shape_error("producer row offset overflows"))?;
            let selected = scratch
                .selected
                .get_mut(..hot.len())
                .ok_or_else(|| shape_error("producer selector scratch too short"))?;
            let masks = scratch
                .masks
                .get_mut(..rows)
                .ok_or_else(|| shape_error("producer mask scratch too short"))?;
            fill(first, selected, masks)?;
            selected
                .chunks(one_ring_bytes)
                .zip(masks.chunks(32))
                .zip(hot.chunks_mut(one_ring_bytes))
                .zip(kinds.chunks_mut(columns))
                .try_fold(
                    RingCounts::default(),
                    |counts, (((selected, masks), hot), kinds)| {
                        counts.merge(encode_ring(selected, masks, columns, hot, kinds)?)
                    },
                )
        };
        #[cfg(feature = "parallel")]
        let counts = hot_slots
            .par_chunks_mut(tile_bytes)
            .zip(kind_slots.par_chunks_mut(ring_tile_bytes))
            .enumerate()
            .map(|(index, (hot, kinds))| ProducerChunk { index, hot, kinds })
            .map_init(scratch, produce)
            .try_reduce(RingCounts::default, RingCounts::merge)?;
        #[cfg(not(feature = "parallel"))]
        let counts = {
            let mut scratch = scratch();
            hot_slots
                .chunks_mut(tile_bytes)
                .zip(kind_slots.chunks_mut(ring_tile_bytes))
                .enumerate()
                .map(|(index, (hot, kinds))| ProducerChunk { index, hot, kinds })
                .try_fold(RingCounts::default(), |counts, task| {
                    counts.merge(produce(&mut scratch, task)?)
                })?
        };
        drop(filling);
        tracing::info!(
            uniform_rings = counts.uniform,
            mixed_rings = counts.mixed,
            absent_rings = counts.absent,
            "Metal one-hot ring classes"
        );
        // SAFETY: aligned chunks partition both exact buffer extents. Each
        // successful encoder writes every byte in its slices, including the
        // final partial ring. All workers have joined; failures returned before
        // this point. Empty buffers require no writes. No GPU work has run.
        let hot = unsafe { hot.assume_init()? };
        let ring_kinds = unsafe { ring_kinds.assume_init()? };
        let sources = (0..columns)
            .map(|column| SourceLayout {
                hot_offset: column,
                hot_stride: columns,
                chunks: stored_rows,
                fields,
            })
            .collect();
        Ok(Self {
            hot,
            ring_kinds,
            sources,
        })
    }
}

/// The matrix view selected by the canonical inner commitment plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneHotCommitShape {
    pub n_a: usize,
    pub active_a_cols: usize,
    pub num_digits_inner: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) digits: usize,
    pub(crate) positions: usize,
}

impl Geometry {
    pub(crate) fn new(shape: OneHotCommitShape) -> Result<Self, AkitaMetalError> {
        let positions = checked::exact_div(shape.active_a_cols, shape.num_digits_inner)
            .ok_or_else(|| shape_error("A width must be a multiple of a positive digit count"))?;
        if !positions.is_power_of_two() {
            return Err(shape_error("positions per block must be a power of two"));
        }
        Ok(Self {
            rows: shape.n_a,
            columns: shape.active_a_cols,
            digits: shape.num_digits_inner,
            positions,
        })
    }

    pub(crate) fn blocks<const D: usize>(
        &self,
        source: &SourceLayout,
    ) -> Result<usize, AkitaMetalError> {
        let rings = checked::exact_div(source.fields, D)
            .ok_or_else(|| shape_error("logical domain must contain complete commitment rings"))?;
        checked::div_ceil(rings, self.positions).ok_or_else(|| shape_error("block count overflows"))
    }
}

fn shape_error(reason: impl Into<String>) -> AkitaMetalError {
    AkitaMetalError::Shape(reason.into())
}

pub(crate) fn to_u32(value: usize) -> Result<u32, AkitaMetalError> {
    u32::try_from(value).map_err(|_| shape_error(format!("{value} exceeds 32-bit indexing")))
}

pub(crate) fn to_u64(value: usize) -> Result<u64, AkitaMetalError> {
    u64::try_from(value).map_err(|_| shape_error(format!("{value} exceeds 64-bit indexing")))
}
