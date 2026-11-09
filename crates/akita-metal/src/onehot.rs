//! Compact row-major K16 sources and their commitment geometry.

use akita_error::checked;
use jolt_metal::runtime::DeviceBuffer;

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

impl DeviceK16Sources {
    /// Validates and uploads a shared row-major prefix without transposing it.
    ///
    /// `selected_rows` contains `masks.len()` rows of `columns` selectors.
    /// Positive selectors are present. A zero selector is present only when
    /// its column's bit is set in that row's mask. Rows beyond the stored
    /// prefix are absent; `logical_rows` still determines output dimensions.
    pub fn from_row_major(
        metal: &AkitaMetal,
        mut selected_rows: Vec<u8>,
        masks: &[u64],
        columns: usize,
        logical_rows: usize,
    ) -> Result<Self, AkitaMetalError> {
        if !(1..=64).contains(&columns) {
            return Err(shape_error("row-major sources require 1..=64 columns"));
        }
        if checked::product([masks.len(), columns]) != Some(selected_rows.len())
            || masks.len() > logical_rows
        {
            return Err(shape_error(
                "selector prefix, masks and logical rows disagree",
            ));
        }
        let fields = checked::product([logical_rows, 16])
            .ok_or_else(|| shape_error("one-hot logical domain overflows"))?;
        let ring_len = checked::product([masks.len().div_ceil(32), columns])
            .ok_or_else(|| shape_error("ring classification size overflows"))?;
        let row_tile = checked::product([32, columns])
            .ok_or_else(|| shape_error("selector tile size overflows"))?;
        let mut ring_kinds = vec![255; ring_len];
        let encode = |((tile, masks), kinds): ((&mut [u8], &[u64]), &mut [u8])| {
            for (row, (selectors, &mask)) in tile.chunks_exact_mut(columns).zip(masks).enumerate() {
                for (column, (selector, kind)) in
                    selectors.iter_mut().zip(kinds.iter_mut()).enumerate()
                {
                    if *selector >= 16 {
                        return Err(shape_error("a K16 selector is outside 0..16"));
                    }
                    if *selector == 0 && mask & (1u64 << column) == 0 {
                        *selector = 255;
                    }
                    if row == 0 {
                        *kind = *selector;
                    } else if *kind != *selector {
                        *kind = 254;
                    }
                }
            }
            // An omitted row is absent, not another copy of a live selector.
            if masks.len() < 32 {
                for kind in kinds.iter_mut().filter(|kind| **kind < 16) {
                    *kind = 254;
                }
            }
            Ok(())
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            selected_rows
                .par_chunks_mut(row_tile)
                .zip(masks.par_chunks(32))
                .zip(ring_kinds.par_chunks_mut(columns))
                .try_for_each(encode)?;
        }
        #[cfg(not(feature = "parallel"))]
        selected_rows
            .chunks_mut(row_tile)
            .zip(masks.chunks(32))
            .zip(ring_kinds.chunks_mut(columns))
            .try_for_each(encode)?;
        tracing::info!(
            uniform_rings = ring_kinds.iter().filter(|&&kind| kind < 16).count(),
            mixed_rings = ring_kinds.iter().filter(|&&kind| kind == 254).count(),
            absent_rings = ring_kinds.iter().filter(|&&kind| kind == 255).count(),
            "Metal one-hot ring classes"
        );
        let sources = (0..columns)
            .map(|column| SourceLayout {
                hot_offset: column,
                hot_stride: columns,
                chunks: masks.len(),
                fields,
            })
            .collect();
        Ok(Self {
            hot: DeviceBuffer::from_slice(metal.device(), &selected_rows)?,
            ring_kinds: DeviceBuffer::from_slice(metal.device(), &ring_kinds)?,
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
