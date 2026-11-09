//! Integer root folding of K16/D512 sources by sparse ring challenges.

use std::{ops::Range, time::Duration};

use akita_error::checked;
use bytemuck::{Pod, Zeroable};
use jolt_metal::runtime::{Batch, Binding, DeviceBuffer, Grid};

use crate::onehot::{to_u32, to_u64, DeviceK16Sources};
use crate::{AkitaMetal, AkitaMetalError};

/// Borrowed terms of a degree-512 challenge polynomial. Repeated positions
/// add their coefficients; sampler-family validation stays with the caller.
#[derive(Clone, Copy, Debug)]
pub struct OneHotFoldChallenge<'a> {
    pub positions: &'a [u32],
    pub coefficients: &'a [i8],
}

pub(crate) const FOLD: &str = "akita_k16_sparse_fold";
const D: usize = 512;
const THREADS: usize = 256;

/// Logical geometry shared by every column; chunk ranges partition its blocks.
#[derive(Clone, Copy, Debug)]
pub struct OneHotFoldShape {
    pub num_positions: usize,
    pub blocks_per_column: usize,
    pub num_columns: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FoldParams {
    prefix_rows: u64,
    positions: u32,
    columns: u32,
    blocks_per_column: u32,
    first_block: u32,
    chunks: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SparseTerm {
    position: u16,
    coefficient: i16,
}

/// Owns one compressed integer output, reused across consecutive trace windows.
/// Challenges are sparse degree-512 rings in column/block order, without an
/// embedding restriction. Output is position/chunk/coefficient.
pub struct OneHotFoldWorkspace<'a> {
    metal: &'a AkitaMetal,
    shape: OneHotFoldShape,
    ranges: Vec<Range<usize>>,
    terms: DeviceBuffer<SparseTerm>,
    offsets: DeviceBuffer<u32>,
    output: DeviceBuffer<i32>,
    next_block: usize,
}

impl<'a> OneHotFoldWorkspace<'a> {
    pub fn new(
        metal: &'a AkitaMetal,
        shape: OneHotFoldShape,
        challenges: &[OneHotFoldChallenge<'_>],
        chunk_ranges: &[Range<usize>],
    ) -> Result<Self, AkitaMetalError> {
        let OneHotFoldShape {
            num_positions,
            blocks_per_column,
            num_columns,
        } = shape;
        let count = checked::product([num_columns, blocks_per_column])
            .ok_or_else(|| AkitaMetalError::Shape("fold challenge count overflows".into()))?;
        if num_positions == 0
            || blocks_per_column == 0
            || !(1..=64).contains(&num_columns)
            || count >= 1 << 26
            || count != challenges.len()
            || chunk_ranges.is_empty()
        {
            return Err(AkitaMetalError::Shape(
                "invalid sparse fold geometry".into(),
            ));
        }
        let mut terms = Vec::new();
        let mut offsets = Vec::with_capacity(challenges.len().saturating_add(1));
        offsets.push(0);
        for challenge in challenges {
            if challenge.positions.len() != challenge.coefficients.len()
                || challenge
                    .positions
                    .iter()
                    .any(|&position| position >= D as u32)
            {
                return Err(AkitaMetalError::Shape(
                    "invalid sparse challenge terms".into(),
                ));
            }
            for (&position, &coefficient) in challenge.positions.iter().zip(challenge.coefficients)
            {
                terms.push(SparseTerm {
                    position: u16::try_from(position).map_err(|_| {
                        AkitaMetalError::Shape("challenge index exceeds u16".into())
                    })?,
                    coefficient: i16::from(coefficient),
                });
            }
            offsets.push(to_u32(terms.len())?);
        }
        let mut end = 0;
        for range in chunk_ranges {
            if range.start != end || range.start > range.end || range.end > blocks_per_column {
                return Err(AkitaMetalError::Shape(
                    "fold chunks must partition the blocks".into(),
                ));
            }
            end = range.end;
            // Each challenge coefficient can contribute at most once to a
            // fixed output coefficient: one-hot row exponents are distinct.
            let bound = challenges
                .iter()
                .enumerate()
                .filter(|(index, _)| {
                    index
                        .checked_rem(blocks_per_column)
                        .is_some_and(|block| range.contains(&block))
                })
                .flat_map(|(_, challenge)| challenge.coefficients)
                .try_fold(0u32, |sum, value| {
                    sum.checked_add(u32::from(value.unsigned_abs()))
                });
            if bound.is_none_or(|value| value > i32::MAX as u32) {
                return Err(AkitaMetalError::Shape(
                    "fold coefficients exceed i32 accumulation".into(),
                ));
            }
        }
        if end != blocks_per_column {
            return Err(AkitaMetalError::Shape(
                "fold chunks omit logical blocks".into(),
            ));
        }
        let len = checked::product([num_positions, chunk_ranges.len(), D])
            .ok_or_else(|| AkitaMetalError::Shape("fold output size overflows".into()))?;
        to_u32(len)?;
        to_u32(num_positions)?;
        to_u32(blocks_per_column)?;
        to_u32(chunk_ranges.len())?;
        let pipeline = metal.pipeline(FOLD)?;
        if pipeline.thread_execution_width() != 32
            || pipeline.max_total_threads_per_threadgroup() < THREADS
        {
            return Err(AkitaMetalError::Shape(
                "root fold requires SIMD32 and 256 threads per group".into(),
            ));
        }
        Ok(Self {
            metal,
            shape,
            ranges: chunk_ranges.to_vec(),
            terms: DeviceBuffer::from_slice(metal.device(), &terms)?,
            offsets: DeviceBuffer::from_slice(metal.device(), &offsets)?,
            output: DeviceBuffer::zeroed(metal.device(), len)?,
            next_block: 0,
        })
    }

    /// Adds the next whole-block window, whose stored suffix may be omitted.
    /// Consecutive windows prevent double counting and preserve the checked
    /// integer bound. The final window may contain an incomplete block.
    pub fn accumulate(
        &mut self,
        sources: &DeviceK16Sources,
        first_block: usize,
    ) -> Result<Duration, AkitaMetalError> {
        let source = sources
            .sources
            .first()
            .ok_or_else(|| AkitaMetalError::Shape("fold has no source columns".into()))?;
        let rings = checked::exact_div(source.fields, D)
            .ok_or_else(|| AkitaMetalError::Shape("fold sources require whole rings".into()))?;
        let blocks = checked::div_ceil(rings, self.shape.num_positions)
            .ok_or_else(|| AkitaMetalError::Shape("fold window size overflows".into()))?;
        let end = checked::sum([first_block, blocks])
            .ok_or_else(|| AkitaMetalError::Shape("fold window end overflows".into()))?;
        if first_block != self.next_block
            || end > self.shape.blocks_per_column
            || blocks == 0
            || sources.sources.len() != self.shape.num_columns
            || sources
                .sources
                .iter()
                .any(|column| column.fields != source.fields || column.chunks != source.chunks)
            || (!rings.is_multiple_of(self.shape.num_positions)
                && end != self.shape.blocks_per_column)
        {
            return Err(AkitaMetalError::Shape(
                "fold source window disagrees with geometry".into(),
            ));
        }
        let ranges = self
            .ranges
            .iter()
            .map(|range| {
                let start = range.start.max(first_block).min(end);
                let end = range.end.min(end).max(start);
                Ok([to_u32(start)?, to_u32(end)?])
            })
            .collect::<Result<Vec<_>, AkitaMetalError>>()?;
        let params = FoldParams {
            prefix_rows: to_u64(source.chunks)?,
            positions: to_u32(self.shape.num_positions)?,
            columns: to_u32(self.shape.num_columns)?,
            blocks_per_column: to_u32(self.shape.blocks_per_column)?,
            first_block: to_u32(first_block)?,
            chunks: to_u32(self.ranges.len())?,
            padding: 0,
        };
        let grid = checked::product([self.shape.num_positions, self.ranges.len(), THREADS])
            .ok_or_else(|| AkitaMetalError::Shape("fold grid size overflows".into()))?;
        let device_ranges = DeviceBuffer::from_slice(self.metal.device(), &ranges)?;
        let mut batch = Batch::new(self.metal.device())?;
        // SAFETY: each group owns one position/chunk pair and writes exactly
        // 512 coefficients. Prefix checks guard every selector load. The
        // constructor bounds challenge indices and every integer partial sum;
        // consecutive windows preserve that bound across completed dispatches.
        unsafe {
            batch.dispatch_unchecked(
                self.metal.pipeline(FOLD)?,
                &[
                    Binding::buffer(&sources.hot),
                    Binding::buffer(&self.terms),
                    Binding::buffer(&self.output),
                    Binding::value(&params),
                    Binding::buffer(&device_ranges),
                    Binding::buffer(&self.offsets),
                ],
                Grid::linear(grid, THREADS),
            )?;
        }
        let duration = batch.commit_and_wait()?;
        self.next_block = end;
        Ok(duration)
    }

    pub fn read(&mut self) -> Result<Vec<i32>, AkitaMetalError> {
        Ok(self.output.read()?.to_vec())
    }
}
