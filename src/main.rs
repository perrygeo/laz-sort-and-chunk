use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::{collections::HashMap, path::PathBuf};

use indicatif::ProgressBar;
use las::{Bounds, Header, Point, Read, Reader, Write, Writer};
use ndarray::prelude::*;

/// Epsilon to deal with literal edge cases
const EPS: f64 = 1e-9;

/// Normalize coordinates
/// from native x, y, z coordinates
/// to 0..1 floating points
fn normalize_coords(point: &Point, bounds: Bounds) -> (f64, f64, f64) {
    let nx = (point.x - bounds.min.x) / (bounds.max.x + EPS - bounds.min.x);
    assert!(nx >= 0.0);
    assert!(nx < 1.0);
    let ny = (point.y - bounds.min.y) / (bounds.max.y + EPS - bounds.min.y);
    assert!(ny >= 0.0);
    assert!(ny < 1.0);
    let nz = (point.z - bounds.min.z) / (bounds.max.z + EPS - bounds.min.z);
    assert!(nz >= 0.0);
    assert!(nz < 1.0);
    (nx, ny, nz)
}

fn grid_idx(coords: (f64, f64, f64), grid_size: usize) -> (usize, usize, usize) {
    (
        (grid_size as f64 * coords.0) as usize,
        (grid_size as f64 * coords.1) as usize,
        (grid_size as f64 * coords.2) as usize,
    )
}

fn downsample_sum(
    arr: &ndarray::Array<usize, ndarray::Dim<[usize; 3]>>,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 3]>> {
    let og_size = arr.shape();
    let new_size = og_size[0] / 2;
    let mut new_arr = Array::<usize, _>::zeros((new_size, new_size, new_size));

    for (new_idx, new_cell) in new_arr.indexed_iter_mut() {
        // calculate the indexes of the original data
        let x0 = new_idx.0 * 2;
        let x1 = (new_idx.0 + 1) * 2;
        let y0 = new_idx.1 * 2;
        let y1 = (new_idx.1 + 1) * 2;
        let z0 = new_idx.2 * 2;
        let z1 = (new_idx.2 + 1) * 2;
        let total = arr.slice(s![x0..x1, y0..y1, z0..z1,]).sum();
        *new_cell = total
    }

    assert_eq!(new_arr.sum(), arr.sum());
    new_arr
}

type ChunkId = (usize, usize, usize, usize);

struct ChunkWriter {
    outdir: String,
    header: Header,
    writers: HashMap<ChunkId, Writer<BufWriter<File>>>,
}

impl ChunkWriter {
    fn new(outdir: String, header: Header) -> Self {
        std::fs::create_dir_all(&outdir).expect("Output directory cannot be created");
        ChunkWriter {
            outdir,
            header,
            writers: HashMap::default(),
        }
    }

    fn path(&self, chunkid: &ChunkId) -> PathBuf {
        let mut outpath = Path::new(&self.outdir).to_path_buf();
        outpath.push(format!(
            "{}-{}-{}-{}.laz",
            chunkid.0, chunkid.1, chunkid.2, chunkid.3
        ));
        outpath
    }

    fn close(&mut self) -> Result<(), Box<dyn Error>> {
        for (_chunk, writer) in self.writers.iter_mut() {
            writer.close()?;
        }
        Ok(())
    }

    /// Write point to the associated chunk file
    fn send(&mut self, chunkid: ChunkId, point: Point) -> Result<(), Box<dyn Error>> {
        let outpath = self.path(&chunkid);
        // Look for existing laz writer
        match self.writers.get_mut(&chunkid) {
            Some(laz_writer) => {
                // write to it
                laz_writer.write(point)?;
            }
            None => {
                // println!("Creating {chunkid:?}");
                // create it
                let rheader = self.header.clone();
                let mut writer = Writer::from_path(outpath, rheader)?;
                // write to it
                writer.write(point)?;
                // cache the open writer
                self.writers.insert(chunkid, writer);
            }
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_path(
        // "tests/data/USGS_LPC_CO_SoPlatteRiver_Lot2a_2013_13TDE484485_LAS_2015.laz",
        "tests/data/red-rocks.laz",
    )?;
    let outdir = "/tmp/out";

    let grid_size = 256;
    let n_levels = (grid_size as f64).log2() as usize;
    let max_points_per_chunk = 1_000_000;

    let header = reader.header().clone();
    let bounds = header.bounds();
    let count = header.number_of_points() as usize;
    let mut counting_grid = Array::<usize, _>::zeros((grid_size, grid_size, grid_size));

    if (count / max_points_per_chunk) > 128 {
        // If this is set too low relative to the input points, you might see > 1024 chunks open
        // Error: Io(Os { code: 24, kind: Uncategorized, message: "Too many open files" })
        // late in the process, that is not a fun surprise, better catch it here.
        panic!("max_points_per_chunk is too low");
    }

    eprintln!("First pass...");
    let bar = ProgressBar::new(count as u64);
    for wrapped_point in reader.points() {
        let point = wrapped_point?;
        assert!(point.matches(header.point_format()));
        let norm_coords = normalize_coords(&point, bounds);
        let idx = grid_idx(norm_coords, grid_size);
        counting_grid[[idx.0, idx.1, idx.2]] += 1;
        bar.inc(1);
    }
    assert_eq!(counting_grid.sum(), count);
    bar.finish();

    eprintln!("Create pyramids");
    let mut levels = vec![counting_grid];
    for ogl in 0..n_levels {
        levels.push(downsample_sum(&levels[ogl]));
    }
    levels.reverse(); // level 0 == lowest resolution

    eprintln!("Second pass...");
    reader.seek(0)?;
    let mut chunk_writer = ChunkWriter::new(outdir.into(), header.clone());
    let bar = ProgressBar::new(count as u64);
    for wrapped_point in reader.points() {
        let point = wrapped_point?;
        let norm_coords = normalize_coords(&point, bounds);
        let chunkidx = {
            // Query pyramids to find: L-X-Y-Z chunk
            let mut chunkidx = (0, 0, 0, 0);
            for (l, level) in levels.iter().enumerate() {
                let grid_size = level.shape()[0];
                let nidx = grid_idx(norm_coords, grid_size);
                let val = level[[nidx.0, nidx.1, nidx.2]];
                if val < max_points_per_chunk {
                    chunkidx = (l, nidx.0, nidx.1, nidx.2);
                    break;
                }
            }
            assert_ne!(chunkidx, (0, 0, 0, 0));
            chunkidx
        };

        chunk_writer.send(chunkidx, point)?;
        bar.inc(1);
    }
    bar.finish();

    eprintln!("Closing writers");
    chunk_writer.close()?;

    eprintln!("Validate chunks...");
    let n_chunks = chunk_writer.writers.len();
    let mut total_chunked = 0;
    let bar = ProgressBar::new(n_chunks as u64);
    for (chunk, _writer) in chunk_writer.writers.iter() {
        let path = chunk_writer.path(chunk);
        let r = Reader::from_path(path)?;
        let count = r.header().number_of_points() as usize;
        // Each chunk file should be below the max
        assert!(count < max_points_per_chunk);
        total_chunked += count;
        bar.inc(1);
    }
    // All points accounted for.
    assert_eq!(total_chunked, count);
    bar.finish();
    eprintln!("Outputs: {}/*.laz", chunk_writer.outdir);

    Ok(())
}
