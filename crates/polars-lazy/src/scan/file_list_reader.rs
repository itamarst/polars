use std::fmt::Display;
use std::fs::File;
use std::path::{PathBuf, Path};

use polars_core::error::to_compute_err;
use polars_core::prelude::*;
use polars_io::cloud::CloudOptions;
use polars_io::mmap::{MmapBytesReader, ReaderFactory};
use polars_io::{is_cloud_url, RowCount};

use crate::prelude::*;

pub type ReaderIterator = Box<dyn Iterator<Item = PolarsResult<ReaderFactory>>>;

// cloud_options is used only with async feature
#[allow(unused_variables)]
fn polars_glob(
    pattern: &str,
    cloud_options: Option<&CloudOptions>,
) -> PolarsResult<ReaderIterator> {
    if is_cloud_url(pattern) {
        #[cfg(feature = "async")]
        {
            let paths = polars_io::async_glob(pattern, cloud_options)?;
            Ok(Box::new(
                paths
                    .into_iter()
                    .map(|a| Ok(ReaderFactory::Remote { uri: a })),
            ))
        }
        #[cfg(not(feature = "async"))]
        panic!("Feature `async` must be enabled to use globbing patterns with cloud urls.")
    } else {
        let paths = glob::glob(pattern)
            .map_err(|_| polars_err!(ComputeError: "invalid glob pattern given"))?;
        let paths = paths.map(|v| {
            v.map_err(to_compute_err).and_then(|p| {
                File::open(p)
                    .map(|f| ReaderFactory::Local {
                        path: p,
                        reader: Box::new(f),
                    })
                    .map_err(to_compute_err)
            })
        });
        Ok(Box::new(paths))
    }
}

/// The source for one or more files (or remote files).
pub enum MultipleReaderFactories {
    /// A path that can be globbed to get multiple files or cloud URIs.
    GlobPath { path: PathBuf },
    /// Multiple specific file paths or cloud URIs.
    SpecificPaths { paths: Vec<PathBuf> },
    // TODO eventually also support things that aren't paths, e.g. Python
    // file-like objects.
}

impl Display for MultipleReaderFactories {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MultipleReaderFactories::GlobPath { path } => write!(f, "{}", path.to_string_lossy()),
            MultipleReaderFactories::SpecificPaths { paths } => {
                write!(f, "[")?;
                for path in paths {
                    write!(f, "{}, ", path.to_string_lossy())?;
                }
                write!(f, "]")
            },
        }
    }
}

impl MultipleReaderFactories {
    /// Create a new ReaderSources with a specific glob path.
    pub fn new_glob(path: impl AsRef<Path>) -> Self {
        Self::GlobPath { path: path.as_ref().to_owned() }
    }

    /// Create a new ReaderSources with a set of specific, non-glob paths.
    pub fn new_paths(paths: Vec<PathBuf>) -> Self {
        Self::SpecificPaths { paths }
    }

    /// Get list of specific reader factories.
    ///
    /// Returns [None] if path is not a glob pattern.
    fn iter_readers(&self, cloud_options: Option<&CloudOptions>) -> PolarsResult<ReaderIterator> {
        match self {
            MultipleReaderFactories::GlobPath { path } => {
                let path_str = path.to_string_lossy();
                polars_glob(&path_str, cloud_options)
            },
            MultipleReaderFactories::SpecificPaths { paths } => {
                Ok(Box::new(paths.clone().into_iter().map(|path| {
                    let path_str = path.to_string_lossy();
                    if is_cloud_url(path_str.as_ref()) {
                        Ok(ReaderFactory::Remote {
                            location: path_str.to_string(),
                        })
                    } else {
                        let f = File::open(path)?;
                        Ok(ReaderFactory::Local {
                            path,
                            reader: Box::new(f),
                        })
                    }
                })))
            },
        }
        // if paths.is_empty() {
        //     let path_str = self.path().to_string_lossy();
        //     if path_str.contains('*') || path_str.contains('?') || path_str.contains('[') {
        //         polars_glob(&path_str, self.cloud_options()).map(Some)
        //     } else {
        //         Ok(None)
        //     }
        // } else {
        //     polars_ensure!(self.path().to_string_lossy() == "", InvalidOperation: "expected only a single path argument");
        //     // Lint is incorrect as we need static lifetime.
        //     #[allow(clippy::unnecessary_to_owned)]
        //     Ok(Some(Box::new(paths.to_vec().into_iter().map(Ok))))
        // }
    }
}
/// A LazyFileListReader can be in one of two states: either it has a set of
/// sources (a filesystem glob, or a list of file paths/uris it needs to iterate
/// over), or it has a single, specific ReaderFactory. This isn't a great design
/// and there should really be separate structs for the two states, e.g. a
/// builder and a final state.
#[derive(Clone)]
pub(crate) enum SpecificOrMultipleReaderFactories {
    Specific { reader: Arc<ReaderFactory> },
    Multiple { sources: Arc<MultipleReaderFactories> },
}

/// Reads [LazyFrame] from a filesystem or a cloud storage.
/// Supports glob patterns.
///
/// Use [LazyFileListReader::finish] to get the final [LazyFrame].
pub trait LazyFileListReader: Clone {
    /// Get the final [LazyFrame].
    fn finish(self) -> PolarsResult<LazyFrame> {
        let readers = self.multiple_readers().iter_readers(self.cloud_options())?;
        let lfs = readers
            .map(|r| {
                let r = r?;
                self.clone()
                    .with_rechunk(false)
                    .finish_no_glob(r)
                    .map_err(|e| {
                        polars_err!(
                            ComputeError: "error while reading {}: {}", r, e
                        )
                    })
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        polars_ensure!(
            !lfs.is_empty(),
            ComputeError: "no matching files found in {}", self.multiple_readers()
        );

        let mut lf = self.concat_impl(lfs)?;
        if let Some(n_rows) = self.n_rows() {
            lf = lf.slice(0, n_rows as IdxSize)
        };
        if let Some(rc) = self.row_count() {
            lf = lf.with_row_count(&rc.name, Some(rc.offset))
        };

        Ok(lf)
    }

    /// Recommended concatenation of [LazyFrame]s from many input files.
    ///
    /// This method should not take into consideration [LazyFileListReader::n_rows]
    /// nor [LazyFileListReader::row_count].
    fn concat_impl(&self, lfs: Vec<LazyFrame>) -> PolarsResult<LazyFrame> {
        concat_impl(&lfs, self.rechunk(), true, true, false)
    }

    /// Get the final [LazyFrame].
    /// This method assumes, that path is *not* a glob.
    ///
    /// It is recommended to always use [LazyFileListReader::finish] method.
    fn finish_no_glob(self, reader: ReaderFactory) -> PolarsResult<LazyFrame>;

    /// Reader of the scanned file.
    fn reader(&mut self) -> &mut ReaderFactory;

    /// A source of multiple readers.
    fn multiple_readers(&self) -> &MultipleReaderFactories;

    /// Set the reader for a _specific_ file (local or remote), e.g. it's not a
    /// glob.
    #[must_use]
    fn with_reader(self, reader: Arc<ReaderFactory>) -> Self;

    /// Set sources of the scanned files.
    #[must_use]
    fn with_multiple_readers(self, sources: Arc<MultipleReaderFactories>) -> Self;

    /// Rechunk the memory to contiguous chunks when parsing is done.
    fn rechunk(&self) -> bool;

    /// Rechunk the memory to contiguous chunks when parsing is done.
    #[must_use]
    fn with_rechunk(self, toggle: bool) -> Self;

    /// Try to stop parsing when `n` rows are parsed. During multithreaded parsing the upper bound `n` cannot
    /// be guaranteed.
    fn n_rows(&self) -> Option<usize>;

    /// Add a `row_count` column.
    fn row_count(&self) -> Option<&RowCount>;

    /// [CloudOptions] used to list files.
    fn cloud_options(&self) -> Option<&CloudOptions> {
        None
    }
}
