#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(ambiguous_glob_reexports)]
#![cfg_attr(feature = "nightly", allow(clippy::non_canonical_partial_ord_impl))] // remove once stable
extern crate core;

#[macro_use]
pub mod utils;
pub mod chunked_array;
pub mod config;
pub mod datatypes;
pub mod error;
pub mod fmt;
pub mod frame;
pub mod functions;
pub mod hashing;
mod named_from;
pub mod prelude;
#[cfg(feature = "random")]
pub mod random;
pub mod scalar;
pub mod schema;
#[cfg(feature = "serde")]
pub mod serde;
pub mod series;
pub mod testing;
#[cfg(test)]
mod tests;

use std::any::Any;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub use datatypes::SchemaExtPl;
pub use hashing::IdBuildHasher;
use once_cell::sync::Lazy;
use rayon::{Scope, ThreadPool as RayonThreadPool, ThreadPoolBuilder};

#[cfg(feature = "dtype-categorical")]
pub use crate::chunked_array::logical::categorical::string_cache::*;

pub static PROCESS_ID: Lazy<u128> = Lazy::new(|| {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
});

/// Same interface as Rayon's ThreadPool, with some extra attempts to reduce
/// compilation time.
pub struct ThreadPool {
    rayon_pool: RayonThreadPool,
}

impl ThreadPool {
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        // By wrapping with a Box, we reduce how much generic code gets
        // generated, and thereby reduce compilation time.
        //
        // real    2m30.241s
        // user    11m20.338s
        // sys     1m29.272s
        let (sender, receiver) = std::sync::mpsc::channel();
        let op = || {
            let result = op();
            let _ = sender.send(Box::new(result));
        };
        let op : Box<dyn FnOnce() -> () + Send> = Box::new(op);
        self.install_uninlined(op);
        *receiver.recv().unwrap()
    }

    #[inline(never)]
    fn install_uninlined(&self, op: Box<dyn FnOnce() -> () + Send>) {
        self.rayon_pool.install(op);
    }

    pub fn current_num_threads(&self) -> usize {
        self.rayon_pool.current_num_threads()
    }

    pub fn current_thread_index(&self) -> Option<usize> {
        self.rayon_pool.current_thread_index()
    }

    pub fn current_thread_has_pending_tasks(&self) -> Option<bool> {
        self.rayon_pool.current_thread_has_pending_tasks()
    }

    pub fn join<A, B, RA, RB>(&self, oper_a: A, oper_b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        self.rayon_pool.join(oper_a, oper_b)
    }

    pub fn spawn<OP>(&self, op: OP)
    where
        OP: FnOnce() + Send + 'static {
        self.rayon_pool.spawn(op)
    }

    pub fn spawn_fifo<OP>(&self, op: OP)
    where
        OP: FnOnce() + Send + 'static {
        self.rayon_pool.spawn_fifo(op);
    }

    pub fn scope<'scope, OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&Scope<'scope>) -> R + Send,
        R: Send {
        self.rayon_pool.scope(op)
    }

    pub fn get_rayon_pool(&self) -> &RayonThreadPool {
        &self.rayon_pool
    }
}

// this is re-exported in utils for polars child crates
#[cfg(not(target_family = "wasm"))] // only use this on non wasm targets
pub static POOL: Lazy<ThreadPool> = Lazy::new(|| {
    let thread_name = std::env::var("POLARS_THREAD_NAME").unwrap_or_else(|_| "polars".to_string());
    let rayon_pool = ThreadPoolBuilder::new()
        .num_threads(
            std::env::var("POLARS_MAX_THREADS")
                .map(|s| s.parse::<usize>().expect("integer"))
                .unwrap_or_else(|_| {
                    std::thread::available_parallelism()
                        .unwrap_or(std::num::NonZeroUsize::new(1).unwrap())
                        .get()
                }),
        )
        .thread_name(move |i| format!("{}-{}", thread_name, i))
        .build()
        .expect("could not spawn threads");
    ThreadPool { rayon_pool }
});

#[cfg(all(target_os = "emscripten", target_family = "wasm"))] // Use 1 rayon thread on emscripten
pub static POOL: Lazy<ThreadPool> = Lazy::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(1)
        .use_current_thread()
        .build()
        .expect("could not create pool")
});

#[cfg(all(not(target_os = "emscripten"), target_family = "wasm"))] // use this on other wasm targets
pub static POOL: Lazy<polars_utils::wasm::Pool> = Lazy::new(|| polars_utils::wasm::Pool);

// utility for the tests to ensure a single thread can execute
pub static SINGLE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

/// Default length for a `.head()` call
pub(crate) const HEAD_DEFAULT_LENGTH: usize = 10;
/// Default length for a `.tail()` call
pub(crate) const TAIL_DEFAULT_LENGTH: usize = 10;
