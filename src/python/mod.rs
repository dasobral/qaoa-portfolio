mod bridge;

pub use bridge::*;

use pyo3::prelude::*;

#[pymodule]
pub fn qaoa_portfolio_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    bridge::register(module)
}
