use nalgebra::DMatrix;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::QaoaError;
use crate::optimization::{
    BruteForceSolver, ContinuousResult, MarkowitzSolver, OptimizationResult, SimulatedAnnealing,
};
use crate::portfolio::ReturnSeries;
use crate::qubo::{QUBOFormulation, QUBOMatrix};

create_exception!(qaoa_portfolio_core, OptimizationError, PyException);

#[pyclass(
    name = "PyReturnSeries",
    module = "qaoa_portfolio_core",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyReturnSeries {
    inner: ReturnSeries,
}

#[pymethods]
impl PyReturnSeries {
    #[new]
    pub fn new(symbols: Vec<String>, returns: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let matrix = matrix_from_pyarray(returns)?;
        let inner = ReturnSeries::from_returns(symbols, matrix).map_err(map_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn num_periods(&self) -> usize {
        self.inner.num_periods()
    }

    #[getter]
    pub fn num_assets(&self) -> usize {
        self.inner.num_assets()
    }

    pub fn mean_returns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .inner
            .mean_returns()
            .iter()
            .copied()
            .collect::<Vec<_>>();
        PyArray1::from_vec(py, values)
    }

    pub fn covariance_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        matrix_to_pyarray(py, &self.inner.covariance_matrix().map_err(map_err)?)
    }
}

#[pyclass(
    name = "PyQUBOMatrix",
    module = "qaoa_portfolio_core",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyQUBOMatrix {
    inner: QUBOMatrix,
}

#[pymethods]
impl PyQUBOMatrix {
    #[new]
    pub fn new(num_variables: usize) -> Self {
        Self {
            inner: QUBOMatrix::new(num_variables),
        }
    }

    #[getter]
    pub fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }

    #[getter]
    pub fn offset(&self) -> f64 {
        self.inner.offset()
    }

    pub fn evaluate(&self, solution: Vec<bool>) -> PyResult<f64> {
        self.inner.evaluate(&solution).map_err(map_err)
    }

    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        matrix_to_pyarray(py, self.inner.as_matrix())
    }

    pub fn to_list(&self) -> Vec<Vec<f64>> {
        self.inner.to_vec()
    }
}

#[pyclass(
    name = "PyOptimizationResult",
    module = "qaoa_portfolio_core",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyOptimizationResult {
    inner: OptimizationResult,
}

#[pymethods]
impl PyOptimizationResult {
    #[getter]
    pub fn solution(&self) -> Vec<bool> {
        self.inner.solution().to_vec()
    }

    #[getter]
    pub fn objective_value(&self) -> f64 {
        self.inner.objective_value()
    }

    #[getter]
    pub fn selected_assets(&self) -> Vec<String> {
        self.inner.selected_symbols().to_vec()
    }

    #[getter]
    pub fn solver_name(&self) -> String {
        self.inner.solver_name().to_string()
    }

    #[getter]
    pub fn iterations(&self) -> usize {
        self.inner.iterations()
    }

    pub fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("solution", self.solution())?;
        dict.set_item("objective_value", self.objective_value())?;
        dict.set_item("selected_assets", self.selected_assets())?;
        dict.set_item("solver_name", self.solver_name())?;
        dict.set_item("iterations", self.iterations())?;
        dict.set_item("elapsed_ms", self.inner.elapsed_ms())?;

        let metadata = PyDict::new(py);
        metadata.set_item(
            "solutions_evaluated",
            self.inner.metadata().solutions_evaluated(),
        )?;
        metadata.set_item(
            "best_found_at_iteration",
            self.inner.metadata().best_found_at_iteration(),
        )?;
        metadata.set_item(
            "convergence_history",
            self.inner.metadata().convergence_history(),
        )?;
        dict.set_item("metadata", metadata)?;
        Ok(dict.into_any().unbind())
    }
}

#[pyfunction]
pub fn build_qubo(
    prices: PyReadonlyArray2<'_, f64>,
    symbols: Vec<String>,
    risk_aversion: f64,
    target_assets: usize,
) -> PyResult<PyQUBOMatrix> {
    let price_matrix = matrix_from_pyarray(prices)?;
    let returns = ReturnSeries::from_prices(symbols, price_matrix).map_err(map_err)?;
    let qubo = QUBOFormulation::new(risk_aversion, target_assets)
        .map_err(map_err)?
        .build(&returns)
        .map_err(map_err)?;
    Ok(PyQUBOMatrix { inner: qubo })
}

#[pyfunction]
pub fn solve_brute_force(qubo: PyRef<'_, PyQUBOMatrix>) -> PyResult<PyOptimizationResult> {
    let inner = BruteForceSolver::solve(&qubo.inner).map_err(map_err)?;
    Ok(PyOptimizationResult { inner })
}

#[pyfunction(signature = (
    qubo,
    initial_temperature = None,
    cooling_rate = None,
    max_iterations = None,
    seed = None
))]
pub fn solve_simulated_annealing(
    qubo: PyRef<'_, PyQUBOMatrix>,
    initial_temperature: Option<f64>,
    cooling_rate: Option<f64>,
    max_iterations: Option<usize>,
    seed: Option<u64>,
) -> PyResult<PyOptimizationResult> {
    let mut solver = SimulatedAnnealing::new();
    if let Some(value) = initial_temperature {
        solver = solver.with_temperature(value);
    }
    if let Some(value) = cooling_rate {
        solver = solver.with_cooling_rate(value);
    }
    if let Some(value) = max_iterations {
        solver = solver.with_max_iterations(value);
    }
    if let Some(value) = seed {
        solver = solver.with_seed(value);
    }

    let inner = solver.solve(&qubo.inner).map_err(map_err)?;
    Ok(PyOptimizationResult { inner })
}

#[pyfunction]
pub fn solve_markowitz(
    prices: PyReadonlyArray2<'_, f64>,
    symbols: Vec<String>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    let price_matrix = matrix_from_pyarray(prices)?;
    let returns = ReturnSeries::from_prices(symbols, price_matrix).map_err(map_err)?;
    let result = MarkowitzSolver::new().solve(&returns).map_err(map_err)?;
    continuous_result_to_dict(py, &result)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("OptimizationError", py.get_type::<OptimizationError>())?;
    module.add_class::<PyReturnSeries>()?;
    module.add_class::<PyQUBOMatrix>()?;
    module.add_class::<PyOptimizationResult>()?;
    module.add_function(wrap_pyfunction!(build_qubo, module)?)?;
    module.add_function(wrap_pyfunction!(solve_brute_force, module)?)?;
    module.add_function(wrap_pyfunction!(solve_simulated_annealing, module)?)?;
    module.add_function(wrap_pyfunction!(solve_markowitz, module)?)?;
    Ok(())
}

fn continuous_result_to_dict(py: Python<'_>, result: &ContinuousResult) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("weights", result.weights())?;
    dict.set_item("expected_return", result.expected_return())?;
    dict.set_item("volatility", result.volatility())?;
    dict.set_item("sharpe_ratio", result.sharpe_ratio())?;
    dict.set_item("symbols", result.symbols())?;
    Ok(dict.into_any().unbind())
}

fn matrix_from_pyarray(array: PyReadonlyArray2<'_, f64>) -> PyResult<DMatrix<f64>> {
    let view = array.as_array();
    let shape = view.shape();
    let rows = shape[0];
    let cols = shape[1];
    let mut values = Vec::with_capacity(rows * cols);
    for row in view.outer_iter() {
        values.extend(row.iter().copied());
    }
    Ok(DMatrix::from_row_slice(rows, cols, &values))
}

fn matrix_to_pyarray<'py>(
    py: Python<'py>,
    matrix: &DMatrix<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let values = (0..matrix.nrows())
        .flat_map(|row| (0..matrix.ncols()).map(move |col| matrix[(row, col)]))
        .collect::<Vec<_>>();
    let array = ndarray::Array2::from_shape_vec((matrix.nrows(), matrix.ncols()), values)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array(py, array))
}

fn map_err(error: QaoaError) -> PyErr {
    match error {
        QaoaError::InvalidInput(_) | QaoaError::DimensionMismatch { .. } => {
            PyValueError::new_err(error.to_string())
        }
        QaoaError::OptimizationFailed(_) => OptimizationError::new_err(error.to_string()),
        QaoaError::NumericalError(_) | QaoaError::SerializationError(_) => {
            PyRuntimeError::new_err(error.to_string())
        }
    }
}
