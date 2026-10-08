import numpy as np
import pytest

from qaoa_portfolio.exceptions import QuantumBackendError
from qaoa_portfolio.params import QAOAParams
from qaoa_portfolio.quantum_backend import (
    QAOAConfig,
    QAOAQuantumBackend,
    QAOAResult,
    bitstring_to_solution,
    build_cost_hamiltonian,
    build_mixer_hamiltonian,
    decode_solution,
    evaluate_qubo_bitstring,
    hamming_weights,
    solve_qubo_qaoa,
)

try:
    import qaoa_portfolio_core
except ImportError:  # pragma: no cover - exercised only without the extension installed
    qaoa_portfolio_core = None

pytestmark = pytest.mark.unit


TOY_QUBO = np.array(
    [
        [-1.0, 2.0],
        [2.0, -0.5],
    ],
    dtype=float,
)


def constant_prices(periods=70, assets=4):
    return np.full((periods, assets), 100.0, dtype=np.float64)


def hamiltonian_terms(hamiltonian):
    coeffs, ops = hamiltonian.terms()
    terms = {}

    for coeff, op in zip(coeffs, ops):
        if op.name == "Identity":
            key = ("I",)
        elif op.name in {"PauliX", "PauliZ"}:
            key = ((op.name.removeprefix("Pauli"), int(op.wires[0])),)
        elif op.name == "Prod":
            key = tuple(
                (operand.name.removeprefix("Pauli"), int(operand.wires[0]))
                for operand in op.operands
            )
        else:
            key = (op.name, tuple(op.wires.tolist()))
        terms[key] = terms.get(key, 0.0) + float(coeff)

    return terms


def test_quantum_backend_public_api_exported_from_package():
    import qaoa_portfolio as qp

    assert qp.QAOAConfig is QAOAConfig
    assert qp.QAOAQuantumBackend is QAOAQuantumBackend
    assert qp.solve_qubo_qaoa is solve_qubo_qaoa
    assert qp.evaluate_qubo_bitstring is evaluate_qubo_bitstring


def test_qaoa_config_defaults_and_validation():
    config = QAOAConfig()

    assert config.layers == QAOAParams.DEFAULT_LAYERS
    assert config.max_iterations == QAOAParams.DEFAULT_MAX_ITERATIONS
    assert config.convergence_threshold == QAOAParams.DEFAULT_CONVERGENCE_THRESHOLD
    assert config.optimizer == "adam"
    assert config.backend == "default.qubit"

    invalid_configs = [
        {"layers": 0},
        {"max_iterations": 0},
        {"convergence_threshold": 0.0},
        {"shots": 0},
        {"optimizer": "bad-optimizer"},
        {"backend": "bad.backend"},
        {"num_restarts": 0},
    ]

    for kwargs in invalid_configs:
        with pytest.raises(QuantumBackendError):
            QAOAConfig(**kwargs)


def test_qaoa_result_serializes_to_json_safe_dict():
    result = QAOAResult(
        best_bitstring="10",
        best_solution=[True, False],
        selected_indices=[0],
        selected_assets=["AAPL"],
        objective_value=-1.0,
        probabilities={"10": 0.75, "01": 0.25},
        top_solutions=[
            {
                "bitstring": "10",
                "solution": [True, False],
                "selected_indices": [0],
                "selected_assets": ["AAPL"],
                "objective_value": -1.0,
                "probability": 0.75,
            }
        ],
        optimal_parameters={"gammas": [0.1], "betas": [0.2]},
        convergence_history=[0.0, -1.0],
        iterations=2,
        elapsed_ms=3,
        metadata={"backend": "default.qubit"},
    )

    payload = result.to_dict()

    assert payload["best_bitstring"] == "10"
    assert payload["selected_assets"] == ["AAPL"]
    assert payload["probabilities"] == {"10": 0.75, "01": 0.25}
    assert payload["optimal_parameters"] == {"gammas": [0.1], "betas": [0.2]}


def test_normalize_qubo_accepts_numpy_and_rust_qubo_matrix():
    backend = QAOAQuantumBackend(QAOAConfig(layers=1, max_iterations=1, num_restarts=1))

    matrix, offset, labels = backend._normalize_qubo(TOY_QUBO, labels=["A", "B"])
    assert np.allclose(matrix, TOY_QUBO)
    assert offset == 0.0
    assert labels == ["A", "B"]

    if qaoa_portfolio_core is None:
        pytest.skip("qaoa_portfolio_core is required for PyQUBOMatrix normalization")

    rust_qubo = qaoa_portfolio_core.build_qubo(
        constant_prices(),
        ["A", "B", "C", "D"],
        0.5,
        2,
    )
    matrix, offset, labels = backend._normalize_qubo(rust_qubo)

    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)
    assert offset == rust_qubo.offset
    assert labels == ["x0", "x1", "x2", "x3"]


def test_normalize_qubo_rejects_invalid_matrices():
    backend = QAOAQuantumBackend(QAOAConfig(layers=1, max_iterations=1, num_restarts=1))
    invalid_matrices = [
        np.array([]),
        np.array([1.0, 2.0]),
        np.ones((2, 3)),
        np.array([[1.0, np.nan], [np.nan, 2.0]]),
        np.array([[1.0, 2.0], [3.0, 1.0]]),
    ]

    for matrix in invalid_matrices:
        with pytest.raises(QuantumBackendError):
            backend._normalize_qubo(matrix)

    with pytest.raises(QuantumBackendError):
        backend._normalize_qubo(TOY_QUBO, labels=["only-one-label"])


def test_cost_hamiltonian_matches_hand_computed_two_variable_qubo():
    hamiltonian = build_cost_hamiltonian(TOY_QUBO)
    terms = hamiltonian_terms(hamiltonian)

    assert terms[("I",)] == pytest.approx(-0.25)
    assert terms.get((("Z", 0),), 0.0) == pytest.approx(0.0)
    assert terms[(("Z", 1),)] == pytest.approx(-0.25)
    assert terms[(("Z", 0), ("Z", 1))] == pytest.approx(0.5)


def test_mixer_hamiltonian_has_one_x_term_per_wire():
    hamiltonian = build_mixer_hamiltonian(3)
    terms = hamiltonian_terms(hamiltonian)

    assert terms == {
        (("X", 0),): pytest.approx(1.0),
        (("X", 1),): pytest.approx(1.0),
        (("X", 2),): pytest.approx(1.0),
    }


def test_bitstring_decoding_and_qubo_evaluation():
    assert bitstring_to_solution("101") == [True, False, True]

    decoded = decode_solution("101", labels=["AAPL", "MSFT", "NVDA"])
    assert decoded == {
        "bitstring": "101",
        "solution": [True, False, True],
        "selected_indices": [0, 2],
        "selected_assets": ["AAPL", "NVDA"],
    }

    assert evaluate_qubo_bitstring(TOY_QUBO, "00") == pytest.approx(0.0)
    assert evaluate_qubo_bitstring(TOY_QUBO, "10") == pytest.approx(-1.0)
    assert evaluate_qubo_bitstring(TOY_QUBO, "01") == pytest.approx(-0.5)
    assert evaluate_qubo_bitstring(TOY_QUBO, "11") == pytest.approx(0.5)


def test_qubo_evaluation_matches_rust_bridge_upper_triangle_convention():
    if qaoa_portfolio_core is None:
        pytest.skip("qaoa_portfolio_core is required for Rust evaluation comparison")

    rust_qubo = qaoa_portfolio_core.build_qubo(
        constant_prices(),
        ["A", "B", "C", "D"],
        0.5,
        2,
    )
    matrix = rust_qubo.to_numpy()
    solution = [True, True, False, False]
    bitstring = "1100"

    assert evaluate_qubo_bitstring(
        matrix, bitstring, rust_qubo.offset
    ) == pytest.approx(rust_qubo.evaluate(solution))


def test_seeded_qaoa_run_is_deterministic_for_toy_qubo():
    config = QAOAConfig(
        layers=1,
        optimizer="gradient_descent",
        max_iterations=8,
        convergence_threshold=1e-12,
        seed=123,
        num_restarts=2,
    )

    first = solve_qubo_qaoa(TOY_QUBO, labels=["A", "B"], config=config)
    second = solve_qubo_qaoa(TOY_QUBO, labels=["A", "B"], config=config)

    assert first.best_bitstring == "10"
    assert first.best_bitstring == second.best_bitstring
    assert first.objective_value == pytest.approx(-1.0)
    assert first.probabilities == pytest.approx(second.probabilities)
    assert first.convergence_history == pytest.approx(second.convergence_history)
    assert first.optimal_parameters["gammas"] == pytest.approx(
        second.optimal_parameters["gammas"]
    )
    assert first.optimal_parameters["betas"] == pytest.approx(
        second.optimal_parameters["betas"]
    )


def test_hamming_weights_match_popcount():
    weights = hamming_weights(6)
    assert weights.dtype == np.uint8
    assert weights.tolist() == [bin(index).count("1") for index in range(64)]


def test_feasible_decoding_requires_target_assets():
    with pytest.raises(QuantumBackendError):
        QAOAConfig(feasible_decoding=True)
    with pytest.raises(QuantumBackendError):
        QAOAConfig(target_assets=-1)


def test_feasibility_metadata_and_feasible_decoding():
    # Unconstrained minimum is "11" (weight 2); with k = 1 the best feasible
    # state is "10". A tiny stored head forces the feasible filter to matter.
    qubo = np.array([[-1.0, -1.0], [-1.0, -1.0]])
    base = dict(
        layers=1,
        optimizer="gradient_descent",
        max_iterations=5,
        num_restarts=1,
        max_stored_solutions=1,
        target_assets=1,
    )

    plain = solve_qubo_qaoa(
        qubo, config=QAOAConfig(**base), reference_bitstrings=["10", "01"]
    )
    distribution = plain.metadata["hamming_weight_distribution"]
    assert sum(distribution) == pytest.approx(1.0)
    assert plain.metadata["feasible_probability"] == pytest.approx(distribution[1])
    references = plain.metadata["reference_probabilities"]
    assert sum(references.values()) == pytest.approx(distribution[1])

    feasible = solve_qubo_qaoa(qubo, config=QAOAConfig(**base, feasible_decoding=True))
    assert feasible.best_bitstring.count("1") == 1
    assert all(key.count("1") == 1 for key in feasible.probabilities)
