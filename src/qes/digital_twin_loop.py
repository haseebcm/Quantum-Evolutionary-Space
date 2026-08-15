"""Phase 9 -- Digital Twin 2.0 closed-loop utilities.

This module extends QES's existing `DigitalTwin` anchor into a fuller
classical estimation-and-prediction loop:

    real world -> sensors -> fused observation -> digital twin anchoring
    -> QES reality generation -> simulation/prediction -> action suggestion
    -> next observation

All algorithms here are ordinary numerical estimation methods running on CPU
and RAM: inverse-variance sensor fusion, recursive least squares, a linear
Kalman filter, finite-difference covariance propagation, residual-based
anomaly detection, and simple rolling-window drift checks. They are useful for
synthetic or practical digital-twin workflows, but they are not literal
quantum computing and not zero-cost computation.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import isfinite
from statistics import NormalDist

import numpy as np

from qes.convergence import qes_entropy
from qes.digital_twin import DigitalTwin
from qes.reality_generator import RealityGenerator
from qes.room import Room

_EPSILON = 1e-12


def _as_vector(name: str, values: object) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be array-like and numeric") from exc
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_matrix(name: str, values: object, size: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be array-like and numeric") from exc
    if array.ndim == 0:
        if size is None:
            raise ValueError(f"{name} scalar input requires an explicit size")
        array = np.eye(size, dtype=float) * float(array)
    elif array.ndim == 1:
        array = np.diag(array)
    elif array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square")
    if size is not None and array.shape != (size, size):
        raise ValueError(f"{name} must have shape {(size, size)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _validate_non_negative_real(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return numeric


def _model_name(model: Callable[[np.ndarray], np.ndarray]) -> str:
    named = getattr(model, "name", None)
    if isinstance(named, str) and named:
        return named
    fallback = getattr(model, "__name__", None)
    if isinstance(fallback, str) and fallback:
        return fallback
    return model.__class__.__name__


@dataclass
class Observation:
    """One timestamped sensor observation for a vector-valued state.

    Attributes:
        values: observed sensor values.
        step: optional discrete step index.
        timestamp: optional wall-clock or simulation timestamp.
        uncertainty: optional per-channel standard deviation estimate.
        sensor_id: optional sensor label.
    """

    values: np.ndarray
    step: int | None = None
    timestamp: float | None = None
    uncertainty: np.ndarray | None = None
    sensor_id: str | None = None

    def __post_init__(self) -> None:
        self.values = _as_vector("values", self.values)
        if self.step is not None:
            self.step = _validate_positive_int("step", self.step + 1) - 1
        if self.timestamp is not None:
            self.timestamp = _validate_non_negative_real("timestamp", self.timestamp)
        if self.uncertainty is not None:
            self.uncertainty = _as_vector("uncertainty", self.uncertainty)
            if self.uncertainty.shape != self.values.shape:
                raise ValueError("uncertainty must have the same shape as values")
            if np.any(self.uncertainty <= 0.0):
                raise ValueError("uncertainty entries must be > 0")
        if self.sensor_id is not None and not isinstance(self.sensor_id, str):
            raise TypeError("sensor_id must be a string when provided")


@dataclass
class PropagationResult:
    """Result of propagating a mean/covariance pair through a transition."""

    mean: np.ndarray
    covariance: np.ndarray


@dataclass
class ModelSelectionResult:
    """Fit summary for one candidate predictive model."""

    model_name: str
    model: Callable[[np.ndarray], np.ndarray]
    residual_sum_squares: float
    score: float


@dataclass
class AnomalyReport:
    """Residual-based anomaly decision for one observation."""

    flagged: bool
    residual: np.ndarray
    z_scores: np.ndarray
    threshold: float
    max_z_score: float


@dataclass
class DriftReport:
    """Rolling-window drift diagnostic over prediction errors."""

    flagged: bool
    baseline_mean: float
    baseline_std: float
    recent_mean: float
    recent_std: float
    mean_shift_sigma: float
    std_ratio: float


@dataclass
class LoopStepResult:
    """One completed digital-twin loop step."""

    step: int
    fused_observation: Observation
    estimated_state: np.ndarray
    estimated_covariance: np.ndarray
    predicted_state: np.ndarray
    predicted_observation: np.ndarray
    prediction_interval: tuple[np.ndarray, np.ndarray]
    anomaly_report: AnomalyReport
    drift_report: DriftReport
    selected_model_name: str | None
    parameter_estimate: np.ndarray | None
    uncertainty_score: float
    calibration_score: float | None
    recommended_action: str
    reality_weights: np.ndarray = field(repr=False)


def sensor_fusion(
    readings: list[Observation],
    weights: Sequence[float] | None = None,
) -> Observation:
    """Fuse several noisy observations with inverse-variance weighting.

    Each sensor contributes a precision proportional to

        weight_i / variance_i

    per channel. When explicit uncertainties are absent, unit variance is
    assumed for that sensor. The fused mean is the precision-weighted average
    and the fused uncertainty is the square root of the reciprocal summed
    precision.
    """

    if not isinstance(readings, list):
        raise TypeError("readings must be a list of Observation instances")
    if not readings:
        raise ValueError("readings must contain at least one Observation")
    for reading in readings:
        if not isinstance(reading, Observation):
            raise TypeError("readings must contain only Observation instances")

    dim = readings[0].values.shape[0]
    for reading in readings[1:]:
        if reading.values.shape[0] != dim:
            raise ValueError("all observations must have the same dimensionality")

    if weights is None:
        external_weights = np.ones(len(readings), dtype=float)
    else:
        external_weights = _as_vector("weights", weights)
        if external_weights.shape[0] != len(readings):
            raise ValueError("weights must have the same length as readings")
        if np.any(external_weights <= 0.0):
            raise ValueError("weights must be > 0")

    means = np.vstack([reading.values for reading in readings])
    stds = np.vstack(
        [
            np.ones(dim, dtype=float)
            if reading.uncertainty is None
            else reading.uncertainty
            for reading in readings
        ]
    )
    variances = np.maximum(stds**2, _EPSILON)
    precisions = external_weights[:, None] / variances
    total_precision = np.sum(precisions, axis=0)
    fused_values = np.sum(precisions * means, axis=0) / total_precision
    fused_uncertainty = np.sqrt(1.0 / total_precision)

    latest = max(
        readings,
        key=lambda item: (
            -1 if item.step is None else item.step,
            -1.0 if item.timestamp is None else item.timestamp,
        ),
    )
    return Observation(
        values=fused_values,
        step=latest.step,
        timestamp=latest.timestamp,
        uncertainty=fused_uncertainty,
        sensor_id="fused",
    )


class OnlineParameterEstimator:
    """Recursive least-squares online estimator for linear parameters.

    The model is

        y_t = phi_t^T theta + noise_t

    and `update(features, target)` performs one classical RLS update with a
    forgetting factor.
    """

    def __init__(
        self,
        initial_parameters: Sequence[float],
        forgetting_factor: float = 0.99,
        covariance_scale: float = 1000.0,
    ):
        self.parameters = _as_vector("initial_parameters", initial_parameters)
        if not 0.0 < float(forgetting_factor) <= 1.0:
            raise ValueError("forgetting_factor must satisfy 0 < forgetting_factor <= 1")
        self.forgetting_factor = float(forgetting_factor)
        scale = _validate_non_negative_real("covariance_scale", covariance_scale)
        self.covariance = np.eye(self.parameters.shape[0], dtype=float) * max(scale, _EPSILON)

    def predict(self, features: Sequence[float]) -> float:
        """Predict a scalar target from a feature vector."""

        phi = _as_vector("features", features)
        if phi.shape != self.parameters.shape:
            raise ValueError("features must have the same length as parameters")
        return float(phi @ self.parameters)

    def update(self, features: Sequence[float], target: float) -> np.ndarray:
        """Assimilate one `(features, target)` pair and return the new parameter estimate."""

        phi = _as_vector("features", features)
        if phi.shape != self.parameters.shape:
            raise ValueError("features must have the same length as parameters")
        target_value = float(target)
        if not isfinite(target_value):
            raise ValueError("target must be finite")

        phi_column = phi[:, None]
        denominator = self.forgetting_factor + float(
            (phi_column.T @ self.covariance @ phi_column).item()
        )
        gain = (self.covariance @ phi_column) / max(denominator, _EPSILON)
        residual = target_value - float(phi @ self.parameters)
        self.parameters = self.parameters + gain[:, 0] * residual
        identity = np.eye(self.parameters.shape[0], dtype=float)
        self.covariance = (
            (identity - gain @ phi_column.T) @ self.covariance / self.forgetting_factor
        )
        return self.parameters.copy()


class StateEstimator:
    """Classical linear Kalman filter with predict/update steps."""

    def __init__(
        self,
        initial_state: Sequence[float],
        initial_covariance: np.ndarray,
        transition_matrix: np.ndarray | None = None,
        observation_matrix: np.ndarray | None = None,
        process_covariance: np.ndarray | None = None,
        observation_covariance: np.ndarray | None = None,
    ):
        self.state = _as_vector("initial_state", initial_state)
        dim = self.state.shape[0]
        self.covariance = _as_matrix("initial_covariance", initial_covariance, size=dim)
        self.transition_matrix = (
            np.eye(dim, dtype=float)
            if transition_matrix is None
            else _as_matrix("transition_matrix", transition_matrix, size=dim)
        )
        self.observation_matrix = (
            np.eye(dim, dtype=float)
            if observation_matrix is None
            else _as_matrix("observation_matrix", observation_matrix, size=dim)
        )
        self.process_covariance = (
            np.eye(dim, dtype=float) * 1e-3
            if process_covariance is None
            else _as_matrix("process_covariance", process_covariance, size=dim)
        )
        self.observation_covariance = (
            np.eye(dim, dtype=float)
            if observation_covariance is None
            else _as_matrix("observation_covariance", observation_covariance, size=dim)
        )

    def predict(
        self,
        control_input: Sequence[float] | None = None,
        control_matrix: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance the current state estimate one step forward."""

        self.state = self.transition_matrix @ self.state
        if control_input is not None:
            control = _as_vector("control_input", control_input)
            if control_matrix is None:
                if control.shape != self.state.shape:
                    raise ValueError(
                        "control_input must match state shape when control_matrix is omitted"
                    )
                self.state = self.state + control
            else:
                matrix = np.asarray(control_matrix, dtype=float)
                self.state = self.state + matrix @ control
        self.covariance = (
            self.transition_matrix @ self.covariance @ self.transition_matrix.T
            + self.process_covariance
        )
        return self.state.copy(), self.covariance.copy()

    def update(
        self,
        observation: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assimilate a new observation into the state estimate."""

        observed = _as_vector("observation", observation)
        expected_shape = (self.observation_matrix.shape[0],)
        if observed.shape != expected_shape:
            raise ValueError(f"observation must have shape {expected_shape}")

        innovation = observed - self.observation_matrix @ self.state
        innovation_covariance = (
            self.observation_matrix @ self.covariance @ self.observation_matrix.T
            + self.observation_covariance
        )
        gain = (
            self.covariance
            @ self.observation_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.state = self.state + gain @ innovation
        identity = np.eye(self.state.shape[0], dtype=float)
        self.covariance = (identity - gain @ self.observation_matrix) @ self.covariance
        return self.state.copy(), self.covariance.copy()


def uncertainty_propagation(
    state_mean: Sequence[float] | np.ndarray,
    covariance: np.ndarray,
    transition_function: Callable[[np.ndarray], Sequence[float] | np.ndarray],
    process_covariance: np.ndarray | None = None,
    epsilon: float = 1e-5,
) -> PropagationResult:
    """Propagate mean/covariance through a transition via finite differences.

    This is a classical first-order approximation: evaluate the transition at
    the current mean, estimate its Jacobian numerically, and map covariance as

        P_next = J P J^T + Q
    """

    mean = _as_vector("state_mean", state_mean)
    covariance_matrix = _as_matrix("covariance", covariance, size=mean.shape[0])
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")

    propagated_mean = _as_vector("transition output", transition_function(mean))
    output_dim = propagated_mean.shape[0]
    jacobian = np.zeros((output_dim, mean.shape[0]), dtype=float)

    for index in range(mean.shape[0]):
        offset = np.zeros_like(mean)
        offset[index] = epsilon
        plus = _as_vector("transition output", transition_function(mean + offset))
        minus = _as_vector("transition output", transition_function(mean - offset))
        if plus.shape != propagated_mean.shape or minus.shape != propagated_mean.shape:
            raise ValueError("transition_function must preserve a consistent output shape")
        jacobian[:, index] = (plus - minus) / (2.0 * epsilon)

    if process_covariance is None:
        process_noise = np.zeros((output_dim, output_dim), dtype=float)
    else:
        process_noise = _as_matrix("process_covariance", process_covariance, size=output_dim)
    propagated_covariance = jacobian @ covariance_matrix @ jacobian.T + process_noise
    return PropagationResult(mean=propagated_mean, covariance=propagated_covariance)


class ModelSelector:
    """Scores candidate next-step models by sum of squared residuals."""

    def score_model(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        inputs: Sequence[Sequence[float] | np.ndarray],
        targets: Sequence[Sequence[float] | np.ndarray],
    ) -> ModelSelectionResult:
        """Return one candidate model's residual score over recent history."""

        if not callable(model):
            raise TypeError("model must be callable")
        if len(inputs) == 0 or len(targets) == 0:
            raise ValueError("inputs and targets must be non-empty")
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same length")

        residual_sum_squares = 0.0
        for current_input, target in zip(inputs, targets, strict=True):
            predicted = _as_vector("predicted", model(_as_vector("input", current_input)))
            actual = _as_vector("target", target)
            if predicted.shape != actual.shape:
                raise ValueError("model predictions must have the same shape as targets")
            residual = actual - predicted
            residual_sum_squares += float(residual @ residual)

        return ModelSelectionResult(
            model_name=_model_name(model),
            model=model,
            residual_sum_squares=residual_sum_squares,
            score=-residual_sum_squares,
        )

    def select_best(
        self,
        models: Sequence[Callable[[np.ndarray], np.ndarray]],
        inputs: Sequence[Sequence[float] | np.ndarray],
        targets: Sequence[Sequence[float] | np.ndarray],
    ) -> ModelSelectionResult:
        """Score every candidate model and return the best-fitting one."""

        if not models:
            raise ValueError("models must contain at least one callable")
        scored = [self.score_model(model, inputs, targets) for model in models]
        return sorted(scored, key=lambda item: (-item.score, item.model_name))[0]

    def rank_models(
        self,
        models: Sequence[Callable[[np.ndarray], np.ndarray]],
        inputs: Sequence[Sequence[float] | np.ndarray],
        targets: Sequence[Sequence[float] | np.ndarray],
    ) -> list[ModelSelectionResult]:
        """Return all candidate models sorted best to worst."""

        if not models:
            raise ValueError("models must contain at least one callable")
        scored = [self.score_model(model, inputs, targets) for model in models]
        return sorted(scored, key=lambda item: (-item.score, item.model_name))


def anomaly_detection(
    observation: Sequence[float] | np.ndarray,
    expected: Sequence[float] | np.ndarray,
    uncertainty: Sequence[float] | np.ndarray,
    threshold: float = 3.0,
) -> AnomalyReport:
    """Flag statistically large residuals using per-channel z-scores."""

    observed = _as_vector("observation", observation)
    expected_vector = _as_vector("expected", expected)
    if observed.shape != expected_vector.shape:
        raise ValueError("observation and expected must have the same shape")
    if threshold <= 0.0:
        raise ValueError("threshold must be > 0")

    if np.asarray(uncertainty).ndim == 2:
        covariance = _as_matrix("uncertainty", uncertainty, size=observed.shape[0])
        standard_deviations = np.sqrt(np.maximum(np.diag(covariance), _EPSILON))
    else:
        standard_deviations = _as_vector("uncertainty", uncertainty)
        if standard_deviations.shape != observed.shape:
            raise ValueError("uncertainty must have the same shape as observation")
        standard_deviations = np.maximum(standard_deviations, _EPSILON)

    residual = observed - expected_vector
    z_scores = np.abs(residual) / standard_deviations
    max_z_score = float(np.max(z_scores))
    return AnomalyReport(
        flagged=bool(max_z_score > threshold),
        residual=residual,
        z_scores=z_scores,
        threshold=float(threshold),
        max_z_score=max_z_score,
    )


def prediction_interval(
    mean: float | Sequence[float] | np.ndarray,
    std: float | Sequence[float] | np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """Return a Gaussian prediction interval for a mean and standard deviation."""

    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must satisfy 0 < confidence < 1")
    quantile = NormalDist().inv_cdf((1.0 + float(confidence)) / 2.0)

    mean_array = np.asarray(mean, dtype=float)
    std_array = np.asarray(std, dtype=float)
    if mean_array.shape != std_array.shape:
        raise ValueError("mean and std must have the same shape")
    if np.any(std_array < 0.0):
        raise ValueError("std must be >= 0")

    lower = mean_array - quantile * std_array
    upper = mean_array + quantile * std_array
    if mean_array.ndim == 0:
        return float(lower), float(upper)
    return lower.astype(float), upper.astype(float)


def calibration_check(
    predicted_intervals: Sequence[tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]],
    actual_outcomes: Sequence[Sequence[float] | np.ndarray],
) -> float:
    """Measure empirical interval coverage across outcomes."""

    if len(predicted_intervals) == 0 or len(actual_outcomes) == 0:
        raise ValueError("predicted_intervals and actual_outcomes must be non-empty")
    if len(predicted_intervals) != len(actual_outcomes):
        raise ValueError("predicted_intervals and actual_outcomes must have the same length")

    covered = 0
    total = 0
    for (lower, upper), actual in zip(predicted_intervals, actual_outcomes, strict=True):
        lower_array = _as_vector("interval lower bound", lower)
        upper_array = _as_vector("interval upper bound", upper)
        actual_array = _as_vector("actual outcome", actual)
        if lower_array.shape != upper_array.shape or lower_array.shape != actual_array.shape:
            raise ValueError("interval bounds and actual outcomes must share the same shape")
        covered += int(np.sum((actual_array >= lower_array) & (actual_array <= upper_array)))
        total += actual_array.size
    return covered / total


def drift_detection(
    residuals: Sequence[float],
    window: int = 5,
    mean_shift_threshold: float = 2.0,
    std_ratio_threshold: float = 2.0,
) -> DriftReport:
    """Compare a recent residual window against an older baseline window."""

    window = _validate_positive_int("window", window)
    mean_shift_threshold = _validate_non_negative_real(
        "mean_shift_threshold", mean_shift_threshold
    )
    std_ratio_threshold = _validate_non_negative_real("std_ratio_threshold", std_ratio_threshold)
    if std_ratio_threshold == 0.0:
        raise ValueError("std_ratio_threshold must be > 0")

    values = _as_vector("residuals", residuals)
    if values.shape[0] < 2 * window:
        return DriftReport(
            flagged=False,
            baseline_mean=0.0,
            baseline_std=0.0,
            recent_mean=0.0,
            recent_std=0.0,
            mean_shift_sigma=0.0,
            std_ratio=1.0,
        )

    baseline = values[-2 * window:-window]
    recent = values[-window:]
    baseline_mean = float(np.mean(baseline))
    recent_mean = float(np.mean(recent))
    baseline_std = float(np.std(baseline, ddof=0))
    recent_std = float(np.std(recent, ddof=0))
    scale = max(baseline_std, _EPSILON)
    mean_shift_sigma = abs(recent_mean - baseline_mean) / scale
    std_ratio = max(recent_std, _EPSILON) / scale
    flagged = bool(
        mean_shift_sigma > mean_shift_threshold
        or std_ratio > std_ratio_threshold
        or std_ratio < 1.0 / std_ratio_threshold
    )
    return DriftReport(
        flagged=flagged,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        recent_mean=recent_mean,
        recent_std=recent_std,
        mean_shift_sigma=float(mean_shift_sigma),
        std_ratio=float(std_ratio),
    )


class DigitalTwinLoop:
    """Full closed-loop digital twin orchestrator for Phase 9.

    `observe(sensor_readings)` performs one loop tick:

    1. fuse raw sensor readings,
    2. compare them against the current twin prediction,
    3. update the state estimate,
    4. branch nearby QES realities,
    5. score those realities against the latest observation,
    6. simulate the next state,
    7. emit prediction intervals, anomaly/drift diagnostics, and an action hint.
    """

    def __init__(
        self,
        state_estimator: StateEstimator,
        twin: DigitalTwin | None = None,
        reality_generator: RealityGenerator | None = None,
        parameter_estimator: OnlineParameterEstimator | None = None,
        model_selector: ModelSelector | None = None,
        models: Sequence[Callable[[np.ndarray], np.ndarray]] | None = None,
        transition_function: Callable[[np.ndarray, np.ndarray | None], Sequence[float]] | None = None,
        observation_function: Callable[[np.ndarray], Sequence[float]] | None = None,
        reference_state: Sequence[float] | None = None,
        lower_bounds: Sequence[float] | None = None,
        upper_bounds: Sequence[float] | None = None,
        reality_count: int = 8,
        reality_scale: float = 0.05,
        prediction_confidence: float = 0.95,
        anomaly_threshold: float = 3.0,
        drift_window: int = 5,
        drift_mean_shift_threshold: float = 2.0,
        drift_std_ratio_threshold: float = 2.0,
    ):
        if not isinstance(state_estimator, StateEstimator):
            raise TypeError("state_estimator must be a StateEstimator")
        self.state_estimator = state_estimator
        self.twin = DigitalTwin(
            noise_covariance=state_estimator.observation_covariance.copy(),
            window=max(drift_window * 2, 2),
        ) if twin is None else twin
        self.reality_generator = reality_generator or RealityGenerator()
        self.parameter_estimator = parameter_estimator
        self.model_selector = model_selector or ModelSelector()
        self.models = list(models or [])
        self.transition_function = transition_function
        self.observation_function = (
            observation_function if observation_function is not None else lambda state: state
        )

        dim = self.state_estimator.state.shape[0]
        self.reference_state = (
            np.zeros(dim, dtype=float)
            if reference_state is None
            else _as_vector("reference_state", reference_state)
        )
        if self.reference_state.shape != (dim,):
            raise ValueError(f"reference_state must have shape {(dim,)}")

        default_extent = np.maximum(np.abs(self.reference_state) + 1.0, 1.0)
        self.lower_bounds = (
            -default_extent if lower_bounds is None else _as_vector("lower_bounds", lower_bounds)
        )
        self.upper_bounds = (
            default_extent if upper_bounds is None else _as_vector("upper_bounds", upper_bounds)
        )
        if self.lower_bounds.shape != (dim,) or self.upper_bounds.shape != (dim,):
            raise ValueError(f"lower_bounds and upper_bounds must have shape {(dim,)}")
        if np.any(self.lower_bounds >= self.upper_bounds):
            raise ValueError("lower_bounds must be strictly less than upper_bounds")

        self.reality_count = _validate_positive_int("reality_count", reality_count)
        self.reality_scale = _validate_non_negative_real("reality_scale", reality_scale)
        self.prediction_confidence = float(prediction_confidence)
        self.anomaly_threshold = float(anomaly_threshold)
        self.drift_window = _validate_positive_int("drift_window", drift_window)
        self.drift_mean_shift_threshold = float(drift_mean_shift_threshold)
        self.drift_std_ratio_threshold = float(drift_std_ratio_threshold)

        self.history: list[LoopStepResult] = []
        self._observation_history: list[np.ndarray] = []
        self._prediction_intervals: list[tuple[np.ndarray, np.ndarray]] = []
        self._interval_outcomes: list[np.ndarray] = []
        self._error_history: list[float] = []

    def _make_room(self, state: np.ndarray) -> Room:
        clipped = np.clip(state, self.lower_bounds, self.upper_bounds)
        return Room(
            x=clipped.copy(),
            x_star=self.reference_state.copy(),
            lower=self.lower_bounds.copy(),
            upper=self.upper_bounds.copy(),
            activation=np.ones_like(clipped),
        )

    def _simulate_transition(
        self,
        state: np.ndarray,
        selected_model: Callable[[np.ndarray], np.ndarray] | None,
    ) -> np.ndarray:
        if selected_model is not None:
            return _as_vector("model prediction", selected_model(state))
        if self.transition_function is not None:
            parameters = None if self.parameter_estimator is None else self.parameter_estimator.parameters
            return _as_vector("transition output", self.transition_function(state, parameters))
        return self.state_estimator.transition_matrix @ state

    def _recommend_action(
        self,
        anomaly_report: AnomalyReport,
        drift_report: DriftReport,
        predicted_state: np.ndarray,
    ) -> str:
        if anomaly_report.flagged:
            return "Inspect sensors and recalibrate the twin before applying aggressive control."
        if drift_report.flagged:
            return "Re-estimate model structure or parameters because residual drift is rising."
        if float(np.linalg.norm(predicted_state - self.reference_state)) > 0.25 * predicted_state.size:
            return "Apply a corrective real-world action that nudges the system toward the reference state."
        return "Hold the current operating point and continue observing."

    def observe(
        self,
        sensor_readings: list[Observation],
        parameter_features: Sequence[float] | None = None,
        parameter_target: float | None = None,
    ) -> LoopStepResult:
        """Run one full observe-update-predict loop step."""

        fused_observation = sensor_fusion(sensor_readings)
        observed = fused_observation.values

        if self._prediction_intervals:
            self._interval_outcomes.append(observed.copy())

        if self.history:
            self.state_estimator.predict()

        expected_observation = _as_vector(
            "expected observation", self.observation_function(self.state_estimator.state)
        )
        residual = self.twin.twin_residual(observed, expected_observation)
        weighted_error = self.twin.weighted_twin_error(residual)
        self.twin.record_residual(weighted_error)
        self._error_history.append(weighted_error)

        uncertainty = (
            self.state_estimator.observation_covariance
            if fused_observation.uncertainty is None
            else np.diag(fused_observation.uncertainty**2)
        )
        anomaly_report = anomaly_detection(
            observed,
            expected_observation,
            uncertainty=uncertainty,
            threshold=self.anomaly_threshold,
        )

        estimated_state, estimated_covariance = self.state_estimator.update(observed)
        self._observation_history.append(observed.copy())

        if self.parameter_estimator is not None:
            if parameter_features is None and parameter_target is None:
                parameter_estimate = self.parameter_estimator.parameters.copy()
            elif parameter_features is None or parameter_target is None:
                raise ValueError(
                    "parameter_features and parameter_target must both be provided when updating parameters"
                )
            else:
                parameter_estimate = self.parameter_estimator.update(parameter_features, parameter_target)
        else:
            parameter_estimate = None

        selected_model_name: str | None = None
        selected_model: Callable[[np.ndarray], np.ndarray] | None = None
        if self.models and len(self._observation_history) >= 2:
            model_result = self.model_selector.select_best(
                self.models,
                self._observation_history[:-1],
                self._observation_history[1:],
            )
            selected_model_name = model_result.model_name
            selected_model = model_result.model

        anchor_room = self._make_room(estimated_state)
        children = self.reality_generator.branch(
            anchor_room,
            count=self.reality_count,
            scale=self.reality_scale,
        )
        observation_covariance = (
            self.state_estimator.observation_covariance
            if fused_observation.uncertainty is None
            else np.diag(fused_observation.uncertainty**2)
        )
        likelihoods: list[float] = []
        next_states: list[np.ndarray] = []
        for child in children:
            current_observation = _as_vector(
                "current observation", self.observation_function(child.x)
            )
            child_residual = self.twin.twin_residual(observed, current_observation)
            likelihoods.append(
                self.twin.likelihood_gaussian(child_residual, observation_covariance)
            )
            next_states.append(self._simulate_transition(child.x, selected_model))

        reality_weights = self.twin.evidence_update(
            (np.ones(len(children), dtype=float) / len(children)).tolist(),
            likelihoods,
        )
        stacked_states = np.vstack(next_states)
        predicted_state = np.sum(reality_weights[:, None] * stacked_states, axis=0)

        centered = stacked_states - predicted_state
        ensemble_covariance = (
            centered.T @ (centered * reality_weights[:, None])
            if len(children) > 1
            else np.zeros((predicted_state.shape[0], predicted_state.shape[0]), dtype=float)
        )
        propagated = uncertainty_propagation(
            estimated_state,
            estimated_covariance,
            lambda state: self._simulate_transition(state, selected_model),
            process_covariance=self.state_estimator.process_covariance,
        )
        predicted_covariance = propagated.covariance + ensemble_covariance

        predicted_observation_result = uncertainty_propagation(
            predicted_state,
            predicted_covariance,
            self.observation_function,
        )
        predicted_observation = predicted_observation_result.mean
        observation_std = np.sqrt(
            np.maximum(np.diag(predicted_observation_result.covariance), _EPSILON)
        )
        interval = prediction_interval(
            predicted_observation,
            observation_std,
            confidence=self.prediction_confidence,
        )
        if not isinstance(interval[0], np.ndarray) or not isinstance(interval[1], np.ndarray):
            raise AssertionError("vector prediction intervals must return numpy arrays")
        self._prediction_intervals.append((interval[0].copy(), interval[1].copy()))

        if np.sum(np.diag(predicted_covariance)) > 0.0:
            normalized_variance = np.diag(predicted_covariance) / np.sum(np.diag(predicted_covariance))
            uncertainty_score = qes_entropy(normalized_variance)
        else:
            uncertainty_score = 0.0

        drift_report = drift_detection(
            self._error_history,
            window=self.drift_window,
            mean_shift_threshold=self.drift_mean_shift_threshold,
            std_ratio_threshold=self.drift_std_ratio_threshold,
        )
        calibration_score = (
            calibration_check(self._prediction_intervals[:-1], self._interval_outcomes)
            if self._interval_outcomes
            else None
        )

        result = LoopStepResult(
            step=len(self.history),
            fused_observation=fused_observation,
            estimated_state=estimated_state.copy(),
            estimated_covariance=estimated_covariance.copy(),
            predicted_state=predicted_state.copy(),
            predicted_observation=predicted_observation.copy(),
            prediction_interval=(interval[0].copy(), interval[1].copy()),
            anomaly_report=anomaly_report,
            drift_report=drift_report,
            selected_model_name=selected_model_name,
            parameter_estimate=None if parameter_estimate is None else parameter_estimate.copy(),
            uncertainty_score=float(uncertainty_score),
            calibration_score=calibration_score,
            recommended_action=self._recommend_action(anomaly_report, drift_report, predicted_state),
            reality_weights=np.asarray(reality_weights, dtype=float).copy(),
        )
        self.history.append(result)
        return result


__all__ = [
    "AnomalyReport",
    "DigitalTwinLoop",
    "DriftReport",
    "LoopStepResult",
    "ModelSelectionResult",
    "ModelSelector",
    "Observation",
    "OnlineParameterEstimator",
    "PropagationResult",
    "StateEstimator",
    "anomaly_detection",
    "calibration_check",
    "drift_detection",
    "prediction_interval",
    "sensor_fusion",
    "uncertainty_propagation",
]
