"""Tests for `qes.digital_twin_loop` (Phase 9 -- Digital Twin 2.0)."""
from __future__ import annotations

import numpy as np
import pytest

import qes.digital_twin_loop as digital_twin_loop_module
from qes.digital_twin_loop import (
    DigitalTwinLoop,
    Observation,
    OnlineParameterEstimator,
    StateEstimator,
    anomaly_detection,
    calibration_check,
    drift_detection,
    prediction_interval,
    sensor_fusion,
    uncertainty_propagation,
)
from qes.reality_generator import RealityGenerator


def make_state_estimator(
    initial_state: list[float] | np.ndarray,
    *,
    initial_covariance: float | np.ndarray = 1.0,
    transition_matrix: np.ndarray | None = None,
    process_covariance: float | np.ndarray = 0.01,
    observation_covariance: float | np.ndarray = 0.04,
) -> StateEstimator:
    state = np.asarray(initial_state, dtype=float)
    dim = state.shape[0]
    covariance = (
        np.eye(dim, dtype=float) * float(initial_covariance)
        if np.asarray(initial_covariance).ndim == 0
        else np.asarray(initial_covariance, dtype=float)
    )
    transition = np.eye(dim, dtype=float) if transition_matrix is None else transition_matrix
    process = (
        np.eye(dim, dtype=float) * float(process_covariance)
        if np.asarray(process_covariance).ndim == 0
        else np.asarray(process_covariance, dtype=float)
    )
    observation = (
        np.eye(dim, dtype=float) * float(observation_covariance)
        if np.asarray(observation_covariance).ndim == 0
        else np.asarray(observation_covariance, dtype=float)
    )
    return StateEstimator(
        initial_state=state,
        initial_covariance=covariance,
        transition_matrix=transition,
        process_covariance=process,
        observation_covariance=observation,
    )


def good_decay_model(state: np.ndarray) -> np.ndarray:
    return 0.8 * np.asarray(state, dtype=float)


def biased_growth_model(state: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=float) + 0.7


class TestObservation:
    def test_normalizes_valid_inputs(self) -> None:
        observation = Observation(
            values=[1.0, 2.0],
            step=0,
            timestamp=1.5,
            uncertainty=[0.1, 0.2],
            sensor_id="sensor-a",
        )
        np.testing.assert_allclose(observation.values, [1.0, 2.0])
        np.testing.assert_allclose(observation.uncertainty, [0.1, 0.2])
        assert observation.step == 0
        assert observation.timestamp == pytest.approx(1.5)
        assert observation.sensor_id == "sensor-a"

    def test_rejects_negative_step_and_bad_uncertainty_shape(self) -> None:
        with pytest.raises(ValueError, match="step must be > 0"):
            Observation(values=[1.0], step=-1)
        with pytest.raises(ValueError, match="same shape as values"):
            Observation(values=[1.0, 2.0], uncertainty=[0.2])

    def test_rejects_nonpositive_uncertainty_and_bad_sensor_id(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            Observation(values=[1.0], uncertainty=[0.0])
        with pytest.raises(TypeError, match="sensor_id must be a string"):
            Observation(values=[1.0], sensor_id=123)


class TestValidationHelpers:
    def test_vector_and_matrix_helpers_validate_shape_and_finiteness(self) -> None:
        np.testing.assert_allclose(digital_twin_loop_module._as_vector("scalar", 2.0), [2.0])
        np.testing.assert_allclose(
            digital_twin_loop_module._as_matrix("diag", [1.0, 2.0]),
            np.diag([1.0, 2.0]),
        )
        np.testing.assert_allclose(
            digital_twin_loop_module._as_matrix("scalar-matrix", 3.0, size=2),
            np.eye(2) * 3.0,
        )

        with pytest.raises(TypeError, match="array-like and numeric"):
            digital_twin_loop_module._as_vector("values", object())
        with pytest.raises(ValueError, match="one-dimensional"):
            digital_twin_loop_module._as_vector("values", [[1.0], [2.0]])
        with pytest.raises(ValueError, match="finite values"):
            digital_twin_loop_module._as_vector("values", [1.0, np.nan])
        with pytest.raises(TypeError, match="array-like and numeric"):
            digital_twin_loop_module._as_matrix("matrix", object())
        with pytest.raises(ValueError, match="explicit size"):
            digital_twin_loop_module._as_matrix("matrix", 1.0)
        with pytest.raises(ValueError, match="one- or two-dimensional"):
            digital_twin_loop_module._as_matrix("matrix", np.zeros((1, 1, 1)))
        with pytest.raises(ValueError, match="must be square"):
            digital_twin_loop_module._as_matrix("matrix", [[1.0, 2.0]])
        with pytest.raises(ValueError, match="must have shape"):
            digital_twin_loop_module._as_matrix("matrix", np.eye(2), size=1)
        with pytest.raises(ValueError, match="finite values"):
            digital_twin_loop_module._as_matrix("matrix", [[1.0, np.inf], [0.0, 1.0]])

    def test_scalar_and_model_name_helpers_cover_fallbacks(self) -> None:
        with pytest.raises(TypeError, match="count must be an integer"):
            digital_twin_loop_module._validate_positive_int("count", 1.5)
        with pytest.raises(TypeError, match="rate must be a real number"):
            digital_twin_loop_module._validate_non_negative_real("rate", True)
        with pytest.raises(TypeError, match="rate must be a real number"):
            digital_twin_loop_module._validate_non_negative_real("rate", "bad")
        with pytest.raises(ValueError, match="rate must be finite"):
            digital_twin_loop_module._validate_non_negative_real("rate", float("inf"))
        with pytest.raises(ValueError, match="rate must be >="):
            digital_twin_loop_module._validate_non_negative_real("rate", -1.0)

        named_model = type("NamedModel", (), {"name": "custom-model", "__call__": lambda self, x: x})()
        anonymous_model = type("AnonymousModel", (), {"__call__": lambda self, x: x})()
        assert digital_twin_loop_module._model_name(named_model) == "custom-model"
        assert digital_twin_loop_module._model_name(good_decay_model) == "good_decay_model"
        assert digital_twin_loop_module._model_name(anonymous_model) == "AnonymousModel"


class TestSensorFusion:
    def test_inverse_variance_weighting_favors_more_precise_sensor(self) -> None:
        readings = [
            Observation(values=[10.0], uncertainty=[1.0]),
            Observation(values=[14.0], uncertainty=[2.0]),
        ]

        fused = sensor_fusion(readings)

        assert fused.values[0] == pytest.approx(10.8)
        assert fused.uncertainty is not None
        assert fused.uncertainty[0] == pytest.approx(np.sqrt(0.8))

    def test_external_weights_shift_fused_mean(self) -> None:
        readings = [
            Observation(values=[0.0], uncertainty=[1.0]),
            Observation(values=[10.0], uncertainty=[1.0]),
        ]

        fused = sensor_fusion(readings, weights=[1.0, 3.0])

        assert fused.values[0] == pytest.approx(7.5)
        assert fused.uncertainty is not None
        assert fused.uncertainty[0] == pytest.approx(0.5)

    def test_carries_forward_latest_step_and_timestamp_metadata(self) -> None:
        earlier = Observation(values=[1.0], step=1, timestamp=1.0, uncertainty=[0.5])
        later = Observation(values=[2.0], step=3, timestamp=0.5, uncertainty=[0.5])

        fused = sensor_fusion([earlier, later])

        assert fused.step == 3
        assert fused.timestamp == pytest.approx(0.5)
        assert fused.sensor_id == "fused"

    def test_rejects_mismatched_dimensions_and_invalid_weights(self) -> None:
        with pytest.raises(ValueError, match="same dimensionality"):
            sensor_fusion(
                [
                    Observation(values=[1.0], uncertainty=[0.2]),
                    Observation(values=[1.0, 2.0], uncertainty=[0.2, 0.2]),
                ]
            )
        with pytest.raises(ValueError, match="same length as readings"):
            sensor_fusion([Observation(values=[1.0])], weights=[1.0, 2.0])
        with pytest.raises(ValueError, match="weights must be > 0"):
            sensor_fusion([Observation(values=[1.0])], weights=[0.0])

    def test_requires_observation_list_with_at_least_one_entry(self) -> None:
        with pytest.raises(TypeError, match="list of Observation"):
            sensor_fusion("bad")
        with pytest.raises(ValueError, match="at least one"):
            sensor_fusion([])
        with pytest.raises(TypeError, match="Observation instances"):
            sensor_fusion([Observation(values=[1.0]), object()])


class TestOnlineParameterEstimator:
    def test_predict_uses_current_parameter_estimate(self) -> None:
        estimator = OnlineParameterEstimator(initial_parameters=[2.0, -1.0])

        prediction = estimator.predict([3.0, 4.0])

        assert prediction == pytest.approx(2.0)

    def test_converges_toward_true_linear_parameters(self) -> None:
        rng = np.random.default_rng(123)
        true_parameters = np.array([2.0, -1.5], dtype=float)
        estimator = OnlineParameterEstimator(
            initial_parameters=[0.0, 0.0],
            forgetting_factor=1.0,
            covariance_scale=100.0,
        )

        for _ in range(250):
            features = rng.normal(size=2)
            target = float(features @ true_parameters)
            estimator.update(features, target)

        np.testing.assert_allclose(estimator.parameters, true_parameters, atol=5e-2)

    def test_rejects_bad_feature_length_and_non_finite_target(self) -> None:
        estimator = OnlineParameterEstimator(initial_parameters=[0.0, 0.0])

        with pytest.raises(ValueError, match="same length as parameters"):
            estimator.predict([1.0])
        with pytest.raises(ValueError, match="target must be finite"):
            estimator.update([1.0, 2.0], float("nan"))

    def test_rejects_invalid_forgetting_factor_and_update_feature_length(self) -> None:
        with pytest.raises(ValueError, match="forgetting_factor must satisfy"):
            OnlineParameterEstimator(initial_parameters=[0.0], forgetting_factor=0.0)

        estimator = OnlineParameterEstimator(initial_parameters=[0.0, 0.0])
        with pytest.raises(ValueError, match="same length as parameters"):
            estimator.update([1.0], 1.0)


class TestStateEstimator:
    def test_predict_applies_transition_and_control_input(self) -> None:
        estimator = make_state_estimator(
            [0.0, 1.0],
            transition_matrix=np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float),
            process_covariance=np.eye(2, dtype=float) * 0.01,
            observation_covariance=np.eye(2, dtype=float) * 0.1,
        )

        state, covariance = estimator.predict(
            control_input=[0.5],
            control_matrix=np.array([[0.5], [1.0]], dtype=float),
        )

        np.testing.assert_allclose(state, [1.25, 1.5])
        assert covariance.shape == (2, 2)

    def test_update_moves_estimate_toward_observation_and_reduces_variance(self) -> None:
        estimator = make_state_estimator(
            [0.0],
            initial_covariance=np.array([[1.0]], dtype=float),
            process_covariance=np.array([[0.0]], dtype=float),
            observation_covariance=np.array([[0.25]], dtype=float),
        )

        state, covariance = estimator.update([1.0])

        assert state[0] == pytest.approx(0.8)
        assert covariance[0, 0] == pytest.approx(0.2)

    def test_update_rejects_wrong_observation_shape(self) -> None:
        estimator = make_state_estimator([0.0, 0.0])

        with pytest.raises(ValueError, match="observation must have shape"):
            estimator.update([1.0])

    def test_predict_rejects_control_shape_without_matrix(self) -> None:
        estimator = make_state_estimator([0.0, 0.0])

        with pytest.raises(ValueError, match="match state shape"):
            estimator.predict(control_input=[1.0])

    def test_predict_accepts_control_input_without_matrix_when_shapes_match(self) -> None:
        estimator = make_state_estimator([0.0, 1.0])

        state, _covariance = estimator.predict(control_input=[0.5, -0.5])

        np.testing.assert_allclose(state, [0.5, 0.5])


class TestUncertaintyPropagation:
    def test_matches_linear_covariance_propagation(self) -> None:
        transition = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=float)
        covariance = np.diag([0.2, 0.3])
        process_covariance = np.diag([0.05, 0.01])

        result = uncertainty_propagation(
            state_mean=[1.0, -1.0],
            covariance=covariance,
            transition_function=lambda state: transition @ state,
            process_covariance=process_covariance,
        )

        np.testing.assert_allclose(result.mean, [2.0, -0.5], atol=1e-10)
        np.testing.assert_allclose(
            result.covariance,
            transition @ covariance @ transition.T + process_covariance,
            atol=1e-8,
        )

    def test_larger_input_uncertainty_increases_output_uncertainty(self) -> None:
        small = uncertainty_propagation(
            state_mean=[1.0],
            covariance=np.array([[0.1]], dtype=float),
            transition_function=lambda state: np.array([state[0] ** 2], dtype=float),
        )
        large = uncertainty_propagation(
            state_mean=[1.0],
            covariance=np.array([[0.5]], dtype=float),
            transition_function=lambda state: np.array([state[0] ** 2], dtype=float),
        )

        assert large.covariance[0, 0] > small.covariance[0, 0]

    def test_rejects_non_positive_epsilon_and_shape_changes(self) -> None:
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            uncertainty_propagation(
                state_mean=[0.0],
                covariance=np.array([[1.0]], dtype=float),
                transition_function=lambda state: np.array([state[0]], dtype=float),
                epsilon=0.0,
            )

        def inconsistent_transition(state: np.ndarray) -> np.ndarray:
            if abs(state[0]) < 1e-12:
                return np.array([state[0]], dtype=float)
            return np.array([state[0], state[0]], dtype=float)

        with pytest.raises(ValueError, match="consistent output shape"):
            uncertainty_propagation(
                state_mean=[0.0],
                covariance=np.array([[1.0]], dtype=float),
                transition_function=inconsistent_transition,
            )


class TestModelSelector:
    def test_score_model_reports_zero_residual_for_exact_model(self) -> None:
        selector = DigitalTwinLoop(
            state_estimator=make_state_estimator([0.0]),
        ).model_selector

        result = selector.score_model(
            good_decay_model,
            inputs=[np.array([1.0]), np.array([2.0])],
            targets=[np.array([0.8]), np.array([1.6])],
        )

        assert result.model_name == "good_decay_model"
        assert result.residual_sum_squares == pytest.approx(0.0)
        assert result.score == pytest.approx(0.0)

    def test_select_best_prefers_lower_residual_model(self) -> None:
        selector = DigitalTwinLoop(
            state_estimator=make_state_estimator([0.0]),
        ).model_selector

        result = selector.select_best(
            [good_decay_model, biased_growth_model],
            inputs=[np.array([1.0]), np.array([0.8])],
            targets=[np.array([0.8]), np.array([0.64])],
        )

        assert result.model_name == "good_decay_model"

    def test_rank_models_and_reject_shape_mismatch(self) -> None:
        selector = DigitalTwinLoop(
            state_estimator=make_state_estimator([0.0]),
        ).model_selector
        ranked = selector.rank_models(
            [good_decay_model, biased_growth_model],
            inputs=[np.array([1.0])],
            targets=[np.array([0.8])],
        )

        assert [item.model_name for item in ranked] == [
            "good_decay_model",
            "biased_growth_model",
        ]

        with pytest.raises(ValueError, match="same shape as targets"):
            selector.score_model(
                lambda state: np.array([state[0], state[0]], dtype=float),
                inputs=[np.array([1.0])],
                targets=[np.array([0.8])],
            )

    def test_validates_callable_and_non_empty_batches(self) -> None:
        selector = DigitalTwinLoop(state_estimator=make_state_estimator([0.0])).model_selector

        with pytest.raises(TypeError, match="model must be callable"):
            selector.score_model(None, inputs=[np.array([1.0])], targets=[np.array([1.0])])
        with pytest.raises(ValueError, match="must be non-empty"):
            selector.score_model(good_decay_model, inputs=[], targets=[])
        with pytest.raises(ValueError, match="same length"):
            selector.score_model(
                good_decay_model,
                inputs=[np.array([1.0]), np.array([2.0])],
                targets=[np.array([1.0])],
            )
        with pytest.raises(ValueError, match="at least one callable"):
            selector.select_best([], inputs=[np.array([1.0])], targets=[np.array([1.0])])
        with pytest.raises(ValueError, match="at least one callable"):
            selector.rank_models([], inputs=[np.array([1.0])], targets=[np.array([1.0])])


class TestAnomalyDetection:
    def test_flags_clear_outlier(self) -> None:
        report = anomaly_detection(
            observation=[10.0],
            expected=[0.0],
            uncertainty=[1.0],
            threshold=3.0,
        )

        assert report.flagged is True
        assert report.max_z_score == pytest.approx(10.0)

    def test_leaves_normal_observation_unflagged(self) -> None:
        report = anomaly_detection(
            observation=[0.2, -0.1],
            expected=[0.0, 0.0],
            uncertainty=[1.0, 1.0],
            threshold=3.0,
        )

        assert report.flagged is False
        np.testing.assert_allclose(report.residual, [0.2, -0.1])

    def test_accepts_covariance_matrix_and_rejects_bad_threshold(self) -> None:
        report = anomaly_detection(
            observation=[1.0, 0.0],
            expected=[0.0, 0.0],
            uncertainty=np.diag([0.25, 1.0]),
            threshold=3.0,
        )
        assert report.z_scores[0] == pytest.approx(2.0)

        with pytest.raises(ValueError, match="threshold must be > 0"):
            anomaly_detection([1.0], [0.0], [1.0], threshold=0.0)
        with pytest.raises(ValueError, match="same shape"):
            anomaly_detection([1.0], [0.0, 1.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="same shape as observation"):
            anomaly_detection([1.0], [0.0], [1.0, 2.0])


class TestPredictionAndCalibration:
    def test_prediction_interval_returns_expected_scalar_bounds(self) -> None:
        lower, upper = prediction_interval(mean=0.0, std=1.0, confidence=0.95)

        assert lower == pytest.approx(-1.959963984540054)
        assert upper == pytest.approx(1.959963984540054)

    def test_prediction_interval_supports_vectors_and_rejects_bad_confidence(self) -> None:
        lower, upper = prediction_interval(
            mean=np.array([1.0, -1.0], dtype=float),
            std=np.array([0.5, 0.25], dtype=float),
            confidence=0.9,
        )

        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert lower.shape == (2,)
        assert upper.shape == (2,)

        with pytest.raises(ValueError, match="0 < confidence < 1"):
            prediction_interval(mean=0.0, std=1.0, confidence=1.0)
        with pytest.raises(ValueError, match="same shape"):
            prediction_interval(mean=np.array([0.0]), std=np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="std must be >="):
            prediction_interval(mean=0.0, std=-1.0)

    def test_calibration_check_reports_empirical_coverage(self) -> None:
        coverage = calibration_check(
            predicted_intervals=[
                (np.array([0.0]), np.array([1.0])),
                (np.array([0.0]), np.array([1.0])),
            ],
            actual_outcomes=[np.array([0.5]), np.array([1.5])],
        )

        assert coverage == pytest.approx(0.5)

    def test_calibration_check_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            calibration_check(
                predicted_intervals=[
                    (np.array([0.0]), np.array([1.0])),
                    (np.array([0.0]), np.array([1.0])),
                ],
                actual_outcomes=[np.array([0.5])],
            )

    def test_calibration_check_rejects_empty_inputs_and_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must be non-empty"):
            calibration_check([], [])
        with pytest.raises(ValueError, match="must share the same shape"):
            calibration_check(
                predicted_intervals=[(np.array([0.0]), np.array([1.0, 2.0]))],
                actual_outcomes=[np.array([0.5])],
            )


class TestDriftDetection:
    def test_flags_shifted_residual_distribution(self) -> None:
        report = drift_detection(
            residuals=[0.0, 0.1, -0.1, 0.0, 2.0, 2.2, 1.9, 2.1],
            window=4,
            mean_shift_threshold=2.0,
        )

        assert report.flagged is True
        assert report.mean_shift_sigma > 2.0

    def test_keeps_stable_distribution_unflagged(self) -> None:
        report = drift_detection(
            residuals=[0.0, 0.1, -0.1, 0.05, 0.0, 0.1, -0.05, 0.02],
            window=4,
            mean_shift_threshold=2.0,
        )

        assert report.flagged is False

    def test_returns_neutral_report_when_history_is_too_short(self) -> None:
        report = drift_detection(residuals=[0.1, 0.2, 0.3], window=2)

        assert report.flagged is False
        assert report.mean_shift_sigma == pytest.approx(0.0)
        assert report.std_ratio == pytest.approx(1.0)

    def test_rejects_zero_std_ratio_threshold(self) -> None:
        with pytest.raises(ValueError, match="std_ratio_threshold must be > 0"):
            drift_detection(residuals=[0.0, 0.0, 0.0, 0.0], window=2, std_ratio_threshold=0.0)


class TestDigitalTwinLoop:
    def test_init_validates_state_estimator_and_bounds(self) -> None:
        with pytest.raises(TypeError, match="state_estimator must be a StateEstimator"):
            DigitalTwinLoop(state_estimator="bad")
        with pytest.raises(ValueError, match="reference_state must have shape"):
            DigitalTwinLoop(
                state_estimator=make_state_estimator([0.0]),
                reference_state=[0.0, 1.0],
            )
        with pytest.raises(ValueError, match="must have shape"):
            DigitalTwinLoop(
                state_estimator=make_state_estimator([0.0]),
                lower_bounds=[-1.0, -2.0],
            )
        with pytest.raises(ValueError, match="strictly less"):
            DigitalTwinLoop(
                state_estimator=make_state_estimator([0.0]),
                lower_bounds=[1.0],
                upper_bounds=[1.0],
            )

    def test_observe_returns_sane_loop_step_result(self) -> None:
        loop = DigitalTwinLoop(
            state_estimator=make_state_estimator([0.0]),
            reality_generator=RealityGenerator(np.random.default_rng(0)),
            reality_count=4,
            reality_scale=0.0,
        )

        result = loop.observe([Observation(values=[1.0], step=0, uncertainty=[0.2])])

        assert result.step == 0
        assert result.selected_model_name is None
        assert result.fused_observation.sensor_id == "fused"
        assert result.estimated_state.shape == (1,)
        assert result.estimated_covariance.shape == (1, 1)
        assert result.predicted_observation.shape == (1,)
        assert result.prediction_interval[0].shape == (1,)
        assert result.prediction_interval[1].shape == (1,)
        assert result.reality_weights.shape == (4,)
        assert len(loop.history) == 1

    def test_observe_selects_best_model_and_computes_calibration_after_second_step(self) -> None:
        loop = DigitalTwinLoop(
            state_estimator=make_state_estimator([0.0]),
            reality_generator=RealityGenerator(np.random.default_rng(1)),
            models=[good_decay_model, biased_growth_model],
            reality_count=3,
            reality_scale=0.0,
        )

        loop.observe([Observation(values=[1.0], step=0, uncertainty=[0.2])])
        result = loop.observe([Observation(values=[0.8], step=1, uncertainty=[0.2])])

        assert result.selected_model_name == "good_decay_model"
        assert result.calibration_score == pytest.approx(1.0)

    def test_observe_updates_online_parameter_estimator_when_data_supplied(self) -> None:
        parameter_estimator = OnlineParameterEstimator(initial_parameters=[0.0, 0.0])
        loop = DigitalTwinLoop(
            state_estimator=make_state_estimator([0.0]),
            reality_generator=RealityGenerator(np.random.default_rng(2)),
            parameter_estimator=parameter_estimator,
            reality_count=2,
            reality_scale=0.0,
        )

        result = loop.observe(
            [Observation(values=[0.5], step=0, uncertainty=[0.2])],
            parameter_features=[1.0, 2.0],
            parameter_target=5.0,
        )

        assert result.parameter_estimate is not None
        assert result.parameter_estimate[0] > 0.0
        assert result.parameter_estimate[1] > 0.0

    def test_observe_reuses_current_parameter_estimate_when_no_update_requested(self) -> None:
        parameter_estimator = OnlineParameterEstimator(initial_parameters=[1.0])
        loop = DigitalTwinLoop(
            state_estimator=make_state_estimator([0.0]),
            parameter_estimator=parameter_estimator,
            reality_generator=RealityGenerator(np.random.default_rng(20)),
            reality_count=1,
            reality_scale=0.0,
        )

        result = loop.observe([Observation(values=[0.5], step=0, uncertainty=[0.2])])

        np.testing.assert_allclose(result.parameter_estimate, [1.0])

    def test_observe_requires_parameter_features_and_target_together(self) -> None:
        loop = DigitalTwinLoop(
            state_estimator=make_state_estimator([0.0]),
            parameter_estimator=OnlineParameterEstimator(initial_parameters=[0.0]),
            reality_generator=RealityGenerator(np.random.default_rng(3)),
            reality_count=2,
            reality_scale=0.0,
        )

        with pytest.raises(ValueError, match="must both be provided"):
            loop.observe(
                [Observation(values=[0.5], step=0, uncertainty=[0.2])],
                parameter_features=[1.0],
            )

    def test_observe_recommends_sensor_inspection_for_obvious_anomaly(self) -> None:
        loop = DigitalTwinLoop(
            state_estimator=make_state_estimator([0.0]),
            reality_generator=RealityGenerator(np.random.default_rng(4)),
            anomaly_threshold=2.0,
            reality_count=2,
            reality_scale=0.0,
        )

        result = loop.observe([Observation(values=[10.0], step=0, uncertainty=[0.1])])

        assert result.anomaly_report.flagged is True
        assert "Inspect sensors" in result.recommended_action

    def test_observe_surfaces_drift_when_residuals_shift_over_time(self) -> None:
        loop = DigitalTwinLoop(
            state_estimator=make_state_estimator([0.0]),
            reality_generator=RealityGenerator(np.random.default_rng(5)),
            anomaly_threshold=100.0,
            drift_window=2,
            drift_mean_shift_threshold=0.5,
            reality_count=2,
            reality_scale=0.0,
        )

        result = None
        for step, value in enumerate([0.0, 0.0, 0.0, 0.0, 5.0, 5.0]):
            result = loop.observe([Observation(values=[value], step=step, uncertainty=[0.2])])

        assert result is not None
        assert result.drift_report.flagged is True
        assert "Re-estimate model structure or parameters" in result.recommended_action

    def test_observe_uses_transition_function_with_parameter_estimator(self) -> None:
        seen_parameters = []

        def transition(state: np.ndarray, parameters: np.ndarray | None) -> np.ndarray:
            seen_parameters.append(None if parameters is None else parameters.copy())
            return state + parameters

        loop = DigitalTwinLoop(
            state_estimator=make_state_estimator([0.0]),
            reality_generator=RealityGenerator(np.random.default_rng(30)),
            parameter_estimator=OnlineParameterEstimator(initial_parameters=[0.25]),
            transition_function=transition,
            reality_count=1,
            reality_scale=0.0,
        )

        result = loop.observe([Observation(values=[0.0], step=0, uncertainty=[0.2])])

        np.testing.assert_allclose(seen_parameters[0], [0.25])
        assert result.predicted_state[0] == pytest.approx(0.25)

    def test_observe_raises_when_prediction_interval_contract_is_broken(self, monkeypatch) -> None:
        loop = DigitalTwinLoop(
            state_estimator=make_state_estimator([0.0]),
            reality_generator=RealityGenerator(np.random.default_rng(31)),
            reality_count=1,
            reality_scale=0.0,
        )
        monkeypatch.setattr(digital_twin_loop_module, "prediction_interval", lambda *args, **kwargs: (0.0, 1.0))

        with pytest.raises(AssertionError, match="vector prediction intervals"):
            loop.observe([Observation(values=[0.0], step=0, uncertainty=[0.2])])

    def test_observe_reports_zero_uncertainty_score_when_covariance_is_zero(self) -> None:
        loop = DigitalTwinLoop(
            state_estimator=make_state_estimator(
                [0.0],
                initial_covariance=np.array([[0.0]], dtype=float),
                process_covariance=np.array([[0.0]], dtype=float),
                observation_covariance=np.array([[1.0]], dtype=float),
            ),
            reality_generator=RealityGenerator(np.random.default_rng(32)),
            reality_count=1,
            reality_scale=0.0,
        )

        result = loop.observe([Observation(values=[0.0], step=0, uncertainty=[1.0])])

        assert result.uncertainty_score == pytest.approx(0.0)
