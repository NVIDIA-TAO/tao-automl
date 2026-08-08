# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bayesian AutoML algorithm modules"""
import copy
import hashlib
import json
import numpy as np
import math
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm, qmc
from scipy.optimize import minimize

from tao_automl.brain import network_utils
from tao_automl.utils.math_utils import (
    JobStates, get_valid_range, clamp_value,
    get_valid_options, get_option_weights, fix_input_dimension
)
from tao_automl.brain.base import (
    OBSERVATION_UTILITY_VERSION,
    AutoMLAlgorithmBase,
    is_nan_value,
)
from tao_automl.brain.objective_acquisition import (
    constrained_latency_ei,
    default_calibration_points,
    expected_improvement_maximize,
    parego_utilities,
    retained_accuracy_threshold,
    valid_accuracy_observations,
    valid_objective_observations,
)
from tao_automl.recommendation_audit import canonical_audit_sha256
from tao_automl.selection import accuracy_feasibility_boundary
from tao_automl.utils.spec_utils import get_flatten_specs

logger = logging.getLogger(__name__)


def _get_total_epochs_from_specs(specs):
    """Extract total epochs from spec dict"""
    max_epoch = 100.0
    for key1 in specs:
        if key1 in ("training_config", "train_config", "train"):
            for key2 in specs[key1]:
                if key2 in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch"):
                    max_epoch = int(specs[key1][key2])
                elif key2 == "train_config":
                    for key3 in specs[key1][key2]:
                        if key3 == "runner":
                            for key4 in specs[key1][key2][key3]:
                                if key4 == "max_epochs":
                                    max_epoch = int(specs[key1][key2][key3][key4])
        elif key1 == "num_epochs":
            max_epoch = int(specs[key1])
    return max_epoch


class Bayesian(AutoMLAlgorithmBase):
    """Bayesian AutoML algorithm class"""

    def __init__(
        self,
        context,
        state_store,
        network,
        parameters,
        metric="kpi",
        direction=None,
        objective_config=None,
        acquisition_settings=None,
    ):
        """Initialize the Bayesian algorithm class

        Args:
            context: AutoMLContext instance
            state_store: StateStore instance
            network: model we are running AutoML on
            parameters: automl sweepable parameters
        """
        super().__init__(context, state_store, network, parameters)
        # Keep the constructor-time schema immutable for resume compatibility.
        # Recommendation conversion may apply range overrides to the live
        # ``self.parameters`` records in place.
        self._parameter_schema_identity = copy.deepcopy(parameters)
        self._configure_objective(metric, direction)
        self.objective_config = objective_config
        self.objective_acquisition_mode = self._resolve_objective_acquisition_mode()
        self.acquisition_settings = self._normalize_acquisition_settings(
            acquisition_settings
        )
        self.xi = self.acquisition_settings["xi"]
        self.calibration_points = self.acquisition_settings["calibration_points"]
        self.parego_augmentation_rho = self.acquisition_settings[
            "augmentation_rho"
        ]
        self._recommendation_count = 0
        self._model_based_iteration = 0
        self._rng = np.random.RandomState(self.random_seed)
        self._last_acquisition_value = None
        self._pending_recommendation_audits = []

        selection_config = (
            objective_config.selection_config
            if objective_config is not None
            else None
        )
        self.accuracy_metric = (
            selection_config.accuracy_metric
            if selection_config is not None
            else None
        )
        self.latency_metric = (
            selection_config.latency_metric
            if selection_config is not None
            else None
        )

        self._acquisition_audit = {
            "version": 1,
            "mode": self.objective_acquisition_mode,
            "method": self._configured_acquisition_method(),
            "active_method": None,
            "stage": "initialized",
            "uses_raw_objectives": self._uses_native_objective_acquisition,
            "selector_score_used": False
            if self._uses_native_objective_acquisition
            else None,
            "accuracy_metric": self.accuracy_metric,
            "latency_metric": self.latency_metric,
            "configuration": copy.deepcopy(self.acquisition_settings),
            "recommendations_issued": 0,
            "model_based_iterations": 0,
            "observation_count": 0,
        }

        self.gp = self._new_gp()
        self.accuracy_gp = self._new_gp()
        self.latency_gp = self._new_gp()
        # The following 2 need to be stored
        self.Xs = []
        self.ys = []

        self.num_restarts = 5

        self.num_epochs_per_experiment = _get_total_epochs_from_specs(self.default_train_spec)

    @property
    def _uses_native_objective_acquisition(self):
        """Return whether accuracy/latency acquisition uses raw objectives."""
        return self.objective_acquisition_mode in {
            "accuracy",
            "latency",
            "multi_objective",
        }

    @property
    def acquisition_audit(self):
        """Return a copy of the current recommendation-policy audit."""
        return copy.deepcopy(self._acquisition_audit)

    def consume_last_recommendation_audits(self):
        """Return and clear immutable per-proposal acquisition audit payloads."""
        audits = copy.deepcopy(self._pending_recommendation_audits)
        self._pending_recommendation_audits.clear()
        return audits

    def _new_gp(self):
        """Return an independently fitted GP with the frozen search seed."""
        length_scale = [1.0] * len(self.parameters)
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=length_scale,
            nu=2.5,
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=10,
            random_state=self.random_seed,
        )

    def _resolve_objective_acquisition_mode(self):
        """Resolve the production acquisition mode from ObjectiveConfig."""
        if (
            self.objective_config is None
            or not self.objective_config.is_multi_objective
            or not self.objective_config.has_archive_selector
        ):
            return "single_objective"
        return self.objective_config.selection_config.mode

    def _normalize_acquisition_settings(self, raw_settings):
        """Validate deterministic Bayesian acquisition settings."""
        if raw_settings is None:
            settings = {}
        elif isinstance(raw_settings, dict):
            settings = dict(raw_settings)
        else:
            raise TypeError("objective acquisition settings must be a mapping")

        allowed = {"calibration_points", "xi", "augmentation_rho"}
        unknown = sorted(set(settings) - allowed)
        if unknown:
            raise ValueError(
                "Unsupported objective acquisition setting(s): "
                + ", ".join(unknown)
            )

        raw_calibration = settings.get("calibration_points")
        if raw_calibration is None:
            calibration_points = default_calibration_points(
                len(self.parameters)
            )
        else:
            if isinstance(raw_calibration, (bool, np.bool_)):
                raise TypeError("calibration_points must be an integer")
            try:
                calibration_points = int(raw_calibration)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("calibration_points must be an integer") from exc
            if (
                isinstance(raw_calibration, (float, np.floating))
                and not float(raw_calibration).is_integer()
            ):
                raise ValueError("calibration_points must be an integer")
            if calibration_points < 2:
                raise ValueError("calibration_points must be at least 2")

        xi = self._finite_observation_value(settings.get("xi", 0.01))
        if xi is None or xi < 0.0:
            raise ValueError("objective acquisition xi must be finite and >= 0")

        selection_config = (
            self.objective_config.selection_config
            if self.objective_config is not None
            else None
        )
        default_rho = (
            selection_config.augmentation_rho
            if selection_config is not None
            else 1e-6
        )
        augmentation_rho = self._finite_observation_value(
            settings.get("augmentation_rho", default_rho)
        )
        if augmentation_rho is None or augmentation_rho < 0.0:
            raise ValueError(
                "objective acquisition augmentation_rho must be finite and >= 0"
            )
        return {
            "calibration_points": calibration_points,
            "xi": xi,
            "augmentation_rho": augmentation_rho,
        }

    def _configured_acquisition_method(self):
        return {
            "accuracy": "accuracy_expected_improvement",
            "latency": "constrained_latency_expected_improvement",
            "multi_objective": "parego_expected_improvement",
            "single_objective": "single_objective_expected_improvement",
        }[self.objective_acquisition_mode]

    def _acquisition_signature(self):
        """Return the persisted configuration identity used for safe resume."""
        # ``generate_automl_param_rec_value`` applies custom range overrides to
        # parameter records in place.  Build the identity from an effective
        # copy so the signature is identical before and after a recommendation
        # while still binding every range-defining field.
        effective_parameters = copy.deepcopy(self._parameter_schema_identity)
        for parameter in effective_parameters:
            if not isinstance(parameter, dict):
                continue
            parameter_name = parameter.get("parameter")
            overrides = self.custom_ranges.get(parameter_name)
            if isinstance(overrides, dict):
                for key, value in overrides.items():
                    if value is not None:
                        parameter[key] = copy.deepcopy(value)

        return {
            "version": 2,
            "mode": self.objective_acquisition_mode,
            "objective_config": (
                self.objective_config.to_dict()
                if self._uses_native_objective_acquisition
                else None
            ),
            "settings": copy.deepcopy(self.acquisition_settings),
            "random_seed": self.random_seed,
            "network": self.network,
            "action": getattr(self.context, "action", None),
            "handler_id": getattr(self.context, "handler_id", None),
            "parameter_count": len(effective_parameters),
            "parameter_schema_sha256": canonical_audit_sha256(
                self._parameter_schema_identity
            ),
            "search_space_sha256": canonical_audit_sha256(
                effective_parameters
            ),
            "custom_ranges_sha256": canonical_audit_sha256(
                self.custom_ranges
            ),
            "train_spec_sha256": canonical_audit_sha256(
                self.default_train_spec
            ),
        }

    def _rng_state_to_dict(self):
        algorithm, keys, position, has_gauss, cached_gaussian = (
            self._rng.get_state()
        )
        return {
            "algorithm": algorithm,
            "keys": keys.tolist(),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        }

    def _restore_rng_state(self, raw_state):
        if not raw_state:
            return
        try:
            state = (
                str(raw_state["algorithm"]),
                np.asarray(raw_state["keys"], dtype=np.uint32),
                int(raw_state["position"]),
                int(raw_state["has_gauss"]),
                float(raw_state["cached_gaussian"]),
            )
            self._rng.set_state(state)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid persisted Bayesian RNG state") from exc

    def _calibration_point(self, design_index):
        """Return one deterministic scrambled-Halton initial-design point."""
        if (
            isinstance(design_index, (bool, np.bool_))
            or not isinstance(design_index, int)
            or design_index < 0
        ):
            raise ValueError("calibration design index must be a non-negative integer")
        sampler = qmc.Halton(
            d=len(self.parameters),
            scramble=True,
            seed=self.random_seed,
        )
        if design_index:
            sampler.fast_forward(design_index)
        point = sampler.random(1)[0]
        self._acquisition_audit.update({
            "calibration_design": "scrambled_halton",
            "calibration_design_seed": self.random_seed,
            "calibration_design_index": design_index,
        })
        return point

    def _objective_observations(self, history):
        """Return the raw observations required by the active native mode."""
        if self.objective_acquisition_mode == "accuracy":
            return valid_accuracy_observations(
                history,
                accuracy_metric=self.accuracy_metric,
            )
        observations = valid_objective_observations(
            history,
            accuracy_metric=self.accuracy_metric,
            latency_metric=self.latency_metric,
        )
        return observations

    def _record_objective_audit(
        self,
        *,
        observations,
        stage,
        active_method,
        **details,
    ):
        """Record the exact raw archive and acquisition state for this proposal."""
        for key in (
            "calibration_design",
            "calibration_design_seed",
            "calibration_design_index",
            "accuracy_reference",
            "accuracy_threshold",
            "accuracy_feasibility_boundary",
            "feasible_latency_incumbent",
            "feasible_observation_count",
            "incumbent_accuracy",
            "optimization_direction",
            "parego",
            "acquisition_index",
        ):
            self._acquisition_audit.pop(key, None)
        self._acquisition_audit.update({
            "stage": stage,
            "active_method": active_method,
            "recommendations_issued": self._recommendation_count,
            "model_based_iterations": self._model_based_iteration,
            "observation_count": len(observations),
            "observations": [
                dict(
                    {
                        "candidate_id": item.candidate_id,
                        "accuracy": item.accuracy,
                    },
                    **(
                        {"latency": item.latency}
                        if hasattr(item, "latency")
                        else {}
                    ),
                )
                for item in observations
            ],
        })
        self._acquisition_audit.update(copy.deepcopy(details))

    def _append_suggestion(self, suggestions):
        """Record one normalized proposal and convert it to user specifications."""
        suggestions = np.asarray(suggestions, dtype=float).reshape(-1)
        if (
            len(suggestions) != len(self.parameters)
            or not np.all(np.isfinite(suggestions))
        ):
            raise ValueError(
                "Bayesian acquisition must return one finite value per parameter"
            )
        suggestions = np.clip(suggestions, 0.0, 1.0)
        self.Xs.append(suggestions)
        recommendation_index = self._recommendation_count
        self._recommendation_count += 1
        self._acquisition_audit["recommendations_issued"] = (
            self._recommendation_count
        )

        recommendations = []
        for param_dict, suggestion in zip(self.parameters, suggestions):
            recommendation_value = self.generate_automl_param_rec_value(
                param_dict,
                suggestion,
            )
            logger.info(
                "Recommendation param: %s value: %s",
                param_dict["parameter"],
                recommendation_value,
            )
            recommendations.append(recommendation_value)
        rng_state = self._rng_state_to_dict()
        rng_state_hash = hashlib.sha256(
            json.dumps(
                rng_state,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        observation_rows = self._acquisition_audit.get("observations", [])
        observation_summary = {
            "count": len(observation_rows),
            "candidate_ids": [
                item.get("candidate_id") for item in observation_rows
            ],
        }
        if observation_rows:
            accuracies = [item["accuracy"] for item in observation_rows]
            objective_bounds = {
                self.accuracy_metric: {
                    "minimum": min(accuracies),
                    "maximum": max(accuracies),
                }
            }
            latencies = [
                item["latency"]
                for item in observation_rows
                if "latency" in item
            ]
            if latencies:
                objective_bounds[self.latency_metric] = {
                    "minimum": min(latencies),
                    "maximum": max(latencies),
                }
            observation_summary["objective_bounds"] = objective_bounds
        if self.objective_acquisition_mode == "accuracy":
            objectives = [
                {"metric": self.accuracy_metric, "direction": "maximize"},
            ]
        elif self._uses_native_objective_acquisition:
            objectives = [
                {"metric": self.accuracy_metric, "direction": "maximize"},
                {"metric": self.latency_metric, "direction": "minimize"},
            ]
        else:
            objectives = [
                {"metric": self.metric, "direction": self.metric_direction}
            ]
        audit = {
            "schema_version": 1,
            "recommendation_index": recommendation_index,
            "acquisition_mode": self.objective_acquisition_mode,
            "acquisition_configuration": copy.deepcopy(
                self.acquisition_settings
            ),
            "stage": self._acquisition_audit.get("stage"),
            "calibration_or_acquisition_index": self._acquisition_audit.get(
                "calibration_design_index",
                self._acquisition_audit.get(
                    "acquisition_index",
                    recommendation_index,
                ),
            ),
            "objectives": objectives,
            "observation_summary": observation_summary,
            "seed": self.random_seed,
            "rng_state_sha256": rng_state_hash,
            "acquisition_function": self._acquisition_audit.get(
                "active_method"
            ),
            "chosen_acquisition_value": self._last_acquisition_value,
            "normalized_suggestion": suggestions.tolist(),
            # Freeze the exact decision state with this proposal. The rolling
            # brain-level acquisition audit is replaced by the next call, so
            # it cannot by itself prove which retained-accuracy threshold,
            # feasible incumbent, optimization direction, or ParEGO
            # normalization/weights produced an earlier recommendation.
            "decision_state": copy.deepcopy(self._acquisition_audit),
        }
        self._pending_recommendation_audits.append(audit)
        return [
            dict(
                zip(
                    [param["parameter"] for param in self.parameters],
                    recommendations,
                )
            )
        ]

    def _optimize_acquisition(self, acquisition):
        """Maximize a deterministic acquisition function over the unit cube."""
        bounds = [(0.0, 1.0)] * len(self.parameters)
        best_value = math.inf
        best_x = None
        for _ in range(self.num_restarts):
            x0 = self._rng.rand(len(self.parameters))
            result = minimize(
                lambda point: -float(
                    np.asarray(acquisition(point), dtype=float).reshape(-1)[0]
                ),
                x0=x0,
                bounds=bounds,
                method="L-BFGS-B",
            )
            value = self._finite_observation_value(result.fun)
            if (
                value is not None
                and np.all(np.isfinite(result.x))
                and value < best_value
            ):
                best_value = value
                best_x = np.asarray(result.x, dtype=float)
        if best_x is None:
            logger.warning(
                "Bayesian acquisition optimizer returned no finite point; "
                "using the next seeded fallback point"
            )
            best_x = self._rng.rand(len(self.parameters))
            fallback_value = self._finite_observation_value(
                np.asarray(acquisition(best_x), dtype=float).reshape(-1)[0]
            )
            self._last_acquisition_value = fallback_value
        else:
            self._last_acquisition_value = -float(best_value)
        return best_x.reshape(-1)

    def _maximize_expected_improvement(self, gp, observed_values):
        """Optimize maximize-oriented expected improvement for one fitted GP."""
        incumbent = float(np.max(np.asarray(observed_values, dtype=float)))

        def acquisition(point):
            mean, stddev = gp.predict(
                np.asarray(point, dtype=float).reshape(1, -1),
                return_std=True,
            )
            return expected_improvement_maximize(
                mean,
                stddev,
                incumbent=incumbent,
                xi=self.xi,
            )

        return self._optimize_acquisition(acquisition)

    def _latency_accuracy_threshold(self, accuracies):
        """Resolve the self-calibrating latency constraint from this archive."""
        constraint = (
            self.objective_config.selection_config.latency_accuracy_retention
        )
        if constraint.kind == "relative":
            return retained_accuracy_threshold(accuracies, constraint.value)
        if not accuracies:
            return None
        reference = float(max(accuracies))
        return reference, constraint.threshold(reference)

    def _generate_objective_recommendations(self, history):
        """Generate one mode-aware proposal directly from raw objectives."""
        self._last_acquisition_value = None
        get_flatten_specs(
            self.default_train_spec,
            self.default_train_spec_flattened,
        )
        terminal_statuses = {
            JobStates.success,
            JobStates.done,
            JobStates.failure,
            JobStates.error,
            JobStates.canceled,
        }
        if history and history[-1].status not in terminal_statuses:
            return []

        observations = self._objective_observations(history)
        if len(self.Xs) > len(observations):
            # The only permissible surplus is the most recently issued point.
            # It is paired only when the active mode's required successful raw
            # observation exists; failures and invalid required measurements
            # stay in the controller archive but never enter a surrogate.
            self._discard_pending_observation()
        if len(self.Xs) != len(observations):
            raise ValueError(
                "Native objective archive is misaligned: "
                f"{len(self.Xs)} parameter point(s), "
                f"{len(observations)} valid mode-required observation(s)"
            )

        accuracies = [item.accuracy for item in observations]
        latencies = [
            item.latency
            for item in observations
            if hasattr(item, "latency")
        ]
        if len(observations) < self.calibration_points:
            # Keep one finite placeholder response per completed design point
            # solely so X/Y pairing survives persistence. No selector score is
            # fitted or used during this deterministic initial design.
            self.ys = list(accuracies)
            self._record_objective_audit(
                observations=observations,
                stage="calibration",
                active_method="deterministic_low_discrepancy_design",
                calibration_points_required=self.calibration_points,
            )
            return self._append_suggestion(
                self._calibration_point(self._recommendation_count)
            )

        if not observations:
            self.ys = []
            self._record_objective_audit(
                observations=observations,
                stage="initial_design",
                active_method="deterministic_low_discrepancy_design",
                calibration_points_required=1,
            )
            return self._append_suggestion(self._calibration_point(0))

        Xs_npy = np.asarray(self.Xs, dtype=float)
        accuracy_npy = np.asarray(accuracies, dtype=float)
        latency_npy = np.asarray(latencies, dtype=float)

        if self.objective_acquisition_mode == "accuracy":
            self.ys = accuracy_npy.tolist()
            self.accuracy_gp.fit(Xs_npy, accuracy_npy)
            suggestion = self._maximize_expected_improvement(
                self.accuracy_gp,
                accuracy_npy,
            )
            self._record_objective_audit(
                observations=observations,
                stage="model_based",
                active_method="accuracy_expected_improvement",
                optimization_direction={"accuracy": "maximize"},
                incumbent_accuracy=float(np.max(accuracy_npy)),
                acquisition_index=self._model_based_iteration,
            )
        elif self.objective_acquisition_mode == "latency":
            self.ys = (-latency_npy).tolist()
            self.accuracy_gp.fit(Xs_npy, accuracy_npy)
            threshold_state = self._latency_accuracy_threshold(accuracies)
            if threshold_state is None:
                # No positive relative accuracy reference yet. Continue
                # discovering a viable quality reference instead of declaring
                # zero-quality fast candidates feasible.
                suggestion = self._maximize_expected_improvement(
                    self.accuracy_gp,
                    accuracy_npy,
                )
                self._record_objective_audit(
                    observations=observations,
                    stage="quality_discovery",
                    active_method="accuracy_expected_improvement",
                    optimization_direction={"accuracy": "maximize"},
                    accuracy_reference=None,
                    accuracy_threshold=None,
                    feasible_latency_incumbent=None,
                    acquisition_index=self._model_based_iteration,
                )
            else:
                accuracy_reference, accuracy_threshold = threshold_state
                accuracy_tolerance = (
                    self.objective_config.selection_config.accuracy_tolerance
                )
                feasibility_boundary = accuracy_feasibility_boundary(
                    accuracy_threshold,
                    accuracy_tolerance,
                )
                feasible_latencies = [
                    item.latency
                    for item in observations
                    if item.accuracy
                    >= feasibility_boundary
                ]
                feasible_incumbent = (
                    min(feasible_latencies) if feasible_latencies else None
                )
                self.latency_gp.fit(Xs_npy, latency_npy)

                def acquisition(point):
                    point = np.asarray(point, dtype=float).reshape(1, -1)
                    latency_mean, latency_stddev = self.latency_gp.predict(
                        point,
                        return_std=True,
                    )
                    accuracy_mean, accuracy_stddev = self.accuracy_gp.predict(
                        point,
                        return_std=True,
                    )
                    return constrained_latency_ei(
                        latency_mean,
                        latency_stddev,
                        accuracy_mean,
                        accuracy_stddev,
                        accuracy_threshold=feasibility_boundary,
                        feasible_latency_incumbent=feasible_incumbent,
                        xi=self.xi,
                    )

                suggestion = self._optimize_acquisition(acquisition)
                self._record_objective_audit(
                    observations=observations,
                    stage="model_based",
                    active_method="constrained_latency_expected_improvement",
                    optimization_direction={
                        "accuracy": "constraint_maximize",
                        "latency": "minimize",
                    },
                    accuracy_reference=accuracy_reference,
                    accuracy_threshold=accuracy_threshold,
                    accuracy_feasibility_boundary=feasibility_boundary,
                    feasible_latency_incumbent=feasible_incumbent,
                    feasible_observation_count=len(feasible_latencies),
                    acquisition_index=self._model_based_iteration,
                )
        else:
            utilities, parego_audit = parego_utilities(
                accuracy_npy,
                latency_npy,
                iteration=self._model_based_iteration,
                augmentation_rho=self.parego_augmentation_rho,
            )
            self.ys = utilities.tolist()
            self.gp.fit(Xs_npy, utilities)
            suggestion = self._maximize_expected_improvement(
                self.gp,
                utilities,
            )
            self._record_objective_audit(
                observations=observations,
                stage="model_based",
                active_method="parego_expected_improvement",
                optimization_direction={
                    "accuracy": "maximize",
                    "latency": "minimize",
                },
                parego=parego_audit,
                acquisition_index=self._model_based_iteration,
            )

        self._model_based_iteration += 1
        self._acquisition_audit["model_based_iterations"] = (
            self._model_based_iteration
        )
        return self._append_suggestion(suggestion)

    def generate_automl_param_rec_value(self, parameter_config, suggestion):
        """Convert 0 to 1 GP prediction into a possible value"""
        parameter_name = parameter_config.get("parameter")
        # Apply custom overrides if provided
        if self.custom_ranges and parameter_name in self.custom_ranges:
            for override_key, override_value in self.custom_ranges[parameter_name].items():
                if override_value is not None:
                    parameter_config[override_key] = override_value

        data_type = parameter_config.get("value_type")
        default_value = parameter_config.get("default_value", None)
        math_cond = parameter_config.get("math_cond", None)
        parent_param = parameter_config.get("parent_param", None)

        if data_type == "float":
            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")

            # If no valid range, generate values around default using suggestion
            if v_min == "" or v_max == "":
                if default_value is not None and default_value != "":
                    default_val = float(default_value)
                    # Use suggestion to vary around default
                    if default_val > 0:
                        v_min = default_val / 10.0
                        v_max = default_val * 10.0
                    elif default_val < 0:
                        v_min = default_val * 10.0
                        v_max = default_val / 10.0
                    else:  # default is 0
                        v_min = -1.0
                        v_max = 1.0
                    # Use suggestion to pick value in range
                    quantized = suggestion * (v_max - v_min) + v_min
                    logger.info(
                        f"Generated float for {parameter_name} (no range): "
                        f"{quantized} from suggestion {suggestion}"
                    )
                    return quantized
                # No default, use suggestion in [0, 1]
                return float(suggestion)

            # Check for NaN ranges (skip if v_min/v_max are lists - handled by network-specific logic)
            if is_nan_value(v_min) or is_nan_value(v_max):
                # NaN ranges, use default-based range
                if default_value is not None:
                    default_val = float(default_value)
                    if default_val > 0:
                        v_min = default_val / 10.0
                        v_max = default_val * 10.0
                    else:
                        v_min = 0.0
                        v_max = 1.0
                    quantized = suggestion * (v_max - v_min) + v_min
                    return quantized
                return float(suggestion)

            # Handle list-based ranges (e.g., per-model-part learning rates)
            # Generate a base value and let network-specific handler convert to list
            if isinstance(v_min, list) or isinstance(v_max, list):
                # Use first element of list for base range, or default if available
                if isinstance(v_min, list) and isinstance(v_max, list):
                    base_min = float(v_min[0]) if v_min else 0.0
                    base_max = float(v_max[0]) if v_max else 1.0
                elif isinstance(v_min, list):
                    base_min = float(v_min[0]) if v_min else 0.0
                    base_max = float(v_max) if v_max not in (None, '', "") else base_min * 10
                else:
                    base_min = float(v_min) if v_min not in (None, '', "") else 0.0
                    base_max = float(v_max[0]) if v_max else 1.0

                # Generate base value using log-uniform sampling (better for LR)
                if base_min > 0 and base_max > 0:
                    log_min = np.log10(base_min)
                    log_max = np.log10(base_max)
                    base_value = float(10 ** (suggestion * (log_max - log_min) + log_min))
                else:
                    base_value = float(suggestion * (base_max - base_min) + base_min)

                # Check for disable_list option - if True, skip network-specific logic
                # and return pure float value for Bayesian optimization
                disable_list = parameter_config.get("disable_list", False)
                if disable_list:
                    logger.info(
                        f"disable_list=True for {parameter_name}: "
                        f"returning pure float {base_value} (skipping network-specific logic)"
                    )
                    return base_value

                # Let network-specific handler convert to list format
                return network_utils.apply_network_specific_param_logic(
                    network=self.network,
                    data_type=data_type,
                    parameter_name=parameter_name,
                    value=base_value,
                    v_max=v_max,
                    default_train_spec=self.default_train_spec,
                    parent_params=self.parent_params
                )

            v_min, v_max = get_valid_range(parameter_config, self.parent_params, self.custom_ranges)

            # Check for disable_list option early - log the parameter config for debugging
            disable_list = parameter_config.get("disable_list", False)
            logger.debug(
                f"[BAYESIAN] Parameter {parameter_name}: v_min={v_min}, v_max={v_max}, "
                f"disable_list={disable_list}, parameter_config keys={list(parameter_config.keys())}"
            )

            # Apply math condition if specified
            # Skip relational constraints (like "> depends_on") as they're handled in base class
            if math_cond and type(math_cond) is str and "depends_on" not in math_cond:
                parts = math_cond.split(" ")
                if len(parts) >= 2:
                    operator = parts[0]
                    factor = int(float(parts[1]))
                    if operator == "^":
                        # Use helper function for power constraints with equal priority
                        normalized = suggestion * (v_max - v_min) + v_min
                        fallback = clamp_value(normalized, v_min, v_max)
                        quantized = float(self._apply_power_constraint_with_equal_priority(
                            v_min, v_max, factor, fallback))
                    else:
                        # Regular sampling for non-power constraints
                        normalized = suggestion * (v_max - v_min) + v_min
                        quantized = clamp_value(normalized, v_min, v_max)
                else:
                    # Invalid math condition format, fall back to regular sampling
                    normalized = suggestion * (v_max - v_min) + v_min
                    quantized = clamp_value(normalized, v_min, v_max)
            else:
                # No math condition, regular sampling
                normalized = suggestion * (v_max - v_min) + v_min
                quantized = clamp_value(normalized, v_min, v_max)

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if (isinstance(parent_param, str) and parent_param != "nan" and parent_param == "TRUE") or (
                    isinstance(parent_param, bool) and parent_param
                ):
                    self.parent_params[parameter_name] = quantized

            # Check for disable_list option - if True, skip network-specific logic
            # and return pure float value (works for both scalar and list ranges)
            if disable_list:
                logger.info(
                    f"disable_list=True for {parameter_name}: "
                    f"returning pure float {quantized} (skipping network-specific logic)"
                )
                return quantized

            # Apply network-specific parameter logic
            return network_utils.apply_network_specific_param_logic(
                network=self.network,
                data_type=data_type,
                parameter_name=parameter_name,
                value=quantized,
                v_max=v_max,
                default_train_spec=self.default_train_spec,
                parent_params=self.parent_params
            )

        if data_type in ("int", "integer"):
            # Preserve explicit discrete integer domains supplied by the
            # schema/custom range adapter. Mapping the normalized GP proposal
            # to the ordered option set prevents unsealed intermediate values.
            valid_options = get_valid_options(
                parameter_config, self.custom_ranges
            )
            if valid_options:
                index = min(
                    int(suggestion * len(valid_options)),
                    len(valid_options) - 1,
                )
                quantized_int = int(valid_options[index])
                if not (
                    type(parent_param) is float
                    and math.isnan(parent_param)
                ):
                    if (
                        isinstance(parent_param, str)
                        and parent_param != "nan"
                        and parent_param == "TRUE"
                    ) or (
                        isinstance(parent_param, bool)
                        and parent_param
                    ):
                        self.parent_params[parameter_name] = quantized_int
                return network_utils.apply_network_specific_param_logic(
                    network=self.network,
                    data_type=data_type,
                    parameter_name=parameter_name,
                    value=quantized_int,
                    v_max=max(int(item) for item in valid_options),
                    default_train_spec=self.default_train_spec,
                    parent_params=self.parent_params,
                )

            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")

            # If no valid range, generate values around default using suggestion
            if v_min == "" or v_max == "":
                if default_value is not None and default_value != "":
                    default_val = int(default_value)
                    # Use suggestion to vary around default
                    if default_val > 0:
                        v_min = max(1, default_val // 2)
                        v_max = default_val * 2
                    else:
                        v_min = 1
                        v_max = 100
                    # Use suggestion to pick value in range
                    continuous_value = suggestion * (v_max - v_min) + v_min
                    quantized_int = int(round(continuous_value))
                    logger.info(
                        f"Generated int for {parameter_name} (no range): "
                        f"{quantized_int} from suggestion {suggestion}"
                    )
                    return quantized_int
                # No default, use suggestion in [1, 100]
                return int(round(suggestion * 99 + 1))

            if is_nan_value(v_min) or is_nan_value(v_max):
                # NaN ranges, use default-based range
                if default_value is not None:
                    default_val = int(default_value)
                    v_min = max(1, default_val // 2)
                    v_max = default_val * 2
                    continuous_value = suggestion * (v_max - v_min) + v_min
                    return int(round(continuous_value))
                return int(round(suggestion * 99 + 1))

            v_min, v_max = get_valid_range(parameter_config, self.parent_params, self.custom_ranges)

            # Map GP suggestion to discrete integer
            continuous_value = suggestion * (v_max - v_min) + v_min
            quantized_int = int(round(continuous_value))

            # Apply math condition if specified
            # Skip relational constraints (like "> depends_on") as they're handled later
            if math_cond and type(math_cond) is str and "depends_on" not in math_cond:
                parts = math_cond.split(" ")
                if len(parts) >= 2:
                    operator = parts[0]
                    factor = int(float(parts[1]))
                    if operator == "^":
                        quantized_int = int(self._apply_power_constraint_with_equal_priority(
                            v_min, v_max, factor, quantized_int))
                    elif operator == "/":
                        quantized_int = fix_input_dimension(quantized_int, factor)

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if ((type(parent_param) is str and parent_param != "nan" and parent_param == "TRUE") or
                        (type(parent_param) is bool and parent_param)):
                    self.parent_params[parameter_name] = quantized_int

            return network_utils.apply_network_specific_param_logic(
                network=self.network,
                data_type=data_type,
                parameter_name=parameter_name,
                value=quantized_int,
                default_train_spec=self.default_train_spec,
                parent_params=self.parent_params
            )

        if data_type in ("categorical", "ordered"):
            valid_options = get_valid_options(parameter_config, self.custom_ranges)
            if not valid_options or valid_options == "":
                return default_value

            # Map GP suggestion to discrete index
            idx = int(suggestion * len(valid_options))
            idx = min(idx, len(valid_options) - 1)

            # Handle weighted options
            weights = get_option_weights(parameter_config, self.custom_ranges)
            if weights and len(weights) == len(valid_options):
                sorted_pairs = sorted(zip(valid_options, weights), key=lambda x: x[1], reverse=True)
                cumulative = 0
                total_weight = sum(weights)
                for option, weight in sorted_pairs:
                    cumulative += weight / total_weight
                    if suggestion <= cumulative:
                        return option
                return sorted_pairs[0][0]

            return valid_options[idx]

        if data_type == "ordered_int":
            valid_options = get_valid_options(parameter_config, self.custom_ranges)
            if not valid_options or valid_options == "":
                return int(default_value) if default_value else 0

            idx = int(suggestion * len(valid_options))
            idx = min(idx, len(valid_options) - 1)

            weights = get_option_weights(parameter_config, self.custom_ranges)
            if weights and len(weights) == len(valid_options):
                sorted_pairs = sorted(zip(valid_options, weights), key=lambda x: x[1], reverse=True)
                cumulative = 0
                total_weight = sum(weights)
                for option, weight in sorted_pairs:
                    cumulative += weight / total_weight
                    if suggestion <= cumulative:
                        return int(option)
                return int(sorted_pairs[0][0])

            return int(valid_options[idx])

        return super().generate_automl_param_rec_value(parameter_config)

    def save_state(self):
        """Save the Bayesian algorithm related variables to brain metadata"""
        state_dict = {}
        state_dict["Xs"] = np.array(self.Xs).tolist()  # List of np arrays
        state_dict["ys"] = np.array(self.ys).tolist()  # Oriented utilities
        state_dict["metric"] = self.metric
        state_dict["metric_direction"] = self.metric_direction
        state_dict["observation_utility_version"] = OBSERVATION_UTILITY_VERSION
        state_dict["random_seed"] = self.random_seed
        state_dict["objective_acquisition_signature"] = (
            self._acquisition_signature()
        )
        state_dict["objective_acquisition_audit"] = self.acquisition_audit
        state_dict["recommendation_count"] = self._recommendation_count
        state_dict["model_based_iteration"] = self._model_based_iteration
        state_dict["rng_state"] = self._rng_state_to_dict()
        state_dict["pending_recommendation_audits"] = copy.deepcopy(
            self._pending_recommendation_audits
        )

        self.state_store.save_brain_info(self.context.id, state_dict)

    @staticmethod
    def load_state(
        context,
        state_store,
        network,
        parameters,
        metric="kpi",
        direction=None,
        objective_config=None,
        acquisition_settings=None,
    ):
        """Load the Bayesian algorithm related variables to brain metadata"""
        json_loaded = state_store.get_brain_info(context.id)
        if json_loaded is None:
            return Bayesian(
                context,
                state_store,
                network,
                parameters,
                metric=metric,
                direction=direction,
                objective_config=objective_config,
                acquisition_settings=acquisition_settings,
            )
        if not isinstance(json_loaded, dict):
            raise ValueError("Persisted Bayesian state must be a mapping")

        bayesian = Bayesian(
            context,
            state_store,
            network,
            parameters,
            metric=metric,
            direction=direction,
            objective_config=objective_config,
            acquisition_settings=acquisition_settings,
        )
        stored_metric = json_loaded.get("metric")
        stored_direction = json_loaded.get("metric_direction")
        if stored_metric != bayesian.metric:
            raise ValueError(
                "Cannot resume Bayesian state with a different metric: "
                f"stored={stored_metric!r}, requested={bayesian.metric!r}"
            )
        if stored_direction != bayesian.metric_direction:
            raise ValueError(
                "Cannot resume Bayesian state with a different direction: "
                f"stored={stored_direction!r}, "
                f"requested={bayesian.metric_direction!r}"
            )
        stored_acquisition_signature = json_loaded.get(
            "objective_acquisition_signature"
        )
        if stored_acquisition_signature is None:
            raise ValueError(
                "Cannot safely resume Bayesian state without the complete "
                "objective acquisition and search-space compatibility signature"
            )
        requested_signature = bayesian._acquisition_signature()
        if stored_acquisition_signature != requested_signature:
            stored_keys = (
                set(stored_acquisition_signature)
                if isinstance(stored_acquisition_signature, dict)
                else set()
            )
            changed_fields = sorted(
                key
                for key in stored_keys | set(requested_signature)
                if not isinstance(stored_acquisition_signature, dict)
                or stored_acquisition_signature.get(key)
                != requested_signature.get(key)
            )
            raise ValueError(
                "Cannot resume Bayesian state with a different objective "
                "acquisition configuration or Bayesian search identity; "
                "changed fields: "
                + ", ".join(changed_fields)
            )

        bayesian._restore_observation_state(
            json_loaded.get("Xs", []),
            json_loaded.get("ys", []),
            utilities_oriented=(
                json_loaded.get("observation_utility_version")
                == OBSERVATION_UTILITY_VERSION
            ),
        )
        bayesian._recommendation_count = int(
            json_loaded.get("recommendation_count", len(bayesian.Xs))
        )
        bayesian._model_based_iteration = int(
            json_loaded.get("model_based_iteration", 0)
        )
        stored_audit = json_loaded.get("objective_acquisition_audit")
        if isinstance(stored_audit, dict):
            bayesian._acquisition_audit = copy.deepcopy(stored_audit)
        stored_recommendation_audits = json_loaded.get(
            "pending_recommendation_audits"
        )
        if isinstance(stored_recommendation_audits, list):
            bayesian._pending_recommendation_audits = copy.deepcopy(
                stored_recommendation_audits
            )
        bayesian._restore_rng_state(json_loaded.get("rng_state"))
        if bayesian.ys and not bayesian._uses_native_objective_acquisition:
            bayesian.gp.fit(
                np.array(bayesian.Xs[:len(bayesian.ys)]),
                np.array(bayesian.ys),
            )

        return bayesian

    def generate_recommendations(self, history):
        """Generates parameter values and appends to recommendations"""
        if self._uses_native_objective_acquisition:
            return self._generate_objective_recommendations(history)

        self._last_acquisition_value = None
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)
        if history == []:
            # default recommendation => random points
            # TODO: In production, this must be default values for a baseline
            self._acquisition_audit.update({
                "stage": "initial_design",
                "active_method": "seeded_uniform_design",
                "acquisition_index": 0,
                "observation_count": 0,
            })
            suggestions = self._rng.rand(len(self.parameters))
            return self._append_suggestion(suggestions)
        # This function will be called every 5 seconds or so.
        # If no change in history, dont give a recommendation
        # ie - wait for previous recommendation to finish
        terminal_statuses = {
            JobStates.success,
            JobStates.done,
            JobStates.failure,
            JobStates.error,
            JobStates.canceled,
        }
        if history[-1].status not in terminal_statuses:
            return []

        latest_utility = self._observation_utility(history[-1])
        if latest_utility is None:
            logger.warning(
                "Skipping unusable Bayesian observation for recommendation %s "
                "(status=%s, result=%r)",
                getattr(history[-1], "id", "N/A"),
                getattr(history[-1], "status", None),
                getattr(history[-1], "result", None),
            )
            self._discard_pending_observation()

        # Objective normalization can change every prior scalarized result when
        # the archive grows. Re-read every current success rather than appending
        # only the latest (potentially leaving stale utilities in the GP).
        utilities = self._rebuild_observation_utilities(history)
        if utilities:
            self.update_gp()
            suggestions = self.optimize_ei()
            self._acquisition_audit.update({
                "stage": "model_based",
                "active_method": "single_objective_expected_improvement",
                "acquisition_index": self._model_based_iteration,
                "observation_count": len(utilities),
            })
            self._model_based_iteration += 1
        else:
            suggestions = self._rng.rand(len(self.parameters))
            self._acquisition_audit.update({
                "stage": "fallback_design",
                "active_method": "seeded_uniform_design",
                "acquisition_index": self._recommendation_count,
                "observation_count": 0,
            })
        return self._append_suggestion(suggestions)

    def update_gp(self):
        """Update gaussian regressor parameters"""
        Xs_npy = np.array(self.Xs)
        ys_npy = np.array(self.ys)

        if (
            len(Xs_npy) == 0
            or len(ys_npy) == 0
            or len(Xs_npy) != len(ys_npy)
            or not np.all(np.isfinite(Xs_npy))
            or not np.all(np.isfinite(ys_npy))
        ):
            raise ValueError(
                "Bayesian GP requires complete finite X/Y observation pairs"
            )
        self.gp.fit(Xs_npy, ys_npy)

    def optimize_ei(self):
        """Optmize expected improvement functions"""
        best_ei = 1.0
        best_x = None

        dim = len(self.Xs[0])
        bounds = [(0, 1)] * len(self.parameters)

        for _ in range(self.num_restarts):
            x0 = self._rng.rand(dim)
            res = minimize(self._expected_improvement, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < best_ei:
                best_ei = res.fun
                best_x = res.x
        if best_x is None:
            best_x = self._rng.rand(dim)
            value = self._finite_observation_value(
                self._expected_improvement(best_x)
            )
            self._last_acquisition_value = (
                -value if value is not None else None
            )
            return best_x
        self._last_acquisition_value = -float(best_ei)
        return best_x.reshape(-1)

    """
    Used from:
    http://krasserm.github.io/2018/03/21/bayesian-optimization/
    """
    def _expected_improvement(self, X, xi=0.01):
        """Calculate the expected improvement at points X based on existing samples.

        Args:
            X: Points at which EI shall be calculated (m x d)
            xi: Exploitation-exploration trade-off parameter

        Returns:
            float: Expected improvements at points X
        """
        X = X.reshape(1, -1)

        mu, sigma = self.gp.predict(X, return_std=True)
        mu_sample = self.gp.predict(np.array(self.Xs))

        sigma = sigma.reshape(-1, 1)
        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [1]
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return -1 * ei[0, 0]
