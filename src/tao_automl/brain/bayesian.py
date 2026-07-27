# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bayesian AutoML algorithm modules"""
import numpy as np
import math
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm
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
    ):
        """Initialize the Bayesian algorithm class

        Args:
            context: AutoMLContext instance
            state_store: StateStore instance
            network: model we are running AutoML on
            parameters: automl sweepable parameters
        """
        super().__init__(context, state_store, network, parameters)
        self._configure_objective(metric, direction)
        length_scale = [1.0] * len(self.parameters)
        m52 = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5)
        # m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) # is another option
        self.gp = GaussianProcessRegressor(
            kernel=m52,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=10,
            random_state=self.random_seed,
        )
        # The following 2 need to be stored
        self.Xs = []
        self.ys = []

        self.xi = 0.01
        self.num_restarts = 5

        self.num_epochs_per_experiment = _get_total_epochs_from_specs(self.default_train_spec)

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

        self.state_store.save_brain_info(self.context.id, state_dict)

    @staticmethod
    def load_state(
        context,
        state_store,
        network,
        parameters,
        metric="kpi",
        direction=None,
    ):
        """Load the Bayesian algorithm related variables to brain metadata"""
        json_loaded = state_store.get_brain_info(context.id)
        if not json_loaded:
            return Bayesian(
                context,
                state_store,
                network,
                parameters,
                metric=metric,
                direction=direction,
            )

        bayesian = Bayesian(
            context,
            state_store,
            network,
            parameters,
            metric=metric,
            direction=direction,
        )
        stored_metric = json_loaded.get("metric")
        stored_direction = json_loaded.get("metric_direction")
        if stored_metric is not None and stored_metric != bayesian.metric:
            raise ValueError(
                "Cannot resume Bayesian state with a different metric: "
                f"stored={stored_metric!r}, requested={bayesian.metric!r}"
            )
        if (
            stored_direction is not None
            and stored_direction != bayesian.metric_direction
        ):
            raise ValueError(
                "Cannot resume Bayesian state with a different direction: "
                f"stored={stored_direction!r}, "
                f"requested={bayesian.metric_direction!r}"
            )

        bayesian._restore_observation_state(
            json_loaded.get("Xs", []),
            json_loaded.get("ys", []),
            utilities_oriented=(
                json_loaded.get("observation_utility_version")
                == OBSERVATION_UTILITY_VERSION
            ),
        )
        if bayesian.ys:
            bayesian.gp.fit(
                np.array(bayesian.Xs[:len(bayesian.ys)]),
                np.array(bayesian.ys),
            )

        return bayesian

    def generate_recommendations(self, history):
        """Generates parameter values and appends to recommendations"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)
        if history == []:
            # default recommendation => random points
            # TODO: In production, this must be default values for a baseline
            suggestions = np.random.rand(len(self.parameters))
            self.Xs.append(suggestions)
            recommendations = []
            for param_dict, suggestion in zip(self.parameters, suggestions):
                recommendation_value = self.generate_automl_param_rec_value(param_dict, suggestion)
                logger.info(f"Recommendation param: {param_dict['parameter']} value: {recommendation_value}")
                recommendations.append(recommendation_value)
            return [dict(zip([param["parameter"] for param in self.parameters], recommendations))]
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
        else:
            suggestions = np.random.rand(len(self.parameters))
        self.Xs.append(suggestions)
        # Convert the suggestions to recommendations based on parameter type
        # Assume one:one mapping between self.parameters and suggestions
        recommendations = []
        assert len(self.parameters) == len(suggestions), (
            f"Number of parameters ({len(self.parameters)}) does not match "
            f"number of suggestions ({len(suggestions)})"
        )
        for param_dict, suggestion in zip(self.parameters, suggestions):
            recommendation_value = self.generate_automl_param_rec_value(param_dict, suggestion)
            logger.info(f"Recommendation param: {param_dict['parameter']} value: {recommendation_value}")
            recommendations.append(recommendation_value)

        return [dict(zip([param["parameter"] for param in self.parameters], recommendations))]

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
            x0 = np.random.rand(dim)
            res = minimize(self._expected_improvement, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < best_ei:
                best_ei = res.fun
                best_x = res.x
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
