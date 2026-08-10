# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BOHB (Bayesian Optimization and HyperBand) AutoML algorithm modules"""
import copy
import numpy as np
import math
import logging
from scipy.stats import gaussian_kde

from tao_automl.utils.math_utils import (
    ResumeRecommendation, JobStates, get_valid_range, clamp_value,
    get_valid_options, get_option_weights, fix_input_dimension
)
from tao_automl.brain.base import AutoMLAlgorithmBase, is_nan_value
from tao_automl.utils.spec_utils import get_flatten_specs
from tao_automl.brain import network_utils

logger = logging.getLogger(__name__)


class BOHB(AutoMLAlgorithmBase):
    """BOHB (Bayesian Optimization and HyperBand) AutoML algorithm class

    BOHB combines the resource allocation strategy of HyperBand with
    a Tree-structured Parzen Estimator (TPE) model for smart configuration sampling.
    """

    def __init__(self, context, state_store, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                 kde_samples=64, top_n_percent=15.0, min_points_in_model=10, metric="loss"):
        """Initialize the BOHB algorithm class"""
        super().__init__(context, state_store, network, parameters)
        self.epoch_multiplier = int(epoch_multiplier)
        self.metric = metric
        self.ni = {}
        self.ri = {}
        self.brackets_and_sh_sequence(max_epochs, reduction_factor)
        self.epoch_number = 0

        # State variables
        self.bracket = "0"
        self.override_num_epochs(self.ri[self.bracket][-1] * self.epoch_multiplier)
        self.sh_iter = 0
        self.experiments_considered = []
        self.expt_iter = 0
        self.complete = False

        self.reverse_sort = True
        if metric == "loss" or "loss" in metric.lower() or metric.lower() in ("evaluation_cost",):
            self.reverse_sort = False
        self.last_launched_count = 0

        # TPE-specific variables
        self.observations = []
        self.quantile = top_n_percent / 100.0
        self.min_bandwidth = 0.01
        self.num_samples = int(kde_samples)
        self.min_points_in_model = int(min_points_in_model)

        logger.info(
            f"BOHB initialized with max_epochs={max_epochs}, "
            f"reduction_factor={reduction_factor}, epoch_multiplier={self.epoch_multiplier}, "
            f"kde_samples={self.num_samples}, top_n_percent={top_n_percent}, "
            f"min_points_in_model={self.min_points_in_model}"
        )

    def brackets_and_sh_sequence(self, max_epochs, reduction_factor):
        """Generate ni,ri arrays based on max_epochs and reduction_factor values"""
        smax = int(np.log(max_epochs) / np.log(reduction_factor))
        for itr, s in enumerate(range(smax, 0, -1)):
            self.ni[str(itr)] = []
            self.ri[str(itr)] = []
            n = int(math.ceil(int((smax + 1) / (s + 1)) * (reduction_factor**s)))
            r = int(max_epochs / (reduction_factor**s))
            for s_idx in range(s + 1):
                ni = int(n * (reduction_factor**(-s_idx)))
                ri = int(r * (reduction_factor**s_idx))
                self.ni[str(itr)].append(ni)
                self.ri[str(itr)].append(ri)

    def override_num_epochs(self, num_epochs):
        """Override num epochs parameter in train spec file"""
        spec = self.state_store.get_job_specs(self.context.id)
        for key1 in spec:
            if key1 in ("training_config", "train_config", "train"):
                for key2 in spec[key1]:
                    if key2 in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch"):
                        spec[key1][key2] = num_epochs
                    elif key2 in ("train_config"):
                        for key3 in spec[key1][key2]:
                            if key3 == "runner":
                                for key4 in spec[key1][key2][key3]:
                                    if key4 == "max_epochs":
                                        spec[key1][key2][key3][key4] = num_epochs
            elif key1 in ("num_epochs"):
                spec[key1] = num_epochs
        self.state_store.save_job_specs(self.context.id, spec)

    def _build_kde(self, data, bandwidth=None):
        """Build Kernel Density Estimator for TPE model"""
        if len(data) < max(2, self.min_points_in_model):
            return None
        try:
            if bandwidth is not None:
                kde = gaussian_kde(data.T, bw_method=bandwidth)
            else:
                kde = gaussian_kde(data.T)
            return kde
        except Exception as e:
            logger.warning(f"Failed to build KDE: {e}")
            return None

    def _sample_from_kde(self, kde, n_samples):
        """Sample configurations from KDE"""
        if kde is None:
            return None
        try:
            samples = kde.resample(n_samples).T
            samples = np.clip(samples, 0.0, 1.0)
            return samples
        except Exception as e:
            logger.warning(f"Failed to sample from KDE: {e}")
            return None

    def _tpe_suggest(self):
        """Use Tree-structured Parzen Estimator to suggest next configuration"""
        if len(self.observations) < max(2, self.min_points_in_model):
            min_required = max(2, self.min_points_in_model)
            logger.info(
                f"Insufficient observations for TPE ({len(self.observations)} < {min_required}), "
                "using random sampling"
            )
            return np.random.rand(len(self.parameters))

        sorted_obs = sorted(self.observations, key=lambda x: x[1], reverse=self.reverse_sort)
        n_good = max(1, int(self.quantile * len(sorted_obs)))
        good_obs = np.array([obs[0] for obs in sorted_obs[:n_good]])
        bad_obs = np.array([obs[0] for obs in sorted_obs[n_good:]])

        good_kde = self._build_kde(good_obs)
        bad_kde = self._build_kde(bad_obs)

        if good_kde is None:
            logger.info("Failed to build good KDE, using random sampling")
            return np.random.rand(len(self.parameters))

        candidates = self._sample_from_kde(good_kde, self.num_samples)
        if candidates is None:
            logger.info("Failed to sample from good KDE, using random sampling")
            return np.random.rand(len(self.parameters))

        best_ei = -np.inf
        best_candidate = None

        for candidate in candidates:
            good_prob = good_kde.pdf(candidate.reshape(-1, 1))[0]
            if bad_kde is not None:
                bad_prob = bad_kde.pdf(candidate.reshape(-1, 1))[0]
                bad_prob = max(bad_prob, 1e-10)
                ei = good_prob / bad_prob
            else:
                ei = good_prob

            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate

        if best_candidate is None:
            logger.warning("No valid candidate found, using random sampling")
            return np.random.rand(len(self.parameters))

        logger.info(f"TPE suggested configuration with EI={best_ei}")
        return best_candidate

    def generate_automl_param_rec_value(self, parameter_config, suggestion=None):
        """Generate parameter value from TPE suggestion or randomly"""
        if suggestion is None:
            return super().generate_automl_param_rec_value(parameter_config)

        parameter_name = parameter_config.get("parameter")

        if self.custom_ranges and parameter_name in self.custom_ranges:
            for override_key, override_value in self.custom_ranges[parameter_name].items():
                if override_value is not None:
                    parameter_config[override_key] = override_value

        tp = parameter_config.get("value_type")
        default_value = parameter_config.get("default_value", None)
        math_cond = parameter_config.get("math_cond", None)
        parent_param = parameter_config.get("parent_param", None)

        if tp == "float":
            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")

            if v_min == "" or v_max == "":
                if default_value is not None and default_value != "":
                    default_val = float(default_value)
                    if default_val > 0:
                        v_min = default_val / 10.0
                        v_max = default_val * 10.0
                    elif default_val < 0:
                        v_min = default_val * 10.0
                        v_max = default_val / 10.0
                    else:
                        v_min = -1.0
                        v_max = 1.0
                    random_float = suggestion * (v_max - v_min) + v_min
                    logger.info(
                        f"Generated float for {parameter_name} (no range): "
                        f"{random_float} from suggestion {suggestion}"
                    )
                    return random_float
                return float(suggestion)

            if is_nan_value(v_min) or is_nan_value(v_max):
                if default_value is not None:
                    default_val = float(default_value)
                    if default_val > 0:
                        v_min = default_val / 10.0
                        v_max = default_val * 10.0
                    else:
                        v_min = 0.0
                        v_max = 1.0
                    random_float = suggestion * (v_max - v_min) + v_min
                    return random_float
                return float(suggestion)

            if isinstance(v_min, list) or isinstance(v_max, list):
                if isinstance(v_min, list) and isinstance(v_max, list):
                    base_min = float(v_min[0]) if v_min else 0.0
                    base_max = float(v_max[0]) if v_max else 1.0
                elif isinstance(v_min, list):
                    base_min = float(v_min[0]) if v_min else 0.0
                    base_max = float(v_max) if v_max not in (None, '', "") else base_min * 10
                else:
                    base_min = float(v_min) if v_min not in (None, '', "") else 0.0
                    base_max = float(v_max[0]) if v_max else 1.0

                if base_min > 0 and base_max > 0:
                    log_min = np.log10(base_min)
                    log_max = np.log10(base_max)
                    base_value = float(10 ** (suggestion * (log_max - log_min) + log_min))
                else:
                    base_value = float(suggestion * (base_max - base_min) + base_min)

                disable_list = parameter_config.get("disable_list", False)
                if disable_list:
                    logger.info(
                        f"disable_list=True for {parameter_name}: "
                        f"returning pure float {base_value} (skipping network-specific logic)"
                    )
                    return base_value

                return network_utils.apply_network_specific_param_logic(
                    network=self.network,
                    data_type=tp,
                    parameter_name=parameter_name,
                    value=base_value,
                    v_max=v_max,
                    default_train_spec=self.default_train_spec,
                    parent_params=self.parent_params
                )

            v_min, v_max = get_valid_range(parameter_config, self.parent_params, self.custom_ranges)

            disable_list = parameter_config.get("disable_list", False)
            logger.debug(
                f"[BOHB] Parameter {parameter_name}: v_min={v_min}, v_max={v_max}, "
                f"disable_list={disable_list}"
            )

            if math_cond and type(math_cond) is str and "depends_on" not in math_cond:
                parts = math_cond.split(" ")
                if len(parts) >= 2:
                    operator = parts[0]
                    factor = int(float(parts[1]))
                    if operator == "^":
                        normalized = suggestion * (v_max - v_min) + v_min
                        fallback = clamp_value(normalized, v_min, v_max)
                        random_float = float(self._apply_power_constraint_with_equal_priority(
                            v_min, v_max, factor, fallback))
                    else:
                        normalized = suggestion * (v_max - v_min) + v_min
                        random_float = clamp_value(normalized, v_min, v_max)
            else:
                normalized = suggestion * (v_max - v_min) + v_min
                random_float = clamp_value(normalized, v_min, v_max)

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if ((type(parent_param) is str and parent_param != "nan" and parent_param == "TRUE") or
                        (type(parent_param) is bool and parent_param)):
                    self.parent_params[parameter_config.get("parameter")] = random_float

            if disable_list:
                logger.info(
                    f"disable_list=True for {parameter_name}: "
                    f"returning pure float {random_float} (skipping network-specific logic)"
                )
                return random_float

            return network_utils.apply_network_specific_param_logic(
                network=self.network,
                data_type=tp,
                parameter_name=parameter_name,
                value=random_float,
                v_max=v_max,
                default_train_spec=self.default_train_spec,
                parent_params=self.parent_params
            )

        if tp in ("int", "integer"):
            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")

            if v_min == "" or v_max == "":
                if default_value is not None and default_value != "":
                    default_val = int(default_value)
                    if default_val > 0:
                        v_min = max(1, default_val // 2)
                        v_max = default_val * 2
                    else:
                        v_min = 1
                        v_max = 100
                    continuous_value = suggestion * (v_max - v_min) + v_min
                    quantized_int = int(round(continuous_value))
                    logger.info(
                        f"Generated int for {parameter_name} (no range): "
                        f"{quantized_int} from suggestion {suggestion}"
                    )
                    return quantized_int
                return int(round(suggestion * 99 + 1))

            if is_nan_value(v_min) or is_nan_value(v_max):
                if default_value is not None:
                    default_val = int(default_value)
                    v_min = max(1, default_val // 2)
                    v_max = default_val * 2
                    continuous_value = suggestion * (v_max - v_min) + v_min
                    return int(round(continuous_value))
                return int(round(suggestion * 99 + 1))

            v_min, v_max = get_valid_range(parameter_config, self.parent_params, self.custom_ranges)

            continuous_value = suggestion * (v_max - v_min) + v_min
            quantized_int = int(round(continuous_value))

            if math_cond and type(math_cond) is str and "depends_on" not in math_cond:
                parts = math_cond.split(" ")
                if len(parts) >= 2:
                    operator = parts[0]
                    factor = int(parts[1])
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
                data_type=tp,
                parameter_name=parameter_name,
                value=quantized_int,
                default_train_spec=self.default_train_spec,
                parent_params=self.parent_params
            )

        if tp in ("categorical", "ordered"):
            valid_options = get_valid_options(parameter_config, self.custom_ranges)
            if not valid_options or valid_options == "":
                return default_value

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
                        return option
                return sorted_pairs[0][0]

            return valid_options[idx]

        if tp == "ordered_int":
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
        """Save the BOHB algorithm related variables to brain metadata"""
        state_dict = {}
        state_dict["bracket"] = self.bracket
        state_dict["sh_iter"] = self.sh_iter
        state_dict["expt_iter"] = self.expt_iter
        state_dict["complete"] = self.complete
        state_dict["epoch_number"] = self.epoch_number
        state_dict["epoch_multiplier"] = self.epoch_multiplier
        state_dict["ni"] = self.ni
        state_dict["ri"] = self.ri
        state_dict["last_launched_count"] = self.last_launched_count
        state_dict["metric"] = self.metric
        state_dict["observations"] = [
            (obs[0].tolist(), obs[1]) for obs in self.observations
        ]

        self.state_store.save_brain_info(self.context.id, state_dict)

    @staticmethod
    def load_state(
        context, state_store, network, parameters, max_epochs,
        reduction_factor, epoch_multiplier, metric="loss"
    ):
        """Load the BOHB algorithm related variables from brain metadata"""
        json_loaded = state_store.get_brain_info(context.id)
        if not json_loaded:
            return BOHB(
                context, state_store, network, parameters, max_epochs,
                reduction_factor, epoch_multiplier, metric=metric
            )

        loaded_metric = json_loaded.get("metric", metric)
        brain = BOHB(
            context, state_store, network, parameters, max_epochs,
            reduction_factor, epoch_multiplier, metric=loaded_metric
        )
        brain.bracket = json_loaded["bracket"]
        brain.sh_iter = json_loaded["sh_iter"]
        brain.expt_iter = json_loaded["expt_iter"]
        brain.complete = json_loaded["complete"]
        brain.epoch_number = json_loaded["epoch_number"]
        brain.last_launched_count = json_loaded.get("last_launched_count", 0)

        if "observations" in json_loaded:
            brain.observations = [
                (np.array(obs[0]), obs[1]) for obs in json_loaded["observations"]
            ]

        return brain

    def _generate_one_recommendation(self, history):
        """Updates the counter variables and performs successive halving with TPE"""
        if self.complete:
            return None

        num = self.ni[self.bracket][self.sh_iter]
        if self.expt_iter == num:
            self.expt_iter = 0
            self.sh_iter += 1
        if self.sh_iter == len(self.ni[self.bracket]):
            self.sh_iter = 0
            self.bracket = str(int(self.bracket) + 1)
            if self.bracket in self.ri.keys():
                self.override_num_epochs(self.ri[self.bracket][-1] * self.epoch_multiplier)
        if int(self.bracket) > int(max(list(self.ni.keys()), key=int)):
            logger.info(f"BOHB: All brackets complete (bracket={self.bracket} > max), setting complete=True")
            self.complete = True
            return None

        if self.sh_iter == 0:
            suggestions = self._tpe_suggest()
            specs = self._generate_parameters_from_suggestions(suggestions)
            self.epoch_number = self.ri[self.bracket][self.sh_iter] * self.epoch_multiplier
            final_epoch = self.ri[self.bracket][-1] * self.epoch_multiplier
            self.override_num_epochs(final_epoch)
            specs.update(self._epoch_spec_overrides(self.epoch_number))
            to_return = specs
        else:
            lower = -1 * self.ni.get(self.bracket, [0])[0]

            if self.expt_iter == 0:
                # Promotion must rank only trials that actually produced a
                # metric: failed trials carry a runner-synthesized sentinel
                # (0.0) which, under minimize metrics, would rank as the best
                # result and be promoted ahead of every real candidate.
                def _promotable(recs):
                    ok = [r for r in recs if getattr(r, "status", None) == JobStates.success]
                    return ok if ok else list(recs)

                if self.sh_iter == 1:
                    self.experiments_considered = sorted(
                        _promotable(history[lower:]),
                        key=lambda rec: rec.result,
                        reverse=self.reverse_sort
                    )[0:self.ni[self.bracket][self.sh_iter]]
                else:
                    for experiment in self.experiments_considered:
                        experiment.result = history[experiment.id].result
                        experiment.status = history[experiment.id].status
                    self.experiments_considered = sorted(
                        _promotable(self.experiments_considered),
                        key=lambda rec: rec.result,
                        reverse=self.reverse_sort
                    )[0:self.ni[self.bracket][self.sh_iter]]

            self.epoch_number = self.ri[self.bracket][self.sh_iter] * self.epoch_multiplier
            final_epoch = self.ri[self.bracket][-1] * self.epoch_multiplier
            resume_from_epoch = (
                self.ri[self.bracket][self.sh_iter - 1] * self.epoch_multiplier
                if self.sh_iter > 0 else 0
            )
            self.override_num_epochs(final_epoch)
            specs = copy.deepcopy(self.experiments_considered[self.expt_iter].specs)
            specs.update(self._epoch_spec_overrides(self.epoch_number))
            resumerec = ResumeRecommendation(
                self.experiments_considered[self.expt_iter].id,
                specs,
                self.experiments_considered[self.expt_iter].job_id,
                resume_from_epoch=resume_from_epoch,
            )
            to_return = resumerec
        self.expt_iter += 1

        return to_return

    def done(self):
        """Return if BOHB algorithm is complete or not."""
        if not self.complete:
            return False
        if self.last_launched_count > 0:
            return False
        return True

    @property
    def max_concurrent(self):
        """Maximum number of concurrent experiments for BOHB."""
        max_ni = 1
        for bracket_ni in self.ni.values():
            if bracket_ni:
                max_ni = max([max_ni] + bracket_ni)
        return max_ni

    def _generate_parameters_from_suggestions(self, suggestions):
        """Generates parameter values from TPE suggestions"""
        hyperparam_dict = {}
        for param, suggestion in zip(self.parameters, suggestions):
            name = param["parameter"]
            rec = self.generate_automl_param_rec_value(param, suggestion)
            logger.info(f"Generated parameter in BOHB: {name} = {rec}")
            hyperparam_dict[name] = rec
        return hyperparam_dict

    def generate_recommendations(self, history):
        """Generates recommendations for the controller to run (supports parallel execution)"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)

        logger.info(
            f"BOHB generate_recommendations: complete={self.complete}, "
            f"last_launched_count={self.last_launched_count}, history_len={len(history)}"
        )

        if self.complete:
            if self.last_launched_count > 0 and history:
                any_running = any(
                    exp.status in [JobStates.pending, JobStates.started, JobStates.running]
                    for exp in history
                )
                if not any_running:
                    self.last_launched_count = 0
            return []

        # Update observations with completed experiments
        for rec in history:
            if rec.status == JobStates.success and rec.result != 0.0:
                config = []
                for param in self.parameters:
                    param_name = param["parameter"]
                    value = rec.specs.get(param_name)
                    if param["value_type"] == "float":
                        v_min, v_max = get_valid_range(param, self.parent_params, self.custom_ranges)
                        if v_max > v_min:
                            normalized = (value - v_min) / (v_max - v_min)
                            config.append(np.clip(normalized, 0.0, 1.0))
                        else:
                            config.append(0.5)
                    else:
                        config.append(0.5)

                if len(config) == len(self.parameters):
                    config_array = np.array(config)
                    is_duplicate = any(
                        np.allclose(obs[0], config_array) for obs in self.observations
                    )
                    if not is_duplicate:
                        self.observations.append((config_array, rec.result))
                        logger.info(f"Added observation: config with result={rec.result}")

        if history == []:
            num_configs_in_rung = self.ni[self.bracket][self.sh_iter]
            recommendations = []
            for _ in range(num_configs_in_rung):
                rec = self._generate_one_recommendation(history)
                if type(rec) is dict:
                    recommendations.append(rec)
            self.last_launched_count = len(recommendations)
            self.track_id = len(recommendations) - 1 if recommendations else 0
            return recommendations

        if self.last_launched_count > 0:
            any_running = any(
                exp.status in [JobStates.pending, JobStates.started, JobStates.running]
                for exp in history
            )
            if any_running:
                return []

        num_configs_before = self.ni[self.bracket][self.sh_iter] if not self.complete else 0

        recommendations = []
        for _ in range(max(num_configs_before, 1)):
            rec = self._generate_one_recommendation(history)
            if rec is None:
                break
            recommendations.append(rec)

            if len(recommendations) == 1:
                num_configs_in_new_rung = self.ni[self.bracket][self.sh_iter] if not self.complete else 0
                if num_configs_in_new_rung != num_configs_before:
                    for _ in range(num_configs_in_new_rung - 1):
                        rec = self._generate_one_recommendation(history)
                        if rec:
                            recommendations.append(rec)
                        else:
                            break
                    break

        self.last_launched_count = len(recommendations)

        if recommendations:
            last_rec = recommendations[-1]
            if type(last_rec) is dict:
                self.track_id = len(history) + len(recommendations) - 1
            elif type(last_rec) is ResumeRecommendation:
                self.track_id = last_rec.id

        return recommendations
