# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ASHA (Asynchronous Successive Halving Algorithm) AutoML algorithm modules"""
import copy
import numpy as np
import math
import logging
from collections import defaultdict

from tao_automl.utils.math_utils import (
    ResumeRecommendation, JobStates, get_valid_range, clamp_value
)
from tao_automl.brain.base import AutoMLAlgorithmBase, is_nan_value
from tao_automl.utils.spec_utils import get_flatten_specs
from tao_automl.brain import network_utils

logger = logging.getLogger(__name__)


class ASHA(AutoMLAlgorithmBase):
    """ASHA (Asynchronous Successive Halving Algorithm) AutoML algorithm class"""

    def __init__(
        self, context, state_store, network, parameters, max_epochs,
        reduction_factor, epoch_multiplier, max_concurrent=4, max_trials=None,
        min_top_configs=5, metric="loss"
    ):
        """Initialize the ASHA algorithm class"""
        super().__init__(context, state_store, network, parameters)
        self.epoch_multiplier = int(epoch_multiplier)
        self.reduction_factor = int(reduction_factor)
        self.max_epochs = int(max_epochs)
        self.max_concurrent = int(max_concurrent)
        self.max_trials = max_trials
        self.min_top_configs = int(min_top_configs)
        self._configure_objective(metric)

        K = int(math.floor(math.log(max_epochs) / math.log(reduction_factor)))
        r0 = max(1, int(math.floor(max_epochs / (reduction_factor ** K))))
        self.rungs = [(r0 * (reduction_factor ** i)) * self.epoch_multiplier for i in range(K + 1)]
        self.rungs[-1] = max_epochs * self.epoch_multiplier
        logger.info(f"ASHA rungs (epochs): {self.rungs}")

        self.rung_results = defaultdict(list)
        self.config_to_rung = {}
        self.active_configs = set()
        self.pending_promotions = []
        self.completed_configs = set()
        self.config_specs = {}
        self.next_config_id = 0
        self.total_configs_started = 0
        self.complete = False

        self.reverse_sort = self.metric_direction == "maximize"
        self.rung_completions = defaultdict(int)
        self.rung_promotions = defaultdict(int)
        self.promoted_from_rung = defaultdict(set)
        self.epoch_number = self.rungs[0]

        self.ni = {"0": [max_concurrent] * len(self.rungs)}
        self.ri = {"0": [r // self.epoch_multiplier for r in self.rungs]}
        self.bracket = "0"
        self.sh_iter = 0
        self.expt_iter = 0

        self.override_num_epochs(self.rungs[-1])

        logger.info(
            f"ASHA initialized with max_epochs={max_epochs}, "
            f"reduction_factor={reduction_factor}, max_concurrent={max_concurrent}"
        )

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

    def generate_automl_param_rec_value(self, parameter_config):
        """Generate a random value for the parameter passed"""
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
                    random_float = np.random.uniform(v_min, v_max)
                    logger.info(f"Generated random float for {parameter_name} (no range): {random_float}")
                    return random_float
                return np.random.uniform(0.0, 1.0)

            if is_nan_value(v_min) or is_nan_value(v_max):
                if default_value is not None:
                    default_val = float(default_value)
                    if default_val > 0:
                        return np.random.uniform(default_val / 10.0, default_val * 10.0)
                    return np.random.uniform(0.0, 1.0)
                return np.random.uniform(0.0, 1.0)

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
                    base_value = float(10 ** np.random.uniform(log_min, log_max))
                else:
                    base_value = float(np.random.uniform(base_min, base_max))

                disable_list = parameter_config.get("disable_list", False)
                if disable_list:
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

            if math_cond and type(math_cond) is str and "depends_on" not in math_cond:
                parts = math_cond.split(" ")
                if len(parts) >= 2:
                    operator = parts[0]
                    factor = int(float(parts[1]))
                    if operator == "^":
                        fallback = np.random.uniform(low=v_min, high=v_max)
                        fallback = clamp_value(fallback, v_min, v_max)
                        random_float = float(self._apply_power_constraint_with_equal_priority(
                            v_min, v_max, factor, fallback))
                    else:
                        random_float = np.random.uniform(low=v_min, high=v_max)
                        random_float = clamp_value(random_float, v_min, v_max)
            else:
                random_float = np.random.uniform(low=v_min, high=v_max)
                random_float = clamp_value(random_float, v_min, v_max)

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if ((type(parent_param) is str and parent_param != "nan" and parent_param == "TRUE") or
                        (type(parent_param) is bool and parent_param)):
                    self.parent_params[parameter_config.get("parameter")] = random_float

            if disable_list:
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

        return super().generate_automl_param_rec_value(parameter_config)

    def save_state(self):
        """Save the ASHA algorithm related variables to brain metadata"""
        state_dict = {}
        state_dict["next_config_id"] = self.next_config_id
        state_dict["total_configs_started"] = self.total_configs_started
        state_dict["complete"] = self.complete
        state_dict["epoch_multiplier"] = self.epoch_multiplier
        state_dict["rungs"] = self.rungs
        state_dict["rung_results"] = {str(k): v for k, v in self.rung_results.items()}
        state_dict["rung_completions"] = {str(k): v for k, v in self.rung_completions.items()}
        state_dict["rung_promotions"] = {str(k): v for k, v in self.rung_promotions.items()}
        state_dict["promoted_from_rung"] = {str(k): list(v) for k, v in self.promoted_from_rung.items()}
        state_dict["config_to_rung"] = {str(k): v for k, v in self.config_to_rung.items()}
        state_dict["active_configs"] = list(self.active_configs)
        state_dict["completed_configs"] = list(self.completed_configs)
        state_dict["config_specs"] = {str(k): v for k, v in self.config_specs.items()}
        state_dict["pending_promotions"] = self.pending_promotions
        state_dict["epoch_number"] = self.epoch_number
        state_dict["ni"] = self.ni
        state_dict["ri"] = self.ri
        state_dict["bracket"] = self.bracket
        state_dict["sh_iter"] = self.sh_iter
        state_dict["expt_iter"] = self.expt_iter
        state_dict["max_trials"] = self.max_trials
        state_dict["min_top_configs"] = self.min_top_configs
        state_dict["metric"] = self.metric

        self.state_store.save_brain_info(self.context.id, state_dict)

    @staticmethod
    def load_state(
        context, state_store, network, parameters, max_epochs, reduction_factor,
        epoch_multiplier, max_concurrent=4, max_trials=None, min_top_configs=5,
        metric="loss"
    ):
        """Load the ASHA algorithm related variables from brain metadata"""
        json_loaded = state_store.get_brain_info(context.id)
        if not json_loaded:
            return ASHA(
                context, state_store, network, parameters, max_epochs,
                reduction_factor, epoch_multiplier, max_concurrent, max_trials,
                min_top_configs, metric
            )

        brain = ASHA(
            context, state_store, network, parameters, max_epochs,
            reduction_factor, epoch_multiplier, max_concurrent, max_trials,
            min_top_configs, metric
        )
        brain.next_config_id = json_loaded["next_config_id"]
        brain.total_configs_started = json_loaded.get("total_configs_started", brain.next_config_id)
        brain.complete = json_loaded["complete"]
        brain.rung_results = defaultdict(list, {int(k): v for k, v in json_loaded["rung_results"].items()})
        brain.rung_completions = defaultdict(int, {int(k): v for k, v in json_loaded.get("rung_completions", {}).items()})
        brain.rung_promotions = defaultdict(int, {int(k): v for k, v in json_loaded.get("rung_promotions", {}).items()})
        brain.promoted_from_rung = defaultdict(set, {int(k): set(v) for k, v in json_loaded.get("promoted_from_rung", {}).items()})
        brain.config_to_rung = {int(k): v for k, v in json_loaded["config_to_rung"].items()}
        brain.active_configs = set(json_loaded["active_configs"])
        brain.completed_configs = set(json_loaded["completed_configs"])
        brain.config_specs = {int(k): v for k, v in json_loaded["config_specs"].items()}
        brain.pending_promotions = json_loaded.get("pending_promotions", [])
        brain.epoch_number = json_loaded.get("epoch_number", brain.rungs[0])
        brain.ni = json_loaded.get("ni", brain.ni)
        brain.ri = json_loaded.get("ri", brain.ri)
        brain.bracket = json_loaded.get("bracket", "0")
        brain.sh_iter = json_loaded.get("sh_iter", 0)
        brain.expt_iter = json_loaded.get("expt_iter", 0)
        brain.max_trials = json_loaded.get("max_trials", max_trials)
        brain.min_top_configs = json_loaded.get("min_top_configs", min_top_configs)
        brain._configure_objective(json_loaded.get("metric", metric))
        brain.reverse_sort = brain.metric_direction == "maximize"

        return brain

    def _generate_random_parameters(self):
        """Generate random parameter values for a new configuration"""
        hyperparam_dict = {}
        for param in self.parameters:
            name = param["parameter"]
            rec = self.generate_automl_param_rec_value(param)
            logger.info(f"Generated random parameter in ASHA: {name} = {rec}")
            hyperparam_dict[name] = rec
        return hyperparam_dict

    def done(self):
        """Return if ASHA algorithm is complete or not"""
        return self.complete

    def generate_recommendations(self, history):
        """Generate recommendations asynchronously"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)

        if history == []:
            recommendations = []
            for _ in range(self.max_concurrent):
                if self.max_trials is not None and self.total_configs_started >= self.max_trials:
                    break
                specs = self._generate_random_parameters()
                self.epoch_number = self.rungs[0]
                specs.update(self._epoch_spec_overrides(self.epoch_number))
                self.config_specs[self.next_config_id] = specs
                self.config_to_rung[self.next_config_id] = 0
                self.active_configs.add(self.next_config_id)
                self.total_configs_started += 1
                self.next_config_id += 1
                recommendations.append(specs)
            self.track_id = 0
            return recommendations

        active_by_rung = defaultdict(list)
        for rec in history:
            if rec.status in [JobStates.pending, JobStates.started, JobStates.running]:
                rung_idx = self.config_to_rung.get(rec.id, 0)
                active_by_rung[rung_idx].append(rec.id)

        recommendations = []
        for rec in history:
            if rec.status in [JobStates.success, JobStates.failure] and rec.id in self.active_configs:
                self.active_configs.discard(rec.id)
                current_rung_idx = self.config_to_rung.get(rec.id, 0)
                rung_epochs = self.rungs[current_rung_idx]

                self.rung_completions[rung_epochs] += 1

                if rec.status == JobStates.success and rec.result is not None and math.isfinite(float(rec.result)):
                    self.rung_results[rung_epochs].append((rec.id, rec.result))

                if current_rung_idx < len(self.rungs) - 1:
                    next_rung_idx = current_rung_idx + 1
                    next_epochs = self.rungs[next_rung_idx]

                    m = self.rung_completions[rung_epochs]
                    quota = int(m / self.reduction_factor)
                    promotions_so_far = self.rung_promotions[rung_epochs]

                    if quota > promotions_so_far:
                        results_at_rung = list(self.rung_results[rung_epochs])
                        results_at_rung.sort(key=lambda x: x[1], reverse=self.reverse_sort)

                        for rank, (config_id, result) in enumerate(results_at_rung):
                            if rank >= quota:
                                break
                            if config_id in self.promoted_from_rung[rung_epochs]:
                                continue

                            self.promoted_from_rung[rung_epochs].add(config_id)
                            self.rung_promotions[rung_epochs] += 1
                            self.config_to_rung[config_id] = next_rung_idx
                            self.pending_promotions.append((config_id, rung_epochs, next_epochs))

                            if self.rung_promotions[rung_epochs] >= quota:
                                break
                else:
                    self.completed_configs.add(rec.id)

        max_trials_reached = self.max_trials is not None and self.total_configs_started >= self.max_trials
        enough_final_results = len(self.completed_configs) >= self.min_top_configs
        exhausted = (
            max_trials_reached
            and not self.active_configs
            and not self.pending_promotions
        )

        if (enough_final_results and (max_trials_reached or self.max_trials is None)) or exhausted:
            if exhausted and not enough_final_results:
                logger.warning(
                    "ASHA exhausted all %d trial(s) with only %d final-rung "
                    "completion(s); stopping with no further recommendations",
                    self.total_configs_started,
                    len(self.completed_configs),
                )
            self.complete = True
            return []

        new_recommendations = []
        while len(self.active_configs) + len(new_recommendations) < self.max_concurrent:
            if self.pending_promotions:
                promotion = self.pending_promotions.pop(0)
                if len(promotion) == 3:
                    config_id, resume_from_epoch, epochs = promotion
                else:
                    config_id, epochs = promotion
                    previous_idx = max(self.config_to_rung.get(config_id, 1) - 1, 0)
                    resume_from_epoch = self.rungs[previous_idx]
                specs = copy.deepcopy(self.config_specs[config_id])
                specs.update(self._epoch_spec_overrides(epochs))
                self.config_specs[config_id] = specs
                config_job_id = None
                for rec in reversed(history):
                    if rec.id == config_id:
                        config_job_id = rec.job_id
                        break
                self.active_configs.add(config_id)
                self.epoch_number = epochs
                resume_rec = ResumeRecommendation(
                    config_id,
                    specs,
                    config_job_id,
                    resume_from_epoch=resume_from_epoch,
                )
                self.track_id = config_id
                new_recommendations.append(resume_rec)

            elif self.max_trials is None or self.total_configs_started < self.max_trials:
                specs = self._generate_random_parameters()
                self.epoch_number = self.rungs[0]
                specs.update(self._epoch_spec_overrides(self.epoch_number))
                self.config_specs[self.next_config_id] = specs
                self.config_to_rung[self.next_config_id] = 0
                self.active_configs.add(self.next_config_id)
                self.total_configs_started += 1
                self.track_id = self.next_config_id
                self.next_config_id += 1
                new_recommendations.append(specs)
            else:
                break

        return new_recommendations
