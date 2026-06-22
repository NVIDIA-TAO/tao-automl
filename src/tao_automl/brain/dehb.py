# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEHB (Differential Evolution HyperBand) AutoML algorithm modules"""
import copy
import numpy as np
import math
import logging

from tao_automl.utils.math_utils import (
    ResumeRecommendation, JobStates, get_valid_range, clamp_value,
    get_valid_options, get_option_weights, fix_input_dimension
)
from tao_automl.brain.base import AutoMLAlgorithmBase
from tao_automl.utils.spec_utils import get_flatten_specs

logger = logging.getLogger(__name__)


class DEHB(AutoMLAlgorithmBase):
    """DEHB (Differential Evolution HyperBand) AutoML algorithm class"""

    def __init__(self, context, state_store, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                 mutation_factor=0.5, crossover_prob=0.5, metric="loss"):
        """Initialize the DEHB algorithm class"""
        super().__init__(context, state_store, network, parameters)
        self.epoch_multiplier = int(epoch_multiplier)
        self.metric = metric
        self.ni = {}
        self.ri = {}
        self.brackets_and_sh_sequence(max_epochs, reduction_factor)
        self.epoch_number = 0

        self.mutation_factor = float(mutation_factor)
        self.crossover_prob = float(crossover_prob)

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

        self.population = []
        self.population_results = []
        self.bracket_populations = {}

        logger.info(
            f"DEHB initialized with max_epochs={max_epochs}, "
            f"reduction_factor={reduction_factor}, epoch_multiplier={self.epoch_multiplier}, "
            f"mutation_factor={mutation_factor}, crossover_prob={crossover_prob}"
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

    def _normalize_config_to_vector(self, specs):
        """Convert a configuration dict to normalized vector [0, 1]^d"""
        vector = []
        for param in self.parameters:
            param = copy.deepcopy(param)
            param_name = param["parameter"]
            if self.custom_ranges and param_name in self.custom_ranges:
                for override_key, override_value in self.custom_ranges[param_name].items():
                    if override_value is not None:
                        param[override_key] = override_value
            param_name = param["parameter"]
            value = specs.get(param_name)
            param_type = param.get("value_type")

            if param_type in ("float", "int", "integer"):
                v_min = param.get("valid_min", 0)
                v_max = param.get("valid_max", 1)
                if v_max > v_min:
                    normalized = (value - v_min) / (v_max - v_min)
                    vector.append(np.clip(normalized, 0.0, 1.0))
                else:
                    vector.append(0.5)
            else:
                vector.append(0.5)

        return np.array(vector)

    def _vector_to_config(self, vector):
        """Convert normalized vector to configuration dict"""
        specs = {}
        for i, param in enumerate(self.parameters):
            param = copy.deepcopy(param)
            param_name = param["parameter"]
            if self.custom_ranges and param_name in self.custom_ranges:
                for override_key, override_value in self.custom_ranges[param_name].items():
                    if override_value is not None:
                        param[override_key] = override_value
            normalized_value = np.clip(vector[i], 0.0, 1.0)

            param_type = param.get("value_type")
            math_cond = param.get("math_cond", None)

            if param_type == "float":
                v_min, v_max = get_valid_range(param, self.parent_params, self.custom_ranges)
                value = normalized_value * (v_max - v_min) + v_min
                value = clamp_value(value, v_min, v_max)
                specs[param_name] = value

            elif param_type in ("int", "integer"):
                v_min, v_max = get_valid_range(param, self.parent_params, self.custom_ranges)
                continuous_value = normalized_value * (v_max - v_min) + v_min
                value = int(round(continuous_value))

                if math_cond and type(math_cond) is str and "depends_on" not in math_cond:
                    parts = math_cond.split(" ")
                    if len(parts) >= 2:
                        operator = parts[0]
                        factor = int(parts[1])
                        if operator == "^":
                            value = int(self._apply_power_constraint_with_equal_priority(
                                v_min, v_max, factor, value))
                        elif operator == "/":
                            value = fix_input_dimension(value, factor)

                value = int(max(v_min, min(v_max, value)))
                specs[param_name] = value

            elif param_type in ("categorical", "ordered"):
                valid_options = get_valid_options(param, self.custom_ranges)
                if valid_options and valid_options != "":
                    idx = int(normalized_value * len(valid_options))
                    idx = min(idx, len(valid_options) - 1)

                    weights = get_option_weights(param, self.custom_ranges)
                    if weights and len(weights) == len(valid_options):
                        sorted_pairs = sorted(zip(valid_options, weights), key=lambda x: x[1], reverse=True)
                        cumulative = 0
                        total_weight = sum(weights)
                        for option, weight in sorted_pairs:
                            cumulative += weight / total_weight
                            if normalized_value <= cumulative:
                                specs[param_name] = option
                                break
                        else:
                            specs[param_name] = sorted_pairs[0][0]
                    else:
                        specs[param_name] = valid_options[idx]
                else:
                    specs[param_name] = param.get("default_value")

            elif param_type == "ordered_int":
                valid_options = get_valid_options(param, self.custom_ranges)
                if valid_options and valid_options != "":
                    idx = int(normalized_value * len(valid_options))
                    idx = min(idx, len(valid_options) - 1)

                    weights = get_option_weights(param, self.custom_ranges)
                    if weights and len(weights) == len(valid_options):
                        sorted_pairs = sorted(zip(valid_options, weights), key=lambda x: x[1], reverse=True)
                        cumulative = 0
                        total_weight = sum(weights)
                        for option, weight in sorted_pairs:
                            cumulative += weight / total_weight
                            if normalized_value <= cumulative:
                                specs[param_name] = int(option)
                                break
                        else:
                            specs[param_name] = int(sorted_pairs[0][0])
                    else:
                        specs[param_name] = int(valid_options[idx])
                else:
                    default_val = param.get("default_value")
                    specs[param_name] = int(default_val) if default_val else 0

            elif param_type == "bool":
                specs[param_name] = normalized_value >= 0.5

            else:
                specs[param_name] = self.generate_automl_param_rec_value(param)

        return specs

    def _differential_evolution_mutation(self):
        """Generate new configuration using DE mutation and crossover"""
        if len(self.population) < 4:
            return self._generate_random_parameters()

        base_idx = np.random.randint(len(self.population))
        base_vector = self.population[base_idx]

        indices = list(range(len(self.population)))
        indices.remove(base_idx)
        r1, r2 = np.random.choice(indices, size=2, replace=False)

        mutant_vector = base_vector + self.mutation_factor * (
            self.population[r1] - self.population[r2]
        )

        mutant_vector = np.clip(mutant_vector, 0.0, 1.0)

        trial_vector = np.copy(base_vector)
        for i in range(len(trial_vector)):
            if np.random.rand() < self.crossover_prob:
                trial_vector[i] = mutant_vector[i]

        if np.random.rand() < self.crossover_prob:
            j_rand = np.random.randint(len(trial_vector))
            trial_vector[j_rand] = mutant_vector[j_rand]

        return self._vector_to_config(trial_vector)

    def _generate_random_parameters(self):
        """Generate random parameter values"""
        hyperparam_dict = {}
        for param in self.parameters:
            name = param["parameter"]
            rec = self.generate_automl_param_rec_value(param)
            hyperparam_dict[name] = rec
        return hyperparam_dict

    def save_state(self):
        """Save the DEHB algorithm related variables to brain metadata"""
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
        state_dict["population"] = [p.tolist() for p in self.population]
        state_dict["population_results"] = self.population_results

        self.state_store.save_brain_info(self.context.id, state_dict)

    @staticmethod
    def load_state(context, state_store, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                   mutation_factor=0.5, crossover_prob=0.5, metric="loss"):
        """Load the DEHB algorithm related variables from brain metadata"""
        json_loaded = state_store.get_brain_info(context.id)
        if not json_loaded:
            return DEHB(context, state_store, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                        mutation_factor, crossover_prob, metric)

        loaded_metric = json_loaded.get("metric", metric)
        brain = DEHB(context, state_store, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                     mutation_factor, crossover_prob, loaded_metric)
        brain.bracket = json_loaded["bracket"]
        brain.sh_iter = json_loaded["sh_iter"]
        brain.expt_iter = json_loaded["expt_iter"]
        brain.complete = json_loaded["complete"]
        brain.epoch_number = json_loaded["epoch_number"]
        brain.last_launched_count = json_loaded.get("last_launched_count", 0)

        if "population" in json_loaded:
            brain.population = [np.array(p) for p in json_loaded["population"]]
            brain.population_results = json_loaded["population_results"]

        return brain

    def _generate_one_recommendation(self, history):
        """Updates the counter variables and performs successive halving with DE"""
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
            self.complete = True
            return None

        if self.sh_iter == 0:
            specs = self._differential_evolution_mutation()
            self.epoch_number = self.ri[self.bracket][self.sh_iter] * self.epoch_multiplier
            final_epoch = self.ri[self.bracket][-1] * self.epoch_multiplier
            self.override_num_epochs(final_epoch)
            specs.update(self._epoch_spec_overrides(self.epoch_number))
            to_return = specs
        else:
            lower = -1 * self.ni.get(self.bracket, [0])[0]

            if self.expt_iter == 0:
                if self.sh_iter == 1:
                    self.experiments_considered = sorted(
                        history[lower:],
                        key=lambda rec: rec.result,
                        reverse=self.reverse_sort
                    )[0:self.ni[self.bracket][self.sh_iter]]
                else:
                    for experiment in self.experiments_considered:
                        experiment.result = history[experiment.id].result
                    self.experiments_considered = sorted(
                        self.experiments_considered,
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
        """Return if DEHB algorithm is complete or not."""
        if not self.complete:
            return False
        if self.last_launched_count > 0:
            return False
        return True

    @property
    def max_concurrent(self):
        """Maximum number of concurrent experiments for DEHB."""
        max_ni = 1
        for bracket_ni in self.ni.values():
            if bracket_ni:
                max_ni = max([max_ni] + bracket_ni)
        return max_ni

    def generate_recommendations(self, history):
        """Generates recommendations for the controller to run"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)

        if self.complete:
            if self.last_launched_count > 0 and history:
                any_running = any(
                    exp.status in [JobStates.pending, JobStates.started, JobStates.running]
                    for exp in history
                )
                if not any_running:
                    self.last_launched_count = 0
            return []

        # Update DE population
        for rec in history:
            if rec.status == JobStates.success and rec.result is not None:
                config_vector = self._normalize_config_to_vector(rec.specs)
                is_duplicate = any(np.allclose(config_vector, p) for p in self.population)
                if not is_duplicate:
                    self.population.append(config_vector)
                    self.population_results.append(rec.result)

                    if len(self.population) > 50:
                        if self.reverse_sort:
                            worst_idx = np.argmin(self.population_results)
                        else:
                            worst_idx = np.argmax(self.population_results)
                        self.population.pop(worst_idx)
                        self.population_results.pop(worst_idx)

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
