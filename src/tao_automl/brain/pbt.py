# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PBT (Population-Based Training) AutoML algorithm modules"""
import numpy as np
import logging
import copy

from tao_automl.utils.math_utils import (
    ResumeRecommendation, JobStates, get_valid_range, clamp_value,
    get_valid_options, get_option_weights
)
from tao_automl.brain.base import AutoMLAlgorithmBase
from tao_automl.utils.spec_utils import get_flatten_specs

logger = logging.getLogger(__name__)


class PBT(AutoMLAlgorithmBase):
    """PBT (Population-Based Training) AutoML algorithm class"""

    def __init__(self, context, state_store, network, parameters,
                 population_size=10, max_generations=20, eval_interval=10, perturbation_factor=1.2, metric="loss"):
        """Initialize the PBT algorithm class"""
        super().__init__(context, state_store, network, parameters)

        self.population_size = int(population_size)
        self.max_generations = int(max_generations)
        self.eval_interval = int(eval_interval)
        self.perturbation_factor = float(perturbation_factor)
        self._configure_objective(metric)

        self.population = {}
        self.generation = 0
        self.complete = False

        self.reverse_sort = self.metric_direction == "maximize"
        self.next_member_id = 0
        self.epoch_number = eval_interval

        self.perturbable_params = [
            p for p in parameters
            if p.get("value_type") in ("float", "int", "integer", "ordered_int")
        ]

        total_epochs = max_generations * eval_interval
        self.override_num_epochs(total_epochs, eval_interval)

        logger.info(
            f"PBT initialized with population_size={population_size}, "
            f"max_generations={max_generations}, eval_interval={eval_interval}, "
            f"perturbation_factor={perturbation_factor}"
        )

    def override_num_epochs(self, total_epochs, interval):
        """Override training parameters in train spec file"""
        spec = self.state_store.get_job_specs(self.context.id)
        for key1 in spec:
            if key1 in ("training_config", "train_config", "train"):
                for key2 in spec[key1]:
                    if key2 in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch"):
                        spec[key1][key2] = total_epochs
                    elif key2 in ("validation_interval", "val_interval"):
                        spec[key1][key2] = interval
                    elif key2 in ("checkpoint_interval", "ckpt_interval"):
                        spec[key1][key2] = interval
                    elif key2 in ("train_config"):
                        for key3 in spec[key1][key2]:
                            if key3 in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch"):
                                spec[key1][key2][key3] = total_epochs
                            elif key3 in ("validation_interval", "val_interval"):
                                spec[key1][key2][key3] = interval
                            elif key3 in ("checkpoint_interval", "ckpt_interval"):
                                spec[key1][key2][key3] = interval
        self.state_store.save_job_specs(self.context.id, spec)

    def save_state(self):
        """Save the PBT algorithm related variables to brain metadata"""
        state_dict = {}
        state_dict["population"] = {str(k): v for k, v in self.population.items()}
        state_dict["generation"] = self.generation
        state_dict["complete"] = self.complete
        state_dict["next_member_id"] = self.next_member_id
        state_dict["population_size"] = self.population_size
        state_dict["max_generations"] = self.max_generations
        state_dict["eval_interval"] = self.eval_interval
        state_dict["epoch_number"] = self.epoch_number
        state_dict["metric"] = self.metric

        self.state_store.save_brain_info(self.context.id, state_dict)

    @staticmethod
    def load_state(context, state_store, network, parameters,
                   population_size=10, max_generations=20, eval_interval=10, perturbation_factor=1.2, metric="loss"):
        """Load the PBT algorithm related variables from brain metadata"""
        json_loaded = state_store.get_brain_info(context.id)
        if not json_loaded:
            return PBT(context, state_store, network, parameters,
                       population_size, max_generations, eval_interval, perturbation_factor, metric)

        loaded_metric = json_loaded.get("metric", metric)
        brain = PBT(context, state_store, network, parameters,
                    population_size, max_generations, eval_interval, perturbation_factor, loaded_metric)
        brain.population = {int(k): v for k, v in json_loaded["population"].items()}
        brain.generation = json_loaded["generation"]
        brain.complete = json_loaded["complete"]
        brain.next_member_id = json_loaded["next_member_id"]
        if "max_generations" in json_loaded:
            brain.max_generations = json_loaded["max_generations"]
        if "epoch_number" in json_loaded:
            brain.epoch_number = json_loaded["epoch_number"]

        return brain

    def _perturb_parameter(self, param_config, current_value):
        """Perturb a parameter value using resample or perturb strategy"""
        param_config = copy.deepcopy(param_config)
        param_name = param_config.get("parameter")
        if self.custom_ranges and param_name in self.custom_ranges:
            for override_key, override_value in self.custom_ranges[param_name].items():
                if override_value is not None:
                    param_config[override_key] = override_value
        data_type = param_config.get("value_type")

        if np.random.rand() < 0.2:
            return self.generate_automl_param_rec_value(param_config)

        if data_type == "float":
            v_min = param_config.get("valid_min", "")
            v_max = param_config.get("valid_max", "")
            if v_min == "" or v_max == "":
                return current_value

            v_min, v_max = get_valid_range(param_config, self.parent_params, self.custom_ranges)

            if np.random.rand() < 0.5:
                new_value = current_value * self.perturbation_factor
            else:
                new_value = current_value / self.perturbation_factor

            new_value = clamp_value(new_value, v_min, v_max)
            return new_value

        if data_type in ("int", "integer"):
            v_min = param_config.get("valid_min", "")
            v_max = param_config.get("valid_max", "")
            if v_min == "" or v_max == "":
                return current_value

            delta = max(1, int(abs(current_value) * (self.perturbation_factor - 1.0)))
            if np.random.rand() < 0.5:
                new_value = current_value + delta
            else:
                new_value = current_value - delta

            new_value = max(int(v_min), min(int(v_max), new_value))
            return new_value

        if data_type == "ordered_int":
            valid_options = get_valid_options(param_config, self.custom_ranges)
            if not valid_options or current_value not in valid_options:
                return current_value

            current_idx = valid_options.index(current_value)
            if np.random.rand() < 0.5 and current_idx < len(valid_options) - 1:
                new_value = valid_options[current_idx + 1]
            elif current_idx > 0:
                new_value = valid_options[current_idx - 1]
            else:
                new_value = current_value

            return new_value

        if data_type == "bool":
            if np.random.rand() < 0.3:
                return not current_value
            return current_value

        if data_type in ("categorical", "ordered"):
            valid_options = get_valid_options(param_config, self.custom_ranges)
            if not valid_options or valid_options == "":
                return current_value

            if isinstance(valid_options, (list, tuple)):
                alternative_options = [opt for opt in valid_options if opt != current_value]
            else:
                alternative_options = []

            if alternative_options:
                if np.random.rand() < 0.5:
                    weights = get_option_weights(param_config, self.custom_ranges)
                    if weights and len(weights) == len(valid_options):
                        alt_weights = []
                        for i, opt in enumerate(valid_options):
                            if opt != current_value:
                                alt_weights.append(weights[i])
                        total_weight = sum(alt_weights)
                        if total_weight > 0:
                            probabilities = [w / total_weight for w in alt_weights]
                            new_value = np.random.choice(alternative_options, p=probabilities)
                        else:
                            new_value = np.random.choice(alternative_options)
                    else:
                        new_value = np.random.choice(alternative_options)
                    return new_value

            return current_value

        return current_value

    def _exploit_and_explore(self, member_id, population_results, member_job_id=None):
        """Apply exploit (copy better member) and explore (perturb) to a member"""
        member_result = self.population[member_id]["result"]
        member_rank = next(
            (i for i, (mid, _) in enumerate(population_results) if mid == member_id),
            len(population_results) - 1
        )

        threshold_rank = int(0.8 * len(population_results))
        if member_rank < threshold_rank:
            return None, None

        top_rank = int(0.2 * len(population_results))
        top_members = population_results[:max(1, top_rank)]
        source_id, source_result = top_members[np.random.randint(len(top_members))]

        new_specs = copy.deepcopy(self.population[source_id]["specs"])

        for param_config in self.perturbable_params:
            param_name = param_config["parameter"]
            if param_name in new_specs:
                current_value = new_specs[param_name]
                new_value = self._perturb_parameter(param_config, current_value)
                new_specs[param_name] = new_value

        return new_specs, source_id

    def _generate_random_parameters(self):
        """Generate random parameter values for a new population member"""
        hyperparam_dict = {}
        for param in self.parameters:
            name = param["parameter"]
            rec = self.generate_automl_param_rec_value(param)
            hyperparam_dict[name] = rec
        return hyperparam_dict

    @property
    def max_concurrent(self):
        """Return maximum number of concurrent experiments for PBT"""
        return self.population_size

    def done(self):
        """Return if PBT algorithm is complete or not"""
        return self.complete

    def generate_recommendations(self, history):
        """Generate recommendations using population-based training"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)

        if history == []:
            recommendations = []
            for _ in range(self.population_size):
                specs = self._generate_random_parameters()
                specs.update(
                    self._training_budget_spec_overrides(
                        num_epochs=self.eval_interval,
                        interval=self.eval_interval,
                    )
                )
                member_id = self.next_member_id
                self.population[member_id] = {
                    "specs": specs,
                    "result": 0.0,
                    "epochs": 0
                }
                self.next_member_id += 1
                recommendations.append(specs)
            self.track_id = 0
            return recommendations

        current_batch = history[-self.population_size:]
        all_complete = all(
            rec.status in [
                JobStates.success,
                JobStates.done,
                JobStates.failure,
                JobStates.error,
                JobStates.canceled,
            ]
            for rec in current_batch
        )

        if not all_complete:
            return []

        valid_results = {}
        for rec in current_batch:
            observation = self._completed_observation_value(rec)
            if observation is None:
                continue
            member_id = rec.id
            if member_id in self.population:
                self.population[member_id]["result"] = observation
                self.population[member_id]["epochs"] = self.population[member_id].get("epochs", 0) + self.eval_interval
                valid_results[member_id] = observation

        self.generation += 1
        self.epoch_number = (self.generation + 1) * self.eval_interval

        if self.generation >= self.max_generations:
            self.complete = True
            return []

        if not valid_results:
            logger.warning(
                "PBT has no successful finite population member to promote; "
                "ending the search"
            )
            self.complete = True
            return []

        population_results = sorted(
            valid_results.items(),
            key=lambda x: x[1],
            reverse=self.reverse_sort
        )

        recommendations = []
        for member_id in self.population.keys():
            member_job_id = None
            for rec in reversed(history):
                if rec.id == member_id:
                    member_job_id = rec.job_id
                    break

            new_specs, source_id = self._exploit_and_explore(member_id, population_results, member_job_id)

            if new_specs is not None:
                source_job_id = None
                for rec in reversed(history):
                    if rec.id == source_id:
                        source_job_id = rec.job_id
                        break

                self.population[member_id]["specs"] = new_specs
                self.population[member_id]["result"] = 0.0
                resume_from_epoch = self.population[source_id].get("epochs")
                new_specs.update(
                    self._training_budget_spec_overrides(
                        num_epochs=self.epoch_number,
                        interval=self.eval_interval,
                    )
                )
                resume_rec = ResumeRecommendation(
                    member_id,
                    new_specs,
                    member_job_id,
                    resume_from_job_id=source_job_id,
                    resume_from_epoch=resume_from_epoch,
                )
                recommendations.append(resume_rec)
            else:
                specs = copy.deepcopy(self.population[member_id]["specs"])
                resume_from_epoch = self.population[member_id].get("epochs")
                specs.update(
                    self._training_budget_spec_overrides(
                        num_epochs=self.epoch_number,
                        interval=self.eval_interval,
                    )
                )
                self.population[member_id]["specs"] = specs
                resume_rec = ResumeRecommendation(
                    member_id,
                    specs,
                    member_job_id,
                    resume_from_epoch=resume_from_epoch,
                )
                recommendations.append(resume_rec)

        self.track_id = list(self.population.keys())[0]
        return recommendations
