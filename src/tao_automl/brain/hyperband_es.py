# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HyperBand with Early Stopping (Learning Curve Prediction) AutoML algorithm modules"""
# pylint: disable=arguments-renamed
import numpy as np
import logging
from scipy.optimize import curve_fit

from tao_automl.utils.math_utils import JobStates
from tao_automl.brain.hyperband import HyperBand

logger = logging.getLogger(__name__)


class HyperBandES(HyperBand):  # pylint: disable=arguments-renamed
    """HyperBand with Early Stopping via Learning Curve Prediction"""

    def __init__(self, context, state_store, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                 early_stop_threshold=0.8, min_early_stop_epochs=3, metric="loss"):
        """Initialize the HyperBand ES algorithm class"""
        super().__init__(
            context, state_store, network, parameters, max_epochs,
            reduction_factor, epoch_multiplier, metric=metric,
        )

        self.min_epochs_for_prediction = int(min_early_stop_epochs)
        self.confidence_threshold = float(early_stop_threshold)

        self.learning_curves = {}
        self.early_stopped_configs = set()
        self.observed_results = set()
        self.early_stop_decisions = []
        self._last_decision = None

        logger.info(
            f"HyperBandES initialized with early_stop_threshold={early_stop_threshold}, "
            f"min_early_stop_epochs={min_early_stop_epochs}, metric={metric}"
        )

    @staticmethod
    def _power_law_model(x, a, b, c):
        """Power law learning curve model: y = a * x^b + c"""
        return a * np.power(x, b) + c

    @staticmethod
    def _exponential_model(x, a, b, c):
        """Exponential learning curve model: y = a * exp(-b * x) + c"""
        return a * np.exp(-b * x) + c

    def _predict_final_performance(self, config_id, current_curve):
        """Predict final performance using learning curve extrapolation"""
        # Three parameters are fitted below, so fewer than three observations
        # cannot produce a meaningful curve even when the user deliberately
        # lowers the prediction gate for a small-budget run.
        if len(current_curve) < max(self.min_epochs_for_prediction, 3):
            return None, 0.0

        epochs = np.array([e for e, _ in current_curve])
        metrics = np.array([m for _, m in current_curve])

        try:
            p0 = [metrics[0] - metrics[-1], -0.5, metrics[-1]]

            popt_power, _ = curve_fit(
                self._power_law_model,
                epochs,
                metrics,
                p0=p0,
                maxfev=1000
            )

            max_epochs = self.ri[self.bracket][-1] * self.epoch_multiplier
            predicted_power = self._power_law_model(max_epochs, *popt_power)

            residuals = metrics - self._power_law_model(epochs, *popt_power)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((metrics - np.mean(metrics)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            confidence = max(0, min(1, r_squared))

            return predicted_power, confidence

        except Exception as e:
            logger.warning(f"Failed to fit learning curve for config {config_id}: {e}")
            return None, 0.0

    def _should_early_stop(self, config_id, current_result, current_epoch):
        """Determine if a configuration should be stopped early"""
        if config_id in self.early_stopped_configs:
            self._last_decision = {
                "decision": "discard", "reason": "already_discarded",
                "predicted_final": None, "confidence": None,
            }
            return False

        if config_id not in self.learning_curves:
            self.learning_curves[config_id] = []

        self.learning_curves[config_id].append((current_epoch, current_result))

        required_points = max(self.min_epochs_for_prediction, 3)
        if len(self.learning_curves[config_id]) < required_points:
            self._last_decision = {
                "decision": "keep", "reason": "insufficient_curve_points",
                "points": len(self.learning_curves[config_id]),
                "required_points": required_points,
                "predicted_final": None, "confidence": 0.0,
            }
            return False

        predicted_final, confidence = self._predict_final_performance(
            config_id,
            self.learning_curves[config_id]
        )

        if predicted_final is None:
            self._last_decision = {
                "decision": "keep", "reason": "prediction_unavailable",
                "predicted_final": None, "confidence": confidence,
            }
            return False
        if confidence < self.confidence_threshold:
            self._last_decision = {
                "decision": "keep", "reason": "low_confidence",
                "predicted_final": float(predicted_final),
                "confidence": float(confidence),
            }
            return False

        all_results = []
        for rec_id, curve in self.learning_curves.items():
            if rec_id != config_id and curve:
                all_results.append(curve[-1][1])

        if not all_results:
            self._last_decision = {
                "decision": "keep", "reason": "no_peer_result",
                "predicted_final": float(predicted_final),
                "confidence": float(confidence),
            }
            return False

        if self.reverse_sort:
            current_best = max(all_results)
            margin = 0.05
            should_stop = predicted_final < current_best * (1 - margin)
        else:
            current_best = min(all_results)
            margin = 0.05
            should_stop = predicted_final > current_best * (1 + margin)

        if should_stop:
            self.early_stopped_configs.add(config_id)
            logger.info(
                f"Early stopping config {config_id}: predicted={predicted_final:.4f}, "
                f"current_best={current_best:.4f}"
            )

        self._last_decision = {
            "decision": "discard" if should_stop else "keep",
            "reason": "predicted_below_peer" if should_stop else "prediction_competitive",
            "predicted_final": float(predicted_final),
            "confidence": float(confidence),
            "current_best": float(current_best),
        }

        return should_stop

    @staticmethod
    def _recommendation_budget(rec, fallback):
        """Resolve the explicit resource budget carried by a recommendation."""
        epoch_keys = {"num_epochs", "epochs", "n_epochs", "max_epochs", "epoch"}
        budgets = []

        def walk(value):
            if not isinstance(value, dict):
                return
            for key, child in value.items():
                if key.split(".")[-1] in epoch_keys and isinstance(child, (int, float)):
                    budgets.append(int(child))
                walk(child)

        walk(rec.specs)
        return max(budgets, default=int(fallback))

    def on_recommendation_result(self, recommendation, history):
        """Observe authoritative rung metrics as soon as the controller records them."""
        rec = recommendation
        if rec.status not in (JobStates.success, JobStates.done):
            return
        observation_id = f"{rec.id}:{rec.job_id or rec.last_modified}"
        if observation_id in self.observed_results:
            return
        self.observed_results.add(observation_id)

        budget = self._recommendation_budget(rec, self.epoch_number)
        self._should_early_stop(rec.id, rec.result, budget)
        decision = dict(self._last_decision or {
            "decision": "keep", "reason": "no_decision",
        })
        decision.update({
            "config_id": rec.id,
            "job_id": rec.job_id,
            "epoch": budget,
            "metric": float(rec.result),
        })
        self.early_stop_decisions.append(decision)
        logger.info(
            "HyperBandES decision config %s at epoch %s: %s (%s)",
            rec.id, budget, decision["decision"], decision["reason"],
        )

    def generate_recommendations(self, history):
        """Generates recommendations with predictive early stopping"""
        recommendations = super().generate_recommendations(history)
        filtered = []
        for rec in recommendations:
            rec_id = getattr(rec, "id", None)
            if rec_id in self.early_stopped_configs:
                logger.info("HyperBandES discard config %s before next rung", rec_id)
                continue
            filtered.append(rec)
        self.last_launched_count = len(filtered)
        return filtered

    @staticmethod
    def load_state(context, state_store, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                   early_stop_threshold=0.8, min_early_stop_epochs=3, metric="loss"):  # pylint: disable=arguments-renamed
        """Load the HyperBandES algorithm related variables from brain metadata"""
        json_loaded = state_store.get_brain_info(context.id)
        if not json_loaded:
            return HyperBandES(
                context, state_store, network, parameters, max_epochs,
                reduction_factor, epoch_multiplier,
                early_stop_threshold=early_stop_threshold,
                min_early_stop_epochs=min_early_stop_epochs,
                metric=metric,
            )

        brain = HyperBandES(
            context, state_store, network, parameters, max_epochs,
            reduction_factor, epoch_multiplier,
            early_stop_threshold=early_stop_threshold,
            min_early_stop_epochs=min_early_stop_epochs,
            metric=metric,
        )
        brain.bracket = json_loaded["bracket"]
        brain.sh_iter = json_loaded["sh_iter"]
        brain.expt_iter = json_loaded["expt_iter"]
        brain.complete = json_loaded["complete"]
        brain.epoch_number = json_loaded["epoch_number"]

        if "learning_curves" in json_loaded:
            brain.learning_curves = {
                int(config_id): curve
                for config_id, curve in json_loaded["learning_curves"].items()
            }
        if "early_stopped_configs" in json_loaded:
            brain.early_stopped_configs = set(json_loaded["early_stopped_configs"])
        brain.observed_results = set(json_loaded.get("observed_results", []))
        brain.early_stop_decisions = json_loaded.get("early_stop_decisions", [])

        return brain

    def save_state(self):
        """Save the HyperBandES algorithm related variables to brain metadata"""
        super().save_state()

        state_dict = self.state_store.get_brain_info(self.context.id)
        state_dict["learning_curves"] = {
            str(config_id): curve
            for config_id, curve in self.learning_curves.items()
        }
        state_dict["early_stopped_configs"] = list(self.early_stopped_configs)
        state_dict["observed_results"] = list(self.observed_results)
        state_dict["early_stop_decisions"] = self.early_stop_decisions

        self.state_store.save_brain_info(self.context.id, state_dict)
