# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HyperBand with Early Stopping (Learning Curve Prediction) AutoML algorithm modules"""
import numpy as np
import logging
from scipy.optimize import curve_fit

from tao_automl.utils.math_utils import JobStates
from tao_automl.brain.hyperband import HyperBand

logger = logging.getLogger(__name__)


class HyperBandES(HyperBand):
    """HyperBand with Early Stopping via Learning Curve Prediction"""

    def __init__(self, context, state_store, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                 early_stop_threshold=0.8, min_early_stop_epochs=3):
        """Initialize the HyperBand ES algorithm class"""
        super().__init__(context, state_store, network, parameters, max_epochs, reduction_factor, epoch_multiplier)

        self.min_epochs_for_prediction = int(min_early_stop_epochs)
        self.confidence_threshold = float(early_stop_threshold)

        self.learning_curves = {}
        self.early_stopped_configs = set()

        logger.info(
            f"HyperBandES initialized with early_stop_threshold={early_stop_threshold}, "
            f"min_early_stop_epochs={min_early_stop_epochs}"
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
        if len(current_curve) < self.min_epochs_for_prediction:
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
            return False

        if config_id not in self.learning_curves:
            self.learning_curves[config_id] = []

        self.learning_curves[config_id].append((current_epoch, current_result))

        if len(self.learning_curves[config_id]) < self.min_epochs_for_prediction:
            return False

        predicted_final, confidence = self._predict_final_performance(
            config_id,
            self.learning_curves[config_id]
        )

        if predicted_final is None or confidence < self.confidence_threshold:
            return False

        all_results = []
        for rec_id, curve in self.learning_curves.items():
            if rec_id != config_id and curve:
                all_results.append(curve[-1][1])

        if not all_results:
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

        return should_stop

    def generate_recommendations(self, history):
        """Generates recommendations with predictive early stopping"""
        recommendations = super().generate_recommendations(history)

        for rec in history:
            if rec.status == JobStates.running and rec.result != 0.0:
                current_epoch = self.epoch_number
                if self._should_early_stop(rec.id, rec.result, current_epoch):
                    logger.info(f"Triggering early stop for config {rec.id}")
                    rec.update_status(JobStates.failure)

        return recommendations

    @staticmethod
    def load_state(context, state_store, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                   metric="loss", min_epochs_for_prediction=3, confidence_threshold=0.8):
        """Load the HyperBandES algorithm related variables from brain metadata"""
        json_loaded = state_store.get_brain_info(context.id)
        if not json_loaded:
            return HyperBandES(
                context, state_store, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
                min_epochs_for_prediction, confidence_threshold
            )

        brain = HyperBandES(
            context, state_store, network, parameters, max_epochs, reduction_factor, epoch_multiplier,
            min_epochs_for_prediction, confidence_threshold
        )
        brain.bracket = json_loaded["bracket"]
        brain.sh_iter = json_loaded["sh_iter"]
        brain.expt_iter = json_loaded["expt_iter"]
        brain.complete = json_loaded["complete"]
        brain.epoch_number = json_loaded["epoch_number"]

        if "learning_curves" in json_loaded:
            brain.learning_curves = json_loaded["learning_curves"]
        if "early_stopped_configs" in json_loaded:
            brain.early_stopped_configs = set(json_loaded["early_stopped_configs"])

        return brain

    def save_state(self):
        """Save the HyperBandES algorithm related variables to brain metadata"""
        super().save_state()

        state_dict = self.state_store.get_brain_info(self.context.id)
        state_dict["learning_curves"] = self.learning_curves
        state_dict["early_stopped_configs"] = list(self.early_stopped_configs)

        self.state_store.save_brain_info(self.context.id, state_dict)
