# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core types for TAO AutoML."""

import datetime
from collections.abc import Mapping
from dataclasses import dataclass

from tao_automl.utils.value_utils import (
    normalize_finite_number,
    normalize_json_value,
)


class JobStates():
    """Various states of an automl job"""

    pending = "pending"
    started = "started"
    running = "running"
    success = "success"
    failure = "failure"
    error = "error"  # alias for failure
    done = "done"  # alias for success
    canceled = "canceled"
    canceling = "canceling"


class Recommendation(Mapping):
    """Recommendation class for AutoML recommendations.

    Implements the ``collections.abc.Mapping`` protocol over ``self.specs``:
    callbacks receive Recommendation objects and near-universally treat them
    as the spec dict (``rec.get("train.optim.lr", 1e-4)``, ``dict(rec)``,
    iteration, ``key in rec``). Attribute access (``rec.id``, ``rec.specs``,
    ``rec.result``, …) is unchanged.
    """

    def __init__(self, identifier, specs, metric):
        """Initialize the Recommendation class

        Args:
            identity: the id of the recommendation
            specs: the specs/config of the recommendation
        """
        assert type(identifier) is int, f"Recommendation identifier must be an integer, got {type(identifier)}"
        self.id = identifier

        assert type(specs) is dict, f"Recommendation specs must be a dictionary, got {type(specs)}"
        self.specs = specs

        self.job_id = None
        self.status = JobStates.pending
        self.result = 0.0
        self.objective_values = {}
        self.objective_score = 0.0
        self.best_epoch_number = ""
        self.metric = metric
        self.resume_from_job_id = None  # For PBT: job ID to resume checkpoint from
        self.resume_from_epoch = None
        self.resume_from_step = None
        # Monotonically increasing batch/window assigned by Controller.  The
        # latest window must remain available until a multi-fidelity brain has
        # made its next promotion decision.
        self.checkpoint_window = 0
        self.early_stop_epoch = None  # For PBT/Hyperband: epoch limit when this rec was launched
        self.failure_reason = None
        self.adjustments = []

        # Add timestamps for timeout tracking
        current_time = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        self.created_on = current_time
        self.last_modified = current_time

    def items(self):
        """Returns specs.items"""
        return self.specs.items()

    def get(self, key, default=None):
        """Returns value of requested key in the spec"""
        return self.specs.get(key, default)

    def __getitem__(self, key):
        return self.specs[key]

    def __iter__(self):
        return iter(self.specs)

    def __len__(self):
        return len(self.specs)

    def __bool__(self):
        # Mapping would make a rec with empty specs falsy; recommendations
        # are objects with identity and callers rely on `best if best else …`.
        return True

    def assign_job_id(self, job_id):
        """Associates provided job id to the class objects job id"""
        assert type(job_id) is str, f"Job ID must be a string, got {type(job_id)}"
        self.job_id = job_id

        # Update last_modified timestamp when job is assigned
        self.last_modified = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    def update_result(self, result):
        """Update the result value"""
        result = normalize_finite_number(result, path="recommendation.result")
        self.result = result
        self.objective_score = result
        self.objective_values = {self.metric: result}

    def update_objectives(self, objective_values, objective_score):
        """Update raw objective values and the scalar optimization score."""
        normalized_values = normalize_json_value(
            objective_values,
            path="recommendation.objective_values",
        )
        if not isinstance(normalized_values, dict):
            raise TypeError(
                "Recommendation objective values must be a dictionary, "
                f"got {type(normalized_values).__name__}"
            )
        self.objective_values = {
            key: normalize_finite_number(
                value,
                path=f"recommendation.objective_values.{key}",
            )
            for key, value in normalized_values.items()
        }
        score = normalize_finite_number(
            objective_score,
            path="recommendation.objective_score",
        )
        self.objective_score = score
        self.result = score

    def primary_metric_value(self):
        """Return the raw value for this recommendation's primary metric."""
        return self.objective_values.get(self.metric, self.result)

    def update_status(self, status):
        """Update the status value"""
        assert type(status) is str, f"Status must be a string, got {type(status)}"
        self.status = status

        # Update last_modified timestamp when status changes
        self.last_modified = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    def __repr__(self):
        """Constructs a dictionary with the class members and returns them"""
        return f"id: {self.id}\njob_id: {self.job_id}\nresult: {self.result}\nstatus: {self.status}"


class ResumeRecommendation:
    """Recommendation class for Hyperband resume experiments"""

    def __init__(
        self,
        identity,
        specs,
        job_id,
        resume_from_job_id=None,
        resume_from_epoch=None,
        resume_from_step=None,
    ):
        """Initialize the ResumeRecommendation class

        Args:
            identity: the id of the recommendation
            specs: the specs/config of the recommendation
            job_id: the job id of the recommendation
            resume_from_job_id: (PBT) the job id to resume checkpoint from if member was replaced
            resume_from_epoch: epoch checkpoint to resume from
            resume_from_step: step checkpoint to resume from
        """
        self.id = identity
        self.specs = specs
        self.job_id = job_id
        self.resume_from_job_id = resume_from_job_id
        self.resume_from_epoch = resume_from_epoch
        self.resume_from_step = resume_from_step


# below code was changed from FTMS version - to remove flask/mongo db related fields


@dataclass
class AutoMLContext:
    """Context object for an AutoML session, replacing the FTMS JobContext."""

    id: str              # session/job ID
    network: str         # network architecture name
    action: str = "train"
    workspace_path: str = ""
    metric: str = "loss"
    handler_id: str = ""  # experiment ID for custom ranges
    num_gpu: int = -1
    random_seed: int | None = None
