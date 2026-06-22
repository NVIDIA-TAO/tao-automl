# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core types for TAO AutoML."""

import datetime
from dataclasses import dataclass


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


class Recommendation:
    """Recommendation class for AutoML recommendations"""

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
        self.best_epoch_number = ""
        self.metric = metric
        self.resume_from_job_id = None  # For PBT: job ID to resume checkpoint from
        self.resume_from_epoch = None
        self.resume_from_step = None
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

    def get(self, key):
        """Returns value of requested key in the spec"""
        return self.specs.get(key, None)

    def assign_job_id(self, job_id):
        """Associates provided job id to the class objects job id"""
        assert type(job_id) is str, f"Job ID must be a string, got {type(job_id)}"
        self.job_id = job_id

        # Update last_modified timestamp when job is assigned
        self.last_modified = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    def update_result(self, result):
        """Update the result value"""
        result = float(result)
        assert type(result) is float, f"Result must be a float value, got {type(result)}"
        self.result = result

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
