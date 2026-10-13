# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAO public names for the shared repository-neutral prompt optimizer.

The implementation is maintained in automl_core; these aliases preserve this
consumer's existing API without requiring TAO in other consumers.
"""

from automl_core.prompts import (
    ActionBatchRunner as TAOActionBatchRunner,
)
from automl_core.prompts import (
    AutoPrompterResult,
    GEPAReflectionLM,
    GEPAutoPrompter,
)
from automl_core.prompts import (
    GEPAAdapter as TAOGEPAAdapter,
)
from automl_core.prompts import (
    RoutedActionBatchRunner as RoutedTAOActionBatchRunner,
)

__all__ = [
    "AutoPrompterResult",
    "GEPAReflectionLM",
    "GEPAutoPrompter",
    "RoutedTAOActionBatchRunner",
    "TAOActionBatchRunner",
    "TAOGEPAAdapter",
]
