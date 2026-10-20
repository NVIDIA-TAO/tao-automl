# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAO public names for the shared repository-neutral prompt optimizer.

The implementation is maintained in packages/automl-core and packaged privately
for TAO; these aliases preserve this consumer's existing API.
"""

from tao_automl._shared.prompts import (
    ActionBatchRunner as TAOActionBatchRunner,
)
from tao_automl._shared.prompts import (
    AutoPrompterResult,
    GEPAReflectionLM,
    GEPAutoPrompter,
)
from tao_automl._shared.prompts import (
    GEPAAdapter as TAOGEPAAdapter,
)
from tao_automl._shared.prompts import (
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
