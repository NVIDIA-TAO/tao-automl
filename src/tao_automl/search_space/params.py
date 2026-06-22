# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AutoML search space parameter extraction module.

Determines which hyperparameters should be included in the AutoML search space
based on the network schema and user configuration.
"""

import logging

import pandas as pd

from tao_automl.schema.generate_schema import generate_schema
from tao_automl.utils.spec_utils import get_flatten_specs, flatten_properties

logger = logging.getLogger(__name__)

# AutoML enablement is owned by model-level metadata in the skill bank. Keep
# the standalone runner free of per-network hard disables so every model with a
# train schema can use its schema-declared search space.
AUTOML_DISABLED_NETWORKS = []

_VALID_TYPES = [
    "int", "integer",
    "float",
    "ordered_int", "bool",
    "ordered", "categorical",
    "list_1_backbone", "list_1_normal", "list_2", "list_3",
    "subset_list", "optional_list",
    "collection", "dict",
]


def _get_network_and_action(network, action):
    """Return (network, action) as-is.

    This is a simplified local helper replacing the FTMS
    ``get_microservices_network_and_action`` mapping.  In the standalone
    wheel the caller already knows the canonical network name and action,
    so no translation is needed.
    """
    return network, action


def generate_hyperparams_to_search(
    network,
    action,
    train_specs,
    automl_hyperparameters,
    override_automl_disabled_params=False,
):
    """Determine which hyperparameters to include in the AutoML search space.

    Works by extracting the hyperparameters from the network's JSON schema,
    checking which are present in the provided training spec, filtering
    deleted / conditionally-excluded parameters, and marking which parameters
    (from *automl_hyperparameters*) should be enabled for AutoML.

    Args:
        network: Network architecture name (e.g. ``"dino"``, ``"deformable_detr"``).
        action: Action string (typically ``"train"``).
        train_specs: The current/updated training spec dict (already loaded).
        automl_hyperparameters: List of parameter names to enable for search.
        override_automl_disabled_params: If True, include parameters even when
            their schema ``automl_enabled`` flag is False.

    Returns:
        Tuple of ``(param_records, param_names)`` where *param_records* is a
        list of dicts (one per searchable parameter) and *param_names* is the
        corresponding list of dotted parameter name strings.

    Note:
        Parameter names use dot notation to represent nested paths in the spec
        structure (e.g. ``"train.optim.lr"``, ``"dataset.batch_size"``).
    """
    network_arch, _ = _get_network_and_action(network, action)
    logger.info("Network arch: %s", network_arch)

    if network_arch in AUTOML_DISABLED_NETWORKS:
        return [{}], []

    try:
        json_schema = generate_schema(network_arch, "train")
    except Exception as e:
        logger.info("Error generating schema for network: %s", network_arch)
        logger.info("Network: %s, Action: %s", network, action)
        raise Exception(e) from e

    # Flatten original (default) spec from the schema
    original_train_spec = json_schema.get("default", {})
    original_spec_with_keys_flattened = {}
    get_flatten_specs(original_train_spec, original_spec_with_keys_flattened)

    # Flatten the caller-provided (updated) training spec
    updated_spec_with_keys_flattened = {}
    get_flatten_specs(train_specs, updated_spec_with_keys_flattened)

    # Parameters that exist in the schema default but were removed by the user
    deleted_params = (
        original_spec_with_keys_flattened.keys() - updated_spec_with_keys_flattened.keys()
    )

    # Build a flat property map from the schema
    format_json_schema = flatten_properties(json_schema["properties"])

    # ---- Network-specific exclusions ----
    params_to_exclude = set()

    # For cosmos-rl: exclude LoRA parameters when LoRA block is absent
    if network_arch == "cosmos-rl":
        has_lora_params = any(
            key.startswith("policy.lora.") for key in updated_spec_with_keys_flattened
        )
        if not has_lora_params:
            logger.info(
                "policy.lora not found in updated spec - excluding LoRA parameters from AutoML"
            )
            params_to_exclude.update(
                p for p in format_json_schema.keys() if p.startswith("policy.lora.")
            )
            logger.info("Excluding %d LoRA parameters: %s", len(params_to_exclude), params_to_exclude)
        if (
            "custom.vision.nframes" in updated_spec_with_keys_flattened
            and "custom.vision.fps" not in set(automl_hyperparameters or [])
        ):
            params_to_exclude.add("custom.vision.fps")
            logger.info(
                "custom.vision.nframes is present - excluding custom.vision.fps "
                "from default Cosmos-RL AutoML search"
            )

    # ---- Build DataFrame and filter ----
    data_frame = pd.DataFrame.from_dict(format_json_schema, orient="index").reset_index()
    data_frame = data_frame[data_frame["value_type"].isin(_VALID_TYPES)]

    if not override_automl_disabled_params:
        data_frame = data_frame.loc[data_frame["automl_enabled"] != False]  # noqa: E712

    if automl_hyperparameters:
        # Caller specified which params to search — enable only those
        data_frame["automl_enabled"] = False
        data_frame.loc[
            data_frame.parameter.isin(automl_hyperparameters), "automl_enabled"
        ] = True
    else:
        # No explicit list — use schema defaults (automl_enabled=TRUE in dataclass)
        # After the filter above, all remaining rows had automl_enabled != False.
        # Convert string "TRUE" to boolean True for consistent filtering.
        data_frame["automl_enabled"] = data_frame["automl_enabled"].apply(
            lambda x: str(x).upper() == "TRUE" if not isinstance(x, bool) else x
        )

    # Keep only enabled, non-deleted, non-excluded parameters
    automl_params = data_frame.loc[data_frame["automl_enabled"] == True]  # noqa: E712
    automl_params = automl_params.loc[~automl_params["parameter"].isin(deleted_params)]
    automl_params = automl_params.loc[~automl_params["parameter"].isin(params_to_exclude)]

    # Sort: parameters that depend on other parameters go last
    automl_params = automl_params.sort_values(by=["depends_on"], na_position="first")

    # Select the columns required by the brain algorithms
    automl_params = automl_params[[
        "parameter",
        "value_type",
        "default_value",
        "valid_min",
        "valid_max",
        "valid_options",
        "option_weights",
        "math_cond",
        "parent_param",
        "depends_on",
    ]]

    logger.info("Automl params enabled: %s", automl_params["parameter"].values)
    return automl_params.to_dict("records"), list(automl_params["parameter"].values)
