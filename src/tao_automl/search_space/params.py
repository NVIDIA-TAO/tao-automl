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
    "ordered_int", "bool", "string",
    "ordered", "categorical",
    "list_1_backbone", "list_1_normal", "list_2", "list_3",
    "subset_list", "optional_list",
    "collection", "dict",
]

_TORCHAO_SUPPORTED_QUANTIZE_MODES = ["weight_only_ptq"]
_TORCHAO_SUPPORTED_QUANTIZE_ALGORITHMS = ["minmax"]
_MODELOPT_PYTORCH_SUPPORTED_QUANTIZE_MODES = ["static_ptq"]
_MODELOPT_PYTORCH_STATIC_PTQ_ALGORITHMS = ["max", "awq_lite", "awq_full"]
_MODELOPT_ONNX_SUPPORTED_QUANTIZE_MODES = ["static_ptq"]
_MODELOPT_ONNX_STATIC_PTQ_ALGORITHMS = [
    "max", "entropy", "awq_clip", "awq_lite", "awq_full", "rtn_dq",
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
    schema=None,
):
    """Determine which hyperparameters to include in the AutoML search space.

    Works by extracting the hyperparameters from the network's JSON schema,
    checking which are present in the provided training spec, filtering
    deleted / conditionally-excluded parameters, and marking which parameters
    (from *automl_hyperparameters*) should be enabled for AutoML.

    Args:
        network: Network architecture name (e.g. ``"dino"``, ``"deformable_detr"``).
        action: Action string (for example ``"train"``, ``"distill"``,
            ``"prune"``, or ``"quantize"``).
        train_specs: The current/updated action spec dict (already loaded).
        automl_hyperparameters: List of parameter names to enable for search.
        override_automl_disabled_params: If True, include parameters even when
            their schema ``automl_enabled`` flag is False.
        schema: Optional pre-built JSON schema. When supplied, it is used as
            the search-space source instead of importing the built-in TAO
            configuration module for ``network``.

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

    if schema is not None:
        json_schema = schema
    else:
        try:
            json_schema = generate_schema(network_arch, action)
        except (ImportError, KeyError, TypeError, ValueError) as e:
            logger.info("Error generating schema for network: %s", network_arch)
            logger.info("Network: %s, Action: %s", network, action)
            raise RuntimeError(
                f"Unable to generate schema for {network_arch!r} action {action!r}"
            ) from e

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
        explicit_params = set(automl_hyperparameters or [])
        for key in updated_spec_with_keys_flattened:
            if key != "vision.nframes" and not key.endswith(".vision.nframes"):
                continue
            fps_key = f"{key[:-len('.nframes')]}.fps"
            if fps_key in format_json_schema and fps_key not in explicit_params:
                params_to_exclude.add(fps_key)
                logger.info(
                    "%s is present - excluding %s from default Cosmos-RL "
                    "AutoML search",
                    key,
                    fps_key,
                )

    # ---- Build DataFrame and filter ----
    data_frame = pd.DataFrame.from_dict(format_json_schema, orient="index").reset_index()
    data_frame = data_frame[data_frame["value_type"].isin(_VALID_TYPES)]

    if not override_automl_disabled_params:
        data_frame = data_frame.loc[data_frame["automl_enabled"].ne(False)]

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
    automl_params = data_frame.loc[data_frame["automl_enabled"].eq(True)]
    automl_params = automl_params.loc[~automl_params["parameter"].isin(deleted_params)]
    automl_params = automl_params.loc[~automl_params["parameter"].isin(params_to_exclude)]
    automl_params = _filter_quantize_options_for_fixed_backend(
        automl_params,
        updated_spec_with_keys_flattened,
    )

    if schema is not None and automl_hyperparameters:
        requested = set(automl_hyperparameters)
        selected = set(automl_params["parameter"])
        missing = sorted(requested - selected)
        if missing:
            raise ValueError(
                "External AutoML schema cannot search the requested parameter(s): "
                f"{missing}. Each requested parameter must exist in the merged "
                "training spec, use a supported scalar type, and not set "
                "automl_enabled=false."
            )

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


def _filter_options(data_frame, parameter, supported_options):
    """Restrict a categorical parameter to supported options."""
    rows = data_frame["parameter"] == parameter
    if not rows.any():
        return data_frame

    filtered = data_frame.copy()
    for idx in filtered.index[rows]:
        current_options = filtered.at[idx, "valid_options"]
        if not current_options:
            continue
        supported = [
            option for option in current_options
            if option in supported_options
        ]
        if supported:
            filtered.at[idx, "valid_options"] = supported
    return filtered


def _effective_options(data_frame, parameter, fixed_value):
    """Return possible values for a parameter after previous filters."""
    rows = data_frame["parameter"] == parameter
    if not rows.any():
        return [fixed_value] if fixed_value is not None else []

    current_options = data_frame.loc[rows, "valid_options"].iloc[0]
    if current_options:
        return list(current_options)
    if fixed_value is not None:
        return [fixed_value]
    return []


def _filter_quantize_options_for_fixed_backend(data_frame, flattened_specs):
    """Drop quantize options known to be incompatible with a fixed backend."""
    searched_params = set(data_frame["parameter"].values)
    backend = flattened_specs.get("quantize.backend")
    mode = flattened_specs.get("quantize.mode")

    if backend == "torchao" and "quantize.backend" not in searched_params:
        data_frame = _filter_options(
            data_frame,
            "quantize.mode",
            _TORCHAO_SUPPORTED_QUANTIZE_MODES,
        )
        effective_modes = _effective_options(data_frame, "quantize.mode", mode)
        if set(effective_modes).issubset(set(_TORCHAO_SUPPORTED_QUANTIZE_MODES)):
            data_frame = _filter_options(
                data_frame,
                "quantize.algorithm",
                _TORCHAO_SUPPORTED_QUANTIZE_ALGORITHMS,
            )

    if backend == "modelopt.pytorch" and "quantize.backend" not in searched_params:
        data_frame = _filter_options(
            data_frame,
            "quantize.mode",
            _MODELOPT_PYTORCH_SUPPORTED_QUANTIZE_MODES,
        )
        effective_modes = _effective_options(data_frame, "quantize.mode", mode)
        if set(effective_modes).issubset(set(_MODELOPT_PYTORCH_SUPPORTED_QUANTIZE_MODES)):
            data_frame = _filter_options(
                data_frame,
                "quantize.algorithm",
                _MODELOPT_PYTORCH_STATIC_PTQ_ALGORITHMS,
            )

    if backend == "modelopt.onnx" and "quantize.backend" not in searched_params:
        data_frame = _filter_options(
            data_frame,
            "quantize.mode",
            _MODELOPT_ONNX_SUPPORTED_QUANTIZE_MODES,
        )
        effective_modes = _effective_options(data_frame, "quantize.mode", mode)
        if set(effective_modes).issubset(set(_MODELOPT_ONNX_SUPPORTED_QUANTIZE_MODES)):
            data_frame = _filter_options(
                data_frame,
                "quantize.algorithm",
                _MODELOPT_ONNX_STATIC_PTQ_ALGORITHMS,
            )

    if (
        backend is not None and
        "quantize.backend" not in searched_params and
        not _effective_options(data_frame, "quantize.mode", mode)
    ):
        data_frame = _filter_options(
            data_frame,
            "quantize.algorithm",
            [],
        )

    return data_frame
