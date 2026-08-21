# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure math utility functions for AutoML parameter generation."""

import math

# Re-export types that brain algorithms import via math_utils
from tao_automl.types import Recommendation, ResumeRecommendation, JobStates  # noqa: F401


def fix_input_dimension(dimension_value, factor=32):
    """Return dimension as a multiple of factor"""
    if int(dimension_value) % factor == 0:
        return dimension_value
    return (int(dimension_value / factor) + 1) * factor


def fix_power_of_factor(value, factor=2):
    """Return the nearest power of factor that is >= value"""
    if value <= 0:
        return factor  # Return the base factor for non-positive values
    # Calculate the power needed: factor^power >= value
    power = math.ceil(math.log(value) / math.log(factor))
    return int(factor ** power)


def clamp_value(value, v_min, v_max):
    """Clamps value within the given range"""
    # A zero-width custom range represents an intentionally fixed parameter.
    # The boundary nudges below would otherwise move it twice (for example,
    # 0.3 -> 0.27 -> 0.33), violating the caller's exact range contract.
    if v_min == v_max:
        return v_min
    if value >= v_max:
        epsilon = v_max / 10
        if epsilon == 0.0:
            epsilon = 0.0000001
        value = v_max - epsilon
    if value <= v_min:
        epsilon = v_min / 10
        if epsilon == 0.0:
            epsilon = 0.0000001
        value = v_min + epsilon
    return value


def get_valid_range(parameter_config, parent_params, custom_ranges=None):
    """Compute the clamp range for the given parameter

    Args:
        parameter_config: Configuration dict for the parameter
        parent_params: Dict of parent parameter values
        custom_ranges: Optional dict of custom parameter ranges from user

    Returns:
        Tuple of (v_min, v_max)
    """
    parameter_name = parameter_config.get("parameter", "")

    # Handle empty strings and None values for numeric parameters
    valid_min = parameter_config.get("valid_min")
    valid_max = parameter_config.get("valid_max")
    default_val = parameter_config.get("default_value")

    # Convert to float, handling empty strings and None
    v_min = float(valid_min) if valid_min not in (None, '', "") else 0.0
    v_max = float(valid_max) if valid_max not in (None, '', "") else float('inf')
    default_value = float(default_val) if default_val not in (None, '', "") else 0.0
    if math.isinf(v_min):
        v_min = default_value
    if math.isinf(v_max):
        v_max = default_value

    # Apply custom ranges if provided
    if custom_ranges and parameter_name in custom_ranges:
        custom_min = custom_ranges[parameter_name].get("valid_min")
        custom_max = custom_ranges[parameter_name].get("valid_max")
        if custom_min is not None:
            v_min = float(custom_min) if not isinstance(custom_min, list) else custom_min
        if custom_max is not None:
            v_max = float(custom_max) if not isinstance(custom_max, list) else custom_max

    # Check for custom depends_on, otherwise use schema depends_on
    dependent_on_param = parameter_config.get("depends_on", None)
    if custom_ranges and parameter_name in custom_ranges:
        custom_depends_on = custom_ranges[parameter_name].get("depends_on")
        if custom_depends_on is not None:
            dependent_on_param = custom_depends_on
    if type(dependent_on_param) is str and dependent_on_param:
        parts = dependent_on_param.split(" ")
        if len(parts) >= 2:
            dependent_on_param_op = parts[0]
            dependent_on_param_name = parts[1]
        else:
            dependent_on_param_name = parts[0]
            math_cond = parameter_config.get("math_cond", "")
            if type(math_cond) is str and "depends_on" in math_cond:
                dependent_on_param_op = math_cond.strip().split()[0]
            else:
                dependent_on_param_op = None

        if dependent_on_param_op is not None:
            if dependent_on_param_name in parent_params.keys():
                limit_value = parent_params[dependent_on_param_name]
            else:
                limit_value = default_value

            epsilon = 0.000001
            if limit_value == epsilon:
                epsilon /= 10

            if dependent_on_param_op == ">":
                v_min = limit_value + epsilon
            elif dependent_on_param_op == ">=":
                v_min = limit_value
            elif dependent_on_param_op == "<":
                v_max = limit_value - epsilon
            elif dependent_on_param_op == "<=":
                v_max = limit_value

    return v_min, v_max


def get_valid_options(parameter_config, custom_ranges=None):
    """Get the valid options for a parameter, considering custom overrides

    Args:
        parameter_config: Configuration dict for the parameter
        custom_ranges: Optional dict of custom parameter ranges from user

    Returns:
        List of valid options (or schema default if no custom options)
    """
    parameter_name = parameter_config.get("parameter", "")
    valid_options = parameter_config.get("valid_options", [])

    # Apply custom valid_options if provided
    if custom_ranges and parameter_name in custom_ranges:
        custom_options = custom_ranges[parameter_name].get("valid_options")
        if custom_options is not None:
            if valid_options:
                schema_options = _as_list(valid_options)
                return [
                    option for option in _as_list(custom_options)
                    if option in schema_options
                ]
            valid_options = custom_options

    return valid_options


def _as_list(value):
    """Normalize a categorical option set while preserving string values."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",")]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def get_option_weights(parameter_config, custom_ranges=None):
    """Get the weights for valid options, considering custom overrides

    Args:
        parameter_config: Configuration dict for the parameter
        custom_ranges: Optional dict of custom parameter ranges from user

    Returns:
        List of weights corresponding to valid_options, or None for uniform sampling
    """
    parameter_name = parameter_config.get("parameter", "")
    option_weights = parameter_config.get("option_weights", None)

    # Apply custom option_weights if provided
    if custom_ranges and parameter_name in custom_ranges:
        custom_weights = custom_ranges[parameter_name].get("option_weights")
        if custom_weights is not None:
            option_weights = custom_weights

    return option_weights
