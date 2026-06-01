# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spec flattening utilities for AutoML."""


def get_flatten_specs(dict_spec, flat_specs, parent=""):
    """Flatten nested dictionary"""
    for key, value in dict_spec.items():
        if isinstance(value, dict):
            get_flatten_specs(value, flat_specs, parent + key + ".")
        else:
            flat_key = parent + key
            flat_specs[flat_key] = value


def flatten_properties(data, parent_key='', sep='.'):
    """Convert schema to a readable dict"""
    flattened = {}

    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict) and v.get('type') in ('object', 'collection', 'dict') and 'properties' in v:
            # Recurse if the value has nested properties
            flattened.update(flatten_properties(v['properties'], new_key, sep))
        elif isinstance(v, dict):
            # Otherwise, gather metadata
            dtype = v.get('type', '')

            # Handle union types (anyOf schemas)
            if 'anyOf' in v and not dtype:
                # For union types, determine the primary type from anyOf
                any_of_types = v.get('anyOf', [])
                if any_of_types:
                    # Use the first type as the primary type for AutoML
                    first_type = any_of_types[0].get('type', '')
                    if first_type == 'integer':
                        dtype = 'int'
                    elif first_type == 'number':
                        dtype = 'float'
                    elif first_type == 'boolean':
                        dtype = 'bool'
                    else:
                        dtype = first_type

            if v.get('type', '') == "number":
                dtype = "float"
            if v.get('type', '') == "boolean":
                dtype = "bool"
            flattened[new_key] = {
                'parameter': new_key,
                'value_type': dtype,
                'default_value': v.get('default', ''),
                'valid_min': v.get('minimum', ''),
                'valid_max': v.get('maximum', ''),
                'valid_options': v.get('enum', []),
                'option_weights': v.get('option_weights', None),
                'automl_enabled': v.get('automl_enabled', ''),
                'math_cond': v.get('math_cond', ''),
                'parent_param': v.get('parent_param', ''),
                'depends_on': v.get('depends_on', ''),
            }

    return flattened
