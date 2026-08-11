# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spec flattening utilities for AutoML."""


def resolve_schema_leaf(node):
    """Return the concrete type and effective metadata for a schema leaf.

    Optional JSON schemas commonly put bounds, enums, and defaults on the
    concrete branch of ``anyOf``.  AutoML needs those fields just as if they
    were declared directly on the property.  Property-level fields take
    precedence when both locations define the same key.
    """
    if not isinstance(node, dict):
        return "", {}, False

    effective = {}
    nullable = False
    dtype = node.get("type", "")
    any_of = node.get("anyOf")
    if not dtype and isinstance(any_of, list):
        concrete = [
            option for option in any_of
            if isinstance(option, dict) and option.get("type") != "null"
        ]
        nullable = any(
            isinstance(option, dict) and option.get("type") == "null"
            for option in any_of
        )
        if concrete:
            # Built-in TAO schemas may offer multiple concrete branches (for
            # example scalar-or-array learning rates). The search-space brain
            # historically tunes the first concrete branch; external schemas
            # apply their stricter ambiguity checks before reaching here.
            effective.update(concrete[0])
            dtype = concrete[0].get("type", "")

    effective.update(node)
    return dtype, effective, nullable


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
            dtype, metadata, _ = resolve_schema_leaf(v)
            valid_options = metadata.get('enum', [])
            if valid_options:
                # Integer enums are discrete ordered choices. Other scalar
                # enums are categorical; treating them as their base type
                # would cause numeric brains to ignore the declared options.
                integer_choices = dtype in ('int', 'integer', 'ordered_int') and all(
                    isinstance(option, int) and not isinstance(option, bool)
                    for option in valid_options
                )
                dtype = 'ordered_int' if integer_choices else 'categorical'
            elif dtype == "number":
                dtype = "float"
            elif dtype == "boolean":
                dtype = "bool"
            flattened[new_key] = {
                'parameter': new_key,
                'value_type': dtype,
                'default_value': metadata.get('default', ''),
                'valid_min': metadata.get('minimum', ''),
                'valid_max': metadata.get('maximum', ''),
                'valid_options': valid_options,
                'option_weights': metadata.get('option_weights', None),
                'automl_enabled': metadata.get('automl_enabled', ''),
                'math_cond': metadata.get('math_cond', ''),
                'parent_param': metadata.get('parent_param', ''),
                'depends_on': metadata.get('depends_on', ''),
            }

    return flattened
