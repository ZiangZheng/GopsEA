# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for utilities for working with dictionaries."""

import collections.abc
import hashlib
import json
import torch
from collections.abc import Iterable, Mapping, Sized
from typing import Any

from .string import callable_to_string, string_to_callable, string_to_slice

"""
Dictionary <-> Class operations.
"""


def class_to_dict(obj: object) -> dict[str, Any]:
    """Convert an object into dictionary recursively.

    Note:
        Ignores all names starting with "__" (i.e. built-in methods).

    Args:
        obj: An instance of a class to convert.

    Raises:
        ValueError: When input argument is not an object.

    Returns:
        Converted dictionary mapping.
    """
    # check that input data is class instance
    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")
    # convert object to dictionary
    if isinstance(obj, dict):
        obj_dict = obj
    elif isinstance(obj, torch.Tensor):
        # We have to treat torch tensors specially because `torch.tensor.__dict__` returns an empty
        # dict, which would mean that a torch.tensor would be stored as an empty dict. Instead we
        # want to store it directly as the tensor.
        return obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return obj

    # convert to dictionary
    data = dict()
    for key, value in obj_dict.items():
        # disregard builtin attributes
        if key.startswith("__"):
            continue
        # check if attribute is callable -- function
        if callable(value):
            data[key] = callable_to_string(value)
        # check if attribute is a dictionary
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)
        # check if attribute is a list or tuple
        elif isinstance(value, (list, tuple)):
            data[key] = type(value)([class_to_dict(v) for v in value])
        else:
            data[key] = value
    return data


def update_class_from_dict(obj, data: dict[str, Any], _ns: str = "") -> None:
    """Reads a dictionary and sets object variables recursively.

    This function performs in-place update of the class member attributes.

    Args:
        obj: An instance of a class to update.
        data: Input dictionary to update from.
        _ns: Namespace of the current object. This is useful for nested configuration
            classes or dictionaries. Defaults to "".

    Raises:
        TypeError: When input is not a dictionary.
        ValueError: When dictionary has a value that does not match default config type.
        KeyError: When dictionary has a key that does not exist in the default config type.
    """
    for key, value in data.items():
        # key_ns is the full namespace of the key
        key_ns = _ns + "/" + key

        # -- A) if key is present in the object ------------------------------------
        if hasattr(obj, key) or (isinstance(obj, dict) and key in obj):
            obj_mem = obj[key] if isinstance(obj, dict) else getattr(obj, key)

            # -- 1) nested mapping → recurse ---------------------------
            if isinstance(value, Mapping):
                # recursively call if it is a dictionary
                update_class_from_dict(obj_mem, value, _ns=key_ns)
                continue

            # -- 2) iterable (list / tuple / etc.) ---------------------
            if isinstance(value, Iterable) and not isinstance(value, str):

                # ---- 2a) flat iterable → replace wholesale ----------
                if all(not isinstance(el, Mapping) for el in value):
                    out_val = tuple(value) if isinstance(obj_mem, tuple) else value
                    if isinstance(obj, dict):
                        obj[key] = out_val
                    else:
                        setattr(obj, key, out_val)
                    continue

                # ---- 2b) existing value is None → abort -------------
                if obj_mem is None:
                    raise ValueError(
                        f"[Config]: Cannot merge list under namespace: {key_ns} because the existing value is None."
                    )

                # ---- 2c) length mismatch → abort -------------------
                if isinstance(obj_mem, Sized) and isinstance(value, Sized) and len(obj_mem) != len(value):
                    raise ValueError(
                        f"[Config]: Incorrect length under namespace: {key_ns}."
                        f" Expected: {len(obj_mem)}, Received: {len(value)}."
                    )

                # ---- 2d) keep tuple/list parity & recurse ----------
                if isinstance(obj_mem, tuple):
                    value = tuple(value)
                else:
                    set_obj = True
                    # recursively call if iterable contains Mappings
                    for i in range(len(obj_mem)):
                        if isinstance(value[i], Mapping):
                            update_class_from_dict(obj_mem[i], value[i], _ns=key_ns)
                            set_obj = False
                    # do not set value to obj, otherwise it overwrites the cfg class with the dict
                    if not set_obj:
                        continue

            # -- 3) callable attribute → resolve string --------------
            elif callable(obj_mem):
                # update function name
                value = string_to_callable(value)

            # -- 4) simple scalar / explicit None ---------------------
            elif value is None or isinstance(value, type(obj_mem)):
                pass

            # -- 5) type mismatch → abort -----------------------------
            else:
                raise ValueError(
                    f"[Config]: Incorrect type under namespace: {key_ns}."
                    f" Expected: {type(obj_mem)}, Received: {type(value)}."
                )

            # -- 6) final assignment ---------------------------------
            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)

        # -- B) if key is not present ------------------------------------
        else:
            raise KeyError(f"[Config]: Key not found under namespace: {key_ns}.")


"""
Dictionary <-> Hashable operations.
"""


def dict_to_md5_hash(data: object) -> str:
    """Convert a dictionary into a hashable key using MD5 hash.

    Args:
        data: Input dictionary or configuration object to convert.

    Returns:
        A string object of double length containing only hexadecimal digits.
    """
    # convert to dictionary
    if isinstance(data, dict):
        encoded_buffer = json.dumps(data, sort_keys=True).encode()
    else:
        encoded_buffer = json.dumps(class_to_dict(data), sort_keys=True).encode()
    # compute hash using MD5
    data_hash = hashlib.md5()
    data_hash.update(encoded_buffer)
    # return the hash key
    return data_hash.hexdigest()

def update_dict(orig_dict: dict, new_dict: collections.abc.Mapping) -> dict:
    """Updates existing dictionary with values from a new dictionary.

    This function mimics the dict.update() function. However, it works for
    nested dictionaries as well.

    Args:
        orig_dict: The original dictionary to insert items to.
        new_dict: The new dictionary to insert items from.

    Returns:
        The updated dictionary.
    """
    for keyname, value in new_dict.items():
        if isinstance(value, collections.abc.Mapping):
            orig_dict[keyname] = update_dict(orig_dict.get(keyname, {}), value)
        else:
            orig_dict[keyname] = value
    return orig_dict


def replace_slices_with_strings(data: dict) -> dict:
    """Replace slice objects with their string representations in a dictionary.

    Args:
        data: The dictionary to process.

    Returns:
        The dictionary with slice objects replaced by their string representations.
    """
    if isinstance(data, dict):
        return {k: replace_slices_with_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_slices_with_strings(v) for v in data]
    elif isinstance(data, slice):
        return f"slice({data.start},{data.stop},{data.step})"
    else:
        return data


def replace_strings_with_slices(data: dict) -> dict:
    """Replace string representations of slices with slice objects in a dictionary.

    Args:
        data: The dictionary to process.

    Returns:
        The dictionary with string representations of slices replaced by slice objects.
    """
    if isinstance(data, dict):
        return {k: replace_strings_with_slices(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_strings_with_slices(v) for v in data]
    elif isinstance(data, str) and data.startswith("slice("):
        return string_to_slice(data)
    else:
        return data


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionary."""
    if isinstance(val, dict):
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        # deal with functions in print statements
        if callable(val):
            print(callable_to_string(val))
        else:
            print(val)
