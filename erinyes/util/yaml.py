from typing import Dict, Tuple, Union

yamlDict = Dict[str, Union[str, bool, float, int, Dict]]  # types: ignore


def split_of(
    dict: yamlDict, key: str
) -> Tuple[Union[str, bool, float, int, yamlDict], yamlDict]:
    """retrieves a value from a yamlDict and returns the rest"""

    if key not in dict:
        raise ValueError(f"could not find {key} in {dict}")

    value = dict[key]
    del dict[key]
    return value, dict
