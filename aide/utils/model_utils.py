#!/usr/bin/env python3
"""
Utility functions for handling model names and IDs.
"""

import os
import re
from pathlib import Path
from typing import Optional, Union


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize model name for use in filenames and paths.
    Replaces slashes and other special characters with safe alternatives.

    Args:
        model_name: Original model name/ID (e.g. "RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic")

    Returns:
        Sanitized model name safe for use in file paths (e.g. "RedHatAI_DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic")
    """
    # Replace slashes with underscores
    sanitized = model_name.replace("/", "_")

    # Replace other potentially problematic characters
    sanitized = re.sub(r'[\\:*?"<>|]', "_", sanitized)

    return sanitized


def get_model_directory(
    model_name: str, base_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Get a directory path for model-related files that's safe regardless of model name format.

    Args:
        model_name: Original model name/ID (may contain slashes)
        base_dir: Base directory to use (default: current working directory)

    Returns:
        Path object for model directory
    """
    safe_name = sanitize_model_name(model_name)

    if base_dir is None:
        base_dir = os.getcwd()

    base_path = Path(base_dir)
    model_dir = base_path / "models" / safe_name

    return model_dir


def parse_model_id(model_name: str) -> tuple:
    """
    Parse a model ID into organization and model name components.

    Args:
        model_name: Full model identifier (e.g. "RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic")

    Returns:
        Tuple of (organization, model_name) or (None, model_name) if no organization
    """
    if "/" in model_name:
        org, model = model_name.split("/", 1)
        return org, model
    else:
        return None, model_name


# Example usage
if __name__ == "__main__":
    model_id = "RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic"

    print(f"Original model ID: {model_id}")
    print(f"Sanitized name: {sanitize_model_name(model_id)}")
    print(f"Model directory: {get_model_directory(model_id)}")

    org, name = parse_model_id(model_id)
    print(f"Organization: {org}")
    print(f"Model name: {name}")
