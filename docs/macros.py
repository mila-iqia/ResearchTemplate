from __future__ import annotations

import importlib
import inspect
import logging
import textwrap
import typing
from pathlib import Path
from typing import Any

if typing.TYPE_CHECKING:
    from mkdocs_macros.plugin import MacrosPlugin

logger = logging.getLogger(__name__)


def define_env(env: MacrosPlugin):
    @env.macro
    def inline(module_or_file: str, indent: int = 0):
        block_type: str | None = None
        # print(f"Inlining reference: {module_or_file}")
        logger.info(f"Inlining reference: {module_or_file}")
        # TODO: need to support adding the indent otherwise we can't use this inside a collapsible block.
        file = Path(env.project_dir) / module_or_file
        if file.exists():
            if not block_type:
                if file.suffix in [".yaml", ".yml"]:
                    block_type = "yaml"
                elif file.suffix == ".py":
                    block_type = "python3"
                elif file.suffix == ".sh":
                    block_type = "bash"
                else:
                    block_type = ""
            content = file.read_text()
        else:
            block_type = block_type or "python3"
            obj: Any = get_object_from_reference(module_or_file)
            content = "".join(inspect.getsourcelines(obj)[0])

        content = f"```{block_type}\n" + textwrap.indent(content + "\n```", " " * indent)
        return content


def get_object_from_reference(reference: str):
    """taken from https://github.com/mkdocs/mkdocs/issues/692"""
    split = reference.split(".")
    right = []
    module = None
    while split:
        try:
            module = importlib.import_module(".".join(split))
            break
        except ModuleNotFoundError:
            right.append(split.pop())
    if module:
        for entry in reversed(right):
            module = getattr(module, entry)
    return module
