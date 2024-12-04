from __future__ import annotations

import importlib
import inspect
import logging
import textwrap
import typing
from pathlib import Path
from typing import Any

import torch

if typing.TYPE_CHECKING:
    from mkdocs_macros.plugin import MacrosPlugin

import lightning

try:
    from mkdocs_autoref_plugin.autoref_plugin import default_reference_sources

    default_reference_sources.extend(
        [
            lightning.Trainer,
            lightning.LightningModule,
            lightning.LightningDataModule,
            torch.nn.Module,
        ]
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)


def define_env(env: MacrosPlugin):
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
            logger.info(f"inlining code for {obj}")
            content = inspect.getsource(obj)
            # BUG: Sometimes using {{ inline('some_module.SomeClass.some_method') }} will show the
            # incorrect source code: it will show the method *above* the one we're looking for.
            # content = "".join(inspect.getsourcelines(obj)[0])

        content = f"```{block_type}\n" + textwrap.indent(content + "\n```", " " * indent)
        return content

    env.macro(inline, name="inline")


def get_object_from_reference(reference: str):
    """taken from https://github.com/mkdocs/mkdocs/issues/692"""
    parts = reference.split(".")
    for i in range(1, len(parts)):
        module_name = ".".join(parts[:i])
        obj_path = parts[i:]
        try:
            obj = importlib.import_module(module_name)
            for part in obj_path:
                obj = getattr(obj, part)
            return obj
        except (ModuleNotFoundError, AttributeError):
            continue
    raise RuntimeError(f"Unable to import the {reference=}")
