import importlib
import inspect
import textwrap
from logging import getLogger as get_logger
from pathlib import Path

from mkdocs_macros.plugin import MacrosPlugin

logger = get_logger(__name__)


def define_env(env: MacrosPlugin):
    @env.macro
    def inline(module_or_file: str, indent: int = 0):
        logger.info(f"Inlining reference: {module_or_file}")
        # TODO: need to support adding the indent otherwise we can't use this inside a collapsible block.
        file = Path(env.project_dir) / module_or_file
        if file.exists():
            return textwrap.indent(file.read_text(), " " * indent)

        obj = get_object_from_reference(module_or_file)
        source = "".join(inspect.getsourcelines(obj)[0])
        return textwrap.indent(source, " " * indent)


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
