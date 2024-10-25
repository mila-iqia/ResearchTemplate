"""IDEA: Tweak the AutoRefsPlugin so that text in backticks like `this` (more IDE-friendly) are
considered refs when possible.
"""

import functools
import inspect
import re
import types

import lightning
import torch
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import (
    BasePlugin,
    get_plugin_logger,
)
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page
from mkdocs_autorefs.plugin import AutorefsPlugin  # noqa

from project.utils.hydra_config_utils import import_object

# Same as in the mkdocs_autorefs plugin.
logger = get_plugin_logger(__name__)

default_reference_sources = [
    lightning.Trainer,
    lightning.LightningModule,
    lightning.LightningDataModule,
    torch.nn.Module,
]
"""These are some "known objects" that can be referenced with backticks anywhere in the docs.

Additionally, if there were modules in here, then any of their public members can also be
referenced.
"""
from mkdocstrings.plugin import MkdocstringsPlugin  # noqa
from mkdocstrings_handlers.python.handler import PythonHandler  # noqa


class CustomAutoRefPlugin(BasePlugin):
    """Small mkdocs plugin that converts backticks to refs when possible."""

    def __init__(self):
        super().__init__()
        self.default_reference_sources = sum(map(_expand, default_reference_sources), [])

    def on_page_markdown(
        self, markdown: str, /, *, page: Page, config: MkDocsConfig, files: Files
    ) -> str | None:
        # Find all instances where backticks are used and try to convert them to refs.

        # Examples:
        # - `package.foo.bar` -> [package.foo.bar][] (only if `package.foo.bar` is importable)
        # - `baz` -> [baz][]

        # TODO: The idea here is to also make all the members of a module referentiable with
        # backticks in the same module. The problem is that the "reference" page we create with
        # mkdocstrings only contains a simple `::: project.path.to.module` and doesn't have any
        # text, so we can't just replace the `backticks` with refs, since mkdocstrings hasn't yet
        # processed the module into a page with the reference docs. This seems to be happening
        # in a markdown extension (the `MkdocstringsExtension`).

        # file = page.file.abs_src_path
        # if file and "reference/project" in file:
        #     relative_path = file[file.index("reference/") :].removeprefix("reference/")
        #     module_path = relative_path.replace("/", ".").replace(".md", "")
        #     if module_path.endswith(".index"):
        #         module_path = module_path.removesuffix(".index")
        #     logger.error(
        #         f"file {relative_path} is the reference page for the python module {module_path}"
        #     )
        #     if "algorithms/example" in file:
        #         assert False, markdown
        #     additional_objects = _expand(module_path)
        if referenced_packages := page.meta.get("additional_python_references", []):
            logger.debug(f"Loading extra references: {referenced_packages}")
            additional_objects: list[object] = _get_referencable_objects_from_doc_page_header(
                referenced_packages
            )
        else:
            additional_objects = []

        if additional_objects:
            additional_objects = [
                obj
                for obj in additional_objects
                if (
                    inspect.isfunction(obj)
                    or inspect.isclass(obj)
                    or inspect.ismodule(obj)
                    or inspect.ismethod(obj)
                )
                # and (hasattr(obj, "__name__") or hasattr(obj, "__qualname__"))
            ]

        known_objects_for_this_module = self.default_reference_sources + additional_objects
        known_object_names = [t.__name__ for t in known_objects_for_this_module]

        new_markdown = []
        # TODO: This changes things inside code blocks, which is not desired!
        in_code_block = False

        for line_index, line in enumerate(markdown.splitlines(keepends=True)):
            # Can't convert `this` to `[this][]` in headers, otherwise they break.
            if line.lstrip().startswith("#"):
                new_markdown.append(line)
                continue
            if "```" in line:
                in_code_block = not in_code_block
            if in_code_block:
                new_markdown.append(line)
                continue

            matches = re.findall(r"`([^`]+)`", line)
            for match in matches:
                thing_name = match
                if any(char in thing_name for char in ["/", " ", "-"]):
                    continue
                if thing_name in known_object_names:
                    # References like `JaxTrainer` (which are in a module that we're aware of).
                    thing = known_objects_for_this_module[known_object_names.index(thing_name)]
                else:
                    thing = _try_import_thing(thing_name)

                if thing is None:
                    logger.debug(f"Unable to import {thing_name}, leaving it as-is.")
                    continue

                new_ref = f"[`{thing_name}`][{_full_path(thing)}]"
                logger.debug(
                    f"Replacing `{thing_name}` with {new_ref} in {page.file.abs_src_path}:{line_index}"
                )
                line = line.replace(f"`{thing_name}`", new_ref)

            new_markdown.append(line)

        return "".join(new_markdown)


def _expand(obj: types.ModuleType | object) -> list[object]:
    if not inspect.ismodule(obj):
        # The ref is something else (a class, function, etc.)
        return [obj]

    # The ref is a package, so we import everything from it.
    # equivalent of `from package import *`
    if hasattr(obj, "__all__"):
        return [getattr(obj, name) for name in obj.__all__]
    else:
        objects_in_global_scope = [v for k, v in vars(obj).items() if not k.startswith("_")]
        # Don't consider any external modules that were imported in the global scope.
        source_file = inspect.getsourcefile(obj)
        # too obtuse, but whatever
        return [
            v
            for v in objects_in_global_scope
            if not (inspect.ismodule(v) and inspect.getsourcefile(v) != source_file)
        ]


def _get_referencable_objects_from_doc_page_header(doc_page_references: list[str]):
    additional_objects: list[object] = []
    for package in doc_page_references:
        additional_ref_source = import_object(package)
        additional_objects.extend(_expand(additional_ref_source))
    return additional_objects


def _full_path(thing) -> str:
    if inspect.ismodule(thing):
        return thing.__name__
    return thing.__module__ + "." + getattr(thing, "__qualname__", thing.__name__)


@functools.cache
def _try_import_thing(thing: str):
    try:
        return import_object(thing)
    except Exception:
        return None
