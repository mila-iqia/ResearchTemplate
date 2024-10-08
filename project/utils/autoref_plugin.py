"""IDEA: Tweak the AutoRefsPlugin so that text in backticks like `this` (more IDE-friendly) are
considered refs when possible.
"""

import functools
import re

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

known_things = [
    lightning.Trainer,
    lightning.LightningModule,
    lightning.LightningDataModule,
    torch.nn.Module,
]
"""
IDEA: IF we see `Trainer`, and we know that that's the `lightning.Trainer`, then we
create the proper ref.

TODO: Ideally this would contain every object / class that we know of in this project.
"""


class CustomAutoRefPlugin(BasePlugin):
    """Small mkdocs plugin that converts backticks to refs when possible."""

    def on_page_markdown(
        self, markdown: str, /, *, page: Page, config: MkDocsConfig, files: Files
    ) -> str | None:
        # Find all instances where backticks are used and try to convert them to refs.

        # Examples:
        # - `package.foo.bar` -> [package.foo.bar][] (only if `package.foo.bar` is importable)
        # - `baz` -> [baz][]

        def _full_path(thing) -> str:
            return thing.__module__ + "." + thing.__qualname__

        known_thing_names = [t.__name__ for t in known_things]

        new_markdown = []
        for line_index, line in enumerate(markdown.splitlines(keepends=True)):
            # Can't convert `this` to `[this][]` in headers, otherwise they break.
            if line.lstrip().startswith("#"):
                new_markdown.append(line)
                continue

            matches = re.findall(r"`([^`]+)`", line)

            for match in matches:
                thing_name = match
                if "." not in thing_name and thing_name in known_thing_names:
                    thing = known_things[known_thing_names.index(thing_name)]
                else:
                    thing = _try_import_thing(thing_name)

                if thing is None:
                    logger.debug(f"Unable to import {thing_name}, leaving it as-is.")
                    continue

                new_ref = f"[{thing_name}][{_full_path(thing)}]"
                logger.info(
                    f"Replacing `{thing_name}` with {new_ref} in {page.file.abs_src_path}:{line_index}"
                )
                line = line.replace(f"`{thing_name}`", new_ref)

            new_markdown.append(line)

        return "".join(new_markdown)


@functools.cache
def _try_import_thing(thing: str):
    try:
        return import_object(thing)
    except Exception:
        return None
