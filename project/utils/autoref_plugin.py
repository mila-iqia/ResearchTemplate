"""IDEA: Tweak the AutoRefsPlugin so that text in backticks like `this` (more IDE-friendly) are
considered refs when possible.
"""

# TODOs / plan:
# - [ ] Replace `this` with [this][] in the markdown file.
# - Also fix refs like `[lightning.Trainer][]` so that they become
#      [lightning.Trainer][lightning.pytorch.trainer.trainer.Trainer]

import functools

# from mkdocs_autorefs.references import *
import re

import lightning
import torch  # noqa
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import (
    BasePlugin,  # noqa
    get_plugin_logger,
)
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page
from mkdocs_autorefs.plugin import AutorefsPlugin  # noqa

from project.utils.hydra_config_utils import import_object

# Same as in the mkdocs_autorefs plugin.
logger = get_plugin_logger(__name__)


class CustomAutoRefPlugin(BasePlugin):
    def on_page_markdown(
        self, markdown: str, /, *, page: Page, config: MkDocsConfig, files: Files
    ) -> str | None:
        # Find all instances where backticks are used and try to convert them to refs.

        # Examples:
        # - `package.foo.bar` -> [package.foo.bar][]
        # - `baz` -> [baz][]
        from lightning import Trainer

        thing = Trainer

        def _full_path(thing) -> str:
            return thing.__module__ + "." + thing.__qualname__

        def best_display_str(thing) -> str:
            return thing.__module__.split(".")[0] + "." + thing.__qualname__

        def _ref_with_name(thing, name: str) -> str:
            full_path = thing.__module__ + "." + thing.__qualname__
            return f"[{name}][{full_path}]"

        known_things = [
            lightning.Trainer,
            lightning.LightningModule,
            lightning.LightningDataModule,
            torch.nn.Module,
        ]
        known_thing_names = [t.__name__ for t in known_things]
        use_translations = True

        # for k, v in translations.items():
        #     markdown = markdown.replace(k, v)
        # messes up the refs!

        new_markdown = []
        for line_index, line in enumerate(markdown.splitlines(keepends=True)):
            # Can't convert `this` to `[this][]` in headers, otherwise they break.
            if line.lstrip().startswith("#"):
                new_markdown.append(line)
                continue

            matches = re.findall(r"`([^`]+)`", line)

            for match in matches:
                thing_name = match
                if not use_translations:
                    # Do something dumber
                    line = line.replace(f"`{thing_name}`", f"[{thing_name}][]")
                    continue
                # line = line.replace(f"`{thing_name}`",)
                if "." not in thing_name and thing_name in known_thing_names:
                    thing = known_things[known_thing_names.index(thing_name)]
                else:
                    thing = _try_import_thing(thing_name)

                if thing is None:
                    logger.debug(f"Unable to import {thing_name}")
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
