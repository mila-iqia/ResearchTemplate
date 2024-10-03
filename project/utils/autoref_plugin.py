"""IDEA: Tweak the AutoRefsPlugin so that text in backticks like `this` (more IDE-friendly) are
considered refs when possible.
"""

# TODOs / plan:
# - [ ] Replace `this` with [this][] in the markdown file.
# - Also fix refs like `[lightning.Trainer][]` so that they become
#      [lightning.Trainer][lightning.pytorch.trainer.trainer.Trainer]

import re
from logging import getLogger as get_logger

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import BasePlugin  # noqa
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page
from mkdocs_autorefs.plugin import AutorefsPlugin  # noqa

logger = get_logger(__name__)


class CustomAutoRefPlugin(BasePlugin):
    def on_page_markdown(
        self, markdown: str, /, *, page: Page, config: MkDocsConfig, files: Files
    ) -> str | None:
        # Find all instances where backticks are used and try to convert them to refs.

        # Examples:
        # - `package.foo.bar` -> [package.foo.bar][]
        # - `baz` -> [baz][]
        _modified = re.sub(r"`([^`]+)`", r"[\1][]", markdown)

        return super().on_page_markdown(markdown, page=page, config=config, files=files)
