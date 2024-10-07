import pytest
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files
from mkdocs.structure.pages import Page

from .autoref_plugin import CustomAutoRefPlugin


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (_header := "## Some header with a ref `lightning.Trainer`", _header),
        (
            "a backtick ref: `lightning.Trainer`",
            "a backtick ref: [lightning.Trainer][lightning.pytorch.trainer.trainer.Trainer]",
        ),
        ("`torch.Tensor`", "[torch.Tensor][torch.Tensor]"),
        (
            "a proper full ref: [lightning.Trainer][lightning.pytorch.core.trainer.trainer.Trainer]",
            "a proper full ref: [lightning.Trainer][lightning.pytorch.core.trainer.trainer.Trainer]",
        ),
        ("`foo.bar`", "`foo.bar`"),
        (
            "`jax.Array`",
            "[jax.Array][jax.Array]",
        ),  # not sure if this will make a proper link in mkdocs though.
    ],
)
def test_autoref_plugin(input: str, expected: str):
    config = MkDocsConfig("mkdocs.yaml")
    plugin = CustomAutoRefPlugin()
    result = plugin.on_page_markdown(
        input,
        page=Page(
            title="Test",
            file=File(
                "test.md",
                src_dir="bob",
                dest_dir="bobo",
                use_directory_urls=False,
            ),
            config=config,
        ),
        config=config,
        files=Files([]),
    )
    assert result == expected
