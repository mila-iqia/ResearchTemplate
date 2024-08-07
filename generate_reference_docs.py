#!/usr/bin/env python
# based on https://github.com/mkdocstrings/mkdocstrings/blob/5802b1ef5ad9bf6077974f777bd55f32ce2bc219/docs/gen_doc_stubs.py#L25


import textwrap
from logging import getLogger as get_logger
from pathlib import Path

import mkdocs_gen_files

from project.utils.env_vars import REPO_ROOTDIR

logger = get_logger(__name__)


def main():
    add_doc_for_module(REPO_ROOTDIR / "project")


def add_doc_for_module(module_path: Path) -> None:
    """Creates a markdown file in the "reference" section for this module and its submodules
    recursively.

    ## TODOs:
    - [ ] We don't currently see the docs from the docstrings of __init__.py files.
    - [ ] Might be nice to show the config files also?
    """

    assert module_path.is_dir()  # and (module_path / "__init__.py").exists(), module_path

    # module_import_path = _get_import_path(module_path)
    # doc_file = module_path.relative_to(REPO_ROOTDIR).with_suffix(".md")
    # write_doc_file = "reference" / doc_file
    # with mkdocs_gen_files.editor.FilesEditor.current().open(str(write_doc_file), "w") as f:
    #     print(
    #         textwrap.dedent(f"""\
    #         ::: {module_import_path}

    #         """),
    #         file=f,
    #     )

    def is_module(p: Path) -> bool:
        return (
            p.suffix == ".py" and not p.name.startswith("__") and not p.name.endswith("_test.py")
        )

    children = list(p for p in module_path.glob("*.py") if is_module(p))
    for child_module_path in children:
        child_module_import_path = _get_import_path(child_module_path)
        doc_file = child_module_path.relative_to(REPO_ROOTDIR).with_suffix(".md")
        write_doc_file = "reference" / doc_file

        with mkdocs_gen_files.editor.FilesEditor.current().open(str(write_doc_file), "w") as f:
            print(
                textwrap.dedent(f"""\
                ::: {child_module_import_path}
                """),
                file=f,
            )

    submodules = list(
        p
        for p in module_path.iterdir()
        if p.is_dir()
        and ((p / "__init__.py").exists() or len(list(p.glob("*.py"))) > 0)
        and not p.name.endswith("_test")
        and not p.name.startswith((".", "__"))
    )
    for submodule in submodules:
        logger.info(f"Creating doc for {submodule}")
        add_doc_for_module(submodule)


def _get_import_path(module_path: Path) -> str:
    """Returns the path to use to import a given (internal) module."""
    return ".".join(module_path.relative_to(REPO_ROOTDIR).with_suffix("").parts)


if __name__ in ["__main__", "<run_path>"]:
    # Run when executed directly or by mkdocs. Seems like the __name__ is <run_path> during `mkdocs serve`
    main()
