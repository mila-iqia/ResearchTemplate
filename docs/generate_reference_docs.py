#!/usr/bin/env python
# based on https://github.com/mkdocstrings/mkdocstrings/blob/5802b1ef5ad9bf6077974f777bd55f32ce2bc219/docs/gen_doc_stubs.py#L25


import textwrap
from pathlib import Path

import mkdocs_gen_files
import mkdocs_gen_files.nav

from project.utils.env_vars import REPO_ROOTDIR

module = "project"
modules = [
    "project/main.py",
    "project/experiment.py",
]
submodules = [
    "project.algorithms",
    "project.configs",
    "project.datamodules",
    "project.networks",
    "project.utils",
]


def _get_import_path(module_path: Path) -> str:
    """Returns the path to use to import a given (internal) module."""
    return ".".join(module_path.relative_to(REPO_ROOTDIR).with_suffix("").parts)


def main():
    nav = mkdocs_gen_files.nav.Nav()

    add_doc_for_module(REPO_ROOTDIR / "project", nav)

    # with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    #     # assert False, "\n".join(nav.build_literate_nav())
    #     nav_file.writelines(nav.build_literate_nav())


def add_doc_for_module(module_path: Path, nav: mkdocs_gen_files.nav.Nav) -> None:
    """TODO."""

    assert module_path.is_dir() and (module_path / "__init__.py").exists(), module_path

    children = list(
        p
        for p in module_path.glob("*.py")
        if not p.name.startswith("__") and not p.name.endswith("_test.py")
    )
    for child_module_path in children:
        child_module_import_path = _get_import_path(child_module_path)
        doc_file = child_module_path.relative_to(REPO_ROOTDIR).with_suffix(".md")
        write_doc_file = f"reference/{doc_file}"

        nav[tuple(child_module_import_path.split("."))] = f"{doc_file}"

        with mkdocs_gen_files.open(write_doc_file, "w") as f:
            print(
                textwrap.dedent(f"""\
                ::: {child_module_import_path}
                """),
                file=f,
            )
        docs_dir = REPO_ROOTDIR / "docs"
        module_path_relative_to_docs_dir = child_module_path.relative_to(docs_dir, walk_up=True)
        mkdocs_gen_files.set_edit_path(write_doc_file, str(module_path_relative_to_docs_dir))

    submodules = list(
        p
        for p in module_path.iterdir()
        if p.is_dir()
        and (p / "__init__.py").exists()
        and not p.name.endswith("_test")
        and not p.name.startswith((".", "__"))
    )
    for submodule in submodules:
        add_doc_for_module(submodule, nav)


if __name__ in ["__main__", "<run_path>"]:
    # Run when executed directly or by mkdocs. Seems like the __name__ is <run_path> during `mkdocs serve`
    main()
