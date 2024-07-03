#!/usr/bin/env python
# based on https://github.com/mkdocstrings/mkdocstrings/blob/5802b1ef5ad9bf6077974f777bd55f32ce2bc219/docs/gen_doc_stubs.py#L25


import textwrap
from pathlib import Path

import mkdocs_gen_files
import mkdocs_gen_files.nav

from project.utils.env_vars import REPO_ROOTDIR

nav = mkdocs_gen_files.nav.Nav()


package = "project"
module = "project/main.py"
submodules = ["project.datamodules", "project.utils", "project.networks", "project.algorithms"]


def _get_import_path(module_path: Path) -> str:
    """Returns the path to use to import a given (internal) module."""
    return ".".join(module_path.relative_to(REPO_ROOTDIR).with_suffix("").parts)


def add_doc_for_module(module_path: Path) -> None:
    a = "reference" / (module_path.relative_to(REPO_ROOTDIR).with_suffix(".md"))
    module_import_path = _get_import_path(module_path)

    with mkdocs_gen_files.open(a, "w") as f:
        print(
            textwrap.dedent(f"""\
            ::: {module_import_path}
            """),
            file=f,
        )
    docs_dir = REPO_ROOTDIR / "docs"
    module_path_relative_to_docs_dir = module_path.relative_to(docs_dir, walk_up=True)
    mkdocs_gen_files.set_edit_path(a, module_path_relative_to_docs_dir)


def get_modules(package: Path) -> list[Path]:
    return [
        p
        for p in package.glob("*.py")
        if not p.name.endswith("_test.py") and not p.name == "__init__.py"
    ]


def get_subpackages(package: Path) -> list[Path]:
    return [
        p
        for p in package.iterdir()
        if p.is_dir() and not p.name.startswith("__") and (p / "__init__.py").exists()
    ]


project_nav = mkdocs_gen_files.nav.Nav()
with mkdocs_gen_files.open("reference/project/main.md", "w") as f:
    print(
        textwrap.dedent("""\
        ::: project.main
        """),
        file=f,
    )
nav["project", "main"] = "project/main.md"
mkdocs_gen_files.set_edit_path("reference/project/main.md", "../project/main.py")

with mkdocs_gen_files.open("reference/project/experiment.md", "w") as f:
    print(
        textwrap.dedent("""\
        ::: project.experiment
        """),
        file=f,
    )
nav["project", "experiment"] = "reference/project/experiment.md"
mkdocs_gen_files.set_edit_path("reference/project/experiment.md", "../project/experiment.py")

project_utils_nav = mkdocs_gen_files.nav.Nav()
with mkdocs_gen_files.open("reference/project/utils/types.md", "w") as f:
    print(
        textwrap.dedent("""\
        ::: project.utils.types
            options:
                show_source: true
        """),
        file=f,
    )
nav["project", "utils", "types"] = "reference/project/utils/types.md"
mkdocs_gen_files.set_edit_path("reference/project/utils/types.md", "../project/utils/types.py")


with mkdocs_gen_files.open("reference.md", "w") as nav_file:
    # assert False, "\n".join(nav.build_literate_nav())
    nav_file.writelines(nav.build_literate_nav())

# with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as project_nav_file:
#     project_nav_file.writelines(project_nav.build_literate_nav())


# project_root = REPO_ROOTDIR / "project"
# for python_module_path in sorted(
#     f
#     # for f in project_root.glob("*.py")
#     for f in [(project_root / "project")]
#     if not f.name.endswith("_test.py") and not f.name == "__init__.py"
# ):
#     doc_path = python_module_path.relative_to(REPO_ROOTDIR).with_suffix(".md")

#     full_doc_path = Path("reference") / doc_path

#     nav[full_doc_path.with_suffix("").parts] = str(full_doc_path)

#     with mkdocs_gen_files.open(full_doc_path, "w") as f:
#         module_import_path = ".".join(
#             python_module_path.relative_to(REPO_ROOTDIR).with_suffix("").parts
#         )
#         print(f"::: {module_import_path}", file=f)

#     mkdocs_gen_files.set_edit_path(
#         full_doc_path, python_module_path.relative_to(REPO_ROOTDIR / "docs", walk_up=True)
#     )

# nav["mkdocs_autorefs", "references"] = "autorefs/references.md"
# nav["mkdocs_autorefs", "plugin"] = "autorefs/plugin.md"

# with mkdocs_gen_files.open("reference.md", "w") as nav_file:
#     nav_file.writelines(nav.build_literate_nav())
