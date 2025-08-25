#!/usr/bin/env python
"""Script used to generate the reference docs for the project from the source code.

Based on
https://github.com/mkdocstrings/mkdocstrings/blob/5802b1ef5ad9bf6077974f777bd55f32ce2bc219/docs/gen_doc_stubs.py#L25
"""

import textwrap
from logging import getLogger as get_logger
from pathlib import Path

logger = get_logger(__name__)


def main():
    """Generate the code reference pages and navigation."""

    import mkdocs_gen_files

    nav = mkdocs_gen_files.nav.Nav()

    root = Path(__file__).parent.parent
    src = root / "project"

    for path in sorted(src.rglob("*.py")):
        module_path = path.relative_to(root).with_suffix("")
        doc_path = path.relative_to(root).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        parts = tuple(module_path.parts)

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(
                textwrap.dedent(
                    # f"""\
                    # ---
                    # additional_python_references:
                    # - {ident}
                    # ---
                    # ::: {ident}
                    # """
                    f"""\
                    ::: {ident}
                    """
                )
            )
            # fd.write(f"::: {ident}\n")

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


if __name__ in ["__main__", "<run_path>"]:
    # Run when executed directly or by mkdocs. Seems like the __name__ is <run_path> during `mkdocs serve`
    main()
