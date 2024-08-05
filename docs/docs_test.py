import pathlib

import pytest
from mktestdocs import check_md_file

# This retrieves all methods/properties that have a docstring.
# todo: Brittle. We'd like something like griffe, that gets all functions / classes / etc in our module.
# members = get_codeblock_members(*[v for k, v in vars(project).items() if k != "__all__"])


def get_pretty_id(obj):
    if hasattr(obj, "__qualname__"):
        return obj.__qualname__
    if hasattr(obj, "__name__"):
        return obj.__name__
    return str(obj)


# todo: do we want to run the tests here? or do we just test the doc pages?
# @pytest.mark.parametrize(
#     "obj",
#     list(itertools.chain(map(getmembers, [project, project.configs, project.algorithms]))),
#     ids=get_pretty_id,
# )
# def test_member(obj):
#     check_docstring(obj)


docs_folder = pathlib.Path(__file__).parent


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize("fpath", docs_folder.rglob("*.md"), ids=str)
def test_documentation_file(fpath):
    check_md_file(fpath=fpath)
