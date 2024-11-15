# Automated Testing

Tests are a vital part of any good codebase, especially in Machine Learning.
They make it easier to explore and try out new ideas, by giving you the security that your codebase
works as intended.

This template comes with some [easy-to-use test suites](#test-suites) as well as some pre-configured
[GitHub Actions workflows](#continuous-integration) to run them:

- [Unit tests](#unit-tests): quick to run and check small functions / modules / classes.
- [Regression tests](#regression-tests): check that your code is reproducible and to let
    you know if something changed while you were developing your code.
- [integration tests](#integration-tests): run your code end-to-end to make sure that all the
    individually-tested components work together as expected.
- [GitHub Actions](#continuous-integration) runs all these tests before you merge your code.


<!--
## TODOs:

- [ ] Described what is tested by the included automated tests (a bit like what is done [here](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template?tab=readme-ov-file#tests))
- [ ] Add some examples of how to run tests
- [ ] describe why the test files are next to the source files, and why TDD is good, and why ML researchers should care more about tests.
- [ ] Explain how the fixtures in `conftest.py` work (indirect parametrization of the command-line overrides, etc).
- [ ] Describe the Github Actions workflows that come with the template, and how to setup a self-hosted runner for template forks.
- [ ] Add links to relevant documentation -->

## Automated testing on SLURM clusters with GitHub CI

> 🔥 NOTE: This is a feature that is entirely unique to this template! 🔥

This template runs all the above-mentioned tests **on an actual Compute Node of the Mila cluster** automatically.
Assuming that you have access to the Mila / DRAC or other Slurm clusters, all you need to do is to
[setup a local self-hosted GitHub runner](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners)
for your fork of this repository, launch it on your local machine with access to a Slurm cluster,
and voila: Your code will now be tested on an ACTUAL slurm cluster whenever you push or update a PR
in your project GitHub repository.

> Detailed instructions on how to set this up in your project will be added soon.


## Test-suites

Unit testing in this template is done with [pytest](https://docs.pytest.org/en/stable/index.html).

To run tests, simply use `pytest` on the command-line. You may want to add some useful flags like
`pytest -x -v`. See [the pytest docs](https://docs.pytest.org/en/stable/contents.html) for more info.

The built-in tests cover the following:

- For each datamodule config, for each data split
    - test that the first batch is always the same
- For each algorithm config, for all compatible network / datamodule config combinations:
    - initialization is deterministic & reproducibile;
    - forward pass is deterministic & reproducibile;
    - backward pass is deterministic & reproducibile;

Take a look at [project.algorithms.testsuites.lightning_module_tests][] to see the included base tests for algorithms.

If you use [Visual Studio Code](https://code.visualstudio.com/), you may want to look into adding
the "test explorer" tab to your editor. Then, you'll be able to see and debug the tests using the GUI.

## Unit tests

```console
pytest -x -v
```

## Regression Tests

We use pytest-regressions to test that code changes don't break things.

- `--gen-missing`: Use this flag when you might be missing some of the regression files (for example on the first test run).
- `--regen-all`: Use this when you want to intentionally re-create the regression files. This should hopefully not be used often!

### First run

On the first run, you might want to run test with the `--gen-missing` files, like so:

```console
pytest --regen-all
```


## integration-tests

To run slower integration tests, use the following:

```console
pytest -x -v --slow
```

## Continuous Integration

<!--
::: project.algorithms.testsuites.lightning_module_tests
    options:
        show_bases: false
        show_source: true
        parameter_headings: true -->
