# Automated Testing

Tests are a vital part of a good codebase, especially in Machine Learning.
They make it easier to explore and try out new ideas, by giving you the security that your codebase works as intended.

- [Unit tests](#unit-tests) are very quick to run and check small functions / modules / classes.
- [Regression tests](#regression-tests) are used to check that your code is reproducible and to let you know if something changed while developing your code.

This template includes some [generic test suites](#test-suites) that are very easy to use with your
own algorithms.

TODOs:

- [ ] Described what is tested by the included automated tests (a bit like what is done [here](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template?tab=readme-ov-file#tests))
- [ ] Add some examples of how to run tests
- [ ] describe why the test files are next to the source files, and why TDD is good, and why ML researchers should care more about tests.
- [ ] Explain how the fixtures in `conftest.py` work (indirect parametrization of the command-line overrides, etc).
- [ ] Describe the Github Actions workflows that come with the template, and how to setup a self-hosted runner for template forks.
- [ ] Add links to relevant documentation ()

https://github.com/gorodnitskiy/yet-another-lightning-hydra-template?tab=readme-ov-file#tests

## Unit tests

Unit testing in this codebase is done with [pytest](https://docs.pytest.org/en/stable/index.html)

## Regression Tests

## Continuous Integration

## Test-suites

- \[project.algorithms.testsuites.algorithm_tests\]\[\]

<!-- ::: project.algorithms.testsuites.algorithm_tests -->

<!--
::: project.algorithms.example_test
    options:
        allow_inspection: true -->
