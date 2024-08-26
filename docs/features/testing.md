# Automated Testing

Tests are a vital part of a good codebase, especially in Machine Learning.
They make it easier to explore and try out new ideas, by giving you the security that your codebase works as intended.

- [Unit tests](#unit-tests) are very quick to run and check small functions / modules / classes.
- [Regression tests](#regression-tests) are used to check that your code is reproducible and to let
    you know if something changed while developing your code.
- [integration tests](#integration-tests) run your code end-to-end to make sure that all the
    individually-tested components work together as expected.
- [Continuous integration](#continuous-integration) runs all these tests before you merge your code.

This template comes with some [easy-to-use test suites](#test-suites) as well as pre-configured
[GitHub Actions workflows](#continuous-integration).

## Automated testing on the Mila / DRAC clusters with GitHub CI

This template runs all the above-mentioned tests **on the Mila cluster** automatically.
If you have access to the Mila / DRAC Slurm clusters, all you need to do is to setup a self-hosted runner on your local machine.

Detailed instructions on how to set this up will be added soon.

## TODOs:

- [ ] Described what is tested by the included automated tests (a bit like what is done [here](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template?tab=readme-ov-file#tests))
- [ ] Add some examples of how to run tests
- [ ] describe why the test files are next to the source files, and why TDD is good, and why ML researchers should care more about tests.
- [ ] Explain how the fixtures in `conftest.py` work (indirect parametrization of the command-line overrides, etc).
- [ ] Describe the Github Actions workflows that come with the template, and how to setup a self-hosted runner for template forks.
- [ ] Add links to relevant documentation ()

https://github.com/gorodnitskiy/yet-another-lightning-hydra-template?tab=readme-ov-file#tests

## Unit tests

Unit testing in this template is done with [pytest](https://docs.pytest.org/en/stable/index.html).

## Regression Tests

## integration-tests

## Continuous Integration

## Test-suites

- [project.algorithms.testsuites.algorithm_tests](../reference/project/algorithms/testsuites/algorithm_tests.md)
