import sys

import setuptools
import versioneer

with open("README.md") as fh:
    long_description = fh.read()


packages = setuptools.find_namespace_packages(include=["project*", "hydra_plugins*"])
print("PACKAGES FOUND:", packages)
print(sys.version_info)

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines() if not line.strip().startswith("#")]

extras_require = {
    "test": ["pytest", "pytest-xdist", "pytest-timeout", "pytest-regressions"],
    "dev": ["pre-commit", "black"],
}
extras_require["all"] = sorted(set(sum(extras_require.values(), [])))

setuptools.setup(
    name="project",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Fabrice Normandin",  # TODO: Replace
    author_email="fabrice.normandin@gmail.com",  # TODO: Replace
    description=("A Research Template repository for PyTorch projects."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mila-iqia/ResearchTemplate",
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require=extras_require,
)
