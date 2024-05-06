import contextlib
import functools
import os
import re
import warnings
from collections.abc import Mapping
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _pytest.outcomes import Failed
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from torch import Tensor

from project.utils.utils import flatten_dict, get_shape_ish

logger = get_logger(__name__)


@functools.singledispatch
def to_ndarray(v: Any) -> np.ndarray | None:
    return np.asarray(v)


@to_ndarray.register(type(None))
def _none_to_ndarray(v: None) -> None:
    return None


@to_ndarray.register(list)
def _list_to_ndarray(v: list) -> np.ndarray:
    if all(isinstance(v_i, list) for v_i in v):
        lengths = [len(v_i) for v_i in v]
        if len(set(lengths)) != 1:
            # List of lists of something, (e.g. a nested tensor-like list of dicts for instance).
            if all(isinstance(v_i_j, dict) and not v_i_j for v_i in v for v_i_j in v_i):
                # all empty dicts!
                return np.asarray([f"list of {len_i} empty dicts" for len_i in lengths])
            raise NotImplementedError(v)
    return np.asarray(v)


@to_ndarray.register(Tensor)
def _tensor_to_ndarray(v: Tensor) -> np.ndarray:
    if v.is_nested:
        v = v.to_padded_tensor(padding=0.0)
    return v.detach().cpu().numpy()


@functools.singledispatch
def _hash(v: Any) -> int:
    return hash(v)


@_hash.register(Tensor)
def tensor_hash(tensor: Tensor) -> int:
    return hash(tuple(tensor.flatten().tolist()))


@_hash.register(np.ndarray)
def ndarray_hash(array: np.ndarray) -> int:
    return hash(tuple(array.flat))


class TensorRegressionFixture:
    """Save some statistics (and a hash) of tensors in a file that is saved with git, but save the
    entire tensors in gitignored files.

    This way, the first time the tests run, they re-generate the full regression files, and check
    that their contents' hash matches what is stored with git!

    TODO: Add a `--regen-missing` option (currently implicitly always true) that decides if we
    raise an error if a file is missing. (for example in unit tests we don't want this to be true!)
    """

    def __init__(
        self,
        datadir: Path,
        original_datadir: Path,
        request: pytest.FixtureRequest,
        ndarrays_regression: NDArraysRegressionFixture,
        data_regression: DataRegressionFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.request = request
        self.datadir = datadir
        self.original_datadir = original_datadir

        self.ndarrays_regression = ndarrays_regression
        self.data_regression = data_regression
        self.monkeypatch = monkeypatch
        self.generate_missing_files: bool | None = self.request.config.getoption(
            "--gen-missing",
            default=None,  # type: ignore
        )

    def get_source_file(self, extension: str, additional_subfolder: str | None = None) -> Path:
        source_file, _test_file = get_test_source_and_temp_file_paths(
            extension=extension,
            request=self.request,
            original_datadir=self.original_datadir,
            datadir=self.datadir,
            additional_subfolder=additional_subfolder,
        )
        return source_file

    # Would be nice if this were a singledispatch method or something similar.

    def check(
        self,
        data_dict: Mapping[str, Any],
        tolerances: dict[str, dict[str, float]] | None = None,
        default_tolerance: dict[str, float] | None = None,
    ) -> None:
        # IDEA:
        # - Get the hashes of each array, and actually run the regression check first with those files.
        # - Then, if that check passes, run the actual check with the full files.
        # NOTE: If the array hash files exist, but the full files don't, then we should just
        # re-create the full files instead of failing.
        # __tracebackhide__ = True

        data_dict = flatten_dict(data_dict)

        if not isinstance(data_dict, dict):
            raise TypeError(
                "Only dictionaries with Tensors, NumPy arrays or array-like objects are "
                "supported on ndarray_regression fixture.\n"
                f"Object with type '{str(type(data_dict))}' was given."
            )

        # File some simple attributes of the full arrays/tensors. This one is saved with git.
        simple_attributes_source_file = self.get_source_file(extension=".yaml")

        # File with the full arrays/tensors. This one is ignored by git.
        arrays_source_file = self.get_source_file(extension=".npz")

        regen_all = self.request.config.getoption("regen_all")
        assert isinstance(regen_all, bool)

        if regen_all:
            assert self.generate_missing_files in [
                True,
                None,
            ], "--gen-missing contradicts --regen-all!"
            # Regenerate everything.
            if arrays_source_file.exists():
                arrays_source_file.unlink()
            if simple_attributes_source_file.exists():
                simple_attributes_source_file.unlink()

        if arrays_source_file.exists():
            logger.info(f"Full arrays file found at {arrays_source_file}.")
            if not simple_attributes_source_file.exists():
                # Weird: the simple attributes file doesn't exist. Re-create it if allowed.
                with dont_fail_if_files_are_missing(enabled=bool(self.generate_missing_files)):
                    self.pre_check(
                        data_dict,
                        simple_attributes_source_file=simple_attributes_source_file,
                    )

            # We already generated the file with the full tensors (and we also already checked
            # that their hashes correspond to what we expect.)
            # 1. Check that they match the data_dict.
            logger.info("Checking the full arrays.")
            self.regular_check(
                data_dict=data_dict,
                fullpath=arrays_source_file,
                tolerances=tolerances,
                default_tolerance=default_tolerance,
            )
            # the simple attributes file should already have been generated and saved in git.
            assert simple_attributes_source_file.exists()
            # NOTE: No need to do this step here. Saves us a super super tiny amount of time.
            # logger.debug("Checking that the hashes of the full arrays still match.")
            # self.pre_check(
            #     data_dict,
            #     simple_attributes_source_file=simple_attributes_source_file,
            # )
            return

        if simple_attributes_source_file.exists():
            logger.debug(f"Simple attributes file found at {simple_attributes_source_file}.")
            logger.debug(f"Regenerating the full arrays at {arrays_source_file}")
            # Go straight to the full check.
            # TODO: Need to get the full error when the tensors change instead of just the check
            # for the hash, which should only be used when re-creating the full regression files.

            with dont_fail_if_files_are_missing():
                self.regular_check(
                    data_dict=data_dict,
                    fullpath=arrays_source_file,
                    tolerances=tolerances,
                    default_tolerance=default_tolerance,
                )
            logger.debug(
                "Checking if the newly-generated full tensor regression files match the expected "
                "attributes and hashes."
            )
            self.pre_check(
                data_dict,
                simple_attributes_source_file=simple_attributes_source_file,
            )
            return

        logger.warning(f"Creating the simple attributes file at {simple_attributes_source_file}.")

        with dont_fail_if_files_are_missing(enabled=bool(self.generate_missing_files)):
            self.pre_check(
                data_dict,
                simple_attributes_source_file=simple_attributes_source_file,
            )

        with dont_fail_if_files_are_missing(enabled=bool(self.generate_missing_files)):
            self.regular_check(
                data_dict=data_dict,
                fullpath=arrays_source_file,
                tolerances=tolerances,
                default_tolerance=default_tolerance,
            )

        test_dir = self.original_datadir
        assert test_dir.exists()
        gitignore_file = test_dir / ".gitignore"
        if not gitignore_file.exists():
            logger.info(f"Making a new .gitignore file at {gitignore_file}")
            gitignore_file.write_text(
                "\n".join(
                    [
                        "# Ignore full tensor files, but not the files with tensor attributes and hashes.",
                        "*.npz",
                    ]
                )
                + "\n"
            )

    def pre_check(self, data_dict: dict[str, Any], simple_attributes_source_file: Path) -> None:
        version_controlled_simple_attributes = get_version_controlled_attributes(data_dict)
        # Run the regression check with the hashes (and don't fail if they don't exist)
        __tracebackhide__ = True
        # TODO: Figure out how to include/use the names of the GPUs:
        # - Should it be part of the hash? Or should there be a subfolder for each GPU type?
        _gpu_names = get_gpu_names(data_dict)
        if len(set(_gpu_names)) == 1:
            gpu_name = _gpu_names[0]
            if any(isinstance(t, Tensor) and t.device.type == "cuda" for t in data_dict.values()):
                version_controlled_simple_attributes["GPU"] = gpu_name

        self.data_regression.check(
            version_controlled_simple_attributes, fullpath=simple_attributes_source_file
        )

    def regular_check(
        self,
        data_dict: dict[str, Any],
        basename: str | None = None,
        fullpath: os.PathLike[str] | None = None,
        tolerances: dict[str, dict[str, float]] | None = None,
        default_tolerance: dict[str, float] | None = None,
    ) -> None:
        array_dict: dict[str, np.ndarray] = {}
        for key, array in data_dict.items():
            if isinstance(key, (int | bool | float)):
                new_key = f"{key}"
                assert new_key not in data_dict
                key = new_key
            assert isinstance(
                key, str
            ), f"The dictionary keys must be strings. Found key with type '{str(type(key))}'"

            ndarray_value = to_ndarray(array)
            if ndarray_value is None:
                logger.debug(
                    f"Got a value of `None` for key {key} not including it in the saved dict."
                )
            else:
                array_dict[key] = ndarray_value
        self.ndarrays_regression.check(
            array_dict,
            basename=basename,
            fullpath=fullpath,
            tolerances=tolerances,
            default_tolerance=default_tolerance,
        )
        return


def get_test_source_and_temp_file_paths(
    extension: str,
    request: pytest.FixtureRequest,
    original_datadir: Path,
    datadir: Path,
    additional_subfolder: str | None = None,
) -> tuple[Path, Path]:
    """Returns the path to the (maybe version controlled) source file and the path to the temporary
    file where test results might be generated during a regression test.

    NOTE: This is different than in pytest-regressions. Here we use a subfolder with the same name
    as the test function.
    """
    basename = re.sub(r"[\W]", "_", request.node.name)
    overrides_name = basename.removeprefix(request.node.function.__name__).lstrip("_")

    if extension.startswith(".") and overrides_name:
        # Remove trailing _'s if the extension starts with a dot.
        overrides_name = overrides_name.rstrip("_")

    if overrides_name:
        # There are overrides, so use a subdirectory.
        relative_path = Path(request.node.function.__name__) / overrides_name
    else:
        # There are no overrides, so use the regular base name.
        relative_path = Path(basename)

    relative_path = relative_path.with_suffix(extension)
    if additional_subfolder:
        relative_path = relative_path.parent / additional_subfolder / relative_path.name

    source_file = original_datadir / relative_path
    test_file = datadir / relative_path
    return source_file, test_file


@functools.singledispatch
def get_simple_attributes(value: Any):
    raise NotImplementedError(
        f"get_simple_attributes doesn't have a registered handler for values of type {type(value)}"
    )


@get_simple_attributes.register(type(None))
def _get_none_attributes(value: None):
    return {"type": "None"}


@get_simple_attributes.register(list)
def list_simple_attributes(some_list: list[Any]):
    return {
        "length": len(some_list),
        "item_types": sorted(set(type(item).__name__ for item in some_list)),
    }


@get_simple_attributes.register(dict)
def dict_simple_attributes(some_dict: dict[str, Any]):
    return {k: get_simple_attributes(v) for k, v in some_dict.items()}


@get_simple_attributes.register(np.ndarray)
def ndarray_simple_attributes(array: np.ndarray) -> dict:
    return {
        "shape": tuple(array.shape),
        "hash": _hash(array),
        "min": array.min().item(),
        "max": array.max().item(),
        "sum": array.sum().item(),
        "mean": array.mean(),
    }


@get_simple_attributes.register(Tensor)
def tensor_simple_attributes(tensor: Tensor) -> dict:
    if tensor.is_nested:
        # assert not [tensor_i.any() for tensor_i in tensor.unbind()], tensor
        # TODO: It might be a good idea to make a distinction here between '0' as the default, and
        # '0' as a value in the tensor? Hopefully this should be clear enough.
        tensor = tensor.to_padded_tensor(padding=0.0)

    return {
        "shape": tuple(tensor.shape) if not tensor.is_nested else get_shape_ish(tensor),
        "hash": _hash(tensor),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "sum": tensor.sum().item(),
        "mean": tensor.float().mean().item(),
        "device": (
            "cpu" if tensor.device.type == "cpu" else f"{tensor.device.type}:{tensor.device.index}"
        ),
    }


def get_gpu_names(data_dict: dict[str, Any]) -> list[str]:
    """Returns the names of the GPUS that tensors in this dict are on."""
    return sorted(
        set(
            torch.cuda.get_device_name(tensor.device)
            for tensor in data_dict.values()
            if isinstance(tensor, Tensor) and tensor.device.type == "cuda"
        )
    )


def get_version_controlled_attributes(data_dict: dict[str, Any]) -> dict[str, Any]:
    return {key: get_simple_attributes(value) for key, value in data_dict.items()}


class FilesDidntExist(Failed):
    pass


@contextlib.contextmanager
def dont_fail_if_files_are_missing(enabled: bool = True):
    try:
        with _catch_fails_with_files_didnt_exist():
            yield
    except FilesDidntExist as exc:
        if enabled:
            logger.warning(exc)
            warnings.warn(RuntimeWarning(exc.msg))
        else:
            raise


@contextlib.contextmanager
def _catch_fails_with_files_didnt_exist():
    try:
        yield
    except Failed as failure_exc:
        if failure_exc.msg and "File not found in data directory, created" in failure_exc.msg:
            raise FilesDidntExist(failure_exc.msg, pytrace=failure_exc.pytrace) from failure_exc
        else:
            raise
