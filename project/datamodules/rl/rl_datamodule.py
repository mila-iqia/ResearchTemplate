from __future__ import annotations

import functools
import warnings
from logging import getLogger as get_logger
from typing import Any, Callable, Generic, Iterable, Literal

import gym
import torch
from gym.utils.colorize import colorize
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from project.datamodules.datamodule import DataModule
from project.datamodules.rl.stacking_utils import stack_dicts
from project.utils.types import StageStr

from .gym_utils import ToTensorsWrapper
from .rl_dataset import RlDataset
from .rl_types import (
    Actor,
    ActorOutput,
    Episode,
    EpisodeBatch,
)

logger = get_logger(__name__)


class RlDataModule(
    LightningDataModule, DataModule[EpisodeBatch[ActorOutput]], Generic[ActorOutput]
):
    """A LightningDataModule for RL environments whose DataLoaders yield `EpisodeBatch` objects.

    This DataModule needs to be passed an actor using either the `actor` argument of the
    constructor or the `set_actor` method, otherwise a random policy is used by default.

    TODO: We might want to actually give the algorithm the opportunity to add wrappers on top
    of the environment. How do we go about doing that in a clean way?
    - @lebrice: I added the `[train/valid/test]_wrappers` arguments and properties. This should be
    enough for now I think.
    """

    def __init__(
        self,
        env: str | Callable[[], gym.Env],
        actor: Actor[Tensor, Tensor, ActorOutput] | None = None,
        episodes_per_epoch: int = 100,
        batch_size: int = 1,
        train_wrappers: list[Callable[[gym.Env], gym.Env]] | None = None,
        valid_wrappers: list[Callable[[gym.Env], gym.Env]] | None = None,
        test_wrappers: list[Callable[[gym.Env], gym.Env]] | None = None,
        train_seed: int | None = 123,
        val_seed: int = 42,
    ):
        """
        Parameters
        ----------

        env: Name of the Gym environment to use, or function that creates a Gym environment.
        actor: The actor that will be used to collect episodes in the environment.
        episodes_per_epoch: Number of episodes per training epoch.
        batch_size : Number of episodes in a batch.
        device: Device to put the tensors on.
        """
        super().__init__()
        self.env_fn = (
            functools.partial(gym.make, env, render_mode="rgb_array")
            if isinstance(env, str)
            else env
        )
        self.env: gym.Env = self.env_fn()
        self.episodes_per_epoch = episodes_per_epoch
        self.batch_size = batch_size

        self.actor: Actor[Tensor, Tensor, ActorOutput] | None = actor

        self.train_env: gym.Env | None = None
        self.valid_env: gym.Env | None = None
        self.test_env: gym.Env | None = None

        self.train_dataset: RlDataset[ActorOutput] | None = None
        self.valid_dataset: RlDataset[ActorOutput] | None = None
        self.test_dataset: RlDataset[ActorOutput] | None = None

        self.train_wrappers = train_wrappers or []
        self.valid_wrappers = valid_wrappers or []
        self.test_wrappers = test_wrappers or []

        self.train_actor = actor
        self.valid_actor = actor
        self.test_actor = actor

        self.train_seed = train_seed
        self.valid_seed = val_seed
        self.test_seed = 111

        self._device: torch.device | None = None

    def set_actor(self, actor: Actor[Tensor, Tensor, ActorOutput]) -> None:
        if self.train_actor is None:
            assert self.test_actor is None
            assert self.valid_actor is None

            self.train_actor = actor
            # Save some computation by disabling gradient computation. Actor can still turn it on
            # internally necessary with a context manager or decorator.
            self.valid_actor = torch.no_grad()(torch.inference_mode()(actor))
            self.test_actor = torch.no_grad()(torch.inference_mode()(actor))
            return

        assert self.test_actor is not None
        assert self.valid_actor is not None

        # TODO: Do we allow changing the actor while the dataset is still "live"?
        # Should we end on-going episode(s)?
        if self.train_dataset is not None:
            self.train_dataset.actor = self.train_actor
        if self.valid_dataset is not None:
            self.valid_dataset.actor = self.valid_actor
        if self.test_dataset is not None:
            self.test_dataset.actor = self.test_actor

    def prepare_data(self) -> None:
        # NOTE: We don't use this hook here.
        ...

    def setup(self, stage: StageStr) -> None:
        """Sets up the environment(s), applying wrappers and such.

        Called at the beginning of each stage (fit, validate, test).
        """

    def _make_env(
        self, wrappers: list[Callable[[gym.Env], gym.Env]]
    ) -> ToTensorsWrapper:
        # TODO: Use gym.vector.make for vector envs, and pass the single-env wrappers to
        # gym.vector.make.
        env = self.env_fn()
        for wrapper in wrappers:
            env = wrapper(env)
        # TODO: Should this wrapper always be mandatory? And should it always be placed at the end?
        logger.debug(f"The episodes will be placed on device {self.device}.")
        env = ToTensorsWrapper(env, device=self.device)
        return env

    def train_dataloader(self) -> Iterable[EpisodeBatch[ActorOutput]]:
        if self.train_actor is None:
            # warn("No actor was set, using a random policy.", color="red")
            # self.train_actor = random_actor
            raise _error_actor_required(self, "train")

        logger.debug(
            f"Creating training environment with wrappers {self.train_wrappers}"
        )
        self.train_env = self.train_env or self._make_env(wrappers=self.train_wrappers)
        self.train_dataset = RlDataset(
            self.train_env,
            actor=self.train_actor,
            episodes_per_epoch=self.episodes_per_epoch,
            seed=self.train_seed,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self) -> Iterable[EpisodeBatch[ActorOutput]]:
        if self.valid_actor is None:
            raise _error_actor_required(self, "valid")

        self.valid_env = self.valid_env or self._make_env(wrappers=self.valid_wrappers)
        self.valid_dataset = RlDataset(
            self.valid_env,
            actor=self.valid_actor,
            episodes_per_epoch=self.episodes_per_epoch,
            seed=self.valid_seed,
        )
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self) -> Iterable[EpisodeBatch[ActorOutput]]:
        if self.test_actor is None:
            raise _error_actor_required(self, "test")

        self.test_env = self.test_env or self._make_env(wrappers=self.test_wrappers)
        self.test_dataset = RlDataset(
            self.test_env,
            actor=self.test_actor,
            episodes_per_epoch=self.episodes_per_epoch,
            seed=self.test_seed,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )

    @property
    def device(self) -> torch.device:
        """Figures out the right device to place the data on based on the Trainer if possible.

        Otherwise, returns the Cuda device if available, else the CPU device.
        """
        from lightning.pytorch.accelerators.cuda import CUDAAccelerator

        if self._device is not None:
            return self._device
        if self.trainer is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(
            f"cuda:{self.trainer.local_rank}"  # TODO: Debug this with a multi-GPU job.
            if isinstance(self.trainer.accelerator, CUDAAccelerator)
            else "cpu"
        )

    def set_device(self, device: torch.device) -> None:
        """Sets the device that the episodes should be transferred to."""
        logger.debug(f"The RL DataLoader will place episode data on device {device}.")
        self._device = device
        for env in [self.train_env, self.valid_env, self.test_env]:
            if env is None:
                continue
            for wrapper in wrappers_in(env):
                if isinstance(wrapper, ToTensorsWrapper):
                    wrapper.device = device

    def teardown(self, stage: StageStr) -> None:
        if stage in ("fit", "validate", None):
            logger.debug("Closing the training environment.")
            self._close(self.train_dataset, self.train_env, "train")
            # NOTE: we don't close the validation environment unless we're explicitly in the
            # "validate" stage, otherwise calling `trainer.validate(...)` overwrites the videos we
            # took with the validation env during training!
        if stage in ("validate", None):
            logger.debug("Closing the validation environment.")
            self._close(self.valid_dataset, self.valid_env, "valid")
        if stage in ("test", None):
            logger.debug("Closing the test environment.")
            self._close(self.test_dataset, self.test_env, "test")

    def _close(
        self,
        dataset: RlDataset[Any] | None,
        env: gym.Env | None,
        name: Literal["train", "valid", "test"],
    ) -> None:
        assert getattr(self, f"{name}_env") is env
        assert getattr(self, f"{name}_dataset") is dataset
        if dataset is not None:
            # One of the dataloader methods was called and the env and dataset were created.
            assert env is not None
            assert env is dataset.env
            env.close()
            setattr(self, f"{name}_env", None)
            setattr(self, f"{name}_dataset", None)
        elif env is not None:
            # The env was created (.setup() was called), but no dataloader methods were used.
            env.close()
            setattr(self, f"{name}_env", None)


def custom_collate_fn(
    episodes: list[Episode[ActorOutput]],
) -> EpisodeBatch[ActorOutput]:
    """Collates a list of episodes into an EpisodeBatch object containing nested tensors."""
    return EpisodeBatch(
        observations=torch.nested.as_nested_tensor(
            [ep["observations"] for ep in episodes]
        ),
        actions=torch.nested.as_nested_tensor([ep["actions"] for ep in episodes]),
        rewards=torch.nested.as_nested_tensor([ep["rewards"] for ep in episodes]),
        # TODO: Could perhaps stack the infos so it mimics what the RecordEpisodeStatistics wrapper
        # does for VectorEnvs.
        infos=[ep["infos"] for ep in episodes],
        terminated=torch.as_tensor([ep["terminated"] for ep in episodes]),
        truncated=torch.as_tensor([ep["terminated"] for ep in episodes]),
        actor_outputs=stack_dicts([ep["actor_outputs"] for ep in episodes]),
        # type(episodes[0]["actor_outputs"])(
        #     **{
        #         key: torch.nested.as_nested_tensor(
        #             [ep["actor_outputs"][key] for ep in episodes]
        #         )
        #         for key in episodes[0]["actor_outputs"].keys()
        #     }
        # ),
    )


def _error_actor_required(
    dm: RlDataModule[Any], name: Literal["train", "valid", "test"]
):
    return RuntimeError(
        "An actor must be set with before we can gather episodes.\n"
        "Either provide a value to the `actor` constructor argument, or call "
        "`datamodule.set_actor(<callable>)` before training, for example in the `on_fit_start` "
        "hook of your Algorithm (a LightningModule)."
    )


def warn(message, warning_type=RuntimeWarning, color="orange", stacklevel=1):
    warnings.warn(warning_type(colorize(message, color)), stacklevel=stacklevel + 1)


def wrappers_in(env: gym.Wrapper | gym.Env) -> list[gym.Wrapper | gym.Env]:
    """Returns the list of wrappers in the given environment."""
    wrappers = []
    while isinstance(env, gym.Wrapper):
        wrappers.append(env)
        env = env.env
    return wrappers
