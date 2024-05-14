from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from logging import getLogger as get_logger
from typing import Any, Generic, Literal, Protocol

import gymnasium
import torch
from gymnasium.spaces.utils import flatdim
from gymnasium.utils.colorize import colorize
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from project.datamodules.rl.envs import make_torch_env, make_torch_vectorenv
from project.datamodules.rl.wrappers.tensor_spaces import TensorSpace
from project.utils.device import default_device
from project.utils.types import StageStr
from project.utils.types.protocols import DataModule

from .rl_dataset import RlEpisodeDataset
from .rl_types import (
    Actor,
    ActorOutput,
    Episode,
    EpisodeBatch,
)
from .wrappers.tensor_spaces import TensorBox, TensorDiscrete, TensorSpace  # noqa
from .wrappers.to_torch import ToTorchWrapper

logger = get_logger(__name__)

type TensorEnv = gymnasium.Env[Tensor, Tensor]


class _EnvFn(Protocol):
    def __call__(self, seed: int) -> TensorEnv: ...


class EnvDataLoader(DataLoader):
    """Yields batches of episodes."""

    def __init__(
        self,
        dataset: RlEpisodeDataset[ActorOutput],
        batch_size: int,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=custom_collate_fn,
            shuffle=False,
        )
        self.env = dataset
        self._iterator: Iterator[EpisodeBatch[ActorOutput]] | None = None

    def __next__(self) -> EpisodeBatch[ActorOutput]:
        if self._iterator is None:
            self._iterator = super().__iter__()
        return next(self._iterator)

    def __iter__(self):
        return self

    def on_actor_update(self) -> None:
        self.env.on_actor_update()
        # force re-creation of the iterator, to prevent different actors in the same batch.
        self._iterator = None


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
        env: str | _EnvFn,
        actor: Actor[Tensor, Tensor, ActorOutput] | None = None,
        num_parallel_envs: int | None = None,
        # todo: also add `steps_per_epoch` (mutually exclusive with episodes_per_epoch).
        episodes_per_epoch: int = 100,
        batch_size: int = 1,
        train_wrappers: list[Callable[[TensorEnv], TensorEnv]] | None = None,
        valid_wrappers: list[Callable[[TensorEnv], TensorEnv]] | None = None,
        test_wrappers: list[Callable[[TensorEnv], TensorEnv]] | None = None,
        train_dataloader_wrappers: list[
            Callable[
                [Iterable[EpisodeBatch[ActorOutput]]],
                Iterable[EpisodeBatch[ActorOutput]],
            ]
        ]
        | None = None,
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

        self.env_fn: _EnvFn = (
            functools.partial(
                make_torch_vectorenv, env_id=env, num_envs=num_parallel_envs, device=self.device
            )
            if isinstance(env, str) and num_parallel_envs is not None
            else functools.partial(
                make_torch_env,
                env,
                device=self.device,
            )
            if isinstance(env, str)
            else env
        )
        # todo: remove this, only use it to get the observation and action spaces.
        self.env: TensorEnv = self.env_fn(seed=0)
        self.episodes_per_epoch = episodes_per_epoch
        self.batch_size = batch_size

        self.actor: Actor[Tensor, Tensor, ActorOutput] | None = actor

        self.train_env: TensorEnv | None = None
        self.valid_env: TensorEnv | None = None
        self.test_env: TensorEnv | None = None

        self.train_dataset: RlEpisodeDataset[ActorOutput] | None = None
        self.valid_dataset: RlEpisodeDataset[ActorOutput] | None = None
        self.test_dataset: RlEpisodeDataset[ActorOutput] | None = None

        self._train_wrappers = tuple(train_wrappers or ())
        self._valid_wrappers = tuple(valid_wrappers or ())
        self._test_wrappers = tuple(test_wrappers or ())

        self.train_dataloader_wrappers = train_dataloader_wrappers

        self.train_actor = actor
        self.valid_actor = actor
        self.test_actor = actor

        self.train_seed = train_seed
        self.valid_seed = val_seed
        self.test_seed = 111

        self._train_dataloader: Iterable[EpisodeBatch[ActorOutput]] | None = None

    def set_actor[SomeActorOutputType: dict](
        self, actor: Actor[Tensor, Tensor, SomeActorOutputType]
    ) -> RlDataModule[SomeActorOutputType]:
        """Sets the actor to be used for collecting episodes.

        Also potentially changes the type of information stored in the `actor_output` entry in the
        EpisodeBatches yielded by the dataloaders.
        """
        # Note: doing this just to save some typing trouble below.
        # _actor = cast(Actor[Tensor, Tensor, ActorOutput], actor)
        _actor = actor
        if self.train_actor is None:
            assert self.test_actor is None
            assert self.valid_actor is None

            self.train_actor = _actor
            # Save some computation by disabling gradient computation. Actor can still turn it on
            # internally necessary with a context manager or decorator.
            self.valid_actor = torch.no_grad()(torch.inference_mode()(_actor))
            self.test_actor = torch.no_grad()(torch.inference_mode()(_actor))
            return self  # type: ignore

        assert self.test_actor is not None
        assert self.valid_actor is not None

        # TODO: notify the dataset that the actor changed so the iterator resets the envs and clears
        # the buffers.
        if self.train_dataset is not None:
            self.train_dataset.actor = self.train_actor
        if self.valid_dataset is not None:
            self.valid_dataset.actor = self.valid_actor
        if self.test_dataset is not None:
            self.test_dataset.actor = self.test_actor
        return self  # type: ignore

    def prepare_data(self) -> None:
        # NOTE: We don't use this hook here.
        ...

    def setup(self, stage: StageStr) -> None:
        """Sets up the environment(s), applying wrappers and such.

        Called at the beginning of each stage (fit, validate, test).
        """
        if stage in ["fit", None]:
            creating = "Recreating" if self.train_env is not None else "Creating"
            logger.debug(f"{creating} training environment with wrappers {self.train_wrappers}")
            self.train_env = self._make_env(wrappers=self.train_wrappers, seed=self.train_seed)
        if stage in ["validate", None]:
            creating = "Recreating" if self.valid_env is not None else "Creating"
            logger.debug(f"{creating} validation environment with wrappers {self.valid_wrappers}")
            self.valid_env = self._make_env(wrappers=self.valid_wrappers, seed=self.valid_seed)
        if stage in ["test", None]:
            creating = "Recreating" if self.test_env is not None else "Creating"
            logger.debug(f"{creating} testing environment with wrappers {self.test_wrappers}")
            self.test_env = self._make_env(wrappers=self.test_wrappers, seed=self.test_seed)

    def train_dataloader(self) -> Iterable[EpisodeBatch[ActorOutput]]:
        if self.train_actor is None:
            # warn("No actor was set, using a random policy.", color="red")
            # self.train_actor = random_actor
            raise _error_actor_required(self, "train")
        self.train_env = self.train_env or self._make_env(wrappers=self.train_wrappers)
        self.train_dataset = RlEpisodeDataset(
            self.train_env,
            actor=self.train_actor,
            episodes_per_epoch=self.episodes_per_epoch,
            seed=self.train_seed,
        )
        dataloader: Iterable[EpisodeBatch[ActorOutput]] = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
        )
        for dataloader_wrapper in self.train_dataloader_wrappers or []:
            logger.debug(f"Applying dataloader wrapper {dataloader_wrapper}")
            dataloader = dataloader_wrapper(dataloader)
        return dataloader

    def val_dataloader(self) -> DataLoader[EpisodeBatch[ActorOutput]]:
        if self.valid_actor is None:
            raise _error_actor_required(self, "valid")

        self.valid_env = self.valid_env or self._make_env(
            wrappers=self.valid_wrappers, seed=self.valid_seed
        )
        self.valid_dataset = RlEpisodeDataset(
            self.valid_env,
            actor=self.valid_actor,
            episodes_per_epoch=self.episodes_per_epoch,
            seed=self.valid_seed,
        )
        assert self.valid_dataset is not None
        dataloader: DataLoader[EpisodeBatch[ActorOutput]] = EnvDataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader[EpisodeBatch[ActorOutput]]:
        if self.test_actor is None:
            raise _error_actor_required(self, "test")

        self.test_env = self.test_env or self._make_env(
            wrappers=self.test_wrappers, seed=self.test_seed
        )
        self.test_dataset = RlEpisodeDataset(
            self.test_env,
            actor=self.test_actor,
            episodes_per_epoch=self.episodes_per_epoch,
            seed=self.test_seed,
        )
        dataloader: DataLoader[EpisodeBatch[ActorOutput]] = EnvDataLoader(  # type: ignore
            self.test_dataset,
            batch_size=self.batch_size,
        )
        return dataloader

    @property
    def train_wrappers(self) -> tuple[Callable[[TensorEnv], TensorEnv], ...]:
        return self._train_wrappers

    @train_wrappers.setter
    def train_wrappers(self, wrappers: Sequence[Callable[[TensorEnv], TensorEnv]]) -> None:
        wrappers = tuple(wrappers)
        if self.train_env is not None and wrappers != self._train_wrappers:
            logger.warn("Training wrappers changed, closing the previous environment.")
            self.train_env.close()
            self.train_env = None
        self._train_wrappers = wrappers

    @property
    def valid_wrappers(self) -> tuple[Callable[[TensorEnv], TensorEnv], ...]:
        return self._valid_wrappers

    @valid_wrappers.setter
    def valid_wrappers(self, wrappers: Sequence[Callable[[TensorEnv], TensorEnv]]) -> None:
        wrappers = tuple(wrappers)
        if self.valid_env is not None and wrappers != self._valid_wrappers:
            logger.warn("validation wrappers changed, closing the previous environment.")
            self.valid_env.close()
            self.valid_env = None
        self._valid_wrappers = wrappers

    @property
    def test_wrappers(self) -> tuple[Callable[[TensorEnv], TensorEnv], ...]:
        return self._test_wrappers

    @test_wrappers.setter
    def test_wrappers(self, wrappers: Sequence[Callable[[TensorEnv], TensorEnv]]) -> None:
        wrappers = tuple(wrappers)
        if self.test_env is not None and wrappers != self._test_wrappers:
            logger.warn("testing wrappers changed, closing the previous environment.")
            self.test_env.close()
            self.test_env = None
        self._valid_wrappers = wrappers

    @property
    def dims(self) -> tuple[int, ...]:
        if self.observation_space.shape:
            return self.observation_space.shape
        return (flatdim(self.observation_space),)

    @property
    def action_dims(self) -> int:
        return flatdim(self.action_space)

    @property
    def observation_space(self) -> TensorSpace:
        if self.train_env is None:
            self.prepare_data()
            self.setup(stage="fit")
        assert self.train_env is not None
        assert isinstance(self.train_env.observation_space, TensorSpace)
        return self.train_env.observation_space

    @property
    def action_space(self) -> TensorSpace:
        if self.train_env is None:
            self.prepare_data()
            self.setup(stage="fit")
        assert self.train_env is not None
        assert isinstance(self.train_env.action_space, TensorSpace)
        return self.train_env.action_space

    def _make_env(
        self, wrappers: Sequence[Callable[[TensorEnv], TensorEnv]], seed: int
    ) -> TensorEnv:
        # TODO: Use gym.vector.make for vector envs, and pass the single-env wrappers to
        # gym.vector.make.
        logger.debug(f"Creating a new environment with {len(wrappers)} wrappers:")
        env = self.env_fn(seed=seed)
        for i, wrapper in enumerate(wrappers):
            env = wrapper(env)
            logger.debug(
                f"after {i} wrappers: {type(env)=}, {env.observation_space=}, {env.action_space=}"
            )
        # TODO: Should this wrapper always be mandatory? And should it always be placed at the end?
        if isinstance(env.observation_space, TensorSpace) and isinstance(
            env.action_space, TensorSpace
        ):
            assert env.observation_space.device == self.device
            assert env.action_space.device == self.device
            logger.debug("Env is already on the right device.")
        else:
            logger.debug(
                f"Add a {ToTorchWrapper.__name__} wrapper to move the numpy arrays to GPU. This isn't ideal!"
            )
            env = ToTorchWrapper(env, device=self.device)
        return env

    @property
    def device(self) -> torch.device:
        from lightning.pytorch.accelerators.cuda import CUDAAccelerator

        if self.trainer is None:
            return default_device()
        return torch.device(
            f"cuda:{self.trainer.local_rank}"  # TODO: Debug this with a multi-GPU job.
            if isinstance(self.trainer.accelerator, CUDAAccelerator)
            else "cpu"
        )

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
        dataset: RlEpisodeDataset[Any] | None,
        env: TensorEnv | None,
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


def custom_collate_fn(episodes: list[Episode[ActorOutput]]) -> EpisodeBatch[ActorOutput]:
    """Collates a list of episodes into an EpisodeBatch object containing (possibly nested)
    tensors."""

    # note: assuming simple tensor observations for now, even though the code below is meant to
    # also support nested dicts observations and actions
    device = episodes[0].observations.device

    def _stack(tensors: list[torch.Tensor]) -> torch.Tensor:
        """Stacks tensors using torch.stack if all the sequences have the same length, otherwise
        uses torch.nested.as_nested_tensor."""
        if all(t_i.shape == tensors[0].shape for t_i in tensors):
            return torch.stack(tensors)
        return torch.nested.as_nested_tensor(tensors)

    return EpisodeBatch(
        observations=_stack([ep["observations"] for ep in episodes]),
        actions=_stack([ep["actions"] for ep in episodes]),
        rewards=_stack([ep["rewards"] for ep in episodes]),
        # TODO: Could perhaps stack the infos so it mimics what the RecordEpisodeStatistics wrapper
        # does for VectorEnvs.
        infos=[ep["infos"] for ep in episodes],
        terminated=torch.as_tensor([ep["terminated"] for ep in episodes], device=device),
        truncated=torch.as_tensor([ep["terminated"] for ep in episodes], device=device),
        actor_outputs=type(episodes[0]["actor_outputs"])(
            **{
                key: _stack([ep["actor_outputs"][key] for ep in episodes])
                for key in episodes[0]["actor_outputs"].keys()
            }
        ),
    )


def _error_actor_required(dm: RlDataModule[Any], name: Literal["train", "valid", "test"]):
    return RuntimeError(
        "An actor must be set with before we can gather episodes.\n"
        "Either provide a value to the `actor` constructor argument, or call "
        "`datamodule.set_actor(<callable>)` before training, for example in the `on_fit_start` "
        "hook of your Algorithm (a LightningModule)."
    )


def warn(message, warning_type=RuntimeWarning, color="orange", stacklevel=1):
    warnings.warn(warning_type(colorize(message, color)), stacklevel=stacklevel + 1)
