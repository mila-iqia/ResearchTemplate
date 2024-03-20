from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from warnings import warn

from hydra.core.config_store import ConfigStore
from hydra.core.utils import JobReturn
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig
from orion.client.experiment import ExperimentClient
from orion.core.worker.trial import Trial

from hydra_plugins.hydra_orion_sweeper.config import (
    AlgorithmConf,
    OrionClientConf,
    OrionSweeperConf,
    StorageConf,
    WorkerConf,
)
from hydra_plugins.hydra_orion_sweeper.implementation import OrionSweeperImpl
from hydra_plugins.hydra_orion_sweeper.orion_sweeper import OrionSweeper

logger = get_logger(__name__)


class CustomOrionSweeper(OrionSweeper):
    def __init__(
        self,
        experiment: OrionClientConf | None,
        worker: WorkerConf,
        algorithm: AlgorithmConf,
        storage: StorageConf,
        parametrization: DictConfig | None,
        params: DictConfig | None,
        orion: OrionClientConf | None = None,
    ):
        # >>> Remove with Issue #8
        if parametrization is not None and params is None:
            warn(
                "`hydra.sweeper.parametrization` is deprecated;"
                "use `hydra.sweeper.params` instead",
                DeprecationWarning,
            )
            params = parametrization

        elif parametrization is not None and params is not None:
            warn(
                "Both `hydra.sweeper.parametrization` and `hydra.sweeper.params` are defined;"
                "using `hydra.sweeper.params`",
                DeprecationWarning,
            )
        # <<<
        params = params or {}
        compat = False
        if orion is not None:
            compat = True
            warn(
                "`hydra.sweeper.orion` as dreprecated in favour of `hydra.sweeper.experiment`."
                "Please update to avoid misconfiguration",
                DeprecationWarning,
            )

        if experiment is None:
            assert orion is not None
            experiment = orion

        self.sweeper = CustomOrionSweeperImpl(
            experiment, worker, algorithm, storage, params, compat
        )

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        return self.sweeper.setup(
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def sweep(self, arguments: list[str]) -> None:
        return self.sweeper.sweep(arguments)


class CustomOrionSweeperImpl(OrionSweeperImpl):
    def setup(
        self, *, hydra_context: HydraContext, task_function: TaskFunction, config: DictConfig
    ) -> None:
        return super().setup(
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def sweep(self, arguments: list[str]) -> None:
        # assert self.config is not None
        # sweep_dir = Path(str(self.config.hydra.sweep.dir))
        # sweep_dir.mkdir(parents=True, exist_ok=True)
        # logger.info(f"Sweep dir : " f"{sweep_dir}")
        return super().sweep(arguments)

    def observe_results(
        self,
        trials: list[Trial],
        returns: Sequence[JobReturn],
        failures: Sequence[JobReturn],
    ):
        """Record the result of each trials."""
        # TODO: The base class assumes that there is the same number of trials and returns, but if
        # we pack multiple trials in a single job e.g. with different random seeds and return all
        # results with the Launcher, then we need to have multiple results per trial or multiple
        # trials (with different seeds) each!
        assert len(trials) == len(returns)
        super().observe_results(trials, returns, failures)

    def show_results(self) -> None:
        assert self.config is not None
        sweep_dir = Path(self.config.hydra.sweep.dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        super().show_results()
        assert self.client is not None
        results = self.client.stats
        best_trial = self.client.get_trial(uid=results.best_trials_id)
        assert best_trial is not None
        best_results = best_trial.results
        logger.info(f"Best trial: {best_trial}")
        logger.info(f"Best results: {best_results}")
        logger.info(f"Best trial working dir: {best_trial.working_dir}")

    def optimize(self, client: ExperimentClient) -> None:
        """Run the hyperparameter search in batches."""
        return super().optimize(client)

    ...


@dataclass
class CustomOrionSweeperConf(OrionSweeperConf):
    ...

    _target_: str = "hydra_plugins.custom_launcher.custom_orion_sweeper.CustomOrionSweeper"


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="custom_orion",
    node=CustomOrionSweeperConf,
    provider="ResearchTemplate",
)
