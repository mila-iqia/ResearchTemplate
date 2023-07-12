from warnings import warn

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from hydra_plugins.hydra_orion_sweeper.config import (
    AlgorithmConf,
    OrionClientConf,
    OrionSweeperConf,
    StorageConf,
    WorkerConf,
)
from hydra_plugins.hydra_orion_sweeper.implementation import OrionSweeperImpl
from hydra_plugins.hydra_orion_sweeper.orion_sweeper import (
    HydraContext,
    OrionSweeper,
    TaskFunction,
)


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
    ...


class CustomOrionSweeperConf(OrionSweeperConf):
    ...

    _target_: str = "hydra_plugins.custom_launcher.custom_orion_sweeper.CustomOrionSweeper"


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="custom_orion",
    node=OrionSweeperConf,
    provider="ResearchTemplate",
)
