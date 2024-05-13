from logging import getLogger as get_logger

import gymnasium
import pytest
import torch

from project.datamodules.rl.envs.gymnax import gymnax_vectorenv
from project.datamodules.rl.rl_dataset import VectorEnvRlDataset
from project.datamodules.rl.wrappers.tensor_spaces import TensorSpace

logger = get_logger(__name__)


@pytest.mark.timeout(10)
def test_set_actor_resets_envs(seed: int, device: torch.device):
    num_envs = 3
    env = gymnax_vectorenv("CartPole-v1", num_envs=num_envs, device=device, seed=seed)
    # target = 3
    # initial_state = 0
    # env = DebugEnv(
    #     min=-10,
    #     max=10,
    #     max_episode_length=5,
    #     target=target,
    #     initial_state=initial_state,
    #     device=device,
    #     randomize_initial_state=False,
    #     randomize_target=False,
    #     wrap_around_state=True,
    #     seed=seed,
    # )

    _update = 0

    def actor_with_update_index_in_output():
        def _actor(observation: torch.Tensor, action_space: gymnasium.Space[torch.Tensor]):
            nonlocal _update
            assert isinstance(action_space, TensorSpace)
            assert action_space.shape is not None
            action = action_space.sample()
            return action, {
                "action": action,
                "_update": _update,
            }

        return _actor

    dataset = VectorEnvRlDataset(
        env, actor=actor_with_update_index_in_output(), seed=seed, episodes_per_epoch=100
    )

    def update_actor():
        nonlocal _update
        _update += 1
        # dataset.actor = new_actor
        dataset.on_actor_update()

    # obs, info = env.reset() # obs #1
    # obs, rew, (...), # obs 2, reward #1
    # obs, rew, (...), # obs 3, reward #2
    ## DONE - reward #3, obs#1 of next episode
    # So len(obs) should be len(rewards)

    actor_outputs_list = []

    for episode_index, episode in enumerate(dataset):
        assert env.single_observation_space.shape is not None
        assert episode.observations.shape == (episode.length, *env.single_observation_space.shape)
        assert env.single_action_space.shape is not None
        assert episode.actions.shape == (episode.length, *env.single_action_space.shape)
        assert episode.rewards.shape == (episode.length,)
        # todo: stack the info dicts? (means that we assume that each steps logs the same keys?)
        assert len(episode.infos) == episode.length
        actor_outputs_list.append(episode.actor_outputs)
        logger.debug(f"{episode_index=}, {episode.actor_outputs=}")

        # Check that the actions are lined up exactly like they were in the actor outputs.
        assert (episode.actor_outputs["action"] == episode.actions).all()

        if episode_index <= 2:
            assert set(episode.actor_outputs["_update"].tolist()) == {0}

        if episode_index == 2:
            # Update the actor to one that outputs a 1
            update_actor()

        if 3 <= episode_index <= 6:
            # There should not be a mix of old and new actions. All actions should have been sampled
            # from the updated actor.
            assert set(episode.actor_outputs["_update"].tolist()) == {1}

        if episode_index == 6:
            update_actor()

        if 6 < episode_index:
            assert set(episode.actor_outputs["_update"].tolist()) == {2}

        if episode_index == 10:
            break
