import gymnasium
import torch

from project.datamodules.rl.wrappers.to_tensor import ToTorchWrapper

from .brax import brax_env, brax_vectorenv
from .gymnax import gymnax_env, gymnax_vectorenv


def make_torch_env(env_id: str, seed: int, device: torch.device, **kwargs):
    import brax.envs
    import gymnax.registration

    if env_id in gymnax.registration.registered_envs:
        return gymnax_env(env_id=env_id, seed=seed, device=device, **kwargs)
    if env_id in brax.envs._envs:
        return brax_env(env_id, device=device, seed=seed, **kwargs)

    env = gymnasium.make(env_id, **kwargs)
    return ToTorchWrapper(env, device=device)


def make_torch_vectorenv(env_id: str, num_envs: int, seed: int, device: torch.device, **kwargs):
    import brax.envs
    import gymnax.registration

    if env_id in gymnax.registration.registered_envs:
        return gymnax_vectorenv(
            env_id=env_id, num_envs=num_envs, seed=seed, device=device, **kwargs
        )
    if env_id in brax.envs._envs:
        return brax_vectorenv(env_id, num_envs=num_envs, seed=seed, device=device, **kwargs)
    env = gymnasium.vector.make(env_id, num_envs=num_envs, **kwargs)
    return ToTorchWrapper(env, device=device)


__all__ = [
    "brax_env",
    "brax_vectorenv",
    "gymnax_env",
    "gymnax_vectorenv",
    "make_torch_env",
    "make_torch_vectorenv",
]
