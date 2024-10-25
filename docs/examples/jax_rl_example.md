---
additional_python_references:
  - project.algorithms.jax_rl_example
  - project.trainers.jax_trainer
---


# Reinforcement Learning in Jax


This example follows the same structure as the other examples:

- An "algorithm" (in this case `JaxRLExample`) is trained with a "trainer" (`JaxTrainer`);


However, there are some very important differences:
- There is no "datamodule". The algorithm accepts an Environment (`gymnax.Environment`) as input.
- The "Trainer" is a `JaxTrainer`, instead of a `lightning.Trainer`.
  - The full training loop is written in Jax;
  - Some (but not all) PyTorch-Lightning callbacks can still be used with the JaxTrainer;
- The `JaxRLExample` class is an algorithm based on rejax.PPO.


## JaxRLExample

The `JaxRLExample` class is a

??? note "Click to show the code for JaxRLExample"
    {{ inline('project.algorithms.jax_rl_example.JaxRLExample', 4) }}


## JaxTrainer

The `JaxTrainer` follows a roughly similar structure as the `lightning.Trainer`:
- `JaxTrainer.fit` is called with a `JaxModule` to train the algorithm.


??? note "Click to show the code for JaxTrainer"
    {{ inline('project.trainers.jax_trainer.JaxTrainer', 4) }}
