"""Configs for learning rate schedulers.

You can add configurations either with a config file or in code using
[hydra-zen.builds](https://mit-ll-responsible-ai.github.io/hydra-zen/generated/hydra_zen.builds.html#).
"""
# import hydra_zen

# Some LR Schedulers have constructors with arguments without a default value (in addition to optimizer).
# In this case, we specify the missing arguments here so we get a nice error message if it isn't passed.

# StepLRConfig = hydra_zen.builds(
#     torch.optim.lr_scheduler.StepLR,
#     populate_full_signature=True,
#     step_size="???",
#     zen_partial=True,
#     zen_dataclass={"cls_name": "StepLRConfig", "frozen": True},
# ),

# lr_scheduler_store = hydra_zen.store(group="lr_scheduler")
# lr_scheduler_store(StepLRConfig, name="step_lr")
