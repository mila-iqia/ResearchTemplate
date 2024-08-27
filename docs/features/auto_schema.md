# Auto Schema for Hydra Configs

> 🔥 NOTE: This is a feature that is entirely unique to this template! 🔥

This project template comes with a really neat feature: Your [Hydra](https://hydra.cc) config files automatically get a [Schema](https://json-schema.org/) associated with them.

This greatly improves the experience of developing a project with Hydra:

- Saves you time by preventing errors caused by unexpected keys in your config files, or values that are of the wrong type
    This can often happen after moving files or renaming a function, for example.
- While writing a config file you get to see:
    - the list of available configuration options in a given config
    - the default values for each value
    - the documentation for each value (taken from the source code of the function!)

Here's a quick demo of what this looks like in practice:

![type:video](https://github.com/user-attachments/assets/08f52d47-ebba-456d-95ef-ac9525d8e983)

Here we have a config that will be used to configure the `lightning.Trainer` class, but any config file in the project will also get a schema automatically, even if it doesn't have a `"_target_"` key directly!
