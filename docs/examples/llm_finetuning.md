---
additional_python_references:
  - project.algorithms.llm_finetuning
---
# Fine-tuning LLMs

This example is based on [this language modeling example from the HuggingFace transformers documentation](https://huggingface.co/docs/transformers/en/tasks/language_modeling).

To better understand what's going on in this example, it is a good idea to read through these tutorials first:

* [Causal language modeling simple example - HuggingFace docs](https://huggingface.co/docs/transformers/en/tasks/language_modeling)
* [Fine-tune a language model - Colab Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb#scrollTo=X6HrpprwIrIz)

The main difference between this example and the original example from HuggingFace is that the `LLMFinetuningExample` is a `LightningModule`, that is trained by a `lightning.Trainer`.

This also means that this example doesn't use [`accelerate`](https://huggingface.co/docs/accelerate/en/index) or the HuggingFace Trainer.


## Running the example

```console
python project/main.py experiment=llm_finetuning_example
```
