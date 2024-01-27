# Compute-Optimal LoRA Adapters for Causal-LM

Chinchilla paper but with low-rank adaptation and for causal language modelling

## To do

- run training with llama (using LoRA)
- run inference with llama
- replace llama with blenderbot (using LoRA)
- run inference with blenderbot
- evaluate the model via Eleuther AI's [evaluation harness](https://colab.research.google.com/drive/1C4OfWDjHOmE8PSuC0TyATrVoADb61B7B)
- train blenderbot with dpo trainer (using the `HuggingFaceH4/ultrachat_200k` dataset's `train_gen` split and [this tutorial](https://huggingface.co/blog/dpo-trl))
- configure sweep training with blenderbot using LoRA hyperparameters (Which ones? – See docs and chatgpt!)
- make sure all results are available on wandb
-

## Done
