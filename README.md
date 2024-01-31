# Compute-Optimal LoRA Adapters for Causal-LM

Chinchilla paper but with low-rank adaptation and for causal language modelling

## To do

- run training with llama (using LoRA)
- run inference with llama (to show actual performance)
- replace llama with blenderbot (using LoRA)
- run inference with blenderbot (to show actual performance)
- evaluate the model via Eleuther AI's [evaluation harness](https://colab.research.google.com/drive/1C4OfWDjHOmE8PSuC0TyATrVoADb61B7B)
- train blenderbot with vanilla trainer (using the `HuggingFaceH4/ultrachat_200k` dataset's `train_gen` split and [this tutorial](https://huggingface.co/blog/dpo-trl))
- OPTIONAL: train blenderbot with dpo trainer (using the `HuggingFaceH4/ultrachat_200k` dataset's `train_gen` split and [this tutorial](https://huggingface.co/blog/dpo-trl))
- configure sweep training with blenderbot using LoRA hyperparameters (Which ones? – See docs and chatgpt!)
- make sure all results are available on wandb
- make sweeps for the following models:
  - mistral
  - zephyr
  - orca
  - mixtral
  - see ChatGPT chat on models
  - see LLM leaderboard for top models
- DPO datasets:
  - [`adamo1139/rawrr_v1`](https://huggingface.co/datasets/adamo1139/rawrr_v1)
  - [`HuggingFaceH4/orca_dpo_pairs`](https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs)
  - [`jondurbin/gutenberg-dpo-v0.1`](https://huggingface.co/datasets/jondurbin/gutenberg-dpo-v0.1)
  - many more, search [datasets](https://huggingface.co/datasets?sort=trending&search=dpo) for `dpo`!
- normal datasets
  - [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) for **vanilla Trainer** (or [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer))

---

- token classification

  - models:
    - `mistralai/Mixtral-8x7B-Instruct-v0.1` (**decoder**) [link](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
    - `FacebookAI/roberta-large` (**encoder**) [link](https://huggingface.co/FacebookAI/roberta-large)
  - evaluation:
    - compare with labels ($\Rightarrow$ precision, recall, accuracy, $F_1$ etc.)
  - datasets
    - `ai4privacy/pii-masking-200k`, large (**209K**)
    - `ai4privacy/pii-masking-200k`, medium (**21K**)
    - `ai4privacy/pii-masking-200k`, small (**2.1K**)
  - trainers
    - `Trainer`

- summarization
  - models:
    - `mistralai/Mixtral-8x7B-Instruct-v0.1` (**decoder**) [link](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
    - `facebook/bart-large-xsum` (**encoder-decoder**) [link](https://huggingface.co/facebook/bart-large-xsum)
  - evaluation:
    - ROUGE [score](https://huggingface.co/spaces/evaluate-metric/rouge)
  - datasets
    - `cnn_dailymail`, large (**311K**)
    - `cnn_dailymail`, medium (**31K**)
    - `cnn_dailymail`, small (**3.1K**)
  - trainers
    - `Trainer`
- conversational
  - model: `mistralai/Mixtral-8x7B-Instruct-v0.1` (**decoder**) [link](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  - evaluation:
    - [eleutherAI LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness)
  - datasets
    - [`Open-Orca/OpenOrca`]("https://huggingface.co/datasets/Open-Orca/OpenOrca"), xlarge (**2.91M**)
    - [`Open-Orca/OpenOrca`]("https://huggingface.co/datasets/Open-Orca/OpenOrca"), large (**291k**)
    - [`Open-Orca/OpenOrca`]("https://huggingface.co/datasets/Open-Orca/OpenOrca"), medium (**29.1k**)
    - [`Open-Orca/OpenOrca`]("https://huggingface.co/datasets/Open-Orca/OpenOrca"), small (**2.9k**)
  - trainers
    - `Trainer`
    - `SFTTrainer`
    - `DPOTrainer`

## Done
