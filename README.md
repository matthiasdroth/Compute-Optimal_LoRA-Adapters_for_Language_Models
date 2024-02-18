# Compute-Optimal LoRA Adapters for Language Models

_Chinchilla paper but with low-rank adaptation and for causal language modelling_

## Overview

- token classification

  - models:
    - `mistralai/Mixtral-8x7B-Instruct-v0.1` (**decoder**, 8$\times$12.1GB) [link](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
    - `FacebookAI/roberta-large` (**encoder**, 1.43GB) [link](https://huggingface.co/FacebookAI/roberta-large)
  - evaluation:
    - **seqeval** [link](https://pypi.org/project/seqeval/)
    - **LightEval** [link](https://github.com/huggingface/lighteval)
  - datasets
    - `DFKI-SLT/few-nerd`, large (**188K**) [link](https://huggingface.co/datasets/DFKI-SLT/few-nerd)
    - `DFKI-SLT/few-nerd`, medium (**18.8K**) [link](https://huggingface.co/datasets/DFKI-SLT/few-nerd)
    - `DFKI-SLT/few-nerd`, small (**1.9K**) [link](https://huggingface.co/datasets/DFKI-SLT/few-nerd)
    - `ai4privacy/pii-masking-200k`, large (**209K**) [link](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
    - `ai4privacy/pii-masking-200k`, medium (**21K**) [link](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
    - `ai4privacy/pii-masking-200k`, small (**2.1K**) [link](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
  - trainers
    - `Accelerate`
    - `Trainer`

- summarization
  - models:
    - `mistralai/Mixtral-8x7B-Instruct-v0.1` (**decoder**, 8$\times$12.1GB) [link](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
    - `facebook/bart-large-xsum` (**encoder-decoder**, 1.63GB) [link](https://huggingface.co/facebook/bart-large-xsum)
  - evaluation:
    - **ROUGE** [link](https://huggingface.co/spaces/evaluate-metric/rouge)
    - **LightEval** [link](https://github.com/huggingface/lighteval)
  - datasets
    - `cnn_dailymail`, large (**311K**) [link](https://huggingface.co/datasets/cnn_dailymail)
    - `cnn_dailymail`, medium (**31K**)
    - `cnn_dailymail`, small (**3.1K**)
  - trainers
    - `Accelerate`
    - `Trainer`
- conversational
  - models:
    - `mistralai/Mixtral-8x7B-Instruct-v0.1` (**decoder**, 8$\times$12.1GB) [link](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
    - `microsoft/DialoGPT-large` (**decoder**, 1.75GB) [link](https://huggingface.co/microsoft/DialoGPT-large/tree/main)
  - evaluation:
    - **EleutherAI LM evaluation harness** [link](https://github.com/EleutherAI/lm-evaluation-harness)
    - **LightEval** [link](https://github.com/huggingface/lighteval)
  - datasets
    - `Open-Orca/OpenOrca`, xlarge (**2.91M**) [link](https://huggingface.co/datasets/Open-Orca/OpenOrca)
    - `Open-Orca/OpenOrca`, large (**291k**)
    - `Open-Orca/OpenOrca`, medium (**29.1k**)
    - `Open-Orca/OpenOrca`, small (**2.9k**)
  - trainers
    - `Accelerate`
    - `Trainer`
    - `SFTTrainer`
    - `DPOTrainer`

## ToDo

- Go through [this guide](https://huggingface.co/docs/peft/task_guides/token-classification-lora) on _LoRA for token classification_ (but use a custom training loop – at best with `accelerate` – to accommodate the deepspeed code for [counting FLOPS](https://www.deepspeed.ai/tutorials/flops-profiler/#example-training-workflow)).
- _If still necessary:_ Review HuggingFace course section 7.2 on Token classification
  - [official](https://huggingface.co/learn/nlp-course/chapter7/2)
  - [own implementation](https://github.com/matthiasdroth/Huggingface-course/blob/main/7.2-Token_classification.ipynb)
- Use the [flops profiler from deepspeed](https://www.deepspeed.ai/tutorials/flops-profiler/#example-training-workflow) or understand how to calculate FLOPS for an entire training run (using LoRA)!
- Change model to [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large).
- Change dataset to [`ai4privacy/pii-masking-200k`](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
- perform sweep on LoRA adapter, trying the following sweep parameters:
  - dataset size
  - PEFT parameters:
    - to
    - be
    - determined
    - but
    - see
    - [here](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) and
    - [here](https://arxiv.org/pdf/2312.03732.pdf)!

---

## Done
