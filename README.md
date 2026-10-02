# ELF Post-Training

Few-step post-training for [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938),
built on the [PyTorch ELF port](https://github.com/lillian039/ELF/tree/pytorch_elf).

A pretrained ELF model is turned into a generator that produces text in a few
*iterative-refinement* passes: every pass predicts clean latents at t=0 and feeds
the prediction back as self-conditioning. Two objectives are provided:

- **MMD**: match the distribution of intermediate ELF features of generated and real
  text with an RBF-kernel MMD loss. A frozen copy of the pretrained model is the
  feature extractor.
- **IRD** (iterative refinement distillation): fit one generator pass to the result
  of several refinement passes of a frozen model, usually an MMD checkpoint.

Both objectives also keep training the model's token decoder with cross-entropy.
[METHOD.md](METHOD.md) describes the algorithms in full, with the exact losses.

## Installation

Create a conda environment named `elf` and install the dependencies:

```bash
conda create -n elf python=3.10 -y
conda activate elf
pip install -r requirements.txt
```

Then log in to WandB to track your experiments if needed:

```bash
wandb login YOUR_WANDB_API_KEY
```

## Checkpoints

`teacher_checkpoint` (training) and `checkpoint_path` (evaluation) accept a local
checkpoint file, a directory (`ckpt.pt` or its latest `checkpoint_<step>` is used), or a
Hugging Face model repo, optionally with a path inside it:
`embedded-language-flows/ELF-B-owt-torch` or `<org>/<repo>/checkpoint_7000`.
HF checkpoints are downloaded on first use. `--checkpoint_path` overrides the
evaluation checkpoint in the config.

**Pretrained ELF models** (initialization for MMD):

| Model | Task | Encoder | Checkpoint |
| --- | --- | --- | --- |
| ELF-B | OpenWebText (unconditional) | T5-small | [embedded-language-flows/ELF-B-owt-torch](https://huggingface.co/embedded-language-flows/ELF-B-owt-torch) |
| ELF-B | OpenWebText (unconditional) | GPT-2 Large | [yresearch/ELF-MMD-OWT/gpt2/elf](https://huggingface.co/yresearch/ELF-MMD-OWT/tree/main/gpt2/elf) |
| ELF-B | TinyGSM (conditional) | GPT-2 | [yresearch/ELF-MMD-TinyGSM/ELF-B/elf](https://huggingface.co/yresearch/ELF-MMD-TinyGSM/tree/main/ELF-B/elf) |
| ELF-M | TinyGSM (conditional) | GPT-2 | [yresearch/ELF-MMD-TinyGSM/ELF-M/elf](https://huggingface.co/yresearch/ELF-MMD-TinyGSM/tree/main/ELF-M/elf) |

**Post-trained models:**

| Model | Task / encoder | MMD | IRD |
| --- | --- | --- | --- |
| ELF-B | OpenWebText / GPT-2 Large | [9,000](https://huggingface.co/yresearch/ELF-MMD-OWT/tree/main/gpt2/elf-mmd) | [6,000](https://huggingface.co/yresearch/ELF-MMD-OWT/tree/main/gpt2/elf-mmd-ird) |
| ELF-B | OpenWebText / T5-small | [10,000](https://huggingface.co/yresearch/ELF-MMD-OWT/tree/main/t5/elf-mmd) | [5,500](https://huggingface.co/yresearch/ELF-MMD-OWT/tree/main/t5/elf-mmd-ird) |
| ELF-B | TinyGSM / GPT-2 | [7,000](https://huggingface.co/yresearch/ELF-MMD-TinyGSM/tree/main/ELF-B/elf-mmd) | [20,000](https://huggingface.co/yresearch/ELF-MMD-TinyGSM/tree/main/ELF-B/elf-mmd-ird) |
| ELF-M | TinyGSM / GPT-2 | [7,000](https://huggingface.co/yresearch/ELF-MMD-TinyGSM/tree/main/ELF-M/elf-mmd) | — |

Numbers are iterations within each post-training stage. All `yresearch` variants
contain EMA-only `ckpt.pt` weights, an `eval_config.yml`, and checkpoint metadata;
TinyGSM variants also include latent normalization statistics. For automatic
download, use a variant path such as
`--checkpoint_path yresearch/ELF-MMD-TinyGSM/ELF-B/elf-mmd-ird`.

To share your own checkpoint, drop the optimizer and RNG state first; the result
loads the same way and is about half the size:

```bash
python scripts/export_checkpoint.py outputs/tinygsm_gpt2-mmd/checkpoint_7000 export/checkpoint_7000
```

For an EMA-only inference release, export a plain tensor state dictionary:

```bash
python scripts/export_checkpoint.py outputs/tinygsm_gpt2-mmd/checkpoint_7000 export/ckpt.pt --ema-only
```

This preserves the EMA tensor values and precision, verifies the saved tensors, and
omits raw weights and all training state. Evaluation and teacher initialization
accept either the `ckpt.pt` file or its directory, locally or on Hugging Face.
Keep the matching evaluation config and latent statistics beside it; relative
statistics paths can resolve from the config directory. Iteration metadata is not
embedded in this format, so inference reports step/epoch zero. Training resume
requires the original full checkpoint.

## Training

| Data / encoder | MMD config | IRD config |
| --- | --- | --- |
| OpenWebText / T5-small | [mmd/owt_t5.yml](src/configs/mmd/owt_t5.yml) | [ird/owt_t5.yml](src/configs/ird/owt_t5.yml) |
| OpenWebText / GPT-2 Large | [mmd/owt_gpt2.yml](src/configs/mmd/owt_gpt2.yml) | [ird/owt_gpt2.yml](src/configs/ird/owt_gpt2.yml) |
| TinyGSM / GPT-2 / ELF-B | [mmd/tinygsm_gpt2.yml](src/configs/mmd/tinygsm_gpt2.yml) | [ird/tinygsm_gpt2.yml](src/configs/ird/tinygsm_gpt2.yml) |
| TinyGSM / GPT-2 / ELF-M | [mmd/tinygsm_gpt2_elf_m.yml](src/configs/mmd/tinygsm_gpt2_elf_m.yml) | — |

The ELF-M preset follows the run behind the published ELF-M-MMD checkpoint: 32 responses
per prompt, global batch 128, one bootstrap pass, learning rate `5e-5`, and 8,000 updates.
It uses this trainer's MMD objective, which differs from that run in three ways: features
are extracted at `t = 1 - t_eps` instead of from noise at `t = 0`, prompts are never
dropped (the original run dropped 10%), and `sigma` is fixed. Its value, 2470, is the
median heuristic on the teacher's real features under this objective, the same rule that
gives the ELF-B preset's 3330. Retraining with this preset is not guaranteed to reproduce
the published checkpoint.

The presets download their datasets and checkpoint weights from Hugging Face:

| Data / encoder | Dataset | Training split | Evaluation split |
| --- | --- | --- | --- |
| OpenWebText / T5-small | [embedded-language-flows/openwebtext-t5](https://huggingface.co/datasets/embedded-language-flows/openwebtext-t5) | Single saved Arrow dataset | — |
| OpenWebText / GPT-2 Large | [yresearch/owt-gpt2](https://huggingface.co/datasets/yresearch/owt-gpt2) | `train` | — (unconditional generation) |
| TinyGSM / GPT-2 | [yresearch/tinygsm-gpt2](https://huggingface.co/datasets/yresearch/tinygsm-gpt2) | `train` | `test` (1,319 GSM8K examples) |

`data_split` and `eval_data_split` select exact split names. For Parquet repositories,
only the selected split's files are downloaded and prepared. Local Arrow paths are
also supported. The architecture and latent normalization in a config must match
its teacher. TinyGSM presets load their per-channel statistics from Hugging Face
and cache them on first use. `latent_mean` accepts a local statistics file or a
Hub file such as `yresearch/ELF-MMD-TinyGSM/ELF-B/elf/latent_stats.pt`;
no checked-in `stats/` file is needed. Existing local files take precedence;
prefix a path with `./` to explicitly select a local file. The `hf://` prefix
is also accepted.

Launch single-GPU training from the repository root:

```bash
bash scripts/launch.sh train src/configs/mmd/owt_t5.yml
```

Launch multi-GPU (single-host) training:

```bash
CUDA_VISIBLE_DEVICES=0,1 NGPU=2 bash scripts/launch.sh train src/configs/mmd/tinygsm_gpt2.yml
```

Any config field can be overridden from the command line (values are parsed as YAML):

```bash
NGPU=2 bash scripts/launch.sh train src/configs/ird/owt_t5.yml \
    --config_override lr=1e-4 \
    --config_override global_batch_size=128 \
    --config_override output_dir=outputs/owt_t5-ird-lr1e-4
```

Notes:

- **IRD starts from MMD.** The IRD presets initialize from the published MMD checkpoints:
  T5-OWT at 10,000, GPT-OWT at 9,000, and TinyGSM at 7,000 updates. Set
  `teacher_checkpoint` to use another checkpoint.
- **Batch size.** `global_batch_size` counts generated samples per optimizer update over
  all GPUs. Each GPU additionally loads `decoder_prob / (1 - decoder_prob)` times as many
  decoder rows (16 extra rows for 64 samples at `decoder_prob: 0.2`). TinyGSM MMD
  generates `mmd_batch_size` responses per prompt, so `global_batch_size` must be a
  multiple of `mmd_batch_size × number of GPUs` (64 × 2 = 128 in the preset).
- **Steps.** `max_iters`, `log_freq`, `save_freq` and `eval_freq` count optimizer updates.
- **Resume.** `--config_override resume=outputs/tinygsm_gpt2-mmd/checkpoint_5000`, with the
  same batch size, GPU count and recipe. The W&B run continues as well.
- **Seeds.** `seed: "42, 43, 44"`: the first seed drives training; every seed is used to
  sample at evaluation, and metrics are reported as the mean and std over seeds.
- **Read-only HF cache.** If the shared Hugging Face cache cannot be written, point
  `data_path` at the dataset's cached Arrow directory instead of the repo id.

**Wall-clock:** the TinyGSM MMD preset (10k steps, global batch 128) takes about 11 hours on
2× A100 80GB, including its 13 five-seed evaluations.

## Evaluation

Evaluation samples from a checkpoint with every setting in the config's sampling grid
and scores the samples: generative perplexity under GPT-2 Large and unigram entropy
for OpenWebText, and program-execution accuracy for TinyGSM. The same runs happen
during training every `eval_freq` steps, using the EMA weights.
Standalone evaluation uses the published `checkpoint_path` in the preset; pass
`--checkpoint_path` to evaluate a different local or Hub checkpoint.

```bash
# A post-trained generator, with the sampling grid of its training config
NGPU=8 bash scripts/launch.sh eval src/configs/mmd/tinygsm_gpt2.yml \
    --config_override output_dir=outputs/eval-tinygsm-mmd

# The pretrained T5 ELF-B with its original SDE sampler
NGPU=8 bash scripts/launch.sh eval src/configs/eval/elf_owt_t5.yml
```

ELF-M TinyGSM evaluation presets use the published EMA checkpoints, all 1,319
GSM8K examples, and seeds 42–46:

| Model | Evaluation config | Sampler |
| --- | --- | --- |
| ELF-M | [elf_m_tinygsm_gpt2.yml](src/configs/eval/elf_m_tinygsm_gpt2.yml) | Original ODE, CFG 2, time shift 32 |
| ELF-M-MMD (7k) | [elf_m_mmd_tinygsm_gpt2.yml](src/configs/eval/elf_m_mmd_tinygsm_gpt2.yml) | Iterative refinement with noise resampling |

```bash
bash scripts/launch.sh eval src/configs/eval/elf_m_tinygsm_gpt2.yml
bash scripts/launch.sh eval src/configs/eval/elf_m_mmd_tinygsm_gpt2.yml
```

Sampling grids live in `src/configs/sampling_configs/`:

| Grid | Sampler | Steps | Notes |
| --- | --- | --- | --- |
| [owt.yml](src/configs/sampling_configs/owt.yml) | iterative refinement | 2–32 | SC-CFG 3, fixed noise |
| [tinygsm.yml](src/configs/sampling_configs/tinygsm.yml) | iterative refinement | 1–64 | fresh noise every pass (`resample_z`) |
| [elf_owt.yml](src/configs/sampling_configs/elf_owt.yml) | original ELF SDE | 32, 64 | logit-normal time grid |
| [elf_tinygsm.yml](src/configs/sampling_configs/elf_tinygsm.yml) | original ELF ODE | 1–64 | CFG 2, shifted time grid |

The [eval configs](src/configs/eval) cover pretrained ELF models with their original
samplers and the published ELF-M-MMD model with iterative refinement.
To score a JSONL of generated OpenWebText afterwards (for example on
another GPU), run `python scripts/eval_ppl.py --input <run>/all_generated_*.jsonl`.

## Outputs and logging

Everything a run produces goes to `output_dir`:

| Path | Content |
| --- | --- |
| `config.yml` | The resolved configuration |
| `training.log` | Timestamped training and evaluation log (`tail -f` it) |
| `checkpoint_<step>` | Generator, EMA, optimizer, scheduler and RNG state |
| `<sampling setting>/all_generated_<epoch>_<step>.jsonl` | Generated samples |
| `<sampling setting>/metrics_summary.jsonl` | Metric mean and std over seeds, one line per evaluation |

With multiple seeds, samples and per-seed metrics go to `seed_<seed>/<sampling setting>/`.

**WandB.** Set `use_wandb: true` (and `wandb_project`, `wandb_entity`, `wandb_run_name`,
`wandb_tag` as needed). A run logs

- `train/*`: losses (`loss`, `mmd_loss` or `mse_loss`, `ce_loss`), `grad_norm`, `lr`,
  and `nfe`, averaged over each `log_freq` window and all GPUs;
- `perf/*`: steps and samples per second, peak GPU memory;
- `eval/<sampling setting>/<metric>` and `..._std`: the mean and std over seeds;
- `samples/<sampling setting>`: a table of generated samples.

All metrics use the optimizer step as x-axis. Standalone evaluation logs to its own run
(job type `eval`). Use `WANDB_MODE=offline` on machines without internet access.

## Configuration

Defaults live in [src/configs/config.py](src/configs/config.py); each YAML overrides
them. The post-training settings:

| Field | Meaning |
| --- | --- |
| `objective` | `mmd` or `ird` |
| `task` | `owt` (unconditional) or `tinygsm` (prompt-conditioned) |
| `teacher_checkpoint` | Initializes the generator and the frozen model |
| `no_bootstrap_prob`, `max_bootstrap_steps` | MMD: per step, 0 gradient-free self-conditioning passes with probability `no_bootstrap_prob`, else uniform in 1..`max_bootstrap_steps` |
| `feature_layer` | MMD: zero-based block of the frozen model whose outputs are compared |
| `mmd_batch_size` | MMD: samples per kernel, i.e. OWT sequences per group or TinyGSM responses per prompt |
| `sigma` | MMD: RBF bandwidth (required, positive) |
| `unbiased_rbf` | MMD: drop kernel pairs within one sample |
| `ird_steps` | IRD: refinement passes of the frozen model that produce the target |
| `decoder_prob`, `decoder_noise_scale` | Fraction of rows used for decoder training, and their noise level |
| `self_cond_cfg_min`, `self_cond_cfg_max` | Range of the self-conditioning CFG scale sampled in training |
| `late_eval_start`, `late_eval_freq` | Switch to a different evaluation interval late in training |
| `use_compile`, `use_bf16` | `torch.compile` for the training and eval models; BF16 autocast |

TF32 is used for training and switched off while sampling and scoring.

## Acknowledgements

This repository builds on the [PyTorch port of ELF](https://github.com/lillian039/ELF/tree/pytorch_elf);
the transformer, optimizer, samplers and evaluation code come from there.
