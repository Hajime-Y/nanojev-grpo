# Predict Position: GRPO and SFT control

Run from the repository root on one CUDA GPU. Uses only local checkpoints and
the existing torch, transformers, safetensors and ViZDoom installation; no downloads.
`eval` batches live policy decisions across CPU environments. `grpo` learns from
binary episode outcomes. `sft` continues supervised training on PP policy rows
only, preserving their original question text. `cases` generates disjoint PP seeds.

The policy is the softmax of candidate logits at temperature 1. GRPO assigns each
episode advantage `A = (reward - group_mean) / (population_std + 1e-8)`, dropping
constant-reward groups. Every collected decision receives its episode advantage.
For each minibatch the minimized objective is the mean of
`-min(exp(new_logp-old_logp)*A, clamp(exp(new_logp-old_logp),1-clip,1+clip)*A)`.
There is no reference model, KL penalty, value head or length normalization.
Collection temperature changes sampling only; both log probabilities in the ratio
are always at temperature 1. SFT minimizes mean gold-action cross entropy.

Each wave sends exactly `policy_request(obs, episode_key)`. The rocket takes time
to leave: decisions continue while ammo is positive, then the worker drains with
`noop` until terminal. Only terminal `info.success` supplies reward. Saved `probs`
are the pure policy probabilities at temperature 1; sampling applies temperature
and then uniform epsilon mixing. Greedy ties choose the first action id in lexical
order. `fire_decision` is a zero-based decision index; `fired` requires observed
ammo zero. `env_steps` includes drained steps. `--save-decisions` retains states
and candidate descriptions; action, logp and probs are always retained.

```bash
python scripts/pp_rl.py cases --output /tmp/pp-heldout.jsonl --count 512 \
  --seed-start 9400000 --split heldout \
  --exclude configs/sonic_predict_supervision_v1_cases.jsonl

python scripts/pp_rl.py eval --checkpoint /models/released \
  --cases configs/sonic_predict_supervision_v1_cases.jsonl --splits test \
  --mode greedy --workers 28 --output runs/pp-test.jsonl

python scripts/pp_rl.py grpo --init-checkpoint /models/released \
  --cases configs/sonic_predict_supervision_v1_cases.jsonl --output-dir runs/pp-grpo \
  --steps 50 --tasks-per-step 16 --group-size 16 --minibatches 8 --workers 28

python scripts/pp_rl.py sft --init-checkpoint /models/released \
  --data data/NanoJev-unified/unified/hard \
  --cases configs/sonic_predict_supervision_v1_cases.jsonl --output-dir runs/pp-sft \
  --updates 2400 --batch-questions 64 --eval-every 200 --save-every 200 --workers 28
```

Both trainers default to BF16 autocast, FP32 parameters/AdamW state, gradient
checkpointing, backbone/head learning rates `1e-5`/`1e-4`, weight decay `.01`,
gradient norm limit `1`, and padded forward/microbatch budgets `262144`/`32768`.
Packing keeps all candidate paths of a question together and rejects oversized
questions. Reduce token budgets if needed. Environments are reused by spec.

Training output:

```text
OUT/config.json                 arguments, initial weight hash, git HEAD, torch/GPU
OUT/heartbeat.json              latest phase, step, time and wave/update detail
OUT/steps.jsonl                 GRPO collection, update, token and memory statistics
OUT/updates.jsonl               SFT updates (instead of steps.jsonl)
OUT/eval.jsonl                  initial, periodic and final greedy + sample results
OUT/ckpt/step-XXXX/             periodic/final GRPO model (SFT: update-XXXX/)
OUT/ckpt/last/                  latest completed step, replaced via a temporary dir
  best.safetensors
  config.json                   original predictor configuration
  tokenizer/
  backbone_config/
  state.json                    completed_step and arguments (also in snapshots)
  optimizer.pt                  last checkpoint only
```

Evaluation uses the same seed/key per case across training steps. SFT evaluation
also records PP dev CE and argmax agreement. Standalone evaluation writes
`OUT.jsonl` and `OUT.summary.json` and refuses existing destinations.

Resume with the same command plus `--resume`; only `--steps` (GRPO) or `--updates`
(SFT) may change. Task epochs shuffle with `seed + epoch`; the last GRPO block can
be shorter. SFT takes fixed-size batches across shuffled epoch boundaries. Tuple
seeds are encoded as compact JSON and hashed with SHA-256; episode RNGs are
independent of batching. Torch is reseeded per update. On resume, logs newer than
the last completed checkpoint are discarded and interrupted steps are replayed.
Worker failures include the episode key and traceback; dead workers and timeouts
abort the pool. A transient `last.old` directory allows recovery during replacement.

CPU contract tests (torch-only numerical tests skip when torch is absent):
`python3 -m unittest scripts/test_pp_rl.py`. Real CUDA memory use, numerical
training behavior and ViZDoom throughput must be checked on the GPU host.
