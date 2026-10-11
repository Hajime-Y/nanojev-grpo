# Predict Position: GRPO and SFT control

Run from the repository root on one CUDA GPU. Uses only local checkpoints and
the existing torch, transformers, safetensors and ViZDoom installation; no downloads.
`eval` batches live policy decisions across CPU environments. `grpo` learns from
binary episode outcomes. `sft` continues supervised training on PP policy rows
only, preserving their original question text. `cases` generates disjoint PP seeds.
All three model commands default to `--forward tree`: each question's shared
prefix is computed once, followed by candidate suffixes with separate causal
attention blocks and positions restarting after the prefix. `--forward paths`
selects the original per-candidate forward. The choice is saved in training
`config.json` under `args.forward` and in evaluation summaries.

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
Episode records also retain `model_decisions`, `fire_decision`, `fired` and `outcome`.

```bash
python scripts/pp_rl.py cases --output /tmp/pp-heldout.jsonl --count 512 \
  --seed-start 9400000 --split heldout \
  --exclude configs/sonic_predict_supervision_v1_cases.jsonl

python scripts/pp_rl.py eval --checkpoint /models/released \
  --cases configs/sonic_predict_supervision_v1_cases.jsonl --splits test \
  --mode greedy --workers 28 --output runs/pp-test.jsonl

python scripts/pp_rl.py check-forward --checkpoint /models/released \
  --data data/NanoJev-unified/unified/hard --count 64 --precision bf16

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
gradient norm limit `1`, and padded forward/microbatch budgets `140000`/`34000`.
These tree budgets target an A100 40 GB shared by two jobs: approximately 256
decisions / 11 GB for inference and 64 decisions / 13 GB for training with gradient
checkpointing. Actual memory depends on sequence lengths and model configuration.
For tree packing, sequence length is `len(shared prefix) + sum(len(suffix))`;
`questions in batch * longest tree sequence <= budget`. Paths packing retains
`total candidate paths * longest candidate path <= budget`. Both keep whole
questions together and reject an oversized question. Token statistics report
unpadded `tokens` and actual `padded_tokens` for the selected forward;
`leaf_paths` still counts the original candidate paths. Reduce budgets if needed.

Standalone and training game evaluation process at most `--max-concurrent 256`
episodes per chunk, with each episode's RNG independent of chunking. GRPO
collection still runs all `tasks-per-step * group-size` episodes together.
Environments are reused by spec.

`check-forward` selects the first `--count` PP policy rows (default 64) plus up to
16 non-PP rows from `DATA/dev.jsonl`, including all supported question types.
It compares both forwards in inference mode, one question at a time, and prints
one JSON line with `max_abs_dlogit`, `max_abs_dprob`, `argmax_agreement` (a fraction),
question/row counts, precision, and synchronized `paths_seconds` / `tree_seconds`.
Timings cover forward calls, including their first-call overhead, without loading
or tokenization. It exits nonzero for any argmax disagreement, probability
difference above `0.02`, or nonfinite result. FP32 is also available.

Training output:

```text
OUT/config.json                 arguments, initial weight hash, git HEAD, torch/GPU
OUT/heartbeat.json              latest phase, step, time and wave/update detail
OUT/steps.jsonl                 GRPO collection, update, token and memory statistics
OUT/updates.jsonl               SFT updates (instead of steps.jsonl)
OUT/eval.jsonl                  initial, periodic and final greedy + sample results
OUT/ckpt/step-XXXX/             periodic/final GRPO model (SFT: update-XXXX/)
OUT/ckpt/last/                  latest saved step/update, replaced via a temporary dir
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
Its summary records `forward`, `max_forward_tokens`, `rollouts`, `epsilon`,
`temperature`, `seed`, the resolved `cases` file path and `max_concurrent`.

Both trainers save an initial `ckpt/last`. GRPO rewrites it with weights and
optimizer state after every step. SFT rewrites it only at `--save-every` updates
and at the end, alongside the model snapshot, avoiding a full checkpoint write
on every update.

Resume with the same command plus `--resume`; only `--steps` (GRPO) or `--updates`
(SFT) may change. Task epochs shuffle with `seed + epoch`; the last GRPO block can
be shorter. SFT takes fixed-size batches across shuffled epoch boundaries. Tuple
seeds are encoded as compact JSON and hashed with SHA-256; episode RNGs are
independent of batching. Torch is reseeded per update. On resume, logs newer than
the last saved checkpoint are discarded and interrupted steps are replayed.
For SFT this also discards unsaved `updates.jsonl` entries and replays their updates.
Worker failures include the episode key and traceback; dead workers and timeouts
abort the pool. A transient `last.old` directory allows recovery during replacement.

CPU contract tests (torch-only numerical tests skip when torch is absent):
`python3 -m unittest scripts/test_pp_rl.py`. Real CUDA memory use, numerical
training behavior and ViZDoom throughput must be checked on the GPU host.

## Predict Position measurement harness

`scripts/pp_probe.py` measures explicit state reading separately from action
consequence prediction. It performs inference and deterministic replays only;
none of its commands trains or updates a model. Use the same cases and mode when
comparing unannotated and annotated play. These measurements provide evidence
about performance on these tasks; annotated success alone does not establish
that a model predicts consequences independently of the supplied simulation.

```bash
# Baseline policy and the same policy with code-written consequences.
python scripts/pp_probe.py play --engine checkpoint --checkpoint /models/released \
  --cases /tmp/pp-heldout.jsonl --splits heldout --mode greedy \
  --annotate none --workers 28 --max-concurrent 128 --output runs/pp-plain.jsonl
python scripts/pp_probe.py play --engine checkpoint --checkpoint /models/released \
  --cases /tmp/pp-heldout.jsonl --splits heldout --mode greedy \
  --annotate lookahead --workers 28 --output runs/pp-lookahead.jsonl

# Simulation-based controller: shoot on a hit; otherwise prefer noop, left, right
# if that action followed by shoot hits; otherwise noop.
python scripts/pp_probe.py play --engine oracle --annotate lookahead \
  --cases /tmp/pp-heldout.jsonl --splits heldout --output runs/pp-oracle.jsonl

# Collect states under policies that never choose shoot.
python scripts/pp_probe.py build --checkpoint /models/released \
  --cases /tmp/pp-probe-cases.jsonl --splits probe \
  --behaviors policy_noshoot,random_noshoot --stride 3 --seed 17 \
  --output runs/pp-probe.jsonl

python scripts/pp_probe.py score --engine checkpoint --checkpoint /models/released \
  --probe runs/pp-probe.jsonl --questions policy,boolean --output runs/pp-local-scores.jsonl
python scripts/pp_probe.py score --engine jev \
  --jev-url http://127.0.0.1:8791/v1/systemone --jev-model typesafe/jev-1.13 \
  --jev-concurrency 12 --probe runs/pp-probe.jsonl --output runs/pp-jev-scores.jsonl

python scripts/pp_probe.py report --probe runs/pp-probe.jsonl \
  --scores local=runs/pp-local-scores.jsonl jev=runs/pp-jev-scores.jsonl \
  --output runs/pp-probe-report.json
```

`play` reuses `pp_rl.run_episodes`, including decisions during the firing
animation and the noop drain after ammo reaches zero. It retains complete
decision states and candidate descriptions. With `--annotate lookahead`, each
candidate is replayed from reset using the live episode's exact action history.
For shoot, the suffix is shoot then noop to termination; for another action it
is that action, shoot, then noop to termination. Only candidate descriptions
change. Each decision saves the resulting `lookahead` Boolean map. Replays
use the existing worker pool and its free environments without disturbing live
episodes, even when replay keys overlap live keys. `EnvPool.replay` returns
success, terminal reason, total environment steps, and whether termination
occurred within the supplied prefix (including its last action).

`play --engine random` returns uniform policy probabilities and always samples,
including with `--mode greedy`; summaries distinguish `mode` from
`effective_mode`. Sampling uses the same per-episode stable seed as PP evaluation,
so batching does not change an episode's random choices. The oracle requires
lookahead annotations. Play summaries include engine, annotation mode, API calls,
input tokens, and the number of decisions with at least one simulated kill.
For unannotated play the latter is zero because no lookahead was measured.

`build` samples the full checkpoint policy renormalized over left/right/noop at
temperature 1, or a uniform distribution over those actions. Saved rows are
pre-action observations at t=0, stride, ...; their history excludes action t.
Their `hit_now` label comes from replaying that history, shoot, then noop to the
end. Features use only the visible non-DoomPlayer label in the newest snapshot;
missing monster centers and motion are JSON null. Side uses a 16-pixel center
band, while explicit Boolean gold uses an 8-pixel exclusion band for side and
a 2-pixel exclusion band for motion. Missing/ambiguous explicit gold is omitted
from the corresponding scoring request. The build summary contains row counts,
hits and hit rates per behavior.

`score` uses the original stored state and unannotated candidates. Checkpoint
Boolean questions use the native `boolean` preparation and the same tree/paths
forward as the policy. Jev receives one request per row containing its policy
question and all defined Boolean questions (sent as `noul`). Choice probabilities
are floored at 1e-6 and renormalized after taking logs to handle relay rounding;
Boolean probabilities are used directly. The relay needs no credentials. HTTP
failures, including 402, receive three client retries after the first attempt,
then raise a hard error. There is no model fallback.

Scoring appends and closes each completed row immediately, skipping IDs already
present on resume. A torn final JSONL line is removed before resuming; malformed
completed lines are errors. Successful in-flight rows are saved before a peer
request's error is raised. Resume with the same engine, checkpoint, question set,
and probe file; use a different output to change them. Score summary API counts
cover the current invocation, count retry attempts as calls, and count input
usage returned by successful responses. Play/build refuse existing output or
summary files. All three write `OUT.summary.json` including their configuration.

Reports contain each named scorer's per-behavior and pooled metrics. AUC uses
average ranks for tied probabilities. Policy argmax ties use lexical action order,
matching the evaluator; a turn-probability tie contributes one half. Boolean
accuracy predicts true at p >= 0.5. Undefined AUC (only one class), empty means,
and zero-denominator precision/recall are JSON null. Reports include total and
scored counts and both base rates so partial score files are visible.

Run offline contracts with:

```bash
python3 -m unittest scripts/test_pp_rl.py scripts/test_pp_probe.py
```

The probe tests use fake environments/scorers and a fake Jev HTTP server on a free
loopback port. If a sandbox denies binding localhost, those server integration
tests skip with the permission error; separate mocked-transport contracts still
check bundling, rounding, retries, hard failures and partial-result persistence.
Existing numerical tests skip when torch is absent.
