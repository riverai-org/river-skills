---
name: river-client-training
description: Write training scripts with the river-client Python package — LoRA fine-tuning, SFT, and RL/GRPO on River-hosted models via the River training API. Use when writing or reviewing code that imports river_client, builds training data for forward_backward / train_step, samples from training weights, or wires up a training loop. Centers on the current train_step API and its pipelining/error semantics; also covers multimodal (image) data, session tags, sampling and prompt logprobs and their uses, MoE expert-routing capture and replay, teacher→student distillation with two models, and fault-tolerant loops that auto-recover from session loss, capacity, and timeout errors.
---

# Train with river_client (the current API)

`river_client` is the Python client for River's training API: create a LoRA
model on remote GPU workers inside a session, push training data through
`train_step`, sample from the live weights, checkpoint. This skill is the
short path to writing a correct script with the **current** API (`train_step`
era, package ≥ 0.5).

## Setup and connection

```bash
pip install river-client        # PyPI, Python 3.12+
```

```python
import os

import river_client as river

client = river.Client(api_key=os.environ["RIVER_API_KEY"], endpoint="api.river.ai")
```

Everything happens inside a session (the GPU allocation) and a model:

```python
with client.session() as session:
    model = session.create_model(
        base_model="Qwen/Qwen3.6-35B-A3B-FP8",
        lora=river.LoraConfig(rank=16, train_unembed=True),  # see note below
    )
    ...
# GPUs freed automatically on exit.
```

Any keyword arguments to `client.session(...)` become **session tags** —
arbitrary string key→value metadata stamped on the session:

```python
with client.session(experiment="grpo-math", run="lr4e-5-r16") as session:
    ...
```

Tags don't change behavior; they exist so runs can be found later: the River
Console can filter training runs by tag, and tags travel with the run for
correlating against an external experiment tracker (a common convention is
`client.session(wandb_project=..., wandb_name=...)`). Tag long-running
experiments — an untagged session is hard to tell apart from every other one
once you have dozens. Key/value counts and lengths are bounded server-side,
so keep them short labels, not payloads.

`LoraConfig` knobs: `rank` (max 32), `train_attn` / `train_mlp` (both default
on), `train_unembed` (default off), `seed` (reproducible adapter init).

**Why `train_unembed=True` for RL.** The unembedding (`lm_head`) is the final
hidden-state → vocab-logits matrix; this flag adds a LoRA adapter on it. With
the head frozen, a policy update can only change token probabilities
indirectly, by bending hidden states through the trunk adapters. The RL losses
(`importance_sampling` / `ppo` / `cispo`) are exactly "push probability toward
or away from these sampled tokens, weighted by advantage" — a gradient that
lands first on the logits — and RL's signal is one scalar reward per sequence,
so you want it expressible as directly as possible. A head adapter gives that
direct lever (suppress premature EOS, boost format tokens, sharpen the
distribution) without spending trunk capacity — and without it, updates burn
more KL (`kl`, `mean_ratio` drift) for less reward gain. For SFT the dense
per-token cross-entropy signal usually makes trunk adapters sufficient, which
is why the flag defaults off; turn it on for RL loops.

## train_step — the default way to take a step

`train_step` is the current, preferred call for a complete training step. It
submits forward+backward **and** the optimizer step back-to-back, so the server
pipelines them without a client round trip in between:

```python
fb_result, optim_result = model.train_step(
    data,                      # list[dict] — see data format below
    lr=1e-4,
    loss_fn="cross_entropy",   # or importance_sampling / ppo / cispo
    grad_clip_norm=1.0,        # optional; also beta1/beta2/eps/weight_decay
)
print(fb_result.metrics["loss_mean"], optim_result.metrics.get("grad_norm"))
```

Semantics you must know (they differ from calling the two ops separately):

- **Both ops are submitted before waiting.** If forward-backward fails, the
  already-queued optimizer step still runs server-side and `model.step` has
  already advanced. `train_step` raises the forward-backward error; use
  `submit_train_step` when you need to inspect both outcomes.
- **A train step always clears gradients first.** It rejects `zero_out` /
  `return_logprobs` kwargs. For micro-batch gradient accumulation, drop down to
  the primitive ops:

  ```python
  p1 = model.submit_forward_backward(micro1, loss_fn="cross_entropy", zero_out=True)
  p2 = model.submit_forward_backward(micro2, loss_fn="cross_entropy", zero_out=False)
  p1.result(); p2.result()
  model.optim_step(lr=1e-4, grad_clip_norm=1.0)
  ```

- **`model.step` advances at submit time**, not on success — a failed or
  timed-out optimizer step leaves it advanced.
- Extra keyword floats (e.g. `eps_max=6.0`, `clip_low=0.2`) pass through as
  loss-function config.

Non-blocking variant for pipelined loops (returns two `PendingOp`s; resolve
with `.result()` whenever you need them):

```python
pending_fb, pending_opt = model.submit_train_step(data, lr=4e-5, loss_fn="cispo", eps_max=6.0)
# ... prepare the next batch / kick off next rollouts here ...
fb_result = pending_fb.result()
optim_result = pending_opt.result()
```

## Training data format

Data is plain dicts. Every loss-aligned field has length `T = len(input_ids)`
and is indexed by **prediction position** — index `i` describes the prediction
made *from* position `i`:

| field | meaning at index `i` |
|---|---|
| `input_ids[i]` | the i-th token |
| `target_tokens[i]` | token to predict from position `i` |
| `weights[i]` | weight of that prediction (`cross_entropy`) |
| `old_logprobs[i]` | sampling logprob of that prediction (RL) |
| `advantages[i]` | advantage of that prediction (RL) |
| `attention_mask[i]` | 1 if `input_ids[i]` is real, 0 if padding |

Two universal rules:

1. **`target_tokens` is optional.** Omitted → server fills next-token targets
   (`input_ids` shifted left by one, trailing slot zero-filled, no wraparound).
   Only pass it explicitly for non-standard targets (e.g. distillation).
2. **The last position is always ignored.** The server force-zeros
   `weights[T-1]` / `advantages[T-1]`; there is no next-token target there.

For chat SFT data, don't hand-roll the template — use the package renderers,
which tokenize, apply the chat template, and mask non-trainable positions
(headers weight 0, trained assistant content weight 1):

```python
from river_client.renderers import get_renderer, TrainOnWhat

renderer = get_renderer("Qwen/Qwen3.6-35B-A3B-FP8")   # Qwen3.5/3.6 + Kimi K2.x
example = renderer.build_training_example(messages, train_on=TrainOnWhat.LAST_ASSISTANT)
datum = example.to_dict()                              # ready for train_step
```

`to_dict()` has two defaults worth knowing. `normalize_weights=True` rescales
each example's weights to sum to 1.0 — per-example token-mean, so batch loss
scales with the number of examples but **not** with sequence length; pass
`normalize_weights=False` if you want the raw SUM regime described under Loss
functions. `shift_weights_for_pre_shift_loss=True` converts the renderer's
completion-position weights to River's prediction-position wire contract —
leave it on, or every completion token's signal lands one position late.

## Images (multimodal)

Served model families are vision-capable (every Qwen3.5/3.6 checkpoint is a
vision-language model; Kimi K2.x likewise). Images are raw PNG or JPEG bytes —
the format is inferred from the bytes' magic header. Let the renderer do the
placeholder bookkeeping: each image needs exactly one un-expanded placeholder
token. The renderer and `model_input` paths validate this client-side and
raise a precise `ValueError` naming the counts; only the raw
`prompt_token_ids=` + `images=` path can carry a mismatch to the backend,
where surplus images are silently dropped and surplus placeholders fail deep
in the engine.

Build messages with `image_part` mixed into the content list. Passing
`height`/`width` explicitly keeps the client Pillow-free (otherwise PIL is
imported lazily to read them):

```python
from river_client.renderers import get_renderer, image_part

renderer = get_renderer(BASE_MODEL)
messages = [
    {"role": "user", "content": [
        image_part(png_bytes, format="png", height=224, width=224),
        {"type": "text", "text": "What color is the circle? One word."},
    ]},
    {"role": "assistant", "content": "red"},
]

# Training: the chunked wire form is emitted automatically; image placeholder
# slots are already in place with weight 0.
datum = renderer.build_training_example(messages).to_dict()
fb, opt = model.train_step([datum], lr=1e-4, loss_fn="cross_entropy")

# Sampling: drop the assistant target turn; to_kwargs() emits {"prompt", "images"}.
sp = renderer.build_sample_prompt(messages[:-1])
groups = model.sample(**sp.to_kwargs(), num_samples=4, max_tokens=32,
                      stop=renderer.get_stop_strings())
```

Notes:

- `SamplePrompt.to_kwargs()` emits `prompt` (singular). To batch several
  multimodal prompts in one `sample` call, pass `prompts=[...]` plus
  per-prompt images as `images=list[list[bytes]]` (a flat `list[bytes]`
  broadcasts the same images to every prompt).
- `model.sample(model_input=...)` also accepts the training-style chunk list
  (`[{"type": "text", "tokens": [...]}, {"type": "image", "data": ...}, ...]`)
  for exact prompt-token control in RL-style multimodal loops.

## Example 1 — minimal SFT loop

```python
import os
import river_client as river
from river_client.renderers import get_renderer, TrainOnWhat

BASE_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

client = river.Client(api_key=os.environ["RIVER_API_KEY"], endpoint="api.river.ai")
renderer = get_renderer(BASE_MODEL)

with client.session() as session:
    model = session.create_model(base_model=BASE_MODEL, lora=river.LoraConfig(rank=16))

    for step, batch in enumerate(batches):   # batch: list of message lists
        data = [
            renderer.build_training_example(
                messages, train_on=TrainOnWhat.LAST_ASSISTANT
            ).to_dict()
            for messages in batch
        ]
        fb, opt = model.train_step(data, lr=1e-4, loss_fn="cross_entropy",
                                   grad_clip_norm=1.0)
        print(f"step {model.step}  loss_mean={fb.metrics['loss_mean']:.4f}  "
              f"grad_norm={opt.metrics.get('grad_norm')}")

        if step % 50 == 0 and step > 0:
            ckpt = model.save_weights(f"step_{step:06d}")   # includes optimizer state
```

## Example 2 — GRPO-style RL loop

Sample from the live training weights (`model.sample` routes through the
inference pool — no separate inference session), score, build pre-shifted RL
datums, `train_step` with `importance_sampling`:

```python
import os
import river_client as river
from river_client.renderers import get_renderer

BASE_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
GROUP_SIZE = 16

client = river.Client(api_key=os.environ["RIVER_API_KEY"], endpoint="api.river.ai")
tokenizer = river.load_tokenizer(base_model=BASE_MODEL)
renderer = get_renderer(BASE_MODEL, tokenizer=tokenizer)

with client.session() as session:
    model = session.create_model(
        base_model=BASE_MODEL,
        lora=river.LoraConfig(rank=16, train_unembed=True),
    )

    for batch_idx, rows in enumerate(batches):   # rows: [{"question", "answer"}, ...]
        # 1. Render + tokenize prompts client-side, sample with prompt_token_ids
        #    so the datum's prompt tokens exactly match what the sampler ran.
        prompt_token_lists = [
            tokenizer.encode(
                renderer.build_prompt_str(
                    [{"role": "user", "content": row["question"]}]),
                # The rendered template already carries its special tokens —
                # the default add_special_tokens=True would prepend a
                # duplicate BOS on tokenizers that define one, malforming
                # the prompt and shifting ob_len by one.
                add_special_tokens=False,
            )
            for row in rows
        ]
        sample_groups = model.sample(
            prompt_token_ids=prompt_token_lists,
            num_samples=GROUP_SIZE,
            max_tokens=512,
            stop=renderer.get_stop_strings(),
        )   # list[list[Sample]] — Sample has .tokens/.text/.logprobs/.stop_reason

        # 2. Score and build training data
        train_data = []
        for prompt_tokens, samples, row in zip(prompt_token_lists, sample_groups, rows):
            rewards = [score(s.text, row["answer"]) for s in samples]
            mean_r = sum(rewards) / len(rewards)
            if all(r == mean_r for r in rewards):
                continue   # zero advantage everywhere → zero gradient, skip

            for sample, reward in zip(samples, rewards):
                advantage = reward - mean_r
                ob_len = len(prompt_tokens)
                full_ids = prompt_tokens + sample.tokens
                # Pre-shifted: the last prompt position predicts the first
                # response token, so response signal starts at ob_len-1;
                # the trailing 0.0 is the always-masked no-next-token slot.
                # Requires ob_len >= 1 (always true for a rendered chat
                # template) — at ob_len == 0 the padding clamps to [] and
                # these arrays end up one longer than input_ids.
                train_data.append({
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "old_logprobs": [0.0] * (ob_len - 1) + sample.logprobs + [0.0],
                    "advantages": ([0.0] * (ob_len - 1)
                                   + [advantage] * len(sample.tokens) + [0.0]),
                })

        # 3. One pipelined step
        if train_data:
            fb, opt = model.train_step(train_data, lr=4e-5,
                                       loss_fn="importance_sampling")
            print(f"step {model.step}  kl={fb.metrics['kl']:.4f}  "
                  f"mean_ratio={fb.metrics['mean_ratio']:.3f}")

        if batch_idx % 20 == 0 and batch_idx > 0:
            model.save_weights(f"step_{batch_idx:06d}")
```

## Example 3 — pipelined steps with submit_train_step

Overlap training with data preparation (or the next rollout phase) and inspect
both outcomes independently:

```python
pending = None
for batch in batches:
    data = build_data(batch)          # overlaps with the in-flight step
    if pending is not None:
        pending_fb, pending_opt = pending
        try:
            fb = pending_fb.result()
            print(f"step {model.step}  loss_mean={fb.metrics['loss_mean']:.4f}")
        except river.RiverError as e:
            # The optimizer step was already submitted and may still have
            # applied — resume from the last checkpoint rather than guessing.
            print(f"forward_backward failed: {e}")
        try:
            pending_opt.result()
        except river.RiverError as e:
            print(f"optim_step failed: {e}")
    pending = model.submit_train_step(data, lr=1e-4, loss_fn="cross_entropy")

if pending is not None:
    pending_fb, pending_opt = pending
    pending_fb.result()
    pending_opt.result()
```

## Example 4 — two models: distill a big teacher into a small student

Only the *student* needs a training model. The teacher is just sampled:
`session.sample(..., base_model=TEACHER)` (or stateless `client.sample`) hits
the inference pool directly — no second `create_model`, no extra training
allocation. One hard requirement: for token-level distillation the teacher
and student must **share a tokenizer/vocab** (typically a larger and smaller
checkpoint of the same model family) — token ids from a foreign vocab train
garbage without any error.

Teacher-generates flavor: sample from the teacher with `logprobs=K` and train
the student on the teacher's **top-K distribution** via the 2-D
`cross_entropy` targets from the loss table:

```python
import math
import os

import river_client as river
from river_client.renderers import get_renderer

TEACHER = "<big-checkpoint>"    # generation only — no training session
STUDENT = "<small-checkpoint>"  # must share the teacher's tokenizer/vocab
K = 8

client = river.Client(api_key=os.environ["RIVER_API_KEY"], endpoint="api.river.ai")
tokenizer = river.load_tokenizer(base_model=STUDENT)
renderer = get_renderer(STUDENT, tokenizer=tokenizer)

with client.session(experiment="distill-teacher-student") as session:
    student = session.create_model(base_model=STUDENT, lora=river.LoraConfig(rank=16))

    for step, questions in enumerate(batches):
        prompt_token_lists = [
            tokenizer.encode(
                renderer.build_prompt_str([{"role": "user", "content": q}]),
                add_special_tokens=False)
            for q in questions
        ]

        # 1. Teacher generates, carrying its top-K distribution per position.
        teacher_groups = session.sample(
            prompt_token_ids=prompt_token_lists,
            base_model=TEACHER,
            max_tokens=512,
            logprobs=K,
            stop=renderer.get_stop_strings(),
        )

        # 2. Soft-target datums: [T, K] target_tokens/weights. Row p-1+j
        #    holds the teacher's top-K for response token j (prediction-
        #    position indexing, as everywhere else). Requires p >= 1
        #    (always true for a rendered chat template): at p == 0 the
        #    p-1+j index wraps to the last row and silently shifts every
        #    teacher target one position early.
        data = []
        for prompt_tokens, group in zip(prompt_token_lists, teacher_groups):
            t = group[0]
            full_ids = prompt_tokens + t.tokens
            T, p = len(full_ids), len(prompt_tokens)
            target_tokens = [[0] * K for _ in range(T)]
            weights = [[0.0] * K for _ in range(T)]
            n_scored = len(t.top_logprobs)
            for j, cands in enumerate(t.top_logprobs):
                probs = [math.exp(c.logprob) for c in cands]
                z = sum(probs) * n_scored     # renormalize over K, mean-style
                for k, c in enumerate(cands):
                    target_tokens[p - 1 + j][k] = c.token_id
                    weights[p - 1 + j][k] = probs[k] / z
            data.append({
                "input_ids": full_ids,
                "attention_mask": [1] * T,
                "target_tokens": target_tokens,   # [T, K] int
                "weights": weights,               # [T, K] float
            })

        fb, opt = student.train_step(data, lr=1e-4, loss_fn="cross_entropy")
        print(f"step {student.step}  loss_mean={fb.metrics['loss_mean']:.4f}")
```

For plain hard-target KD, drop `logprobs=K` and build Example-1-style flat
data from the teacher's `t.tokens` (weight the response predictions,
positions `p-1 … T-2`).

**On-policy distillation** — the stronger variant when the student must be
good on *its own* trajectories: sample from the **student**
(`student.sample(...)`, as in Example 2), score those exact tokens under the
teacher with the prompt-logprob trick, and use the per-token gap as the
advantage in an `importance_sampling` step:

```python
full_ids = prompt_tokens + sample.tokens
scored = session.sample(
    prompt_token_ids=[full_ids],
    base_model=TEACHER,
    max_tokens=1,                  # must generate ≥ 1 token; we only want
    return_prompt_logprobs=True,   # the prompt-position scores
)
teacher_lp = scored[0][0].prompt_logprobs[len(prompt_tokens):len(full_ids)]
advantages = [KL_COEF * (t_lp - s_lp)          # push where teacher > student
              for s_lp, t_lp in zip(sample.logprobs, teacher_lp, strict=True)]

# Example-2 layout, but the advantage slot takes this per-token VECTOR —
# concatenate it in place of Example 2's broadcast scalar:
ob_len = len(prompt_tokens)
datum = {
    "input_ids": full_ids,
    "attention_mask": [1] * len(full_ids),
    "old_logprobs": [0.0] * (ob_len - 1) + sample.logprobs + [0.0],
    "advantages": [0.0] * (ob_len - 1) + advantages + [0.0],
}
# Collect datums across samples, then train with loss_fn="importance_sampling".
```

This is per-token reverse-KL minimization against the teacher — no reward
function needed — and it composes with everything above (batch the scoring
calls, run the loop with `train_step`, wrap it in the auto-recovery
skeleton).

## Logprobs — what comes back and what it's for

Every `Sample` always carries `.logprobs` — the sampling policy's per-token
log probabilities, aligned with `.tokens`. Two opt-ins add more:

```python
groups = model.sample(
    prompts,
    return_prompt_logprobs=True,   # also score every prompt position
    logprobs=5,                    # top-K alternatives at each position
)
s = groups[0][0]
s.logprobs             # sampled-token logprobs (aligned with s.tokens)
s.prompt_logprobs      # per-token logprobs over the prompt
s.prompt_token_ids     # the server's exact prompt tokenization, echoed back
s.top_logprobs         # per position: list of TopLogprob(logprob, token_id, token)
s.prompt_top_logprobs  # same, for prompt positions
s.model_step           # training step the weights had when this was sampled
```

On the training side, `ForwardResult.logprobs` holds per-datum arrays of
per-token logprobs under the *trainer's* current weights (for top-K
cross-entropy: the weight-normalized expected logprob per position). And
`model.forward(data, loss_fn=...)` runs a forward-only pass — loss and
logprobs with no gradient update — when you want pure scoring.

What each is for:

- **`Sample.logprobs` → RL `old_logprobs`.** The denominator of the
  importance-sampling ratio. Use the sampler's values verbatim — recomputing
  them client-side or with a different tokenization silently corrupts the
  ratio.
- **Sampler↔trainer KL diagnostics.** Compare `Sample.logprobs` against
  `ForwardResult.logprobs` for the same response tokens (a low-variance
  estimator like k3 works well). Nonzero divergence at the *first* step of a
  batch means your datums are misaligned (tokenization skew, wrong offsets);
  growing divergence across a pipelined loop measures how off-policy your
  samples have drifted.
- **`prompt_logprobs` → scoring fixed text.** Likelihood/perplexity of a
  candidate answer under the current policy: rerank or best-of-n without a
  reward model, filter training data by model likelihood, or A/B a checkpoint
  against base weights on held-out text (`session.sample(...,
  checkpoint=...)` returns the same fields).
- **`prompt_token_ids` → exact alignment.** When you sampled with `prompts`
  strings, the server's tokenization may differ from your client-side one;
  build RL datums from the echoed ids, not from re-encoding the text.
- **`top_logprobs` → distillation targets.** The teacher's top-K ids and
  probabilities per position feed directly into the 2-D top-K
  `cross_entropy` (`target_tokens`/`weights` of shape `[T, K]`) for
  soft-target distillation.
- **`model_step` → off-policy bookkeeping.** In pipelined RL, records how
  many optimizer steps behind the sampler was when it generated the batch.

Cost: both opt-ins are off by default for a reason. `logprobs=K` roughly
halves sampler throughput; `return_prompt_logprobs=True` materializes
full-vocab logits across the whole prompt during prefill, which is expensive
on long prompts. Request them only when the loop actually consumes them.

## Expert routing (MoE sampler↔trainer fidelity)

The served model families are Mixture-of-Experts: a router picks top-k
experts per token per layer. Sampling and training run on different stacks
(inference engine vs trainer — different kernels, batch shapes, numerics), so
near-tie router decisions can *flip* between them. When that happens in RL,
the trainer computes gradients through a different expert mixture than the
one that actually generated the sample — an MoE-specific source of
sampler↔trainer divergence on top of tokenization and precision. The client
gives you one tool to measure it and one to eliminate it:

```python
# 1. Capture routing while sampling.
sample_groups = model.sample(
    prompt_token_ids=prompt_token_lists,
    num_samples=GROUP_SIZE,
    max_tokens=512,
    return_expert_routing=True,
)

# 2. Splat the routing handle into every datum.
train_data = []
for prompt_tokens, samples in zip(prompt_token_lists, sample_groups):
    for sample in samples:
        train_data.append({
            "input_ids": prompt_tokens + sample.tokens,
            # ... old_logprobs / advantages exactly as in Example 2 ...
            **sample.routing_datum_keys(required=True),
        })

# 3. Replay and/or measure during the training step.
fb, opt = model.train_step(
    train_data, lr=4e-5, loss_fn="importance_sampling",
    force_routing_replay=True,        # gradients flow through the sampled experts
    compute_expert_flip_metric=True,  # measure routing disagreement
)
print(fb.metrics["expert_flip/per_token_expert_rate"])   # ∈ [0, 1]
```

- **`compute_expert_flip_metric=True`** (diagnostic, independent of replay)
  compares the sampled routing against the trainer's own routing per token
  and MoE layer, and emits `expert_flip/per_token_expert_rate` — the fraction
  of individual top-k expert slots that differ.
- **`force_routing_replay=True`** (the fix) makes the trainer replay the
  sampled expert *selection* while recomputing the routing weights at those
  experts with its live gate — the gradient flows through the same experts
  that produced the sample.
- Both require the routing keys on **every** datum in the batch —
  `sample.routing_datum_keys(required=True)` raises a clear error when a
  sample carries no capture (non-MoE model, or capture unavailable).
- The capture is an opaque server-minted `handle` on
  `Sample.expert_routing`; round-trip it via `routing_datum_keys()` and don't
  parse the `topk_ids` bytes (inspection only).
- Bonus: with routing capture enabled, the sampler's exact
  `prompt_token_ids` ride along on the wire — you get exact-alignment prompt
  ids without paying the `return_prompt_logprobs` prefill cost.

## Loss functions and learning rates

All losses are **SUM-reduced over tokens** — loss magnitude scales with batch
size and sequence length, so use lower LRs than with mean-reduced losses:

| loss_fn | use for | config kwargs | starting LR |
|---|---|---|---|
| `cross_entropy` | SFT; also top-K soft targets via 2-D `target_tokens`/`weights` `[T, K]` | — | `1e-4` |
| `importance_sampling` | on-policy single-epoch RL (GRPO-style), unclipped | — | `4e-5` |
| `ppo` | multi-epoch on the same batch, two-sided clip | `clip_low`, `clip_high` | `1e-5`–`4e-5` |
| `cispo` | stale/off-policy samples, upper-only stop-gradient clip | `eps_max` (default 6.0) | `1e-5`–`4e-5` |

RL metrics come back on `ForwardResult.metrics`: raw sums plus derived
`mean_ratio`, `kl`, `entropy` (and `clip_frac` / `truncation_frac` for
ppo/cispo). `cross_entropy` returns `loss`, `loss_sum`, `loss_mean`,
`num_tokens`, `weight_sum`.

## Checkpointing

```python
ckpt = model.save_weights("my_ckpt")                    # training mode: weights + optimizer
model.load_weights(ckpt.path, load_optimizer=True)      # or pass checkpoint= to create_model
sampler_ckpt = model.save_weights("for_infer", mode="inference")   # PEFT, no optimizer
```

`session.create_model(..., checkpoint=ckpt)` restores step and optimizer state
automatically when given a `Checkpoint` object.

## Auto-recovery — write loops that survive failures

Long training runs **will** hit transient failures: lost sessions, capacity
squeezes, connection blips, slow steps that exceed a timeout. A script that
just crashes throws away everything since its last checkpoint. Always
structure training loops to recover and continue automatically.

The durability model is simple: **the only durable state is checkpoints.**
A session is a GPU allocation and model weights live in worker memory — when
the session dies, in-memory state is gone. So recovery means: checkpoint on a
cadence, persist your loop progress next to the checkpoint path, and on
failure rebuild the session and resume from the last checkpoint.

Exception taxonomy (all top-level exports, all subclasses of `river.RiverError`):

| exception | meaning | on catch |
|---|---|---|
| `RiverConnectionError` | gRPC/connection failure — **including capacity squeezes**, which arrive as this with a "Server capacity exceeded" message (the exported `CapacityError` is currently never raised by the client) | back off, rebuild session, resume from checkpoint |
| `SessionHeartbeatError` | session lost (heartbeat rejected). A *subclass* of `RiverConnectionError` — an `except RiverConnectionError` clause listed first will swallow it | rebuild session, resume from checkpoint |
| `RiverTimeoutError` | op timed out; carries `.request_id`, the server-side future **stays retrievable** | if you hold the `PendingOp`, call `.result()` again to keep waiting; otherwise resume from checkpoint |
| `AuthenticationError`, `ModelNotFoundError` | config/auth bug | **fail fast — never retry** |
| other `RiverError` (e.g. rejected datum) | usually a data bug | log + skip or fix; blind-retrying the same batch fails the same way |

The canonical resumable-loop skeleton:

```python
import json, os, time
import river_client as river

PROGRESS_FILE = "progress.json"
CKPT_EVERY = 20

def load_progress() -> tuple[int, river.Checkpoint | None]:
    if not os.path.exists(PROGRESS_FILE):
        return 0, None
    with open(PROGRESS_FILE) as f:
        p = json.load(f)
    # Rebuild the Checkpoint object: create_model only restores model.step
    # from a Checkpoint — a bare path string restores weights + optimizer
    # but restarts the step counter at 0.
    ckpt = river.Checkpoint(path=p["ckpt_path"], step=p["ckpt_step"],
                            checkpoint_type="training")
    return p["next_batch"], ckpt

def save_progress(next_batch: int, ckpt: river.Checkpoint) -> None:
    tmp = PROGRESS_FILE + ".tmp"        # write-then-rename: survive a crash mid-write
    with open(tmp, "w") as f:
        json.dump({"next_batch": next_batch, "ckpt_path": ckpt.path,
                   "ckpt_step": ckpt.step}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, PROGRESS_FILE)

def train(client, start_batch: int, ckpt: river.Checkpoint | None) -> None:
    with client.session() as session:
        model = session.create_model(
            base_model=BASE_MODEL,
            lora=river.LoraConfig(rank=16),
            checkpoint=ckpt,           # None on first run; a Checkpoint object
        )                              # restores weights + optimizer + step
        for batch_idx in range(start_batch, num_batches):
            data = build_batch(batch_idx)
            model.train_step(data, lr=1e-4, loss_fn="cross_entropy")
            if (batch_idx + 1) % CKPT_EVERY == 0:
                ckpt = model.save_weights(f"step_{batch_idx + 1:06d}")
                save_progress(batch_idx + 1, ckpt)
        model.save_weights("final")

backoff = 10.0
while True:
    start_batch, ckpt = load_progress()
    try:
        train(client, start_batch, ckpt)
        break
    except (river.RiverConnectionError, river.RiverTimeoutError) as e:
        # RiverConnectionError covers SessionHeartbeatError (subclass) and
        # capacity squeezes ("Server capacity exceeded").
        print(f"transient failure ({type(e).__name__}): {e} — "
              f"resuming from batch {load_progress()[0]} after {backoff:.0f}s")
        time.sleep(backoff)
        backoff = min(backoff * 2, 300.0)
    # AuthenticationError / ModelNotFoundError / data errors propagate: fix, don't retry.
```

Rules that make this correct:

- **Checkpoint and progress move together.** Write the progress record only
  after `save_weights` returns; name checkpoints by step (`step_%06d`) so a
  resumed run overwrites cleanly instead of forking history.
- **Resume from the checkpoint, not from memory.** After any mid-step error,
  don't try to reason about half-applied state (remember: a failed
  forward-backward inside `train_step` does not cancel the already-queued
  optimizer step). Rebuilding from the last checkpoint is always consistent —
  at worst you redo up to `CKPT_EVERY` batches.
- **A timeout is not proof of lost work.** `RiverTimeoutError` keeps its
  `request_id` and `PendingOp.result()` can simply be called again — for a
  slow-but-healthy step, re-waiting is cheaper than a full session rebuild.
  This only applies where you hold the `PendingOp` (pipelined loops like
  Example 3); the blocking `train_step` in the skeleton above doesn't hand
  the pending ops back, so a timeout there falls through to the
  (always-safe) checkpoint rebuild.
- **Reset the backoff after progress.** If the run trains for a while before
  failing again, treat it as a fresh incident (the skeleton above resets
  naturally on `break`; add `backoff = 10.0` after a successful checkpoint if
  runs are very long).
- **Bound nothing silently.** If you cap retries, log loudly what was
  abandoned and leave the progress file intact so a manual restart continues
  where it stopped.
- In RL loops, the same applies to the sampling phase: `model.sample` raising
  a transient error should re-enter the same batch after the session rebuild,
  not skip it.

## Pitfalls checklist

- **Don't** call `forward_backward` + `optim_step` sequentially in a loop when
  a plain complete step is meant — that costs an extra client round trip per
  step; `train_step` is the default.
- **Do** remember `train_step`'s error path: a failed forward-backward does not
  cancel the optimizer step, and `model.step` has already advanced.
- **Pre-shifted RL layout**: response `old_logprobs`/`advantages` start at
  index `prompt_len - 1`, and end with one trailing `0.0`. Off-by-one here
  silently trains on the wrong positions.
- **Skip zero-advantage groups** (all rewards equal) — they contribute zero
  gradient and waste compute.
- **LR scale**: SUM reduction, not mean. Copying LRs from mean-reduced recipes
  will overshoot badly.
- `model.sample` returns `list[list[Sample]]` (per prompt, then per sample),
  even for a single prompt.
- `return_logprobs` is a deprecated no-op on forward-backward; training losses
  return per-token logprobs when the worker includes them.
- There is no packaged helper for building the RL datum — build the dict by
  hand as in Example 2.
- Sessions are not durable — checkpoints are. A long run without the
  auto-recovery skeleton will eventually lose work to a transient failure.
- Never recompute `old_logprobs` client-side — use `Sample.logprobs`
  verbatim, and build datums from the echoed `prompt_token_ids` when the
  server did the tokenization.
- For images, always go through the renderer (`image_part` +
  `build_training_example` / `build_sample_prompt`) or `model_input` — those
  paths validate the placeholder/image pairing client-side with a precise
  `ValueError`. Only raw `prompt_token_ids=` + `images=` can carry a mismatch
  to the backend (surplus images silently dropped).

## Reference

- Package: <https://pypi.org/project/river-client/>
- River: <https://river.ai>
