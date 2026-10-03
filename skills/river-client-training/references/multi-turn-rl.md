# Multi-turn RL with the public client

This small tool task demonstrates the complete wiring. It uses a separate `Env`
instance for each trajectory: never share mutable browser or game state between
members of a rollout group. Replace this environment with your task and choose
its reward and budgets. Running the script consumes API capacity.

```python
import asyncio
from contextlib import closing
import os

import river_client as river
from river_client import rl
from river_client.renderers import get_renderer

MODEL = "zai-org/GLM-5.3-Flash"


class GuessNumber(rl.Env):
    # Unfinished episodes regenerate after restart; no external state to restore.
    recovery = "drop"

    def __init__(self):
        self.solved = False

        @rl.tool
        async def guess(number: int) -> str:
            """Guess the secret integer; receive higher, lower, or correct."""
            if number == self.target:
                self.solved = True
                return "correct"
            return "higher" if number < self.target else "lower"

        self.tools = (guess,)

    async def reset(self, row):
        self.target = row["target"]
        self.solved = False
        return [{"role": "user", "content":
                 "Find the secret integer from 1 to 100 using the guess tool."}]

    async def on_turn(self, traj):
        # Default dispatch executes tools concurrently. Override it for ordered
        # browser actions or multimodal tool observations.
        observations = await super().on_turn(traj)
        return None if self.solved else observations

    async def reward(self, traj, row):
        return float(self.solved)

    async def on_truncated(self, traj, row, cause):
        return float(self.solved)


def log_eval(result):
    print("eval", result.step, result.checkpoint.step, result.metrics, flush=True)


async def main():
    tokenizer = river.load_tokenizer(base_model=MODEL)
    renderer = get_renderer(MODEL, tokenizer=tokenizer, reasoning_effort="low")
    budget = rl.Budget(max_turns=8, max_generated_tokens=4096,
                       max_turn_tokens=512, segment_tokens=512)
    train_rows = [{"target": n} for n in range(1, 41)]
    holdout = [{"target": n} for n in range(81, 91)]
    with closing(river.Client(api_key=os.environ["RIVER_API_KEY"])) as client:
        with client.session(experiment="tool-rl", role="train") as session, \
                client.session(experiment="tool-rl", role="eval") as eval_session:
            model = session.create_model(
                base_model=MODEL, tokenizer=tokenizer,
                lora=river.LoraConfig(rank=16, train_unembed=False),
            )

            def evaluation_engine(checkpoint, variant):
                return rl.RolloutEngine(
                    rl.CheckpointSampler(eval_session, base_model=MODEL,
                                         checkpoint=checkpoint, tokenizer=tokenizer),
                    env=GuessNumber, renderer=renderer, budget=budget,
                    schedule=rl.Schedule(concurrency=4), temperature=0,
                )

            trainer = rl.AsyncTrainer(
                engine=rl.RolloutEngine(
                    model, env=GuessNumber, renderer=renderer, budget=budget,
                    schedule=rl.Schedule(concurrency=16), temperature=1,
                ),
                optimizer=rl.Adam(lr=1e-5),
                completion=rl.GroupCompletion(mode="wait"),
                advantage=rl.GroupCentered(), normalize="token", loss="cispo",
                groups_per_step=4, group_size=4, max_staleness=0,
                checkpoint=rl.Checkpointing(run_dir="run", weights_every=5),
                evaluator=rl.Evaluator(
                    holdout, engine_factory=evaluation_engine, every=5,
                    group_size=1, final_group_size=1, sink=log_eval,
                ),
                run_config={"task_version": 1, "reasoning_effort": "low"},
            )
            async for step in trainer.run(train_rows, steps=10):
                print("train", step.n, step.model_step, step.metrics, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
```

`Evaluator` saves and measures a specific checkpoint at batch zero, every five
batches, and the final batch. Evaluation can finish after later training batches;
log against `result.step` or `result.checkpoint.step`, not arrival order. Its
sampler uses the evaluation session; do not use the live training model there.

Re-run with the same durable `run` directory, dataset, horizon and configuration
to restore training. Preserve the whole directory, including image files. Include
custom environment, reward and rendering choices in `run_config`; those choices
must stay unchanged across a resume. Use a new directory for a different recipe.
A failed browser or provider is `rl.InfrastructureError`, not a negative reward.
Handle transient errors outside the session contexts and reopen fresh sessions
with the same checkpoint directory. Unknown environment state regenerates;
implement `snapshot`/`restore` only when the environment can actually recover it.

## Screenshots and image handles

For a vision environment, pass an environment factory that captures the owning
session, e.g. `env=lambda: BrowserEnv(session)`. Create a browser per instance,
release it in `Env.close`, and return observations from `reset` and `on_turn`:

```python
from river_client.renderers import image_part

# Inside an async environment method; screenshot_png is your browser's output.
handle = await self.session.upload_image_async(screenshot_png)
observation = {"role": "user", "content": [
    {"type": "text", "text": "Current screenshot"}, image_part(handle),
]}
```

A custom `on_turn` executes the latest assistant's tool calls and returns only
new messages, or `None` when the episode ends. For a screenshot tool result,
return `role="tool"`, its `tool_call_id`, and text/image content parts. `@tool`
functions themselves return strings; multimodal results belong in `on_turn`.
Do not re-render sampled history to append a screenshot. The engine preserves
exact sampled tokens and image positions. Use the evaluation session for eval
uploads. Release handles once no pending rollout or request needs them; local
checkpoint image storage supports recovery in a new session. Handle TTL refresh,
quota and reupload behavior are covered in the main skill's image section.

## Sampling policy and overlap

The example is synchronous: `max_staleness=0`, trajectory-pinned sampling, and
wait semantics. Ready groups can already train while other groups are sampling.
The default `forward_backward_batch="auto"` aggregates their work; increasing
policy staleness is not necessary to enable this overlap. Batch-wide advantages
may require all rewards before training can begin.

For asynchronous multi-turn sampling, choose both bounds deliberately:
`AsyncTrainer(max_staleness=2, ...)` bounds training-policy age, while
`RolloutEngine(sampling_policy="segment", kv_cache=rl.KVCache(max_staleness=1,
on_limit="hold"), ...)` bounds cached KV age relative to sampling weights.
`hold` keeps sampling on compatible weights; `refill` permits a new prefill and requires
`allow_reprefill=True`. Use server/model capability discovery for optional
features; the engine also preflights requirements. These knobs do not provision
additional sampling or training capacity.

GLM-5.3 Flash supports `reasoning_effort="low"`, `"high"`, or `"max"` (default)
through `get_renderer` or `Glm53FlashRenderer`. This controls the checkpoint's
prompt template; it does not disable thinking or impose a token limit. Set
`rl.Budget` separately. Unsupported effort values fail instead of being ignored.

## Automatic sampling metrics

Each trainer `Step.metrics` includes `sampling/prompt_tokens`,
`sampling/cached_prompt_tokens`, `sampling/generated_tokens`, request/prompt
counts, KV refresh counts, and client queue/request durations. Log `step.metrics`
as usual; no separate engine polling is needed. `sampling/cached_prompt_fraction`
is cached prompt tokens divided by prompt tokens (not the fraction of requests
with a hit); it is omitted when no prompt tokens were reported.

These are deltas since the preceding emitted step, not totals to sum repeatedly.
With async lookahead or oversampling, they describe sampling work during that
interval, not just trajectories consumed by that training batch. They include
work while the caller handles a yielded step and start fresh after resume.
Evaluation uses its own engine and is not included. In-flight/ready counts are
current gauges. `sampling/generated_tokens_per_second` uses the entire interval
(including training and caller time), not just time spent decoding. Retention
mode is separate from actual token cache hits.
