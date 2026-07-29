# river-skills

Public [Agent Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview) for the River API. Each skill is a folder with a `SKILL.md` that teaches an agent (Claude Code, the Claude Agent SDK, or any Skills-compatible harness) how to use River correctly — the agent loads it on demand when the task matches.

## Skills

| Skill | What it covers |
|---|---|
| [river-client-training](skills/river-client-training/SKILL.md) | Writing training scripts with the [river-client](https://pypi.org/project/river-client/) Python package — LoRA fine-tuning, SFT, and RL/GRPO on River-hosted models via the River training API. Covers the current `train_step` API and its pipelining/error semantics, training data format, multimodal (image) data, sampling and logprobs, MoE expert-routing capture/replay, teacher→student distillation, and fault-tolerant loops that auto-recover from session loss, capacity, and timeout errors. |

## Install

### Claude Code (plugin)

```
/plugin marketplace add riverai-org/river-skills
/plugin install river@river-skills
```

The skills then trigger automatically whenever you work with `river_client` code.

### Manual (any Skills-compatible agent)

Copy a skill folder into your skills directory:

```bash
git clone https://github.com/riverai-org/river-skills.git
mkdir -p ~/.claude/skills
cp -r river-skills/skills/river-client-training ~/.claude/skills/
```

Or vendor it into a project at `.claude/skills/river-client-training/`.

## Source of truth

The canonical copy of each skill ships inside the `river-client` Python package (`river_client/skills/`), so scripts that install the package get it automatically. This repo republishes the skills for direct agent installation; skill files here are kept byte-identical to the packaged versions.

## Links

- River: <https://river.ai>
- river-client on PyPI: <https://pypi.org/project/river-client/>

## License

[Apache-2.0](LICENSE), same as the `river-client` package.
