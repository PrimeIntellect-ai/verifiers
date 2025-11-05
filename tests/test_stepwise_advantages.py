from pathlib import Path
from typing import Any

import numpy as np
import pytest
from datasets import Dataset

from verifiers.types import GenerateMetadata, GenerateOutputs
from verifiers.rl.trainer.generator import Generator
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.rubrics.math_rubric import MathRubric


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        # Deterministic tokenization proportional to text length
        return list(range(len(text)))

    def apply_chat_template(
        self,
        conversation: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        # Simplified chat template: concatenate contents only
        return "".join(m.get("content", "") for m in conversation)


class StepwiseChatEnv(MultiTurnEnv):
    def __init__(
        self,
        dataset: Dataset,
        prepared: list[tuple[list[dict], list[dict], dict[str, Any]]],
    ):
        rb = MathRubric()
        super().__init__(
            exclude_think=True,
            dataset=dataset,
            message_type="chat",
            rubric=rb,
            parser=rb.parser,
        )
        self._prepared = prepared

    async def env_response(self, messages, state, **kwargs):
        return [], state

    async def generate(self, *args, **kwargs) -> GenerateOutputs:  # type: ignore[override]
        prompts: list[Any] = []
        completions: list[Any] = []
        states: list[dict[str, Any]] = []
        answers: list[str] = []
        tasks: list[str] = []
        infos: list[dict[str, Any]] = []
        example_ids: list[int] = []

        for i, (p, c, s) in enumerate(self._prepared):
            s = dict(s)
            s.setdefault("timing", {})
            s["timing"].setdefault("generation_ms", 0.0)
            s["timing"].setdefault("scoring_ms", 0.0)
            s["timing"].setdefault("total_ms", 0.0)

            if "step_scores" not in s:
                step_scores: list[float] = []
                j = 0
                while j < len(c):
                    msg = c[j]
                    if msg.get("role") == "assistant":
                        k = j + 1
                        while k < len(c) and c[k].get("role") != "assistant":
                            k += 1
                        partial = c[:k]
                        rs = await self.rubric.score_rollout(
                            prompt=p,
                            completion=list(partial),
                            answer=s.get("answer", ""),
                            state=s,
                            task=s.get("task", "default"),
                            info=s.get("info", {}),
                            example_id=i,
                        )
                        step_scores.append(float(rs.reward))
                        j = k
                    else:
                        j += 1
                s["step_scores"] = step_scores

            prompts.append(p)
            completions.append(c)
            states.append(s)
            answers.append(s.get("answer", ""))
            tasks.append(s.get("task", "default"))
            infos.append(s.get("info", {}))
            example_ids.append(i)

        meta = GenerateMetadata(
            env_id="stub-chat",
            env_args={},
            model="test-model",
            base_url="http://localhost/v1",
            num_examples=len(prompts),
            rollouts_per_example=1,
            sampling_args={},
            date="1970-01-01",
            time_ms=0.0,
            avg_reward=0.0,
            avg_metrics={},
            state_columns=[],
            path_to_save=Path("/tmp/stub"),
        )
        return GenerateOutputs(
            prompt=prompts,
            completion=completions,
            answer=answers,
            state=states,
            task=tasks,
            info=infos,
            example_id=example_ids,
            reward=[0.0] * len(prompts),
            metrics={},
            metadata=meta,
        )


def chat_prompt(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def assistant_msg(boxed: str, think: str = "chain-of-thought") -> dict:
    return {
        "role": "assistant",
        "content": f"<think>{think}</think> Visible: \\boxed{{{boxed}}}",
    }


def user_env_msg(text: str = "ack") -> dict:
    return {"role": "user", "content": text}


def make_chat_rollout(
    prompt: list[dict],
    answer: str,
    boxed_sequence: list[str],
    token_lengths: list[int] | None = None,
    step_scores: list[float] | None = None,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    token_lengths = token_lengths or [3] * len(boxed_sequence)
    completion: list[dict] = []
    responses = []
    for b, tlen in zip(boxed_sequence, token_lengths):
        completion.append(assistant_msg(b))
        completion.append(user_env_msg("ok"))
        responses.append({"tokens_len": int(tlen)})

    state = {
        "prompt": prompt,
        "completion": completion,
        "responses": responses,
        "turn": len(boxed_sequence),
        "timing": {"total_ms": 0.0},
        "task": "default",
        "info": {},
        "answer": answer,
    }
    if step_scores is not None:
        state["step_scores"] = [float(x) for x in step_scores]
    return prompt, completion, state


def compute_discounted_returns(
    rewards: list[float], gamma: float, aggregation: str
) -> list[float]:
    if not rewards:
        return []
    g = float(gamma)
    if aggregation == "sum":
        out = [0.0] * len(rewards)
        G = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            G = float(rewards[t]) + g * G
            out[t] = G
        return out
    else:
        out = [0.0] * len(rewards)
        R_next: float | None = None
        for t in range(len(rewards) - 1, -1, -1):
            r = float(rewards[t])
            cand = (g * R_next) if (R_next is not None) else None
            R_t = r if cand is None else max(r, cand)
            out[t] = R_t
            R_next = R_t
        return out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "aggregation,gamma",
    [
        ("sum", 1.0),
        ("max", 0.5),
    ],
)
async def test_stepwise_chat_precomputed_returns_and_metrics_with_advantage_checks(
    monkeypatch: pytest.MonkeyPatch, aggregation: str, gamma: float
):
    import verifiers.rl.trainer.generator as gen_mod

    monkeypatch.setattr(
        gen_mod,
        "parse_chat_completion_tokens",
        lambda resp: list(range(resp.get("tokens_len", 2))),
    )
    monkeypatch.setattr(
        gen_mod,
        "parse_chat_completion_logprobs",
        lambda resp: [-1.0] * resp.get("tokens_len", 2),
    )

    # Two prompts, two rollouts each (total 4)
    pA = chat_prompt("Compute 2+2 and provide a boxed answer.")
    pB = chat_prompt("Compute 1+1 and provide a boxed answer.")

    # Boxed sequences and precomputed step_scores (immediate rewards)
    # A1: wrong then right -> [0,1]
    A1 = make_chat_rollout(pA, answer="4", boxed_sequence=["3", "4"], step_scores=[0.0, 1.0])
    # B1: both right -> [1,1]
    B1 = make_chat_rollout(pB, answer="2", boxed_sequence=["2", "2"], step_scores=[1.0, 1.0])
    # A2: right then wrong -> [1,0]
    A2 = make_chat_rollout(pA, answer="4", boxed_sequence=["4", "3"], step_scores=[1.0, 0.0])
    # B2: wrong then right -> [0,1]
    B2 = make_chat_rollout(pB, answer="2", boxed_sequence=["3", "2"], step_scores=[0.0, 1.0])

    prepared = [A1, B1, A2, B2]

    ds = Dataset.from_dict({"prompt": [pA, pB]})
    env = StepwiseChatEnv(dataset=ds, prepared=prepared)
    tokenizer = DummyTokenizer()

    g = Generator(
        env=env,
        client_base_url="http://localhost/v1",
        client_api_key="test",
        client_limit=1,
        client_timeout=1.0,
        model_name="m",
        sampling_args={},
        rollouts_per_example=2,
        batch_size=4,
        micro_batch_size=2,
        num_processes=1,
        generation_timeout=5.0,
        processing_class=tokenizer,
        mask_env_responses=False,
        max_seq_len=4096,
        max_prompt_len=4096,
        mask_truncated_completions=False,
        zero_truncated_completions=False,
        max_concurrent=1,
        use_stepwise_advantage=True,
        stepwise_gamma=gamma,
        stepwise_aggregation=aggregation,
    )

    monkeypatch.setattr(env, "a_generate", env.generate)
    g.client = object()

    result = await g.generate_batch(batch_id=0)

    all_step_rewards = [s["step_scores"] for _p, _c, s in prepared]
    expected = [
        r for rewards in all_step_rewards for r in compute_discounted_returns(rewards, gamma, aggregation)
    ]

    assert np.allclose(result.rewards_dict["reward"], expected)

    assert pytest.approx(result.metrics_dict["reward"], rel=1e-6) == float(np.mean(expected))
    assert pytest.approx(result.metrics_dict["stepwise/turns_per_rollout"], rel=1e-6) == 2.0
    assert pytest.approx(result.metrics_dict["stepwise/rollout_length"], rel=1e-6) == 2.0

    # Advantage sanity checks
    # - per-token advantages are constant within each sample
    # - lengths of advantages match input_ids and loss_mask
    # - advantages are z-scored within each prompt group (mean≈0, std≈1)
    # - advantage/absmean > 0 for non-degenerate case
    assert result.metrics_dict.get("advantage/absmean", 0.0) > 0.0

    sample_adv_scalars: list[float] = []
    microbatches = result.microbatches[0]
    for mb in microbatches:
        for row_ids, row_mask, row_adv in zip(mb.input_ids, mb.loss_mask, mb.advantages):
            assert len(row_ids) == len(row_mask) == len(row_adv)
            uniq_vals = set(row_adv)
            assert len(uniq_vals) == 1
            sample_adv_scalars.append(row_adv[0])

    # Reconstruct prompt groups: with two steps per rollout and two prompts,
    # grouping used by Generator is i % prompts_in_batch; here i == j // steps_per_rollout
    steps_per_rollout = 2
    prompts_in_batch = len(ds)
    N = len(sample_adv_scalars)
    assert N == 4 * steps_per_rollout

    group0 = [sample_adv_scalars[j] for j in range(N) if ((j // steps_per_rollout) % prompts_in_batch) == 0]
    group1 = [sample_adv_scalars[j] for j in range(N) if ((j // steps_per_rollout) % prompts_in_batch) == 1]

    assert pytest.approx(float(np.mean(group0)), abs=1e-6) == 0.0
    assert pytest.approx(float(np.mean(group1)), abs=1e-6) == 0.0
    assert pytest.approx(float(np.std(group0)), rel=1e-5) == 1.0
    assert pytest.approx(float(np.std(group1)), rel=1e-5) == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "aggregation,gamma",
    [
        ("sum", 1.0),
        ("max", 0.5),
    ],
)
async def test_stepwise_chat_returns_precomputed_by_math_rubric(
    monkeypatch: pytest.MonkeyPatch, aggregation: str, gamma: float
):
    import verifiers.rl.trainer.generator as gen_mod

    monkeypatch.setattr(
        gen_mod,
        "parse_chat_completion_tokens",
        lambda resp: list(range(resp.get("tokens_len", 2))),
    )
    monkeypatch.setattr(
        gen_mod,
        "parse_chat_completion_logprobs",
        lambda resp: [-1.0] * resp.get("tokens_len", 2),
    )

    pA = chat_prompt("Compute 2+2 and provide a boxed answer.")
    pB = chat_prompt("Compute 1+1 and provide a boxed answer.")

    A1 = make_chat_rollout(pA, answer="4", boxed_sequence=["3", "4"])
    B1 = make_chat_rollout(pB, answer="2", boxed_sequence=["2", "2"])
    A2 = make_chat_rollout(pA, answer="4", boxed_sequence=["4", "3"])
    B2 = make_chat_rollout(pB, answer="2", boxed_sequence=["3", "2"])

    prepared = [A1, B1, A2, B2]
    ds = Dataset.from_dict({"prompt": [pA, pB]})
    env = StepwiseChatEnv(dataset=ds, prepared=prepared)
    tokenizer = DummyTokenizer()

    g = Generator(
        env=env,
        client_base_url="http://localhost/v1",
        client_api_key="test",
        client_limit=1,
        client_timeout=1.0,
        model_name="m",
        sampling_args={},
        rollouts_per_example=2,
        batch_size=4,
        micro_batch_size=2,
        num_processes=1,
        generation_timeout=5.0,
        processing_class=tokenizer,
        mask_env_responses=False,
        max_seq_len=4096,
        max_prompt_len=4096,
        mask_truncated_completions=False,
        zero_truncated_completions=False,
        max_concurrent=1,
        use_stepwise_advantage=True,
        stepwise_gamma=gamma,
        stepwise_aggregation=aggregation,
    )

    monkeypatch.setattr(env, "a_generate", env.generate)
    g.client = object()

    result = await g.generate_batch(batch_id=0)

    # Expected immediate rewards derived from MathRubric correctness on each step
    # A1: [0,1], B1: [1,1], A2: [1,0], B2: [0,1]
    step_rewards = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    expected = [
        r for rewards in step_rewards for r in compute_discounted_returns(rewards, gamma, aggregation)
    ]

    assert np.allclose(result.rewards_dict["reward"], expected)
    assert pytest.approx(result.metrics_dict["reward"], rel=1e-6) == float(np.mean(expected))
    assert pytest.approx(result.metrics_dict["stepwise/rollout_length"], rel=1e-6) == 2.0
    assert pytest.approx(result.metrics_dict["stepwise/turns_per_rollout"], rel=1e-6) == 2.0


@pytest.mark.asyncio
async def test_stepwise_chat_truncation_zero_reward(monkeypatch: pytest.MonkeyPatch):
    import verifiers.rl.trainer.generator as gen_mod

    monkeypatch.setattr(
        gen_mod,
        "parse_chat_completion_tokens",
        lambda resp: list(range(resp.get("tokens_len", 2))),
    )
    monkeypatch.setattr(
        gen_mod,
        "parse_chat_completion_logprobs",
        lambda resp: [-1.0] * resp.get("tokens_len", 2),
    )

    pA = chat_prompt("Compute 2+2 and provide a boxed answer.")
    pB = chat_prompt("Compute 1+1 and provide a boxed answer.")

    # All steps 'correct' → immediate rewards [1,1] per rollout
    # Force truncation on first step by assigning large token length
    A1 = make_chat_rollout(
        pA,
        answer="4",
        boxed_sequence=["4", "4"],
        token_lengths=[50, 2],
        step_scores=[1.0, 1.0],
    )
    B1 = make_chat_rollout(
        pB, answer="2", boxed_sequence=["2", "2"], token_lengths=[2, 2], step_scores=[1.0, 1.0]
    )
    A2 = make_chat_rollout(
        pA, answer="4", boxed_sequence=["4", "4"], token_lengths=[2, 2], step_scores=[1.0, 1.0]
    )
    B2 = make_chat_rollout(
        pB, answer="2", boxed_sequence=["2", "2"], token_lengths=[2, 2], step_scores=[1.0, 1.0]
    )

    prepared = [A1, B1, A2, B2]
    ds = Dataset.from_dict({"prompt": [pA, pB]})
    env = StepwiseChatEnv(dataset=ds, prepared=prepared)
    tokenizer = DummyTokenizer()

    g = Generator(
        env=env,
        client_base_url="http://localhost/v1",
        client_api_key="test",
        client_limit=1,
        client_timeout=1.0,
        model_name="m",
        sampling_args={},
        rollouts_per_example=2,
        batch_size=4,
        micro_batch_size=2,
        num_processes=1,
        generation_timeout=5.0,
        processing_class=tokenizer,
        mask_env_responses=False,
        max_seq_len=10,
        max_prompt_len=4096,
        mask_truncated_completions=True,
        zero_truncated_completions=True,
        max_concurrent=1,
        use_stepwise_advantage=True,
        stepwise_gamma=1.0,
        stepwise_aggregation="sum",
    )

    monkeypatch.setattr(env, "a_generate", env.generate)
    g.client = object()

    result = await g.generate_batch(batch_id=0)

    # Without truncation, each rollout [1,1] with gamma=1 → returns [2,1]
    expected_naive = [2.0, 1.0] * 4
    rewards = list(result.rewards_dict["reward"])

    # Only the very first step (A1 step 1) should be truncated → set to 0.0
    expected = expected_naive[:]
    expected[0] = 0.0

    assert np.allclose(rewards, expected)

    # masking should be nonzero due to truncation
    assert "tokens/masked_fraction" in result.metrics_dict
    assert result.metrics_dict["tokens/masked_fraction"] > 0.0
