"""
Ultimatum Game (4-Player): Two back-to-back ultimatum games in one episode.

4 actors across 2 models:
  - proposer_fair + proposer_agg → Model A (Qwen3-4B-Instruct-2507)
  - responder_strict + responder_lenient → Model B (Qwen3-4B)

Turn order:
  1. proposer_fair offers
  2. responder_strict accepts/rejects
  3. proposer_agg offers
  4. responder_lenient accepts/rejects

Used for testing multi-model + multi-LoRA (mode 7).
"""

import re

from datasets import Dataset

from verifiers.envs.agent import Agent
from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.rubrics.multiagent_rubric import MultiAgentRubric
from verifiers.envs.taskset import TaskSet
from verifiers.types import Messages, State
from openai import AsyncOpenAI


TOTAL_AMOUNT = 10


def create_rubric() -> MultiAgentRubric:
    rubric = MultiAgentRubric()

    def proposer_fair_reward(state, **kwargs) -> float:
        return state.get("extras", {}).get("proposer_fair_payoff", 0.0)

    def proposer_agg_reward(state, **kwargs) -> float:
        return state.get("extras", {}).get("proposer_agg_payoff", 0.0)

    def responder_strict_reward(state, **kwargs) -> float:
        return state.get("extras", {}).get("responder_strict_payoff", 0.0)

    def responder_lenient_reward(state, **kwargs) -> float:
        return state.get("extras", {}).get("responder_lenient_payoff", 0.0)

    rubric.add_actor_reward_func("proposer_fair", proposer_fair_reward, weight=1.0)
    rubric.add_actor_reward_func("proposer_agg", proposer_agg_reward, weight=1.0)
    rubric.add_actor_reward_func("responder_strict", responder_strict_reward, weight=1.0)
    rubric.add_actor_reward_func("responder_lenient", responder_lenient_reward, weight=1.0)

    # Metrics (not used for training)
    def round1_deal_rate(state, **kwargs) -> float:
        return 1.0 if state.get("extras", {}).get("round1_accepted", False) else 0.0

    def round2_deal_rate(state, **kwargs) -> float:
        return 1.0 if state.get("extras", {}).get("round2_accepted", False) else 0.0

    rubric.add_reward_func(round1_deal_rate, weight=0.0)
    rubric.add_reward_func(round2_deal_rate, weight=0.0)
    return rubric


class UltimatumTask4Player(TaskSet):

    def __init__(self, num_examples: int = -1):
        dataset = self._create_dataset()
        if num_examples > 0:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        super().__init__(
            name="ultimatum_game_4player",
            dataset=dataset,
            rubric=create_rubric(),
            roles=["proposer_fair", "responder_strict", "proposer_agg", "responder_lenient"],
        )

    @staticmethod
    def _create_dataset() -> Dataset:
        items = [
            {
                "prompt": [{"role": "user", "content": "Negotiate a deal."}],
                "answer": "",
                "info": {"total": TOTAL_AMOUNT},
                "example_id": i,
                "task": "ultimatum_game_4player",
            }
            for i in range(20)
        ]
        return Dataset.from_list(items)

    async def setup_state(self, state: State) -> State:
        state["extras"]["round1_offer"] = 0
        state["extras"]["round1_accepted"] = False
        state["extras"]["round2_offer"] = 0
        state["extras"]["round2_accepted"] = False
        state["extras"]["proposer_fair_payoff"] = 0.0
        state["extras"]["proposer_agg_payoff"] = 0.0
        state["extras"]["responder_strict_payoff"] = 0.0
        state["extras"]["responder_lenient_payoff"] = 0.0
        return state

    async def build_prompt(self, role: str, state: State) -> Messages:
        if role == "proposer_fair":
            return [
                {"role": "system", "content": (
                    f"You have ${TOTAL_AMOUNT} to split with another player. "
                    "You must propose how much to give them. "
                    "If they reject your offer, you BOTH get nothing.\n\n"
                    "You value fairness — make a reasonable offer.\n\n"
                    "Output ONLY a single number (0-10) representing how much you offer."
                )},
                {"role": "user", "content": f"You have ${TOTAL_AMOUNT}. How much do you offer?"},
            ]
        elif role == "responder_strict":
            offer = state["extras"]["round1_offer"]
            return [
                {"role": "system", "content": (
                    f"Someone is splitting ${TOTAL_AMOUNT} with you. "
                    f"They are offering you ${offer} (they keep ${TOTAL_AMOUNT - offer}).\n\n"
                    "You have high standards — only accept genuinely fair offers.\n\n"
                    "Output ONLY 'Accept' or 'Reject'."
                )},
                {"role": "user", "content": f"/no_think They offer you ${offer} out of ${TOTAL_AMOUNT}. Accept or Reject?"},
            ]
        elif role == "proposer_agg":
            return [
                {"role": "system", "content": (
                    f"You have ${TOTAL_AMOUNT} to split with another player. "
                    "You must propose how much to give them. "
                    "If they reject your offer, you BOTH get nothing.\n\n"
                    "You want to maximize your own earnings.\n\n"
                    "Output ONLY a single number (0-10) representing how much you offer."
                )},
                {"role": "user", "content": f"You have ${TOTAL_AMOUNT}. How much do you offer?"},
            ]
        else:  # responder_lenient
            offer = state["extras"]["round2_offer"]
            return [
                {"role": "system", "content": (
                    f"Someone is splitting ${TOTAL_AMOUNT} with you. "
                    f"They are offering you ${offer} (they keep ${TOTAL_AMOUNT - offer}).\n\n"
                    "You prefer to make a deal rather than walk away empty-handed.\n\n"
                    "Output ONLY 'Accept' or 'Reject'."
                )},
                {"role": "user", "content": f"/no_think They offer you ${offer} out of ${TOTAL_AMOUNT}. Accept or Reject?"},
            ]

    async def on_turn_complete(self, state: State) -> None:
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        actor_id = last_step["extras"].get("actor_id", "")
        completion = last_step.get("completion", [])
        if not completion:
            return
        content = completion[-1].get("content", "").strip()
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()

        if actor_id == "proposer_fair":
            numbers = re.findall(r'\d+', content)
            if numbers:
                offer = min(int(numbers[0]), TOTAL_AMOUNT)
                offer = max(0, offer)
            else:
                offer = TOTAL_AMOUNT
            state["extras"]["round1_offer"] = offer

        elif actor_id == "responder_strict":
            accepted = "accept" in content.lower()
            state["extras"]["round1_accepted"] = accepted
            offer = state["extras"]["round1_offer"]
            if accepted:
                state["extras"]["proposer_fair_payoff"] = (TOTAL_AMOUNT - offer) / TOTAL_AMOUNT
                state["extras"]["responder_strict_payoff"] = offer / TOTAL_AMOUNT

        elif actor_id == "proposer_agg":
            numbers = re.findall(r'\d+', content)
            if numbers:
                offer = min(int(numbers[0]), TOTAL_AMOUNT)
                offer = max(0, offer)
            else:
                offer = TOTAL_AMOUNT
            state["extras"]["round2_offer"] = offer

        elif actor_id == "responder_lenient":
            accepted = "accept" in content.lower()
            state["extras"]["round2_accepted"] = accepted
            offer = state["extras"]["round2_offer"]
            if accepted:
                state["extras"]["proposer_agg_payoff"] = (TOTAL_AMOUNT - offer) / TOTAL_AMOUNT
                state["extras"]["responder_lenient_payoff"] = offer / TOTAL_AMOUNT

    async def should_stop(self, state: State) -> bool:
        actor_history = state.get("extras", {}).get("actor_history", [])
        return len(actor_history) >= 4

    async def on_game_end(self, state: State) -> None:
        pass


def load_environment(num_examples: int = -1, actor_endpoints: dict[str, str] | None = None):
    task = UltimatumTask4Player(num_examples=num_examples)
    actor_endpoints = actor_endpoints or {}

    proposer_fair_url = actor_endpoints.get("proposer_fair")
    proposer_agg_url = actor_endpoints.get("proposer_agg")
    responder_strict_url = actor_endpoints.get("responder_strict")
    responder_lenient_url = actor_endpoints.get("responder_lenient")

    proposer_fair = Agent(
        id="proposer_fair",
        max_tokens=32,
        is_trainable=True,
        model="Qwen/Qwen3-4B-Instruct-2507",
        client=AsyncOpenAI(base_url=proposer_fair_url, api_key="EMPTY") if proposer_fair_url else None,
    )

    proposer_agg = Agent(
        id="proposer_agg",
        max_tokens=32,
        is_trainable=True,
        model="Qwen/Qwen3-4B-Instruct-2507",
        client=AsyncOpenAI(base_url=proposer_agg_url, api_key="EMPTY") if proposer_agg_url else None,
    )

    responder_strict = Agent(
        id="responder_strict",
        max_tokens=32,
        is_trainable=True,
        model="Qwen/Qwen3-4B",
        client=AsyncOpenAI(base_url=responder_strict_url, api_key="EMPTY") if responder_strict_url else None,
    )

    responder_lenient = Agent(
        id="responder_lenient",
        max_tokens=32,
        is_trainable=True,
        model="Qwen/Qwen3-4B",
        client=AsyncOpenAI(base_url=responder_lenient_url, api_key="EMPTY") if responder_lenient_url else None,
    )

    return MultiAgentEnv(
        task=task,
        agents={
            "proposer_fair": proposer_fair,
            "proposer_agg": proposer_agg,
            "responder_strict": responder_strict,
            "responder_lenient": responder_lenient,
        },
        max_turns=8,
    )
