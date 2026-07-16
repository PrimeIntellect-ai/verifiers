"""user-sim: a modeled user converses with the assistant under evaluation.

The generic two-sided conversation (`--env.id user-sim` over any taskset) — the
substrate a tau2-style benchmark builds on. The taskset's row is read as the USER's
side of the world: its prompt text becomes the scenario in the user's system prompt
(`--env.persona`), and the assistant gets the same task **without** the prompt — it
learns the user's goal only through conversation (its own instructions stay in
`system_prompt`). The user role rides the in-process `direct` harness by default
(`trainable=False`), opens the conversation, and ends it with the done marker; the
assistant's trace is then judged by the task's own rewards, exactly as in any eval.

"The user is just another agent": the user's run is a real chat-session rollout with
its own role-stamped trace — both sides of the conversation land on the record.
"""

import verifiers.v1 as vf

PERSONA = """You are role-playing a USER talking to an AI assistant. This is your situation and goal:

{scenario}

Rules:
- Open the conversation with your request, in your own words.
- Stay in character: short, natural user messages; never act as the assistant.
- Reveal details only when asked, as a real user would.
- When the assistant has fully met your goal — or you are convinced it cannot — reply with exactly {done} and nothing else."""

DONE = "###DONE###"


class UserSimParams(vf.EnvParams):
    assistant: vf.AgentConfig = vf.AgentConfig()
    user: vf.AgentConfig = vf.AgentConfig(
        harness=vf.HarnessConfig(id="direct"), trainable=False
    )
    persona: str = PERSONA
    """The user's system prompt; `{scenario}` is replaced with the task's prompt text
    and `{done}` with the done marker (plain replacement, braces in the text are
    safe)."""
    done_marker: str = DONE
    """The user ends the conversation by replying with this marker."""
    max_user_turns: int = 8
    """Hard cap on user replies per conversation (a run-away exchange ends cleanly)."""


class UserSimEnv(vf.Environment[UserSimParams]):
    def roles(self):
        return {"assistant": self.params.assistant, "user": self.params.user}

    async def rollout(self, task, agents):
        scenario = task.data.prompt_text
        user_task = vf.Task(
            vf.TaskData(
                idx=task.data.idx,
                prompt=None,  # the user opens through the chat session
                system_prompt=self.params.persona.replace(
                    "{scenario}", scenario
                ).replace("{done}", self.params.done_marker),
            )
        )
        # The assistant plays the SAME task minus the prompt (the scenario is the
        # user's knowledge, not the assistant's) — its hooks and rewards still run.
        assistant_task = type(task)(
            task.data.model_copy(update={"prompt": None}), task.config
        )
        turns = 0
        async with agents["user"].chat(user_task) as sim:

            async def relay(text: str) -> vf.Messages:
                nonlocal turns
                if turns >= self.params.max_user_turns:
                    return []
                # The assistant's opening ping is empty (its task has no prompt);
                # seed the user-model with a neutral greeting instead — the tau
                # convention: the assistant "answers the phone", the user states
                # the goal. The greeting exists only on the user's side.
                reply = await sim.turn(text or "Hello! How can I help you today?")
                turns += 1
                if reply.stopped or self.params.done_marker in reply.text:
                    return []
                return [{"role": "user", "content": reply.text}]

            trace = await agents["assistant"].run(assistant_task, user=relay)
        return [trace, sim.trace]

    async def score(self, task, traces):
        assistant, user = traces
        # One conversation-shape fact per side; judgement stays on the task's rewards.
        assistant.record_metric("user_turns", float(user.num_turns))
