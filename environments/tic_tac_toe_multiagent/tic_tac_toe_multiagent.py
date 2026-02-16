from datasets import Dataset

import verifiers as vf
from verifiers.types import Messages, State, SystemMessage, UserMessage


class TicTacToeMultiAgentEnv(vf.MultiAgentEnv):
    PLAYER_X_ID = "player_x"
    PLAYER_O_ID = "player_o"

    def __init__(self, max_turns: int = 9, **kwargs):
        super().__init__(tools=[], max_turns=max_turns, **kwargs)
        self.add_tool(self.make_move, args_to_skip=["state"])

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        state["board_state"] = None
        self.logger.debug("ttt.setup board_state_initialized=false")
        return state

    def get_all_actors(self, state: State) -> dict[str, str]:
        return {
            self.PLAYER_X_ID: (
                "You are player X in tic-tac-toe. "
                "Use make_move with an integer from 1 to 9."
            ),
            self.PLAYER_O_ID: (
                "You are player O in tic-tac-toe. "
                "Use make_move with an integer from 1 to 9."
            ),
        }

    def get_initial_actor_id(self, actors: dict[str, str], state: State) -> str:
        return self.PLAYER_X_ID

    def get_next_actor_id(self, state: State) -> str:
        if state["trajectory_id"] == self.PLAYER_X_ID:
            return self.PLAYER_O_ID
        return self.PLAYER_X_ID

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: Messages,
        state: State,
        **kwargs,
    ) -> dict:
        if tool_name == "make_move":
            return {**tool_args, "state": state}
        return tool_args

    def actor_mark(self, actor_id: str) -> str:
        if actor_id == self.PLAYER_X_ID:
            return "X"
        return "O"

    def empty_board(self) -> list[str]:
        return [" "] * 9

    def render_board(self, board_state: list[str] | None) -> str:
        board = self.empty_board() if board_state is None else board_state
        shown = [cell if cell != " " else str(i + 1) for i, cell in enumerate(board)]
        rows = [
            f"{shown[0]} | {shown[1]} | {shown[2]}",
            f"{shown[3]} | {shown[4]} | {shown[5]}",
            f"{shown[6]} | {shown[7]} | {shown[8]}",
        ]
        return "\n---------\n".join(rows)

    def winner(self, board_state: list[str]) -> str | None:
        wins = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]
        for a, b, c in wins:
            if (
                board_state[a] != " "
                and board_state[a] == board_state[b] == board_state[c]
            ):
                return board_state[a]
        return None

    def is_draw(self, board_state: list[str]) -> bool:
        return all(cell != " " for cell in board_state)

    async def make_move(self, position: int, state: State) -> str:
        """
        Play a tic-tac-toe move.

        Args:
            position: Integer 1 through 9 indicating board position.
            state: Rollout state (injected by environment).

        Returns:
            Success or error message.
        """
        board_state = state["board_state"]
        if board_state is None:
            board_state = self.empty_board()
            state["board_state"] = board_state

        actor_id = state["trajectory_id"]
        mark = self.actor_mark(actor_id)
        self.logger.debug(
            "ttt.move attempt actor=%s mark=%s position=%s", actor_id, mark, position
        )

        if position < 1 or position > 9:
            self.logger.warning(
                "ttt.move illegal actor=%s position=%s reason=out_of_range",
                actor_id,
                position,
            )
            return "Error: illegal move. Position must be an integer from 1 to 9."

        index = position - 1
        if board_state[index] != " ":
            self.logger.warning(
                "ttt.move illegal actor=%s position=%s reason=occupied",
                actor_id,
                position,
            )
            return f"Error: illegal move. Position {position} is already occupied."

        board_state[index] = mark
        state["board_state"] = board_state

        game_winner = self.winner(board_state)
        if game_winner is not None:
            board_text = self.render_board(board_state)
            state["final_env_response"] = (
                f"Game over. Winner: {game_winner}\nFinal board state:\n{board_text}"
            )
            self.logger.info("ttt.game_over winner=%s", game_winner)
            return f"Successfully played move {position}."

        if self.is_draw(board_state):
            board_text = self.render_board(board_state)
            state["final_env_response"] = (
                f"Game over. Draw.\nFinal board state:\n{board_text}"
            )
            self.logger.info("ttt.game_over result=draw")
            return f"Successfully played move {position}."

        self.logger.debug("ttt.move applied actor=%s position=%s", actor_id, position)
        return f"Successfully played move {position}."

    def get_prompt_for_actor(self, messages: Messages, state: State) -> Messages:
        actor_id = state["trajectory_id"]
        system_prompt = state["system_prompts"][actor_id]
        system_message = SystemMessage(content=system_prompt)
        board_state = state["board_state"]
        self.logger.debug(
            "ttt.prompt actor=%s has_history=%s board_initialized=%s",
            actor_id,
            len(messages) > 0,
            board_state is not None,
        )

        if len(messages) == 0:
            if board_state is None:
                prompt_messages = state["prompt"]
                if (
                    len(prompt_messages) != 1
                    or getattr(prompt_messages[0], "role", None) != "user"
                ):
                    raise ValueError(
                        "Expected state['prompt'] to contain exactly one user message"
                    )
                return [system_message, prompt_messages[0]]
            return [
                system_message,
                UserMessage(
                    content=f"Current board state:\n{self.render_board(board_state)}"
                ),
            ]

        return [
            UserMessage(
                content=f"Current board state:\n{self.render_board(board_state)}"
            )
        ]


def load_environment() -> vf.Environment:
    starting_board = "1 | 2 | 3\n---------\n4 | 5 | 6\n---------\n7 | 8 | 9"
    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Starting board state:\n{starting_board}",
                    }
                ],
                "task": "tic_tac_toe_multiagent",
            }
        ]
    )

    async def finished_game_reward(state: State) -> float:
        return 1.0 if state.get("final_env_response") is not None else 0.0

    rubric = vf.Rubric(funcs=[finished_game_reward])
    return TicTacToeMultiAgentEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt=None,
    )
