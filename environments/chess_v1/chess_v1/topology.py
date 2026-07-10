"""chess-v1 — two agents play chess through live sessions (example topology).

The back-and-forth shape: both episodes stay open for the whole game
(`run.agent(...).interact(...)`), each seat suspended between its moves, and `go` is the
referee — a host-side `chess.Board` is the single source of truth for legality and
outcome (judge the world, not the transcript). White's assistant turns arrive as black's
user turns and vice versa, so each seat's trace is ONE multi-turn trajectory with the
opponent baked into its context — the training-sample shape self-play wants, not one
episode per move.

Contracts on display:
  - **Sessions**: `turn()` per move, illegal-move feedback is just another user turn,
    a seat that won't produce a legal move forfeits via `session.end()`.
  - **Outcome as data, judgement declared**: `go` stamps each seat's result into
    `trace.info["chess"]`; the per-seat `outcome` rewards read it at instance end.
  - **Per-seat routing**: `--topology.black.model <id>` plays mixed-model games.

Budgets span the whole game (a seat's episode includes time suspended while the
opponent thinks): size `--max-turns` >= max_plies and the rollout timeout for a full
game, not a solo run.

    uv run eval --topology.id chess-v1 -n 1 --max-turns 64 --timeout.rollout 900 \\
        --sampling.max-tokens 4096   # reasoning room per move — a tight cap truncates turns
"""

import re

import chess

import verifiers.v1 as vf

SEAT_SYSTEM = """You are playing chess as {color}. Each message shows the board and the \
opponent's last move; the listed legal moves are exact. Reply with your reasoning if you \
like, but end your reply with exactly one line of the form `MOVE: <uci>` (e.g. `MOVE: \
e2e4`), choosing one of the listed legal moves."""

_UCI = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


def parse_move(reply: str, board: chess.Board) -> chess.Move | None:
    """The last legal UCI move mentioned on the reply's final `MOVE:` line (any legal
    UCI in the reply as fallback), or None."""
    candidates: list[str] = []
    for line in reversed(reply.splitlines()):
        stripped = line.strip().strip("*`").strip()
        if stripped.upper().startswith("MOVE:"):
            candidates = _UCI.findall(stripped)
            break
    if not candidates:
        candidates = _UCI.findall(reply)
    for text in reversed(candidates):
        move = chess.Move.from_uci(text)
        if move in board.legal_moves:
            return move
    return None


def render(board: chess.Board, last: str) -> str:
    """One seat's view for its move: the opponent's last move, the position, and the
    exact legal moves (the referee's affordance — legality is never the model's guess)."""
    legal = " ".join(m.uci() for m in board.legal_moves)
    return (
        f"{last}\n\nPosition (FEN): {board.fen()}\n{board}\n\n"
        f"Legal moves: {legal}\nYour move."
    )


class SeatTask(vf.Task):
    """One player's episode: framing only (`prompt=None` — sessions open on the first
    `turn()`). No ground truth here; the outcome belongs to the game, i.e. the topology."""

    color: str


class ChessConfig(vf.TopologyConfig):
    white: vf.DirectAgentConfig = vf.DirectAgentConfig()
    black: vf.DirectAgentConfig = vf.DirectAgentConfig()
    """Per-seat routing: `--topology.black.model <id>` plays mixed-model games."""
    num_games: int = 4
    """Seed games (one topology instance each); `-n` slices them."""
    max_plies: int = 40
    """Adjudicate a draw beyond this many half-moves — keeps an episode inside its
    budgets (each seat's single program request spans the whole game)."""
    illegal_retries: int = 2
    """Feedback turns offered for an illegal/unparseable move before the seat forfeits."""


class ChessTopology(vf.Topology[ChessConfig]):
    def load_tasks(self) -> list[vf.Task]:
        """Self-seeding: a seed is just a game id."""
        return [vf.Task(idx=i, prompt=None) for i in range(self.config.num_games)]

    async def go(self, task: vf.Task, run: vf.TopologyRun) -> None:
        board = chess.Board()
        illegal = {"white": 0, "black": 0}
        forfeited: str | None = None
        async with (
            run.agent("white").interact(
                SeatTask(
                    idx=task.idx,
                    color="white",
                    prompt=None,
                    system_prompt=SEAT_SYSTEM.format(color="white"),
                )
            ) as white,
            run.agent("black").interact(
                SeatTask(
                    idx=task.idx,
                    color="black",
                    prompt=None,
                    system_prompt=SEAT_SYSTEM.format(color="black"),
                )
            ) as black,
        ):
            seats = {"white": white, "black": black}
            last = "You open the game."
            while (
                not board.is_game_over()
                and len(board.move_stack) < self.config.max_plies
            ):
                name = "white" if board.turn == chess.WHITE else "black"
                try:
                    reply = await seats[name].turn(render(board, last))
                    move = parse_move(reply, board)
                    for _ in range(self.config.illegal_retries):
                        if move is not None:
                            break
                        illegal[name] += 1
                        reply = await seats[name].turn(
                            "That was not a legal move. End with `MOVE: <uci>`, one of: "
                            + " ".join(m.uci() for m in board.legal_moves)
                        )
                        move = parse_move(reply, board)
                except vf.SessionEnded:
                    move = None  # seat died (budget/error) — treated as a forfeit
                if move is None:
                    forfeited = name
                    await seats[name].end("forfeit")
                    break
                board.push(move)
                last = f"Opponent played {move.uci()}."
        # Outcome as data (judgement stays declared): the referee's verdict, per seat.
        scores = self.adjudicate(board, forfeited)
        for name, session in (("white", white), ("black", black)):
            session.trace.info["chess"] = {
                "score": scores[name],
                "result": "forfeit" if forfeited else board.result(claim_draw=True),
                "plies": len(board.move_stack),
                "illegal_moves": illegal[name],
            }

    @staticmethod
    def adjudicate(board: chess.Board, forfeited: str | None) -> dict[str, float]:
        """Score the finished game: forfeit loses, checkmate wins, anything else
        (stalemate, repetition, the max-plies cutoff) is a draw."""
        if forfeited is not None:
            return {forfeited: 0.0, ("black" if forfeited == "white" else "white"): 1.0}
        result = board.result(claim_draw=True)
        if result == "1-0":
            return {"white": 1.0, "black": 0.0}
        if result == "0-1":
            return {"white": 0.0, "black": 1.0}
        return {"white": 0.5, "black": 0.5}

    @vf.reward(agent="white")
    async def outcome_white(self, trace: vf.Trace) -> dict[str, float]:
        return {"outcome": trace.info["chess"]["score"]}

    @vf.reward(agent="black")
    async def outcome_black(self, trace: vf.Trace) -> dict[str, float]:
        return {"outcome": trace.info["chess"]["score"]}

    @vf.metric(agent="white")
    async def illegal_white(self, trace: vf.Trace) -> dict[str, float]:
        return {"illegal_moves": float(trace.info["chess"]["illegal_moves"])}

    @vf.metric(agent="black")
    async def illegal_black(self, trace: vf.Trace) -> dict[str, float]:
        return {"illegal_moves": float(trace.info["chess"]["illegal_moves"])}


__all__ = ["ChessTopology"]
