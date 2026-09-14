"""Thin local-GameState adapter for OpenSpiel's Python PUCT implementation."""

# Native bindings are built in the Docker runtime.
# pyright: reportMissingImports=false, reportUntypedBaseClass=false

from __future__ import annotations

import importlib.metadata
import os
import time
from dataclasses import dataclass

import numpy as np
import pyspiel
from numpy.typing import NDArray
from open_spiel.python.algorithms import mcts
from zero_ttt.config import GameConfig, SearchConfig
from zero_ttt.game.rules import ACTION_SIZE, BOARD_SIZE, PASS_ACTION, Color
from zero_ttt.game.state import GameResult, GameState

from zero_ttt_selfplay_worker.inference.batching import BatchedInferenceBroker, StateEvaluation

OPEN_SPIEL_VERSION = "2.0.1"
OPEN_SPIEL_REVISION = "112b77704631fc2ce7ad8e4581f6ca09798ce15a"


def validate_open_spiel_runtime() -> None:
    if importlib.metadata.version("open-spiel") != OPEN_SPIEL_VERSION:
        raise RuntimeError(f"Zero-TTT requires OpenSpiel {OPEN_SPIEL_VERSION}")
    runtime_revision = os.environ.get("ZERO_TTT_OPEN_SPIEL_REVISION")
    if runtime_revision is not None and runtime_revision != OPEN_SPIEL_REVISION:
        raise RuntimeError("OpenSpiel runtime revision does not match the audited gitlink")


_GAME_TYPE = pyspiel.GameType(
    short_name="zero_ttt_python_go",
    long_name="Zero-TTT Python Tromp-Taylor Go",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=False,
    provides_observation_tensor=False,
    parameter_specification={},
)


class OpenSpielGoGame(pyspiel.Game):
    def __init__(self, config: GameConfig) -> None:
        validate_open_spiel_runtime()
        if config.board_size != BOARD_SIZE:
            raise ValueError("OpenSpiel adapter only supports 19x19")
        self.config = config
        self.rules_seconds = 0.0
        info = pyspiel.GameInfo(
            num_distinct_actions=ACTION_SIZE,
            max_chance_outcomes=0,
            num_players=2,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=config.max_moves,
        )
        super().__init__(_GAME_TYPE, info, {})

    def new_initial_state(self) -> OpenSpielGoState:
        return OpenSpielGoState(self, GameState.new(self.config))


class OpenSpielGoState(pyspiel.State):
    def __init__(self, game: OpenSpielGoGame, local_state: GameState) -> None:
        super().__init__(game)
        self._game = game
        self.local_state = local_state
        self._legal_cache: tuple[int, ...] | None = None

    def clone(self) -> OpenSpielGoState:
        clone = OpenSpielGoState(self._game, self.local_state)
        clone._legal_cache = self._legal_cache
        return clone

    def current_player(self) -> int:
        if self.local_state.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        return 0 if self.local_state.to_play is Color.BLACK else 1

    def _legal_actions(self, player: int) -> list[int]:
        del player
        if self.local_state.is_terminal():
            return []
        if self._legal_cache is None:
            started = time.perf_counter()
            self._legal_cache = tuple(
                action for action, legal in enumerate(self.local_state.legal_actions()) if legal
            )
            self._game.rules_seconds += time.perf_counter() - started
        return list(self._legal_cache)

    def _apply_action(self, action: int) -> None:
        started = time.perf_counter()
        self.local_state = self.local_state.play(action)
        self._game.rules_seconds += time.perf_counter() - started
        self._legal_cache = None

    def _action_to_string(self, player: int, action: int) -> str:
        color = "B" if player == 0 else "W"
        if action == PASS_ACTION:
            return f"{color}(pass)"
        row, column = divmod(action, BOARD_SIZE)
        return f"{color}({row},{column})"

    def is_terminal(self) -> bool:
        return self.local_state.is_terminal()

    def returns(self) -> list[float]:
        if not self.local_state.is_terminal():
            return [0.0, 0.0]
        winner = self.local_score().winner
        if winner is None:
            return [0.0, 0.0]
        black = 1.0 if winner is Color.BLACK else -1.0
        return [black, -black]

    def local_score(self) -> GameResult:
        started = time.perf_counter()
        result = self.local_state.score()
        self._game.rules_seconds += time.perf_counter() - started
        return result

    def __str__(self) -> str:
        return (
            f"ZeroTTTGo(move={self.local_state.move_number},"
            f"to_play={self.local_state.to_play.name})"
        )


class OpenSpielEvaluator(mcts.Evaluator):
    def __init__(self, broker: BatchedInferenceBroker) -> None:
        self.broker = broker

    @staticmethod
    def _local(state: pyspiel.State) -> GameState:
        if not isinstance(state, OpenSpielGoState):
            raise TypeError("OpenSpiel evaluator received an unrelated game state")
        return state.local_state

    def state_evaluation(self, state: pyspiel.State) -> StateEvaluation:
        return self.broker.evaluate(self._local(state))

    def prior(self, state: pyspiel.State) -> list[tuple[int, float]]:
        local = self._local(state)
        legal = np.asarray(local.legal_actions(), dtype=np.bool_)
        logits = np.asarray(self.broker.evaluate(local).policy_logits, dtype=np.float64)
        legal_actions = np.flatnonzero(legal)
        legal_logits = logits[legal_actions]
        maximum = float(np.max(legal_logits))
        masses = np.exp(legal_logits - maximum)
        total = float(masses.sum())
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("model produced an invalid legal policy")
        masses /= total
        return [
            (int(action), float(mass)) for action, mass in zip(legal_actions, masses, strict=True)
        ]

    def evaluate(self, state: pyspiel.State) -> NDArray[np.float32]:
        local = self._local(state)
        current_value = self.broker.evaluate(local).value
        black_value = current_value if local.to_play is Color.BLACK else -current_value
        return np.asarray([black_value, -black_value], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class MCTSSearchResult:
    action: int
    policy_actions: tuple[int, ...]
    policy_values: tuple[float, ...]
    simulations: int
    root_value: float
    root_score_margin: float
    root_score_available: bool
    temperature: float
    search_seed: int


def _select_action(
    visits: NDArray[np.int64],
    move_number: int,
    config: SearchConfig,
    selection_seed: int,
) -> tuple[int, float]:
    temperature = config.temperature if move_number < config.temperature_drop_ply else 0.0
    if temperature <= 0:
        return int(np.argmax(visits)), temperature
    weights = visits.astype(np.float64) ** (1.0 / temperature)
    weights /= weights.sum()
    action = int(np.random.default_rng(selection_seed).choice(ACTION_SIZE, p=weights))
    return action, temperature


def search_position(
    game: OpenSpielGoGame,
    state: OpenSpielGoState,
    evaluator: OpenSpielEvaluator,
    config: SearchConfig,
    *,
    search_seed: int,
    selection_seed: int,
) -> MCTSSearchResult:
    if state.is_terminal():
        raise ValueError("cannot search a terminal state")
    random_state = np.random.RandomState(search_seed & 0xFFFF_FFFF)
    bot = mcts.MCTSBot(
        game,
        config.uct_c,
        config.max_simulations,
        evaluator,
        solve=False,
        random_state=random_state,
        child_selection_fn=mcts.SearchNode.puct_value,
        dirichlet_noise=(
            None
            if config.dirichlet_epsilon == 0
            else (config.dirichlet_epsilon, config.dirichlet_alpha)
        ),
        dont_return_chance_node=True,
    )
    root = bot.mcts_search(state)
    visits = np.zeros(ACTION_SIZE, dtype=np.int64)
    for child in root.children:
        visits[child.action] = child.explore_count
    positive = np.flatnonzero(visits > 0)
    total = int(visits.sum())
    if total <= 0:
        raise RuntimeError("MCTS root has no visited legal child")
    policy = visits[positive].astype(np.float64) / total
    action, temperature = _select_action(
        visits, state.local_state.move_number, config, selection_seed
    )
    root_value = float(root.total_reward / root.explore_count) if root.explore_count else 0.0
    raw_root = evaluator.state_evaluation(state)
    return MCTSSearchResult(
        action=action,
        policy_actions=tuple(int(value) for value in positive),
        policy_values=tuple(float(value) for value in policy.astype(np.float32)),
        simulations=int(root.explore_count),
        root_value=root_value,
        root_score_margin=(0.0 if raw_root.score_margin is None else float(raw_root.score_margin)),
        root_score_available=raw_root.score_margin is not None,
        temperature=temperature,
        search_seed=search_seed,
    )
