"""
hexpath — игровая логика стратегии на графе Q6

Правила:
- Доска = граф Q6 (64 узла, 192 ребра)
- Два игрока: PLAYER_A (◉) и PLAYER_B (◎)
- Каждый ход = переход в соседний узел (изменение 1 черты)
- Победа: достичь целевой гексаграммы соперника ИЛИ заблокировать соперника

Вариант «Захват»:
- Посещённый узел захватывается игроком
- Соперник не может ходить на захваченный узел
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

from libs.hexcore.hexcore import neighbors, hamming, shortest_path, antipode, render, to_bits


class Player(Enum):
    A = 'A'
    B = 'B'

    def other(self) -> 'Player':
        return Player.B if self == Player.A else Player.A

    def symbol(self) -> str:
        return '◉' if self == Player.A else '◎'


class GameResult(Enum):
    ONGOING = 'ongoing'
    A_WINS = 'a_wins'
    B_WINS = 'b_wins'
    DRAW = 'draw'


@dataclass
class GameState:
    pos_a: int
    pos_b: int
    target_a: int          # цель игрока A
    target_b: int          # цель игрока B
    current_player: Player = Player.A
    captured: dict[int, Player] = field(default_factory=dict)
    history_a: list[int] = field(default_factory=list)
    history_b: list[int] = field(default_factory=list)
    capture_mode: bool = True

    def __post_init__(self) -> None:
        if not self.history_a:
            self.history_a = [self.pos_a]
        if not self.history_b:
            self.history_b = [self.pos_b]
        if self.capture_mode:
            self.captured.setdefault(self.pos_a, Player.A)
            self.captured.setdefault(self.pos_b, Player.B)

    def current_pos(self) -> int:
        return self.pos_a if self.current_player == Player.A else self.pos_b

    def current_target(self) -> int:
        return self.target_a if self.current_player == Player.A else self.target_b

    def legal_moves(self, player: Player | None = None) -> list[int]:
        """Все допустимые ходы для игрока."""
        p = player if player is not None else self.current_player
        pos = self.pos_a if p == Player.A else self.pos_b
        moves = []
        for nb in neighbors(pos):
            if self.capture_mode and nb in self.captured:
                owner = self.captured[nb]
                if owner != p:
                    continue  # захвачен соперником — нельзя
            moves.append(nb)
        return moves

    def make_move(self, destination: int) -> 'GameState':
        """Сделать ход. Возвращает новое состояние."""
        moves = self.legal_moves()
        if destination not in moves:
            raise ValueError(f"Ход {destination} недопустим. Допустимые: {moves}")

        new_captured = dict(self.captured)
        p = self.current_player

        if p == Player.A:
            new_pos_a = destination
            new_pos_b = self.pos_b
            new_hist_a = self.history_a + [destination]
            new_hist_b = self.history_b
        else:
            new_pos_a = self.pos_a
            new_pos_b = destination
            new_hist_a = self.history_a
            new_hist_b = self.history_b + [destination]

        if self.capture_mode:
            new_captured[destination] = p

        return GameState(
            pos_a=new_pos_a,
            pos_b=new_pos_b,
            target_a=self.target_a,
            target_b=self.target_b,
            current_player=p.other(),
            captured=new_captured,
            history_a=new_hist_a,
            history_b=new_hist_b,
            capture_mode=self.capture_mode,
        )

    def result(self) -> GameResult:
        """Определить текущий результат игры."""
        # Победа по достижению цели
        if self.pos_a == self.target_a:
            return GameResult.A_WINS
        if self.pos_b == self.target_b:
            return GameResult.B_WINS

        # Проверяем, есть ли ходы у текущего игрока
        moves_current = self.legal_moves(self.current_player)
        if not moves_current:
            # Текущий игрок заблокирован — проигрывает
            return GameResult.B_WINS if self.current_player == Player.A else GameResult.A_WINS

        return GameResult.ONGOING

    def is_over(self) -> bool:
        return self.result() != GameResult.ONGOING


# ---------------------------------------------------------------------------
# AI: minimax с alpha-beta отсечением
# ---------------------------------------------------------------------------

def _evaluate(state: GameState) -> float:
    """
    Эвристика для AI.
    Положительные значения — хорошо для A, отрицательные — хорошо для B.
    """
    res = state.result()
    if res == GameResult.A_WINS:
        return 1000.0
    if res == GameResult.B_WINS:
        return -1000.0
    if res == GameResult.DRAW:
        return 0.0

    # Расстояние до цели (меньше = лучше)
    dist_a = hamming(state.pos_a, state.target_a)
    dist_b = hamming(state.pos_b, state.target_b)

    # Подвижность: число доступных ходов
    mob_a = len(state.legal_moves(Player.A))
    mob_b = len(state.legal_moves(Player.B))

    score = (dist_b - dist_a) * 10 + (mob_a - mob_b) * 1
    return float(score)


def minimax(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
) -> float:
    if depth == 0 or state.is_over():
        return _evaluate(state)

    moves = state.legal_moves()
    if not moves:
        return _evaluate(state)

    if maximizing:
        value = float('-inf')
        for move in moves:
            child = state.make_move(move)
            value = max(value, minimax(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for move in moves:
            child = state.make_move(move)
            value = min(value, minimax(child, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def best_move(state: GameState, depth: int = 4) -> int:
    """Найти лучший ход для текущего игрока."""
    moves = state.legal_moves()
    if not moves:
        raise ValueError("Нет допустимых ходов")

    maximizing = (state.current_player == Player.A)
    best: int = moves[0]
    best_val = float('-inf') if maximizing else float('inf')

    for move in moves:
        child = state.make_move(move)
        val = minimax(child, depth - 1, float('-inf'), float('inf'), not maximizing)
        if maximizing and val > best_val:
            best_val = val
            best = move
        elif not maximizing and val < best_val:
            best_val = val
            best = move

    return best


def new_game(
    pos_a: int = 0,
    pos_b: int = 63,
    target_a: int | None = None,
    target_b: int | None = None,
    capture_mode: bool = True,
) -> GameState:
    """
    Создать новую игру.
    По умолчанию: A стартует с 0 и идёт к 63, B стартует с 63 и идёт к 0.
    """
    if target_a is None:
        target_a = pos_b   # A хочет достичь начальной позиции B
    if target_b is None:
        target_b = pos_a   # B хочет достичь начальной позиции A
    return GameState(
        pos_a=pos_a,
        pos_b=pos_b,
        target_a=target_a,
        target_b=target_b,
        capture_mode=capture_mode,
    )
