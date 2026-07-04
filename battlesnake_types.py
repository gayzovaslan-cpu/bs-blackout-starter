from enum import Enum
from abc import abstractmethod

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Union


# ---------------------------------------------------------
# Misc Models
# ---------------------------------------------------------
class Point(BaseModel):
    x: int
    y: int


class Food(Point):
    # Исправлено: Battlesnake присылает просто координаты, spawn_turn нет в базовом API
    spawn_turn: Optional[int] = None


class EliminatedCause(str, Enum):
    EliminatedByCollision = "snake-collision"
    EliminatedBySelfCollision = "snake-self-collision"
    EliminatedByOutOfHealth = "out-of-health"
    EliminatedByHeadToHeadCollision = "head-collision"
    EliminatedByOutOfBounds = "wall-collision"


class EliminationEvent(BaseModel):
    cause: EliminatedCause
    turn: int
    by: Optional[str] = None


# ---------------------------------------------------------
# Snake Model
# ---------------------------------------------------------
class SnakeCustomizations(BaseModel):
    # Исправлено: игра присылает строку ("#32cd32"), но мы разрешаем и Tuple, если это нужно вашей логике
    color: Union[str, Tuple[int, int, int]]
    head: Optional[str] = None
    tail: Optional[str] = None


class Snake(BaseModel):
    id: str
    name: str
    length: int
    latency: Optional[str] = None
    squad: Optional[str] = None
    health: Optional[int] = None
    head: Optional[Point] = None
    body: List[Optional[Point]]
    customizations: SnakeCustomizations
    elimination_event: Optional[EliminationEvent] = None


# ---------------------------------------------------------
# Game & Ruleset Models
# ---------------------------------------------------------
class RoyaleSettings(BaseModel):
    shrinkEveryNTurns: Optional[int] = None


class SquadSettings(BaseModel):
    allowBodyCollisions: Optional[bool] = None
    sharedElimination: Optional[bool] = None
    sharedHealth: Optional[bool] = None
    sharedLength: Optional[bool] = None


class RulesetSettings(BaseModel):
    # Разрешаем любые дополнительные новые поля от игрового движка движка
    model_config = {"extra": "ignore"}

    foodSpawnChance: Optional[int] = None
    hazardDamagePerTurn: Optional[int] = None
    minimumFood: Optional[int] = None
    viewRadius: Optional[int] = None
    royale: Optional[RoyaleSettings] = None
    squad: Optional[SquadSettings] = None


class Ruleset(BaseModel):
    name: str
    version: Optional[str] = None
    settings: Optional[RulesetSettings] = None


class Game(BaseModel):
    id: str
    source: str
    timeout: int
    ruleset: Ruleset


# ---------------------------------------------------------
# Board & Root GameState Models
# ---------------------------------------------------------
class Board(BaseModel):
    height: int
    width: int
    food: List[Food]
    hazards: List[Point]
    snakes: List[Snake]


class GameState(BaseModel):
    turn: int
    game: Game
    board: Board
    you: Snake


# ---------------------------------------------------------
# Snake Action Model
# ---------------------------------------------------------

class Direction(str, Enum):
    UP = 'up'
    RIGHT = 'right'
    DOWN = 'down'
    LEFT = 'left'

    @property
    def board_delta(self) -> Tuple[int, int]:
        if self is Direction.DOWN:
            return 0, -1
        if self is Direction.UP:
            return 0, 1
        if self is Direction.LEFT:
            return -1, 0
        if self is Direction.RIGHT:
            return 1, 0

    @classmethod
    def from_board_delta(cls, delta: Tuple[int, int]):
        for direction in cls:
            if direction.board_delta == delta:
                return direction

    @property
    def dx(self) -> int:
        return self.board_delta[0]

    @property
    def dy(self) -> int:
        return self.board_delta[1]


class MoveAction(BaseModel):
    move: Direction


# ---------------------------------------------------------
# Base Agent Interface
# ---------------------------------------------------------

class BaseAgent:

    @abstractmethod
    def get_name(self):
        pass

    def get_color(self):
        return '#32CD32'

    def get_author(self):
        return None

    @abstractmethod
    def start(self, game_state: GameState):
        pass

    @abstractmethod
    def move(self, game_state: GameState) -> MoveAction:
        pass

    @abstractmethod
    def end(self, game_state: GameState):
        pass
