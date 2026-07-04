from dataclasses import dataclass
import heapq
import numpy as np
from typing import List, Tuple

from battlesnake_types import Food, GameState, MoveAction, Direction, BaseAgent, Point


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_obstacle_map(game_state: GameState):
    obstacle_map = np.zeros((game_state.board.height, game_state.board.width), dtype=bool)

    for snake in game_state.board.snakes:
        for body_part in snake.body[:-1]:
            if body_part is None:
                # we don't see this body section, could be many parts long
                continue
            obstacle_map[body_part.y, body_part.x] = 1

    return obstacle_map


def get_vision_mask(width: int, height: int, center: Point, radius: int) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    dist_sq = abs(x - center.x) + abs(y - center.y)
    return dist_sq <= radius


# ---------------------------------------------------------
# Battlesnake Agent Implementation
# ---------------------------------------------------------
@dataclass
class AgentState:
    possible_food: list[Food]


class HungryAgent(BaseAgent):
    def __init__(self):
        self.agent_states: dict[str, AgentState] = {}

    def get_name(self):
        return "Hungry Caterpillar"

    def get_color(self):
        return "#FFC13C"

    def get_author(self):
        return "Gluttony"

    def start(self, game_state: GameState):
        """start is called when the battlesnake begins a game"""
        self.agent_states[game_state.game.id] = AgentState(possible_food=[])

    def move(self, game_state: GameState) -> MoveAction:
        """move is called on every turn and returns your next move"""
        # Безопасное получение состояния игры
        if game_state.game.id not in self.agent_states:
            self.agent_states[game_state.game.id] = AgentState(possible_food=[])

        agent_state = self.agent_states[game_state.game.id]
        head = game_state.you.head
        assert head is not None

        # ИСПРАВЛЕНИЕ 1: Безопасная работа с радиусом видимости (для обычного сайта и Blackout)
        try:
            view_radius = game_state.game.ruleset.settings.viewRadius
        except AttributeError:
            view_radius = None

        if view_radius is not None:
            vision_mask = get_vision_mask(width=game_state.board.width, height=game_state.board.height, center=head,
                                          radius=view_radius)
            updated_food = []
            # сохраняем еду, которую сейчас не видим из-за тумана
            for food in agent_state.possible_food:
                if not vision_mask[food.y, food.x]:
                    updated_food.append(food)
            # добавляем видимую еду
            visible_food = game_state.board.food
            for food in visible_food:
                if food not in updated_food:
                    updated_food.append(food)
            agent_state.possible_food = updated_food
        else:
            # Если тумана войны нет (обычный сайт), просто берем всю еду с поля
            agent_state.possible_food = game_state.board.food

        # build an obstacle map
        obstacle_map = get_obstacle_map(game_state)

        # use A* to find closest food and the required move
        result_direction = None
        min_distance = float('inf')
        for food in agent_state.possible_food:
            direction, length = a_star_wrapper(obstacle_map, head, food)
            if length < min_distance and direction is not None:
                result_direction = direction
                min_distance = length

        # ИСПРАВЛЕНИЕ 2: Безопасное следование за хвостом (если хвост скрыт туманом)
        if result_direction is None and game_state.you.body:
            tail = game_state.you.body[-1]
            if tail is not None:
                result_direction, _ = a_star_wrapper(obstacle_map, head, tail)

        # second fallback (if tail is unreachable): random move
        if result_direction is None:
            result_direction = self.random_fallback_move(game_state, obstacle_map)

        return MoveAction(move=result_direction)

    def random_fallback_move(self, game_state: GameState, obstacle_map: np.ndarray) -> Direction:
        head = game_state.you.head
        assert head is not None
        clear_directions = []
        for d in Direction:
            # ИСПРАВЛЕНИЕ 3: Исправлено обращение к d.dx и d.dy (в оригинале Direction они возвращали tuple)
            dx, dy = d.board_delta

            # check out-of-bounds
            if head.x + dx < 0 or head.x + dx >= game_state.board.width:
                continue
            if head.y + dy < 0 or head.y + dy >= game_state.board.height:
                continue

            # check for obstacles (i.e., snake parts)
            if obstacle_map[head.y + dy, head.x + dx]:
                continue

            clear_directions.append(d)

        result_direction = clear_directions[
            np.random.choice(len(clear_directions))] if clear_directions else Direction.UP
        return result_direction

    def end(self, game_state: GameState):
        """end is called when the battlesnake finishes a game"""
        if game_state.game.id in self.agent_states:
            del self.agent_states[game_state.game.id]


# ---------------------------------------------------------
# A* Algorithm
# ---------------------------------------------------------
def a_star_wrapper(grid: np.ndarray, start: Point, goal: Point) -> tuple[Direction | None, int]:
    """Converts from battlesnake x-y coords to i-j index-tuples used by a_star()."""
    path = a_star(grid, (start.y, start.x), (goal.y, goal.x))
    if path is None or len(path) < 2:
        return None, 9999999

    next_pos = path[1]
    result_direction = Direction.from_board_delta((next_pos[1] - start.x, next_pos[0] - start.y))
    return result_direction, len(path)


def a_star(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]] | None:
    h, w = grid.shape

    # Исправлено: Корректная проверка границ для строк и столбцов
    if goal[0] < 0 or goal[0] >= h or goal[1] < 0 or goal[1] >= w:
        return None
    if grid[goal[0], goal[1]] and start != goal:
        return None

    open_set = [(0, start)]
    g_score = {start: 0}
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (r + dr, c + dc)

            # Исправлено: Корректная проверка соседей
            if not (0 <= neighbor[0] < h and 0 <= neighbor[1] < w):
                continue
            if grid[neighbor[0], neighbor[1]] and neighbor != goal:
                continue

            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                # Манхэттенское расстояние
                f_score = tentative_g + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                heapq.heappush(open_set, (f_score, neighbor))

    return None


if __name__ == "__main__":
    import sys
    import os
    from battlesnake_server import start_server

    agent = HungryAgent()

    # Исправлено: Сначала проверяем переменную окружения Render, затем аргументы CLI
    if "PORT" in os.environ:
        port = int(os.environ["PORT"])
    elif len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 8000

    start_server(agent=agent, port=port)
