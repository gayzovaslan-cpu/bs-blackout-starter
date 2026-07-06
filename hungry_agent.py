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


def count_free_space(grid: np.ndarray, start_r: int, start_c: int) -> int:
    """
    Считает количество свободных связанных клеток с помощью быстрого BFS.
    """
    h, w = grid.shape
    if not (0 <= start_r < h and 0 <= start_c < w) or grid[start_r, start_c]:
        return 0

    visited = np.zeros_like(grid, dtype=bool)
    queue = [(start_r, start_c)]
    visited[start_r, start_c] = True
    count = 0

    head_idx = 0
    while head_idx < len(queue):
        r, c = queue[head_idx]
        head_idx += 1
        count += 1

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if not grid[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    return count


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
        if game_state.game.id not in self.agent_states:
            self.agent_states[game_state.game.id] = AgentState(possible_food=[])

        agent_state = self.agent_states[game_state.game.id]
        head = game_state.you.head
        assert head is not None
        my_length = game_state.you.length

        # Работа с радиусом видимости
        try:
            view_radius = game_state.game.ruleset.settings.viewRadius
        except AttributeError:
            view_radius = None

        if view_radius is not None:
            vision_mask = get_vision_mask(width=game_state.board.width, height=game_state.board.height, center=head,
                                          radius=view_radius)
            updated_food = []
            for food in agent_state.possible_food:
                if not vision_mask[food.y, food.x]:
                    updated_food.append(food)
            visible_food = game_state.board.food
            for food in visible_food:
                if food not in updated_food:
                    updated_food.append(food)
            agent_state.possible_food = updated_food
        else:
            agent_state.possible_food = game_state.board.food

        obstacle_map = get_obstacle_map(game_state)

        # --- НОВАЯ СТРАТЕГИЯ: Считаем свободное место вокруг головы ---
        move_space = {}
        for d in Direction:
            dx, dy = d.board_delta
            next_x, next_y = head.x + dx, head.y + dy
            if 0 <= next_x < game_state.board.width and 0 <= next_y < game_state.board.height:
                move_space[d] = count_free_space(obstacle_map, next_y, next_x)
            else:
                move_space[d] = 0

        # Ищем еду через A*, отсекая ловушки
        result_direction = None
        min_distance = float('inf')
        for food in agent_state.possible_food:
            direction, length = a_star_wrapper(obstacle_map, head, food)
            if direction is not None and length < min_distance:
                # Идем к еде, только если там достаточно места для маневра
                if move_space.get(direction, 0) >= my_length:
                    result_direction = direction
                    min_distance = length

        # Фолбэк 1: Преследование хвоста (тоже с проверкой места)
        if result_direction is None and game_state.you.body:
            tail = game_state.you.body[-1]
            if tail is not None:
                direction, _ = a_star_wrapper(obstacle_map, head, tail)
                if direction is not None and move_space.get(direction, 0) >= my_length:
                    result_direction = direction

        # Фолбэк 2: Умный выбор направления с максимальным пространством
        if result_direction is None:
            safe_moves = [d for d, space in move_space.items() if space > 0]
            if safe_moves:
                result_direction = max(safe_moves, key=lambda d: move_space[d])
            else:
                result_direction = Direction.UP

        return MoveAction(move=result_direction)

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

            if not (0 <= neighbor[0] < h and 0 <= neighbor[1] < w):
                continue
            if grid[neighbor[0], neighbor[1]] and neighbor != goal:
                continue

            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                heapq.heappush(open_set, (f_score, neighbor))

    return None


if __name__ == "__main__":
    import sys
    import os
    from battlesnake_server import start_server

    agent = HungryAgent()

    if "PORT" in os.environ:
        port = int(os.environ["PORT"])
    elif len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 8000

    start_server(agent=agent, port=port)
