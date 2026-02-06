from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import BaseScheduler
import time
import json
import heapq


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

 # algoritmo A*
def astar(start, goal, blocked, width, height):
    def neighbors(p):
        x, y = p
        cand = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        valid = []
        for nx, ny in cand:
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in blocked:
                valid.append((nx, ny))
        return valid

    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {}
    g_score = {start: 0}
    in_open = {start}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        in_open.discard(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nb in neighbors(current):
            tentative_g = g_score[current] + 1
            if nb not in g_score or tentative_g < g_score[nb]:
                came_from[nb] = current
                g_score[nb] = tentative_g
                f = tentative_g + manhattan(nb, goal)
                if nb not in in_open:
                    heapq.heappush(open_heap, (f, nb))
                    in_open.add(nb)

    return None


# Agente walle
class WalleAStar(Agent):
    def __init__(self, unique_id, model, goal):
        super().__init__(unique_id, model)
        self.goal = goal
        self.path = None
        self.i = 0

    def plan(self):
        self.path = astar(self.pos, self.goal, self.model.blocked, self.model.width, self.model.height)
        self.i = 0

    def step(self):
        if self.path is None:
            self.plan()
            return

        if self.path is None or self.pos == self.goal:
            return

        if self.i + 1 < len(self.path):
            self.model.grid.move_agent(self, self.path[self.i + 1])
            self.i += 1


 # clase del grid 11x11 con variable modificable
class GridWorld(Model):
    def __init__(self, width=11, height=11, start=(0, 0), goal=(10, 10), obstacles=None):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = BaseScheduler(self)

        self.start = start
        self.goal = goal
        self.blocked = set(obstacles) if obstacles else set()
        self.blocked.discard(self.start)
        self.blocked.discard(self.goal)

        self.walle = WalleAStar(1, self, self.goal)
        self.grid.place_agent(self.walle, self.start)
        self.schedule.add(self.walle)

        self.step_count = 0

    def at_goal(self):
        return self.walle.pos == self.goal


def print_grid(model):
    w = model.walle.pos
    g = model.goal

    print(f"\npaso {model.step_count}, posición: ({w[0]+1},{w[1]+1}), meta: ({g[0]+1},{g[1]+1})")
    for y in reversed(range(model.height)):
        row = []
        for x in range(model.width):
            p = (x, y)
            if p == w:
                row.append("W")
            elif p == g:
                row.append("M")
            elif p in model.blocked:
                row.append("#")
            else:
                row.append(".")
        print(" ".join(row))


def export_json(model, filename="escenario.json"):
    data = {
        "width": model.width,
        "height": model.height,
        "start": [model.start[0], model.start[1]],
        "goal": [model.goal[0], model.goal[1]],
        "obstacles": [[x, y] for (x, y) in sorted(model.blocked)],
        "path": [[x, y] for (x, y) in model.walle.path] if model.walle.path else None
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":

    # definir obstáculos
    obstacles = {
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (6, 6), (7, 6), (8, 6), (9, 6),
        (6, 7), (6, 8), (6, 10),
    }

    # definir grid
    model = GridWorld(width=11, height=11, start=(0, 0), goal=(10, 10), obstacles=obstacles)

    # exportar escenario para Unity
    model.walle.plan()
    export_json(model, "escenario.json")

    # botón para empezar el juego
    print("\n preciona 1 para empezar la simulación")
    op = input("elige opción: ").strip()
    if op != "1":
        print("simulación detenida")
        raise SystemExit

    # simulación por pasos
    print_grid(model)

    running = True
    while running and not model.at_goal():
        time.sleep(1)  # paso cada segundo
        model.step_count += 1
        model.schedule.step()
        print_grid(model)

        if model.walle.path is None:
            print("\nno se puede continuar")
            break

        # “Botón” STOP (durante ejecución)
        cmd = input("siguiente paso: Enter, parar simulación: s").strip().lower()
        if cmd == "s":
            running = False

    if model.at_goal():
        print("\n Walle llegó a la meta")
    elif not running:
        print("\n se detuvo la simulación por el usuario")
