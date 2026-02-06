import numpy as np
import random
import time
import heapq
import csv
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import BaseScheduler

# configuración global
WIDTH = 11
HEIGHT = 11
START = (0, 0)
GOAL = (10, 10)
NUM_MAPS = 10  # cantidad de mapas por densidad
DENSITIES = [0.1, 0.3, 0.5] # 10%, 30%, 50% de obstáculos

# A*

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_search(start, goal, blocked, width, height):
    """Retorna la lista de pasos (camino) o None si no hay camino."""
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


# Q-Learning

ACTIONS = [0, 1, 2, 3]  # Arriba, Abajo, Izquierda, Derecha
ACTION_MOVES = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}

class QL_Params:
    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON = 0.1
    EPISODES = 500
    MAX_STEPS = 100

class WalleQLearner(Agent):
    def __init__(self, unique_id, model, start, goal):
        super().__init__(unique_id, model)
        self.start_pos = start
        self.goal_pos = goal
        self.q_table = {} 

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        return self.q_table[state]

    def choose_action(self, state, is_training=True):
        if is_training and random.uniform(0, 1) < QL_Params.EPSILON:
            return random.choice(ACTIONS)
        else:
            q_values = self.get_q(state)
            max_val = np.max(q_values)
            best_actions = [i for i, val in enumerate(q_values) if val == max_val]
            return random.choice(best_actions)

    def train(self):
        """Entrena al agente y retorna cuánto tardó."""
        start_time = time.time()
        for _ in range(QL_Params.EPISODES):
            state = self.start_pos
            done = False
            steps = 0
            while not done and steps < QL_Params.MAX_STEPS:
                action = self.choose_action(state, is_training=True)
                dx, dy = ACTION_MOVES[action]
                nx, ny = state[0] + dx, state[1] + dy
                
                # recompensa
                reward = -1
                next_state = (nx, ny)
                
                # checar meta
                if (nx, ny) == self.goal_pos:
                    reward = 10
                    done = True
                # checar obstáculo o límite
                elif (not (0 <= nx < self.model.width and 0 <= ny < self.model.height)) or \
                     ((nx, ny) in self.model.blocked):
                    reward = -10
                    next_state = state # Rebote
                
                # regla de actualización Q
                old_q = self.get_q(state)[action]
                max_next_q = np.max(self.get_q(next_state))
                new_q = old_q + QL_Params.ALPHA * (reward + QL_Params.GAMMA * max_next_q - old_q)
                self.q_table[state][action] = new_q
                
                state = next_state
                steps += 1
        end_time = time.time()
        return end_time - start_time

    def run_policy(self):
        """Ejecuta la ruta aprendida y retorna (éxito, pasos)."""
        state = self.start_pos
        steps = 0
        path_taken = [state]
        
        while steps < QL_Params.MAX_STEPS:
            if state == self.goal_pos:
                return True, steps
            
            action = self.choose_action(state, is_training=False)
            dx, dy = ACTION_MOVES[action]
            nx, ny = state[0] + dx, state[1] + dy
            
            # validar movimiento
            if (0 <= nx < self.model.width and 0 <= ny < self.model.height) and \
               ((nx, ny) not in self.model.blocked):
                state = (nx, ny)
            # si choca o se sale, se queda en el mismo estado (rebote)
            path_taken.append(state)
            steps += 1
            
        return False, steps # si no llega en el límite de pasos

class GridWorldQL(Model):
    def __init__(self, width, height, start, goal, obstacles):
        super().__init__()
        self.width = width
        self.height = height
        self.blocked = set(obstacles)
        self.walle = WalleQLearner(1, self, start, goal)

# se genera el mapa y el experimiento

def generate_obstacles(density):
    """genera obstáculos aleatorios"""
    total_cells = WIDTH * HEIGHT
    num_obstacles = int(total_cells * density)
    
    possible_locs = []
    for x in range(WIDTH):
        for y in range(HEIGHT):
            if (x, y) != START and (x, y) != GOAL:
                possible_locs.append((x, y))
    
    if num_obstacles > len(possible_locs):
        num_obstacles = len(possible_locs)
        
    return set(random.sample(possible_locs, num_obstacles))

def run_experiment():
    results = []
    
    print(f"iniciando experimento ({len(DENSITIES) * NUM_MAPS} escenarios)...")
    print("------------------------------------------------------------")
    print(f"{'densidad':<10} | {'algoritmo':<10} | {'exito':<6} | {'pasos':<6} | {'tiempo':<10}")
    print("------------------------------------------------------------")

    for density in DENSITIES:
        for i in range(NUM_MAPS):
            # genera el mapa
            obstacles = generate_obstacles(density)
            
            # prueba A*
            t0 = time.time()
            path = astar_search(START, GOAL, obstacles, WIDTH, HEIGHT)
            t1 = time.time()
            
            astar_time = t1 - t0
            astar_success = path is not None
            # se resta 1 porque el path incluye el start
            astar_steps = (len(path) - 1) if path else 0 
            
            results.append({
                "mapID": i,
                "densidad": density,
                "algoritmo": "A*",
                "exito": astar_success,
                "pasos": astar_steps,
                "tiempo": round(astar_time, 5)
            })
            print(f"{density:<10} | {'A*':<10} | {str(astar_success):<6} | {astar_steps:<6} | {astar_time:.5f}")

            # pureba Q-Learning
            # se instancia el modelo nuevo
            model_ql = GridWorldQL(WIDTH, HEIGHT, START, GOAL, obstacles)
            
            # entrenamiento
            train_time = model_ql.walle.train()
            
            # ejecución
            ql_success, ql_steps = model_ql.walle.run_policy()
            
            results.append({
                "mapID": i,
                "densidad": density,
                "algoritmo": "Q-Learn",
                "exito": ql_success,
                "pasos": ql_steps,
                "tiempo": round(train_time, 5)
            })
            print(f"{density:<10} | {'Q-Learn':<10} | {str(ql_success):<6} | {ql_steps:<6} | {train_time:.5f}")

    # guardar en csv
    csv_file = "resultados_experimento.csv"
    keys = results[0].keys()
    with open(csv_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    print("------------------------------------------------------------")
    print(f"experimento finalizado. los resultados se guardaron en '{csv_file}'")

if __name__ == "__main__":
    run_experiment()