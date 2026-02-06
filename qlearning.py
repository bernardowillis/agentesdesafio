import numpy as np
import random
import json
import time
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import BaseScheduler

ACTIONS = [0, 1, 2, 3]  # arriba, abajo, izquierda, derecha
ACTION_MOVES = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)} # (dx, dy)

class QL_Params:
    ALPHA = 0.1 # tasa de aprendizaje
    GAMMA = 0.9 # factor de descuento
    EPSILON = 0.1 # probabilidad de exploración (epsilon greedy)
    EPISODES = 500 # cantidad de episodios de entrenamiento
    MAX_STEPS = 100 # pasos máximos por episodio

# --- Agente Q-Learning ---
class WalleQLearner(Agent):
    def __init__(self, unique_id, model, start, goal):
        super().__init__(unique_id, model)
        self.start_pos = start
        self.goal_pos = goal
        self.q_table = {}  # Formato: {(x,y): [q0, q1, q2, q3]}
        self.policy_ready = False # Switch para saber si ya entrenamos

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS))
        return self.q_table[state]

    def choose_action(self, state, is_training=True):
        # epsilon-greedy
        if is_training and random.uniform(0, 1) < QL_Params.EPSILON:
            return random.choice(ACTIONS)
        else:
            # explotación
            q_values = self.get_q(state)
            # np.argmax puede ser determinista con empates, añadimos ruido aleatorio mínimo o choice
            max_val = np.max(q_values)
            best_actions = [i for i, val in enumerate(q_values) if val == max_val]
            return random.choice(best_actions)

    def train(self):
        """Ejecuta el ciclo completo de entrenamiento (Episodios)"""
        print(f"Iniciando entrenamiento ({QL_Params.EPISODES} episodios)...")
        
        for episode in range(QL_Params.EPISODES):
            state = self.start_pos
            done = False
            steps = 0
            
            while not done and steps < QL_Params.MAX_STEPS:
                action = self.choose_action(state, is_training=True)
                
                # calcular siguiente estado hipotético
                dx, dy = ACTION_MOVES[action]
                next_x, next_y = state[0] + dx, state[1] + dy
                
                reward = 0
                next_state = state # Si choca, se queda donde mismo
                
                # recompensas
                # si llega a la meta
                if (next_x, next_y) == self.goal_pos:
                    reward = 10
                    next_state = (next_x, next_y)
                    done = True
                
                # si choca o se sale del mapa
                elif (not (0 <= next_x < self.model.width and 0 <= next_y < self.model.height)) or \
                     ((next_x, next_y) in self.model.blocked):
                    reward = -10
                    next_state = state # no se mueve
                
                # movimiento válido
                else:
                    reward = -1
                    next_state = (next_x, next_y)
                
                # actualización Bellman
                # Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
                old_q = self.get_q(state)[action]
                max_next_q = np.max(self.get_q(next_state))
                
                new_q = old_q + QL_Params.ALPHA * (reward + QL_Params.GAMMA * max_next_q - old_q)
                self.q_table[state][action] = new_q
                
                state = next_state
                steps += 1
        
        self.policy_ready = True
        print("Entrenamiento finalizado.")
        self.save_q_table()

    def step(self):
        """Paso para el modo EJECUCIÓN (Visualización)"""
        if not self.policy_ready:
            print("¡Error! Debes entrenar primero.")
            return

        action = self.choose_action(self.pos, is_training=False)
        dx, dy = ACTION_MOVES[action]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        
        # se mueve si es válido
        # el agente aprendió a no chocar, pero validamos por seguridad visual
        if 0 <= nx < self.model.width and 0 <= ny < self.model.height and (nx, ny) not in self.model.blocked:
            self.model.grid.move_agent(self, (nx, ny))

    def save_q_table(self):
        # Creamos una lista limpia para Unity
        unity_data = []
        
        for state, values in self.q_table.items():
            # state es una tupla (x, y)
            entry = {
                "x": int(state[0]),
                "y": int(state[1]),
                "qValues": values.tolist() # Lista de 4 floats
            }
            unity_data.append(entry)
            
        # Envolvemos en un objeto raíz para que JsonUtility lo lea fácil
        final_json = {"rows": unity_data}
        
        with open("q_table_unity.json", "w") as f:
            json.dump(final_json, f, indent=2)
            
        print("Q-Table guardada optimizada para Unity en 'q_table_unity.json'")


# grid
class GridWorldQL(Model):
    def __init__(self, width=11, height=11, start=(0, 0), goal=(10, 10), obstacles=None):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = BaseScheduler(self)
        self.start = start
        self.goal = goal
        self.blocked = set(obstacles) if obstacles else set()
        
        # se asegura que start y goal no estén bloqueados
        self.blocked.discard(self.start)
        self.blocked.discard(self.goal)

        self.walle = WalleQLearner(1, self, self.start, self.goal)
        self.grid.place_agent(self.walle, self.start)
        self.schedule.add(self.walle)
        self.step_count = 0

    def at_goal(self):
        return self.walle.pos == self.goal
    
    def reset_agent(self):
        self.grid.move_agent(self.walle, self.start)
        self.step_count = 0

# visualización
def print_grid(model):
    w = model.walle.pos
    g = model.goal
    print(f"\nPaso {model.step_count} | Pos: {w}")
    for y in reversed(range(model.height)):
        row = []
        for x in range(model.width):
            p = (x, y)
            if p == w: row.append("W")      # Walle
            elif p == g: row.append("M")    # Meta
            elif p in model.blocked: row.append("#") # Obstáculo
            else: row.append(".")           # Libre
        print(" ".join(row))


if __name__ == "__main__":
    # obstáculos predefinidos
    obstacles = {
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (6, 6), (7, 6), (8, 6), (9, 6),
        (6, 7), (6, 8), (6, 10),
    }
    model = GridWorldQL(width=11, height=11, start=(0, 0), goal=(10, 10), obstacles=obstacles)

    while True:
        print("\nmenú q-learning:")
        print("1. entrenar")
        print("2. ejecutar política")
        print("3. resetear posición")
        print("4. salir")
        
        op = input("Selecciona opción: ").strip()

        if op == "1":
            # entrenamiento
            model.walle.train()
            model.reset_agent() # se regresa al inicio para iniciar la ejecución

        elif op == "2":
            # ejecución paso a paso
            if not model.walle.policy_ready:
                print("¡Primero debes entrenar al agente (Opción 1)!")
                continue
            
            print_grid(model)
            running = True
            while running and not model.at_goal():
                cmd = input("[enter] sig. paso | [s] salir al menú: ").lower()
                if cmd == 's': break
                
                model.step_count += 1
                model.schedule.step() # Llama a walle.step() que usa argmax
                print_grid(model)
            
            if model.at_goal(): print("\n¡LLEGÓ A LA META!")

        elif op == "3":
            model.reset_agent()
            print("Agente reiniciado a Start.")

        elif op == "4":
            print("Adiós.")
            break