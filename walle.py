from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import BaseScheduler
import time


class Walle(Agent):
    def __init__(self, unique_id, model, meta):
        super().__init__(unique_id, model)
        self.meta = meta

    def step(self):
        x, y = self.pos
        gx, gy = self.meta

        # se mueve a la derecha si no ha llegado a la meta
        if x < gx:
            self.model.grid.move_agent(self, (x + 1, y))


class Grid(Model):
    def __init__(self):
        super().__init__()
        self.grid = MultiGrid(5, 5, torus=False)
        self.schedule = BaseScheduler(self)

        # las posiciones son de 0 a 4 para simular ese 1 a 5
        start = (0, 2) # lo mismo que (1,3)
        meta = (4, 2) # lo mismo que (5,3)

        self.walle = Walle(1, self, meta)
        self.grid.place_agent(self.walle, start)
        self.schedule.add(self.walle)

        self.meta = meta


def print_grid(model, paso_num):
    w = model.walle.pos
    g = model.meta

    print(f"\npaso {paso_num + 1}, posición: ({w[0]+1},{w[1]+1}), meta: ({g[0]+1},{g[1]+1})")
    for y in reversed(range(5)):
        row = []
        for x in range(5):
            if (x, y) == w:
                row.append("W")
            elif (x, y) == g:
                row.append("M")
            else:
                row.append(".")
        print(" ".join(row))


if __name__ == "__main__":
    model = Grid()

    paso = 0
    print_grid(model, paso)

    while model.walle.pos != model.meta:
        time.sleep(1) # se espera 1 segundo antes del siguiente paso
        paso += 1
        model.schedule.step()
        print_grid(model, paso)

    print("\nWalle llegó a la meta.")
