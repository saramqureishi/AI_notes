"""
Lab 3 - Task 3.3 The Map Colouring Problem with a Genetic Algorithm (DEAP)
PROBLEM: colour 11 regions (people/nodes) using 4 colours so that no two
         NEIGHBOURS share the same colour (classic graph-colouring / CSP).
  Encoding : list of 11 integers in {0,1,2,3}, one colour per node
  Fitness  : number of edges whose two endpoints have the SAME colour
             (MINIMISE; best = 0 = valid colouring)
  Operators: two-point crossover, uniform-int mutation, tournament selection
"""
import array, random, numpy as np
from deap import creator, base, tools, algorithms

numColour = 4
numNames = 11
colours = ('red', 'green', 'blue', 'gray')
names = ('Mark', 'Julia', 'Steve', 'Amanda', 'Brian',
         'Joanne', 'Derek', 'Allan', 'Michelle', 'Kelly', 'Chris')

neighbours = [[0,1,1,0,0,0,0,0,0,0,0],
              [0,0,1,1,1,0,1,0,0,0,0],
              [0,0,0,1,0,0,0,1,1,0,0],
              [0,0,0,0,0,1,1,0,1,0,0],
              [0,0,0,0,0,0,1,0,0,1,0],
              [0,0,0,0,0,0,1,0,1,1,1],
              [0,0,0,0,0,0,0,0,0,1,1],
              [0,0,0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,1],
              [0,0,0,0,0,0,0,0,0,0,0]]


def evalMapColouring(ind):
    val = 0
    for i in range(numNames):
        for j in range(numNames):
            if neighbours[i][j] == 1 and ind[i] == ind[j]:
                val += 1
    return val,


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_colour", random.randint, 0, numColour - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_colour, numNames)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalMapColouring)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=numColour - 1, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == '__main__':
    random.seed(42)
    pop = toolbox.population(n=10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.4, ngen=100,
                                   stats=stats, verbose=False)
    best = tools.selBest(pop, 1)[0]
    print("Best colouring:", best.tolist())
    print("Conflicts (0 = valid):", evalMapColouring(best)[0])
    print()
    for i in range(numNames):
        print(f"  {names[i]:10s} -> {colours[best[i]]}")
