"""
Lab 3 - Task 3.2 The 8-Queens Problem with a Genetic Algorithm (DEAP)
PROBLEM: place 8 queens on an 8x8 board so none attack each other.
  Clever encoding: individual = a PERMUTATION of 0..7.
    position i in the list = column i; value = the row of the queen in that column.
    -> permutation guarantees NO two queens share a column OR a row automatically,
       so the GA only has to eliminate DIAGONAL clashes.
  Fitness  : number of pairs of queens attacking on diagonals (MINIMISE, best = 0)
  Operators: PMX crossover, shuffle-index mutation, tournament selection
"""
import random, numpy as np
from deap import algorithms, base, creator, tools

NB_QUEENS = 8


def evalNQueens(individual):
    fd = np.zeros(2 * NB_QUEENS - 1)   # falling diagonals
    rd = np.zeros(2 * NB_QUEENS - 1)   # rising diagonals
    for i in range(NB_QUEENS):
        fd[i + individual[i]] += 1
        rd[NB_QUEENS - 1 - i + individual[i]] += 1
    # count queens sharing a diagonal (any diagonal with >1 queen contributes a clash)
    return np.sum(fd[fd > 1]) + np.sum(rd[rd > 1]),


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("permutation", random.sample, range(NB_QUEENS), NB_QUEENS)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.permutation)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalNQueens)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / NB_QUEENS)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == '__main__':
    random.seed(64)
    pop = toolbox.population(n=300)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, verbose=False)
    best = tools.selBest(pop, 1)[0]
    print("Best (row of queen in each column):", list(best))
    print("Fitness (diagonal clashes, 0 = solved):", best.fitness.values[0])
    # ASCII board
    print()
    for row in range(NB_QUEENS - 1, -1, -1):
        line = ''.join(' Q ' if best[col] == row else ' . ' for col in range(NB_QUEENS))
        print(line)
