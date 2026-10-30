"""
Lab 2 - Task 2.3 Maze Solver
----------------------------
PROBLEM DESCRIPTION
  Find a route from start 'o' to target 'x' in a grid maze. '#'=wall, ' '=free.
    State     : (x, y) coordinate in the grid
    Action    : up / down / left / right (and optional diagonals)
    Goal test : current coord == target coord
    Path cost : 1.0 per straight move, 1.4 per diagonal move
  Heuristic   : straight-line (Euclidean) distance to the target (admissible).

Run with:  python3 lab2_maze.py            (4 moves)
           python3 lab2_maze.py diagonal   (8 moves, diagonals cost 1.4)
"""
import sys
import math
from simpleai.search import (SearchProblem, astar, greedy,
                             breadth_first, depth_first, uniform_cost)

MAP = """
##############################
#         #              #   #
# ####    ########       #   #
#  o #    #              #   #
#    ###     ####   ######   #
#         ####      #        #
#            #  #   #   #### #
#     ######    #       # x  #
#        #      #            #
##############################
"""
MAP = [list(x) for x in MAP.split("\n") if x]

STRAIGHT = {"up": 1.0, "down": 1.0, "left": 1.0, "right": 1.0}
DIAGONAL = {**STRAIGHT, "up left": 1.4, "up right": 1.4,
            "down left": 1.4, "down right": 1.4}


class Maze(SearchProblem):
    def __init__(self, board, costs):
        self.board = board
        self.costs = costs
        self.goal = (0, 0)
        for y in range(len(board)):
            for x in range(len(board[y])):
                if board[y][x].lower() == "o":
                    self.initial = (x, y)
                elif board[y][x].lower() == "x":
                    self.goal = (x, y)
        super(Maze, self).__init__(initial_state=self.initial)

    def actions(self, state):
        acts = []
        for a in self.costs:
            nx, ny = self.result(state, a)
            if self.board[ny][nx] != "#":
                acts.append(a)
        return acts

    def result(self, state, action):
        x, y = state
        if action.count("up"): y -= 1
        if action.count("down"): y += 1
        if action.count("left"): x -= 1
        if action.count("right"): x += 1
        return (x, y)

    def is_goal(self, state):
        return state == self.goal

    def cost(self, state, action, state2):
        return self.costs[action]

    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)


def draw(result):
    path = [x[1] for x in result.path()]
    for y in range(len(MAP)):
        for x in range(len(MAP[y])):
            print('.' if (x, y) in path[1:-1] else MAP[y][x], end='')
        print()


if __name__ == '__main__':
    costs = DIAGONAL if (len(sys.argv) > 1 and sys.argv[1] == 'diagonal') else STRAIGHT
    label = '8 moves (diagonals=1.4)' if costs is DIAGONAL else '4 moves'
    print(f'Maze solver [{label}]\n')
    for name, fn in [('A*', astar), ('Greedy', greedy),
                     ('Breadth-first', breadth_first),
                     ('Depth-first', depth_first),
                     ('Uniform-cost', uniform_cost)]:
        result = fn(Maze(MAP, costs), graph_search=True)
        print(f'{name:14s}: moves={result.depth}, cost={result.cost}')
    print('\nA* path:')
    draw(astar(Maze(MAP, costs), graph_search=True))
