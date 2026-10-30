"""
Lab 2 - Task 2.1 Route Planning
--------------------------------
PROBLEM DESCRIPTION
  6 cities (1-6). Find a route from a start city to a goal city, ideally the
  shortest. Formulated as a search problem:
    State     : current city (0-indexed internally)
    Action    : the next city to travel to
    Goal test : current city == destination
    Path cost : edge distance from COSTS matrix

IMPLEMENTATION & RESULTS: run this file.
DISCUSSION: see the printed comparison + notes in 02_Week2_Search.md
"""
from simpleai.search import SearchProblem, breadth_first, depth_first, uniform_cost

COSTS = [
    [0, 7, 9, 'inf', 'inf', 14],
    [7, 0, 10, 15, 'inf', 'inf'],
    [9, 10, 0, 11, 'inf', 2],
    ['inf', 15, 11, 0, 6, 'inf'],
    ['inf', 'inf', 'inf', 6, 0, 9],
    [14, 'inf', 2, 'inf', 9, 0],
]


class Route(SearchProblem):
    def __init__(self, initial, goal):
        self.initial = initial - 1
        self.goal = goal - 1
        super(Route, self).__init__(initial_state=self.initial)

    def actions(self, state):
        return [a for a in range(len(COSTS[state]))
                if COSTS[state][a] not in ['inf', 0]]

    def result(self, state, action):
        return action

    def is_goal(self, state):
        return state == self.goal

    def cost(self, state, action, state2):
        return COSTS[state][action]


if __name__ == '__main__':
    print('Route planning: city 1 -> city 5\n')
    for name, fn in [('Uniform-cost', uniform_cost),
                     ('Breadth-first', breadth_first),
                     ('Depth-first', depth_first)]:
        problem = Route(1, 5)
        result = (fn(problem) if name == 'Uniform-cost'
                  else fn(problem, graph_search=True))
        path = [x[1] + 1 for x in result.path()]
        print(f'{name:14s}: route={path}, cost={result.cost}')

    print('\nDiscussion:')
    print(' - UCS finds the true shortest path (cost 20) because it minimises g(n).')
    print(' - BFS/DFS return 1-6-5 (cost 23): fewer hops but NOT cheapest, because')
    print('   they ignore edge costs. BFS is only optimal when all steps cost the same.')
