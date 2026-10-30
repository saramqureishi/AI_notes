from simpleai.search import SearchProblem, astar, greedy, breadth_first, depth_first
import sys

def list_to_string(input_list):
    return '\n'.join(['-'.join(x) for x in input_list])

def string_to_list(input_string):
    return [x.split('-') for x in input_string.split('\n')]

def get_location(rows, input_element):
    for i, row in enumerate(rows):
        for j, item in enumerate(row):
            if item == input_element:
                return i, j

GOAL = 'E-1-2\n3-4-5\n6-7-8'

# which heuristic to use is set by a module-level flag
HEURISTIC = 'h1'

goal_positions = {}
for number in '12345678E':
    goal_positions[number] = get_location(string_to_list(GOAL), number)


class PuzzleSolver(SearchProblem):
    def actions(self, cur_state):
        rows = string_to_list(cur_state)
        row_empty, col_empty = get_location(rows, 'E')
        actions = []
        if row_empty > 0: actions.append(rows[row_empty - 1][col_empty])
        if row_empty < 2: actions.append(rows[row_empty + 1][col_empty])
        if col_empty > 0: actions.append(rows[row_empty][col_empty - 1])
        if col_empty < 2: actions.append(rows[row_empty][col_empty + 1])
        return actions

    def result(self, state, action):
        rows = string_to_list(state)
        row_empty, col_empty = get_location(rows, 'E')
        row_new, col_new = get_location(rows, action)
        rows[row_empty][col_empty], rows[row_new][col_new] = (
            rows[row_new][col_new], rows[row_empty][col_empty])
        return list_to_string(rows)

    def is_goal(self, state):
        return state == GOAL

    def cost(self, state, action, state2):
        return 1

    def heuristic(self, state):
        rows = string_to_list(state)
        if HEURISTIC == 'h1':
            # h1: number of misplaced tiles
            goal = string_to_list(GOAL)
            return sum([rows[i] != goal[i] for i in range(len(rows))]) - 1
        else:
            # h2: sum of Manhattan distances
            distance = 0
            for number in '12345678':
                r, c = get_location(rows, number)
                gr, gc = goal_positions[number]
                distance += abs(r - gr) + abs(c - gc)
            return distance


CASES = {
    'depth4':  '1-4-2\n3-5-8\n6-7-E',
    'depth8':  '1-4-2\n5-E-8\n3-6-7',
    'depth26': '7-2-4\n5-E-6\n8-3-1',
}

ALGOS = {'astar': astar, 'greedy': greedy, 'bfs': breadth_first, 'dfs': depth_first}

if __name__ == '__main__':
    case = sys.argv[1]
    algo = sys.argv[2]
    HEURISTIC = sys.argv[3] if len(sys.argv) > 3 else 'h1'
    initial = CASES[case]
    fn = ALGOS[algo]
    if algo in ('bfs', 'dfs'):
        result = fn(PuzzleSolver(initial), graph_search=True)
    else:
        result = fn(PuzzleSolver(initial))
    print(f'{case} {algo} ({HEURISTIC}): depth={result.depth}, cost={result.cost}')
