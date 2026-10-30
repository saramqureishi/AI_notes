# Week 2 Lab — Complete Results and Discussion

All results verified by running the code (see lab2_route.py, puzzle8.py, lab2_maze.py).

---

## Task 2.1 Route Planning (City 1 → City 5)

| Method | Route | Cost | Notes |
|---|---|---|---|
| Uniform-cost | 1→3→6→5 | **20** | ✅ Optimal; minimises g(n) |
| Breadth-first | 1→6→5 | 23 | Fewest hops (2 edges), NOT cheapest |
| Depth-first | 1→6→5 | 23 | Just found *a* path; no cost awareness |

**Discussion:** UCS is guaranteed optimal because it always expands the lowest-cost-so-far
node. BFS finds the path with fewest edges but ignores edge weights — it only equals UCS
when all edges cost the same. DFS has no optimality guarantee at all.

---

## Task 2.2 8-Puzzle

Two heuristics:
- h1: number of misplaced tiles
- h2: Manhattan distance (sum of |row diff| + |col diff| for each tile)

| Case | A*(h1) | A*(h2) | Greedy(h1) | BFS | DFS |
|---|---|---|---|---|---|
| depth 4 | ✅ 4 | ✅ 4 | ✅ 4 | ✅ 4 | ❌ blew up |
| depth 8 | ✅ 8 | ✅ 8 | ❌ blew up | ✅ 8 | — |
| depth 26 | ❌ too slow | ✅ **26** | — | — | — |

**Discussion:**
1. DFS is dangerous: the 8-puzzle has 181,440 reachable states. DFS dives deep and
   gets lost; it couldn't even solve the easiest (depth 4) case in 20 seconds.
2. Heuristic quality is decisive: A* with h1 fails on depth-26, but A* with h2 solves it
   instantly. h2 **dominates** h1 (h2 ≥ h1 everywhere, both admissible), so h2 prunes
   exponentially more nodes. This is the Russell & Norvig pp.116-117 point.
3. Greedy ignores g(n), so even on depth-8 with h1 it wandered off and blew up.
4. A* + h2 (Manhattan) = best all-rounder: always optimal, always fast enough.

---

## Task 2.3 Maze Solver

### 4-directional (up/down/left/right, all cost=1.0)

| Method | Moves | Cost |
|---|---|---|
| A* | 33 | 33.0 |
| Greedy | 33 | 33.0 |
| BFS | 33 | 33.0 |
| DFS | 33 | 33.0 |
| UCS | 33 | 33.0 |

All same here because the maze is small and all step costs are equal.

### 8-directional (adds diagonals at cost=1.4)

| Method | Moves | Cost |
|---|---|---|
| A* | 23 | 27.0 |
| Greedy | 23 | 27.0 |
| BFS | 23 | 27.0 |
| DFS | 38 | 49.6 |
| UCS | 23 | 27.0 |

**Discussion:** Diagonals reduce the path from 33 to 23 moves (cost 33→27). DFS again
fails: it found a 38-move path costing 49.6, confirming it has no optimality guarantee.
The heuristic (Euclidean distance to target) is admissible for both 4-dir and 8-dir cases
because straight-line distance never overestimates the true path cost.
