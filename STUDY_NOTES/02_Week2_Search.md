# Week 2 — Search  (HIGH exam value)

This is the foundation of the whole first half of the module. Master it.

## 1. Problem formulation — the 4 ingredients (memorise)
| Ingredient | Meaning | Route example |
|---|---|---|
| State | where you are now | "at city 3" |
| Action | what you can do | "go to city 6" |
| Goal test | are we done? | "at city 5?" |
| Path cost | what we minimise | sum of edge distances |

You reuse this framework for route planning, 8-puzzle, maze, genetic algorithms, everything.

## 2. The 6-city route cost matrix
```
       1     2     3     4     5     6
  1 [  0,    7,    9,  inf,  inf,   14]
  2 [  7,    0,   10,   15,  inf,  inf]
  3 [  9,   10,    0,   11,  inf,    2]
  4 [inf,   15,   11,    0,    6,  inf]
  5 [inf,  inf,  inf,    6,    0,    9]
  6 [ 14,  inf,    2,  inf,    9,    0]
```

## 3. Uninformed (blind) search
- **BFS** — expand shallowest node, FIFO queue. Optimal ONLY if all step costs equal
  (it minimises number of steps, not cost).
- **DFS** — expand deepest node, LIFO stack. Finds *a* path, NOT the shortest. Can blow up.
- **UCS** — expand lowest g(n) (cost so far), priority queue. = Dijkstra. Complete AND optimal.

### Route 1->5 results (verified by running simpleai)
| Method | Route | Cost | Note |
|---|---|---|---|
| Uniform-cost | 1-3-6-5 | 20 | optimal |
| Breadth-first | 1-6-5 | 23 | fewest hops, NOT cheapest |
| Depth-first | 1-6-5 | 23 | just *a* path |

Key point: BFS/DFS ignore edge cost; only UCS minimises actual distance.

### UCS hand-trace (1 -> 5) — exam style
```
init  dist=[0,inf,inf,inf,inf,inf]  explored={}
R1 pick 1(0): update 2=7,3=9,6=14      explored={1}
R2 pick 2(7): 4=22                     explored={1,2}
R3 pick 3(9): 4=20(update),6=11(update) explored={1,2,3}
R4 pick 6(11): 5=20                    explored={1,2,3,6}
R5 pick 4(20): no improvement          explored={1,2,3,4,6}
R6 pick 5(20): GOAL. cost=20, path 1-3-6-5
```

## 4. Informed (heuristic) search
Heuristic h(n) = estimate of remaining cost to goal (e.g. straight-line distance).

- **Greedy best-first** — expand lowest h(n) only. Fast but NOT optimal.
  Romania: greedy route = 450 vs true shortest = 418.
- **A\*** — expand lowest f(n) = g(n) + h(n).
    g(n) = cost so far (like UCS)
    h(n) = estimate to goal (like greedy)
  Combines optimality + speed.

### A* conditions for optimality (EXAM FAVOURITE)
- **Admissible**: never overestimates. h(n) <= true cost to goal. Guarantees optimal (tree search).
- **Consistent (monotonic)**: h(n) <= cost(n,a,n') + h(n')  i.e. |h(n)-h(n')| <= cost.
  Stronger; guarantees optimal for graph search. Consistent => admissible (not vice versa).

## 5. 8-puzzle experiment results (verified)
| Case | A*(h1) | A*(h2) | Greedy | BFS | DFS |
|---|---|---|---|---|---|
| depth 4 | 4 | 4 | 4 | 4 | blew up |
| depth 8 | 8 | 8 | blew up (h1) | 8 | - |
| depth 26 | too slow | 26 | - | - | - |

h1 = number of misplaced tiles. h2 = sum of Manhattan distances.
Lessons:
1. DFS blows up on large state spaces (181,440 states) - not practically complete.
2. Heuristic quality is decisive: A* with h2 solves depth-26 instantly; with h1 it stalls.
   h2 dominates h1 (h2 >= h1 everywhere, both admissible) so expands fewer nodes. (R&N p116-117)
3. Greedy ignores g(n); behaviour swings wildly with heuristic quality.
4. A* + good admissible heuristic = best all-rounder; optimal every solvable case.

## 6. The seminar questions (do these for the exam)
1. Formulate the 8-puzzle (state/action/goal/cost).
2. Route 1->5 with BFS, DFS, UCS - show node expansion + path.
3. Change cost(3,4)=9, cost(4,5)=1; redo - does the path change? why?
4. Romania Arad->Bucharest with greedy and A* using straight-line heuristic (Pitesti=98).
5. S->G graph: is the heuristic admissible? consistent? does A* reach G optimally first time?

## Reading
Russell & Norvig Ch.3 "Solving problems by searching" (in repo). pp.116-117 for heuristic comparison.
