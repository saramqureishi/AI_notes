# CS5707 — WEEK 2: SEARCH (Complete Detailed Notes)

> These notes are self-contained. If you read only this file, you have Week 2.
> Reading: Russell & Norvig Ch.3 "Solving problems by searching" (esp. pp.116-117).

---

## 0. The big picture — why "search"?

A huge amount of AI is really this: **you have a starting situation, a goal situation, and
a set of moves that take you from one situation to another. Find a sequence of moves that
reaches the goal — ideally the cheapest one.**

Examples that are ALL the same problem underneath:
- Google Maps finding a route from A to B.
- Solving a sliding-tile puzzle.
- A robot finding its way through a maze.
- A chess engine looking for a winning line.

Because they're all the same underneath, we use ONE framework to describe them, then we
run GENERAL search algorithms on that framework. Learn the framework once, reuse forever.

---

## 1. Problem formulation — the 4 ingredients (MEMORISE THIS)

Every search problem is defined by exactly four things:

| # | Ingredient   | Question it answers              | Route example           |
|---|--------------|----------------------------------|-------------------------|
| 1 | **State**    | "Where am I right now?"          | "at city 3"             |
| 2 | **Action**   | "What moves can I make from here?"| "travel to city 6"     |
| 3 | **Goal test**| "Am I finished?"                 | "am I at city 5?"       |
| 4 | **Path cost**| "What am I trying to minimise?"  | "sum of road distances" |

Extra vocabulary that goes with it:
- **Initial state**: where the search begins.
- **State space**: the set of ALL states reachable from the initial state. (Can be huge.)
- **Successor / result**: the state you land in after applying an action.
- **Solution**: a sequence of actions from initial state to a goal state.
- **Optimal solution**: the solution with the lowest total path cost.

> In the exam, "formulate the problem" means: write out these four things clearly.
> This is worth easy marks — practise writing them for route, 8-puzzle, and maze.

---

## 2. The running examples used in the lecture

### 2a. Romania map (the textbook classic)
Travel from **Arad** to **Bucharest**. Cities are nodes; roads are edges with distances.
- state: "at Arad"
- action: "Arad -> Zerind" (drive to a connected city)
- goal test: "at Bucharest?"
- path cost: sum of road distances

Romania also has a table of **straight-line distances to Bucharest** for every city.
That table is the HEURISTIC we use later for greedy and A* (see section 5).

### 2b. 6-city route problem (used in the lab)
```
Cities 1..6. Direct-connection distances:
  1-2: 7     1-3: 9     1-6: 14
  2-3: 10    2-4: 15
  3-4: 11    3-6: 2
  4-5: 6
  5-6: 9
```
As a cost matrix (0 on the diagonal, 'inf' = not directly connected):
```
       1     2     3     4     5     6
  1 [  0,    7,    9,  inf,  inf,   14]
  2 [  7,    0,   10,   15,  inf,  inf]
  3 [  9,   10,    0,   11,  inf,    2]
  4 [inf,   15,   11,    0,    6,  inf]
  5 [inf,  inf,  inf,    6,    0,    9]
  6 [ 14,  inf,    2,  inf,    9,    0]
```

### 2c. 8-puzzle
A 3x3 grid with tiles 1-8 and one blank. Slide tiles into the blank until you reach the
goal arrangement.
- state: the 3x3 arrangement
- action: move the BLANK up/down/left/right (easier than "move a tile")
- goal test: arrangement == goal
- path cost: 1 per move (every move costs the same)
- state space size: 9!/2 = 181,440 reachable states (this number matters — see DFS)

### 2d. Maze
Grid with walls '#', free spaces, start 'o', target 'x'.
- state: (x, y) coordinate
- action: up/down/left/right (optionally 4 diagonals)
- goal test: coordinate == target
- path cost: 1.0 per straight move (1.4 per diagonal if diagonals allowed)

---

## 3. Two families of search

```
                 SEARCH
         ________|________
        |                 |
  UNINFORMED          INFORMED
  (blind)             (heuristic)
   - BFS               - Greedy best-first
   - DFS               - A*
   - UCS
```

- **Uninformed / blind**: the algorithm knows NOTHING about which direction the goal is in.
  It only knows the states, actions and costs. It explores "mechanically".
- **Informed / heuristic**: the algorithm has an extra hint h(n) — an estimate of how far
  the goal still is — and uses it to search smarter (head roughly toward the goal).

**The frontier (a.k.a. "queue"/"open list")**: the set of nodes we've discovered but not
yet expanded. The DIFFERENCE between the algorithms is basically: *which node do we pick
from the frontier to expand next?* That single choice is what distinguishes BFS/DFS/UCS/A*.

- "Expand a node" = look at it, generate its successors, add new ones to the frontier.
- "Explored set" (closed list) = nodes already expanded, so we don't redo them.

---

## 4. UNINFORMED SEARCH (blind)

### 4.1 Breadth-First Search (BFS)
- **Rule:** always expand the SHALLOWEST unexpanded node (closest to the start in #steps).
- **Data structure:** FIFO queue (first in, first out).
- **Behaviour:** explores in expanding "rings" — all nodes 1 step away, then all 2 steps
  away, etc.
- **Complete?** Yes (will find a solution if one exists).
- **Optimal?** ONLY if every step has the SAME cost (because it minimises number of steps,
  not total cost).
- **Time / space:** O(b^d) where b = branching factor, d = depth of shallowest solution.
  Memory is the killer: it must store the whole frontier.

### 4.2 Depth-First Search (DFS)
- **Rule:** always expand the DEEPEST unexpanded node.
- **Data structure:** LIFO stack (last in, first out).
- **Behaviour:** charges down one path to the bottom, then backtracks.
- **Complete?** Not in infinite/large spaces — it can go down forever or get lost.
- **Optimal?** NO. It returns *a* path, rarely the cheapest.
- **Time / space:** space is cheap (O(bm), only one branch stored) but time can be
  disastrous. On big state spaces it effectively blows up (we saw this on the 8-puzzle).
- **When useful:** memory-limited problems with shallow solutions. Not for shortest-path.

### 4.3 Uniform-Cost Search (UCS)  ⭐ the important one
- **Rule:** always expand the node with the LOWEST PATH COST SO FAR, g(n).
- **Data structure:** priority queue ordered by g(n).
- **This is essentially Dijkstra's algorithm** (lecturer says so explicitly).
- **Complete?** Yes. **Optimal?** YES — it is guaranteed to find the cheapest path.
- **Behaviour:** expands cheapest-reachable states outward, like a cost "wavefront".

### 4.4 Verified route results (city 1 -> city 5)
| Method        | Route     | Cost | Why                                             |
|---------------|-----------|------|-------------------------------------------------|
| Uniform-cost  | 1-3-6-5   | 20   | Optimal; minimises total distance               |
| Breadth-first | 1-6-5     | 23   | Fewest hops (2 edges) but NOT cheapest           |
| Depth-first   | 1-6-5     | 23   | Just *a* path; no cost awareness                 |

> THE key uninformed lesson: BFS/DFS ignore edge weights. Only UCS minimises real cost.
> BFS == optimal only when all steps cost the same.

### 4.5 Worked UCS hand-trace (1 -> 5)  — THIS is exam gold
Notation: dist[i] = best known cost to city i; "(k)" = we reached it via k.
```
INIT   dist = [0, inf, inf, inf, inf, inf]      explored = {}
R1  pick 1 (0):  2 = 7(1), 3 = 9(1), 6 = 14(1)  explored = {1}
R2  pick 2 (7):  4 = 7+15 = 22(2)                explored = {1,2}
R3  pick 3 (9):  4 = 9+11 = 20(3)  [20<22 update]
                 6 = 9+2  = 11(3)  [11<14 update] explored = {1,2,3}
R4  pick 6 (11): 5 = 11+9 = 20(6)                explored = {1,2,3,6}
R5  pick 4 (20): 5 = 20+6 = 26  [26>20, no change] explored = {1,2,3,4,6}
R6  pick 5 (20): GOAL REACHED.
RESULT  cost = 20,  path = 1 -> 3 -> 6 -> 5  (follow the "(via)" tags backwards)
```
How to reconstruct the path: 5 came via 6, 6 came via 3, 3 came via 1. Reverse => 1,3,6,5.

---

## 5. INFORMED (HEURISTIC) SEARCH

### 5.1 What is a heuristic h(n)?
h(n) = a cheap-to-compute ESTIMATE of the remaining cost from node n to the goal.
It is a "hint" about which direction is promising.
- Route/maze: straight-line (as-the-crow-flies) distance to the goal.
- 8-puzzle: how scrambled the board looks (two versions below).

A good heuristic points you roughly toward the goal so you don't waste effort exploring
away from it.

### 5.2 Greedy Best-First Search
- **Rule:** expand the node with the smallest h(n) ALONE (looks closest to goal).
- **Fast**, because it heads straight for the goal.
- **NOT optimal, NOT reliable:** it's short-sighted — it trusts the estimate and ignores
  how much you've already spent (g).
- **Lecture proof:** on Romania, greedy returns a route of length **450**, but the true
  shortest route is **418**. Greedy got fooled by a locally good-looking move.

### 5.3 A* Search  ⭐⭐ the most important algorithm in Week 2
Combine "cost so far" (UCS) with "estimate to go" (greedy):
```
        f(n) = g(n) + h(n)

  g(n) = actual cost from the start to n        (what UCS uses)
  h(n) = estimated cost from n to the goal      (what greedy uses)
  f(n) = estimated total cost of a path going through n
```
- **Rule:** always expand the node with the smallest f(n).
- **Best of both worlds:** as optimal as UCS, as directed as greedy.
- **Intuition:** don't just chase the goal (greedy) and don't ignore the goal (UCS) —
  balance where you've been against where you're going.

### 5.4 Conditions for A* optimality  (EXAM FAVOURITE — learn exactly)

**(a) Admissibility** — the heuristic NEVER OVERESTIMATES the true remaining cost:
```
    h(n) <= true cost from n to goal      ("optimistic" heuristic)
```
If h is admissible, A* (tree search) is guaranteed to return the optimal solution.
Straight-line distance is admissible because a real road can only be longer than the
straight line, never shorter.

**(b) Consistency (a.k.a. monotonicity)** — satisfies the triangle inequality:
```
    h(n) <= cost(n, a, n') + h(n')
    equivalently   |h(n) - h(n')| <= cost(n, a, n')
```
Consistency is STRONGER than admissibility and guarantees optimality even for GRAPH
search (where we skip already-explored states).

**Relationship to remember:**
```
    consistent  =>  admissible        (every consistent heuristic is admissible)
    admissible  =NOT=>  consistent     (admissible does NOT imply consistent)
```

Why consistency matters: with a consistent heuristic, the first time A* reaches a node it
has already found the cheapest way there, so f never decreases along a path. With an only-
admissible (not consistent) heuristic in graph search, A* might reach the goal on a
non-optimal path first — that's the trap the seminar S->G question is testing.

### 5.5 The 8-puzzle heuristics (classic exam material)
- **h1 = number of misplaced tiles.** Count how many tiles are NOT in their goal spot.
- **h2 = total Manhattan distance.** For each tile, add |row diff| + |col diff| from its
  goal position; sum over all tiles.
- Both are admissible. **h2 >= h1 at every state** => we say h2 **DOMINATES** h1.
- A dominating admissible heuristic expands FEWER nodes, so h2 is better than h1.

### 5.6 Verified 8-puzzle results
| Case     | A*(h1)      | A*(h2)   | Greedy(h1)  | BFS   | DFS        |
|----------|-------------|----------|-------------|-------|------------|
| depth 4  | solves (4)  | solves (4)| solves (4) | solves(4) | blew up |
| depth 8  | solves (8)  | solves (8)| blew up    | solves(8) |   -     |
| depth 26 | too slow    | solves(26)|    -       |   -       |   -     |

Lessons (write these in an exam/lab discussion):
1. **DFS blows up** on the 181,440-state space — not practically complete here.
2. **Heuristic quality decides everything:** A*(h1) stalls on depth-26, A*(h2) nails it
   instantly. Same algorithm; only the heuristic changed. (R&N pp.116-117.)
3. **Greedy is reckless:** ignoring g, with the weak h1 it got lost even at depth 8.
4. **A* + good admissible heuristic (h2) = best all-rounder:** optimal and fast.

### 5.7 Verified maze results
4-directional (all moves cost 1.0): A*, Greedy, BFS, DFS, UCS all find 33 moves / cost 33
(small maze, equal costs, so they coincide).

8-directional (diagonals cost 1.4):
| Method | Moves | Cost |
|--------|-------|------|
| A*     | 23    | 27.0 |
| Greedy | 23    | 27.0 |
| BFS    | 23    | 27.0 |
| DFS    | 38    | 49.6 |
| UCS    | 23    | 27.0 |
Diagonals cut 33 moves down to 23. DFS again returns a worse path (38 moves, cost 49.6),
confirming it is not optimal. The Euclidean heuristic is admissible in both cases.

---

## 6. Summary comparison table (one-glance revision)

| Algorithm | Picks next node by | Data structure | Complete? | Optimal? | Notes |
|-----------|--------------------|----------------|-----------|----------|-------|
| BFS       | shallowest         | FIFO queue     | Yes       | Only if equal step costs | high memory |
| DFS       | deepest            | LIFO stack     | No (big/inf spaces) | No | low memory, can blow up |
| UCS       | lowest g(n)        | priority queue | Yes       | Yes      | = Dijkstra |
| Greedy    | lowest h(n)        | priority queue | No        | No       | fast but short-sighted |
| A*        | lowest f=g+h       | priority queue | Yes       | Yes (if h admissible/consistent) | the gold standard |

---

## 7. Exam-style checklist — can you do each of these?
- [ ] State the 4 ingredients of a search problem for any scenario.
- [ ] Hand-trace BFS, DFS, UCS on the 6-city graph showing the frontier each step.
- [ ] Explain why BFS/DFS gave cost 23 but UCS gave 20.
- [ ] Write f(n) = g(n) + h(n) and define each term.
- [ ] Define admissibility and consistency, and state consistent => admissible.
- [ ] Given a graph with heuristic values, decide if h is admissible and/or consistent.
- [ ] Explain why h2 (Manhattan) beats h1 (misplaced tiles) for the 8-puzzle.
- [ ] Compare any two algorithms on completeness, optimality, time and space.

If you can tick all of these, you have Week 2 at exam standard.

---

## 8. The seminar questions (practise these before the exam)
1. Formulate the 8-puzzle (state / action / goal test / path cost).
2. Route 1 -> 5 with BFS, DFS, UCS — show node expansion and the path for each.
3. Change cost(3,4)=9 and cost(4,5)=1; redo (2). Does the path change for each method? Why?
4. Romania Arad -> Bucharest with Greedy and A* using straight-line distance (Pitesti = 98).
   Show node expansion, the path found, and its total distance.
5. S->G graph: given true edge costs (black) and heuristic-to-G values (blue), decide if h
   is (1) admissible and (2) consistent. Run A* S->G and check whether it is on the optimal
   path the first time it reaches G. Explain why.
