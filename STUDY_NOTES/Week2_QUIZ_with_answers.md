# Week 2 Search — Quiz, Correct Answers & Explanations

Score key (one line):
1-C 2-A 3-C 4-C 5-B 6-A 7-C 8-B 9-B 10-B 11-B 12-B 13-A 14-C 15-A 16-A 17-B 18-C
19-C 20-B 21-B 22-A 23-B 24-B 25-C 26-A 27-C 28-B 29-B

Most-missed on exams: Q8 (BFS != cheapest), Q18 (goal test on expansion), Q28 (A* with h=0 -> UCS).

---

## PROBLEM FORMULATION

**Q1. What does the 'transition model' define?**
A: The result of each action in a given state. ✅
The transition model (result function): from state s, action a -> which next state.
("set of all states"=state space; "destination check"=goal test.)

**Q2. In the 8-puzzle, which represents a 'state'?**
A: A specific configuration of the eight tiles and the blank space. ✅
A state is a full board snapshot. Move rule=action; steps taken=path cost; sorted board=goal state.

**Q3. Which component calculates the cost of a sequence of actions from start to a node?**
A: Path Cost. ✅
Path cost sums step costs. Heuristic only estimates; goal test/transition model don't cost.

**Q4. Why is problem formulation critical?**
A: It maps a real-world problem into a formal mathematical state space. ✅
Turns a messy problem into (state, action, goal test, path cost) so general algorithms can run.
It does NOT pick the algorithm or guarantee costs.

**Q5. Maze agent gains diagonal moves - what changes?**
A: The Actions and Transition Model. ✅
New moves = new actions producing new resulting states. Start/goal/state space unchanged.

---

## BREADTH-FIRST SEARCH (BFS)

**Q6. Data structure for BFS frontier?**
A: FIFO Queue. ✅  First-in-first-out => shallowest nodes expand first => level-by-level.

**Q7. Main reason BFS is 'optimal'?**
A: It finds the shallowest goal node first, ensuring the fewest number of steps. ✅
BFS is optimal in NUMBER OF STEPS (not mileage - see Q8).

**Q8. With varied edge costs, does BFS always find the cheapest path?**
A: No - it only considers number of steps, not the cost of those steps. ✅
Lab proof: BFS cost 23 vs UCS cost 20. BFS counts hops, not distance.

**Q9. BFS frontier after a node is expanded?**
A: Its children are added to the back of the queue. ✅  (FIFO: enter at back, leave at front.)

---

## DEPTH-FIRST SEARCH (DFS)

**Q10. Which strategy describes DFS?**
A: Expand the deepest unexpanded node first. ✅

**Q11. Significant advantage of DFS over BFS?**
A: DFS has much lower space complexity. ✅
DFS stores only the current branch O(bm); BFS stores the whole frontier O(b^d).
DFS is NOT optimal and NOT always complete.

**Q12. Data structure for DFS frontier?**
A: LIFO Stack. ✅  Last-in-first-out => deepest (most recent) node expands next.

**Q13. When can DFS fail to be complete?**
A: In an infinite state space or one with loops/cycles. ✅  Can descend/loop forever.

**Q14. Why does DFS give a wiggly/inefficient maze path?**
A: It follows a single path to its limit regardless of whether shorter paths exist. ✅
Diagonal maze: DFS 38 moves / cost 49.6 vs A* 23 / 27.

**Q15. What is 'backtracking' in DFS?**
A: Returning to the most recent junction after reaching a dead end. ✅

---

## UNIFORM-COST SEARCH (UCS)

**Q16. UCS extends which algorithm?**
A: Breadth-First Search. ✅  Orders frontier by cumulative cost not depth. (= Dijkstra.)

**Q17. Which node does UCS always expand?**
A: The node with the lowest cumulative path cost g(n). ✅
(smallest h=greedy; highest f=nonsense; most recent=DFS.)

**Q18. When is the goal test applied in standard UCS?**
A: When a node is SELECTED FOR EXPANSION from the queue. ✅
NOT on generation - a goal found early may be reachable cheaper later, so UCS waits.
This is essential for UCS optimality.

---

## INFORMED SEARCH / GREEDY

**Q19. Uninformed vs informed - primary difference?**
A: Informed search uses a heuristic to estimate the distance to the goal. ✅

**Q20. What does h(n) represent in Greedy Best-First?**
A: The estimated cost from the current node n to the goal. ✅  (cost from start = g(n).)

**Q21. Why is Greedy often faster than BFS/UCS?**
A: It focuses the search toward the goal, expanding fewer irrelevant nodes. ✅

**Q22. Why can Greedy miss the optimal Arad->Bucharest path?**
A: It ignores the actual path cost g(n) and only follows the straight-line estimate. ✅
Lecture: greedy route 450 vs optimal 418.

**Q23. Is Greedy optimal?**
A: No - it can choose a path that looks short initially but is longer overall. ✅
Not optimal even with an admissible heuristic (that guarantee is A*'s, not greedy's).

**Q24. Why might Greedy get stuck in a maze 'pocket'/dead end?**
A: The dead-end nodes have a lower straight-line distance to the goal than the exit path. ✅
The heuristic lures it into a spot that looks closer but requires moving away to escape.

---

## A* SEARCH

**Q25. A* evaluation function?**
A: f(n) = g(n) + h(n). ✅  cost so far + estimate to go = estimated total cost through n.

**Q26. What does 'admissibility' mean?**
A: The heuristic never overestimates the actual cost to reach the goal. ✅  ("optimistic".)

**Q27. A* is guaranteed optimal if the heuristic is:**
A: Admissible and consistent. ✅  Admissible => optimal (tree); consistent => optimal (graph).

**Q28. If h(n) is always 0, A* behaves like:**
A: Uniform-Cost Search. ✅  f(n)=g(n). (h=0 is trivially admissible.)

**Q29. Primary trade-off of A*?**
A: Optimal and efficient, but space complexity stays high as it stores all generated nodes. ✅
A*'s weakness is memory.
