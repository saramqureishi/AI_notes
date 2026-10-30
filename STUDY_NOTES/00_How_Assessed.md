# CS5707 Artificial Intelligence — How You Are Assessed

## The one fact that drives everything
**100% of your grade is a single 2-hour WRITTEN exam.** There is no graded coursework.

| Item | Detail |
|---|---|
| Assessment | Examination = 100% |
| Format | Written, 2 hours |
| When | Winter exam period (date TBC) |
| Labs | NOT graded (practice only) |
| Module / assessment code | CS5707 / CS5807 |
| Credits | 15 |

## Learning outcomes the exam tests
- **LO1** — Understand key AI concepts and fundamental methods.
- **LO2** — Apply AI methods to problems, *justify* the approach, and *critically evaluate* its effectiveness.

## What this means for how you study
1. Labs are ungraded, but DO them — they build the intuition the exam rewards. Use the
   report template: Problem Description -> Implementation & Results -> Discussion.
2. The exam asks you to:
   - Formulate a problem as (state, action, goal test, path cost).
   - Hand-trace search algorithms: show the frontier/queue, expansion order, final path + cost.
   - Compare and evaluate algorithms: completeness, optimality, time & space, "which is best here & why".
   - Reason about heuristics: admissibility and consistency.
3. You will NOT be asked to type a full program from memory. You WILL be asked to trace by hand.
4. "Three tasks per week — model solutions provided." Understand the reasoning well enough to reproduce it.

## The single highest-value exercise
By hand, trace Uniform-Cost Search and A* on the Romania map (Arad -> Bucharest), writing the
frontier and explored set at every step, and f(n)=g(n)+h(n) for every node A* expands.
