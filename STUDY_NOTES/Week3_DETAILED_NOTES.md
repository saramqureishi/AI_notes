# CS5707 — WEEK 3: GENETIC ALGORITHMS (Complete Detailed Notes)

> Self-contained. If you read only this file, you have Week 3.
> This week is taught via the Week 3 Seminar (exercises) + 3 lab notebooks (TSP, 8-Queens,
> Map Colouring) using the Python DEAP library.
> Reading: Negnevitsky Ch.7 (Evolutionary Computation / GAs); R&N Ch.4 (Local Search).

---

## 0. The big idea — "evolution as a search strategy"

A Genetic Algorithm (GA) is an OPTIMISATION / SEARCH method inspired by Darwinian
evolution: "survival of the fittest". Instead of exploring one path at a time (like Week 2
search), a GA keeps a whole POPULATION of candidate solutions and lets them evolve over
many GENERATIONS. Good solutions reproduce; bad ones die out; occasional random mutations
inject new variety. Over time the population drifts toward high-quality solutions.

Why use a GA instead of Week-2 search?
- Great when the search space is HUGE and there's no useful step-by-step path to the goal.
- Great for OPTIMISATION ("find the best arrangement") rather than "find a route".
- Doesn't need gradients, doesn't need the problem to be nicely mathematical.
- Downside: NOT guaranteed to find the optimum; it's a heuristic / "good enough" method.

GA belongs to the family of LOCAL / population-based search (R&N Ch.4), as opposed to the
systematic tree search of Week 2.

---

## 1. The biological vocabulary (learn these mappings)

| GA term        | Meaning in a GA                                            |
|----------------|------------------------------------------------------------|
| **Individual / chromosome** | one candidate solution                        |
| **Gene**       | one component of a solution (one position in the string)   |
| **Allele**     | the value a gene can take (e.g. a bit 0/1, a colour, a city)|
| **Genotype**   | the encoded form (e.g. the bitstring)                      |
| **Phenotype**  | what the encoding means (the decoded real solution)        |
| **Population** | the set of individuals in the current generation          |
| **Generation** | one cycle of selection + crossover + mutation             |
| **Fitness**    | a score saying how good an individual is (we optimise this)|

---

## 2. The GA cycle (the algorithm) — MEMORISE THIS LOOP

```
1. INITIALISE: create a random population of individuals.
2. EVALUATE  : compute each individual's fitness.
3. REPEAT for each generation until a stop condition:
     a. SELECTION  : pick parents, biased toward higher fitness.
     b. CROSSOVER  : combine pairs of parents to make offspring (with prob. cxpb).
     c. MUTATION   : randomly tweak some genes (with prob. mutpb).
     d. EVALUATE   : score the new offspring.
     e. REPLACE    : offspring form the next generation.
4. RETURN the best individual ever found.
```
Stop condition is usually: fixed number of generations (ngen), OR a good-enough fitness,
OR no improvement for a while.

---

## 3. The five design decisions for ANY GA (THIS is what the exam asks)

Whenever you're asked "design a GA for problem X", you must specify:

1. **Encoding (representation)** — how is a candidate solution written as a chromosome?
   - bitstring, list of integers, a permutation, etc.
2. **Fitness function** — how do we score a solution? (and are we maximising or minimising?)
3. **Selection operator** — how do we choose parents? (fitter = more likely)
4. **Crossover operator** — how do we recombine parents into offspring?
5. **Mutation operator** — how do we randomly perturb a chromosome?
(+ parameters: population size, crossover rate cxpb, mutation rate mutpb, #generations.)

If an exam question says "specify the encoding method, fitness function and genetic
operators" — that's exactly points 1-5 above.

---

## 4. The operators in detail

### 4.1 Selection (choose parents, biased by fitness)
- **Roulette-wheel (fitness-proportionate):** probability of being chosen is proportional
  to fitness. Fitter individuals get a bigger slice of the "wheel".
- **Tournament selection:** randomly pick k individuals, keep the best. Repeat. (DEAP labs
  use tournament with tournsize=3.) Simple and robust; the usual default.
- **Elitism:** always copy the best individual(s) unchanged into the next generation so you
  never lose your best-so-far.

### 4.2 Crossover (recombination) — mix two parents
- **Single-point crossover:** pick one cut point; child = left of parent A + right of parent B.
    Parent A: 111|010      ->  E: 111|011
    Parent B: 011|011      ->  F: 011|010   (cut after bit 3)
- **Two-point crossover:** two cut points; swap the middle segment. (map-colouring lab)
- **PMX (Partially Matched Crossover):** special crossover for PERMUTATIONS (TSP, 8-queens)
  that keeps the result a valid permutation (no repeated/missing cities).

### 4.3 Mutation — small random change for diversity
- **Bitwise mutation:** flip each bit with small probability (bitstrings).
- **Uniform-int mutation:** replace a gene with a random value in range (map-colouring).
- **Shuffle-index mutation:** randomly swap positions (permutations: TSP, 8-queens).
Mutation prevents the population getting stuck (premature convergence) by re-introducing
lost variety. Too little mutation -> stagnation; too much -> random search.

---

## 5. WORKED SEMINAR Q1 (bitstring GA) — do this by hand for the exam

Maximise f(x), x in 0..63 encoded as a 6-bit string:
  f(x) = x            if 0 <= x <= 32
  f(x) = 64 - x       if 32 < x <= 63

Initial population:
  A = 111010 -> x=58 -> f = 64-58 = 6
  B = 011011 -> x=27 -> f = 27           (0<=27<=32)
  C = 001001 -> x=9  -> f = 9
  D = 000110 -> x=6  -> f = 6

(1) Fitness: A=6, B=27, C=9, D=6.  Total = 6+27+9+6 = 48.  Best = B (27).

(2) Crossover A,B after the 3rd bit:
    A = 111|010 , B = 011|011
    E = 111|011 = 111011 -> x=59 -> f = 64-59 = 5
    F = 011|010 = 011010 -> x=26 -> f = 26
    (E takes A's first 3 bits + B's last 3; F is the complement.)

(3) B and C clone (copied unchanged). D is NOT selected -> extinct.
    Offspring C mutates at the FIRST bit: C = 001001 -> Cm = 101001
    Cm -> x = 41 -> f = 64 - 41 = 23

(4) New generation after selection+crossover+clone+mutation: {E, F, B, Cm}
    (two children E,F from the A x B crossover; B cloned; C mutated to Cm; D gone)
    Fitness: E=5, F=26, B=27, Cm=23.  Total = 5+26+27+23 = 81.  Best = B (27).
    Note the overall fitness rose from 48 to 81 — the population improved. That's evolution.

---

## 6. THE THREE LABS (all run & verified with DEAP)

All use the SAME DEAP skeleton:
  creator.create(...)  -> define a minimising fitness + an Individual type
  toolbox.register(...) -> plug in how to build individuals + the 5 operators
  algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen) -> run the GA loop

### 6.1 Task 3.1 — Travelling Salesman Problem (TSP)
- Problem: visit 17 cities once each and return to start, minimise total distance.
- **Encoding:** a PERMUTATION of city indices 0..16 (a tour).
- **Fitness:** total round-trip distance (MINIMISE). weights=(-1.0,) means minimise.
- **Operators:** PMX crossover (keeps a valid permutation), shuffle mutation, tournament.
- Params: pop=500, cxpb=0.7, mutpb=0.2, ngen=50, seed=16.
- RESULT (verified): distance = **2085** = the KNOWN OPTIMUM. GA nailed it.

### 6.2 Task 3.2 — 8-Queens
- Problem: place 8 queens on an 8x8 board, none attacking each other.
- **Clever encoding:** a PERMUTATION of 0..7. Position i = column i; value = the row of the
  queen in that column. A permutation AUTOMATICALLY guarantees no two queens share a row or
  a column, so the GA only has to remove DIAGONAL attacks.
- **Fitness:** number of pairs of queens on the same diagonal (MINIMISE; 0 = solved).
- **Operators:** PMX crossover, shuffle-index mutation (indpb=2/8), tournament.
- Params: pop=300, cxpb=0.5, mutpb=0.2, ngen=50, seed=64.
- RESULT (verified): fitness = **0** (a valid solution found), e.g. rows [5,3,1,7,4,6,0,2].

### 6.3 Task 3.3 — Map Colouring (graph colouring / CSP)
- Problem: colour 11 regions with 4 colours so NO two neighbours share a colour.
- **Encoding:** list of 11 integers each in {0,1,2,3} (one colour per node).
- **Fitness:** number of edges with both endpoints the same colour (MINIMISE; 0 = valid).
- **Operators:** two-point crossover, uniform-int mutation (indpb=0.05), tournament.
- RESULT (verified):
   - With small settings (pop=10, ngen=100): got 1 conflict (Julia & Steve both blue).
   - With bigger settings (pop=100, ngen=200): got **0 conflicts** - a perfect colouring.
- LESSON: a GA does NOT guarantee the optimum. Quality depends on population size,
  number of generations, and operator (crossover/mutation) rates. This is the key point to
  write in the lab Discussion.

---

## 7. Strengths, weaknesses, and the big exam talking points

STRENGTHS
- Handles huge, awkward, non-differentiable search spaces.
- Population-based => explores many regions at once; less likely to get stuck than a single
  hill-climb.
- Easy to apply to very different problems by just changing encoding + fitness.

WEAKNESSES
- NOT guaranteed optimal (it's a heuristic). Our map-colouring run proved this.
- Sensitive to parameters (pop size, cxpb, mutpb, ngen).
- Can suffer PREMATURE CONVERGENCE: the population becomes too similar too early and gets
  stuck on a mediocre solution. Mutation + diversity help avoid this.
- Can be slow (many fitness evaluations).

KEY ROLES OF THE OPERATORS (classic exam question):
- Selection pushes toward quality (exploitation).
- Crossover mixes good building blocks from parents (exploitation + some exploration).
- Mutation maintains diversity and escapes local optima (exploration).
Good GA = right BALANCE of exploration vs exploitation.

---

## 8. The seminar questions (practise - these are exam-style)
Q1. Bitstring GA on f(x) (DONE in Section 5 above - learn that worked trace).
Q2. Genetic PROGRAMMING to discover trig identities (evolve expressions/trees, fitness =
    how closely the evolved expression matches sin(x+y) over sample points).
Q3. Delivery route for N parcels: encoding=permutation of addresses; fitness=total route
    distance (minimise); operators = PMX crossover + swap mutation + tournament. (= TSP)
Q4. Delivery with M workers x N parcels: encoding must PARTITION parcels among workers
    (e.g. permutation + split points, or a 2D assignment); fitness = total distance of all
    workers (optionally balance the loads); operators keep assignments valid.
Q5. Maintenance scheduling of 7 power units over 4 intervals: encoding = start interval per
    unit; fitness = feasibility (net reserve >= 0 every interval) + minimise peak load /
    maximise reserve; penalise infeasible schedules.
Q6. CNF satisfaction (SAT): encoding = bitstring of truth values for each literal; fitness =
    number of satisfied clauses (MAXIMISE; all clauses satisfied = solution); operators =
    bitwise mutation + single/two-point crossover + tournament.

For EACH design question, always answer in the 1-2-3 structure:
(1) encoding, (2) fitness function, (3) genetic operators (selection/crossover/mutation).

---

## 9. Exam-style checklist — can you do each?
- [ ] State the GA cycle (init -> evaluate -> [select, crossover, mutate, evaluate, replace]).
- [ ] Define chromosome, gene, allele, population, generation, fitness.
- [ ] Perform single-point crossover and bitwise mutation BY HAND (Seminar Q1).
- [ ] Compute fitness and total/best fitness of a population by hand.
- [ ] For a new problem, specify encoding + fitness + operators (the 1-2-3 answer).
- [ ] Explain roles of selection (exploit), crossover (recombine), mutation (explore/diversity).
- [ ] Explain premature convergence and how to avoid it.
- [ ] Explain why a GA is not guaranteed optimal and what affects its result quality.
- [ ] Give the right encoding for permutation problems (TSP/8-queens) and why PMX is used.
