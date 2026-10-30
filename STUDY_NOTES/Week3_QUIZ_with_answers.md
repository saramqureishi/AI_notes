# Week 3 — Genetic Algorithms: OFFICIAL Quiz, Answers & Explanations

Score key (one line):
1-A 2-B 3-C 4-B 5-C 6-C 7-D 8-B 9-A 10-B 11-B 12-C 13-B 14-B 15-A 16-B 17-B 18-B
19-C 20-B 21-B 22-A 23-B 24-B 25-B 26-C

Most-missed: Q6/Q7 (compute x THEN square it: fitness is x^2 not x),
Q12 (F=1/D because we invert a minimisation), Q16/Q19 (terminals=leaves/variables, functions=operators like sqrt).

---

## GA FUNDAMENTALS

**Q1. Primary purpose of the 'Mutation' operator?**
A: To maintain genetic diversity and prevent the population from [getting stuck in] local optima. ✅
(Combining parents = crossover; choosing survivors = selection.)

**Q2. Best encoding for a solution that is a sequence/order of items?**
A: Permutation Encoding. ✅  (Binary=bitstrings, Tree=GP, Value=reals.)

**Q3. In Roulette Wheel Selection, selection probability is determined by:**
A: The ratio of its fitness to the total fitness of the population. ✅

**Q4. Which component is the 'interface' between the GA and the real-world problem?**
A: Fitness Function. ✅  (The one piece you rewrite per problem.)

**Q5. What happens during single-point crossover?**
A: Two chromosomes are cut at a random point and their tail segments are swapped. ✅

---

## BINARY ENCODING / FITNESS ARITHMETIC

**Q6. Maximising f(x)=x^2 (x in 0..31), fitness of chromosome 11000?**
A: 576. ✅   11000 = 16+8 = 24, and 24^2 = 576.  (Trap: 24 is x, not fitness.)

**Q7. 5-bit string, max possible fitness of f(x)=x^2?**
A: 961. ✅   max value = 11111 = 31, 31^2 = 961.

**Q8. Average fitness increasing but maximum fitness flat -> what's happening?**
A: The population is converging toward the current best solution. ✅
(Does NOT prove the global optimum was found.)

**Q9. Which represents 'cloning'/'elitism'?**
A: Carrying the best individual into the next generation without changes. ✅

**Q10. 6-bit string, which represents x = 9?**
A: 001001. ✅   001001 = 8+1 = 9.  (100100=36, 000111=7, 011001=25.)

---

## TSP (TRAVELLING SALESMAN)

**Q11. Why is single-point crossover unsuitable for TSP?**
A: It can create invalid tours where cities are repeated or missing. ✅

**Q12. TSP minimises total distance D; how is fitness F typically calculated?**
A: F = 1 / D. ✅  (Invert so smaller distance -> larger fitness.)

**Q13. Which mutation 'untangles' crossing paths in a TSP route?**
A: Inversion mutation. ✅  (reverses a segment of the tour.)

**Q14. Which crossover preserves relative ordering of cities in TSP?**
A: Order crossover (OX). ✅

**Q15. What does a 'gene' represent in a TSP permutation chromosome?**
A: An individual city in the sequence. ✅

---

## GENETIC PROGRAMMING (Pythagorean theorem / symbolic regression)

**Q16. Leaf nodes of a GP tree represent:**
A: Terminals (variables a and b). ✅  (Internal nodes = operators/functions.)

**Q17. How is crossover performed in tree-based GP?**
A: By exchanging randomly selected sub-trees between two parents. ✅

**Q18. Evolving the Pythagorean formula, a fitness score of 0 represents:**
A: A perfect match to the target data (no error). ✅  (fitness = error; 0 = perfect.)

**Q19. Which belongs in the 'Function Set' for discovering Pythagoras?**
A: The 'sqrt' operator. ✅  (a, b, pi are terminals, not functions.)

**Q20. Term for finding an expression that fits data points using GP?**
A: Symbolic regression. ✅

**Q21. 'Mutation' of a tree structure in GP:**
A: A random node or sub-tree is replaced with a new random branch. ✅

---

## MAINTENANCE SCHEDULING

**Q22. What is 'Net Reserve'?**
A: Total capacity of units in use minus the maximum expected load. ✅  (must be >= 0 each interval.)

**Q23. How are hard constraints (power shortages) typically handled?**
A: By applying a penalty function that reduces the fitness of the individual. ✅

**Q24. Common 'gene' representation for a maintenance schedule?**
A: The time interval/period in which a specific unit is maintained. ✅

**Q25. Why is maintenance scheduling hard for traditional search?**
A: The search space is large and involves complex, competing constraints. ✅

**Q26. Unit needs 2 consecutive intervals but chromosome places them apart - GA response?**
A: Apply a penalty to the fitness score for violating a duration constraint. ✅
