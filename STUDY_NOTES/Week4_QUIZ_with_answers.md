# Week 4 — Probabilistic Inference: OFFICIAL Quiz, Answers & Explanations

Score key (one line):
1-A 2-B 3-B 4-B 5-B 6-C 7-A 8-C 9-B 10-B 11-B 12-A 13-B 14-A 15-B 16-A 17-B 18-B 19-C 20-B 21-B

Most-missed: Q5 (logical contingency plans blow up), Q13 vs Q15 (likelihood P(E|H) != posterior P(H|E)),
Q16 (independence assumption => linear scaling).

---

## UNCERTAINTY & WHY PROBABILITY

**Q1. Definition of uncertainty?**
A: The lack of exact knowledge needed to reach a perfectly reliable conclusion. ✅

**Q2. Source of uncertainty = difficulty establishing absolute IF-THEN correlations?**
A: Weak implications. ✅  (-> handled with numerical certainty factors.)

**Q3. Why is classical logic unsuitable for real-world expert systems?**
A: It relies on the law of the excluded middle, which does not allow 'grey areas'. ✅

**Q4. Primary advantage of probabilistic over logic-based agents?**
A: It allows the agent to weigh the merits of plans not guaranteed to succeed. ✅

**Q5. Why can a logical 'contingency plan' become intractably large?**
A: Because it must account for every possible eventuality, no matter how unlikely. ✅

---

## BASIC PROBABILITY

**Q6. Range of values for a degree of belief?**
A: 0 to 1. ✅

**Q7. Which random variable has domain exactly <true,false>?**
A: Boolean random variable. ✅

**Q8. Kolmogorov: probability of a necessarily true proposition?**
A: 1. ✅

**Q9. What does P(a | b) represent?**
A: The posterior probability of 'a' given that 'b' is known. ✅

**Q10. Rule for P(a and b) = P(a|b)P(b)?**
A: The Product Rule. ✅

**Q11. If P(Weather=sunny)=0.7, P(Weather=not sunny)=?**
A: 0.3. ✅  (values sum to 1.)

---

## NAIVE BAYES

**Q12. The fundamental 'naive' assumption?**
A: All effect variables are conditionally independent given the cause. ✅

**Q13. What is the 'likelihood'?**
A: The probability of the evidence given the hypothesis is true, P(E|H). ✅
(prior = belief before; posterior = belief after.)

**Q14. Naive Bayes joint P(Cause,E1,E2,E3) is computed by:**
A: Multiplying the prior P(Cause) by the product of individual conditional probabilities. ✅
P(Cause)*P(E1|Cause)*P(E2|Cause)*P(E3|Cause).

**Q15. P(E|H1)=0.9, P(E|H2)=0.1; what happens to belief in H1?**
A: The belief in H1 will increase relative to H2. ✅  (not to certainty; H2 not zeroed.)

**Q16. Why is Naive Bayes efficient for large-scale classification?**
A: It scales linearly with the number of variables due to the independence assumption. ✅

---

## BAYESIAN NETWORKS

**Q17. A Bayesian network is which structure?**
A: A directed acyclic graph (DAG). ✅

**Q18. A directed arrow A -> B represents:**
A: Node A has a direct influence on Node B. ✅

**Q19. What is stored locally at each node?**
A: A Conditional Probability Table (CPT) showing the effect of its parents. ✅

**Q20. Which inference method avoids repeated calculations by saving intermediate results?**
A: Inference by Variable Elimination. ✅

**Q21. How is the full joint distribution calculated?**
A: By the product of the local conditional distributions at every node. ✅
P(x1..xn) = product_i P(x_i | parents(x_i)).
