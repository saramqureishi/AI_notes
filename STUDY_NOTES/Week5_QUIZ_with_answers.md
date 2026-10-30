# Week 5 — Fuzzy Systems: OFFICIAL Quiz, Answers & Explanations

Score key (one line):
1-B 2-C 3-B 4-B 5-A 6-C 7-B 8-A 9-B 10-C 11-B 12-B 13-B 14-A 15-C 16-D 17-B 18-B
19-A 20-C 21-B 22-C 23-B 24-A

Most-missed: Q11/Q17 (OR = MAX), Q8/Q24 (AND = MIN), Q6 ('very' = square, NOT square root),
Q13 vs Q16 (fuzzification is FIRST, defuzzification/COG is LAST).

Golden rules:  NOT A = 1 - A  |  A AND B = min(A,B)  |  A OR B = max(A,B)
Mamdani pipeline: Fuzzification -> Rule evaluation -> Aggregation -> Defuzzification

---

## FUZZY SET BASICS

**Q1. Who introduced fuzzy sets in 1965?**
A: Lotfi Zadeh. ✅  (Lukasiewicz did earlier multi-valued logic; Aristotle = classical two-valued logic.)

**Q2. In a crisp set, the only possible characteristic-function values are:**
A: 0 and 1. ✅  (An element is fully in or fully out. Fuzzy sets allow anything in [0,1].)

**Q3. Which membership function is a smooth bell-shaped curve?**
A: Gaussian. ✅  (Triangular = 3 points/peak; trapezoidal = flat top; singleton = a single spike.)

**Q4. Membership in 'Tall' = 0.6; membership in 'NOT Tall' = ?**
A: 0.4. ✅  Fuzzy complement: NOT A = 1 - A = 1 - 0.6 = 0.4.

**Q5. A 'linguistic variable' is:**
A: A variable that takes words or sentences as values instead of numbers. ✅
e.g. Height can take the values {short, medium, tall}.

**Q6. Applying the hedge 'very' (concentration) does what?**
A: Membership values are squared, narrowing the set. ✅
'very' A = A^2  (e.g. 0.6 -> 0.36). 'somewhat/more or less' = square root (dilation, widens).

**Q7. 'Multi-valued' logic means in practice:**
A: It allows degrees of truth between 0 and 1. ✅

**Q8. Fuzzy AND (intersection) uses:**
A: Minimum (min). ✅

**Q9. Which logic says a statement is either 100% true or 100% false?**
A: Boolean logic. ✅

---

## FUZZY RULES

**Q10. In a fuzzy IF-THEN rule, the 'THEN' part is called:**
A: The consequent. ✅  (The IF part = antecedent / premise.)

**Q11. A = 0.3, B = 0.7. A OR B = ?**
A: 0.7. ✅  Fuzzy OR = max(0.3, 0.7) = 0.7.  (AND would give min = 0.3.)

**Q12. How does a fuzzy rule differ from a classical rule?**
A: Fuzzy rules allow partial firing based on degrees of membership. ✅
A classical rule either fires completely or not at all; a fuzzy rule fires to a degree (e.g. 0.4).

---

## MAMDANI FUZZY INFERENCE (4 steps)

**Q13. First step of Mamdani inference?**
A: Fuzzification. ✅  Convert crisp inputs into degrees of membership.

**Q14. In rule evaluation, 'clipping' (correlation minimum) does what?**
A: Slices the top of the output fuzzy set at the level of the rule's firing strength. ✅
(The alternative 'scaling' multiplies the whole set by the firing strength instead.)

**Q15. Purpose of 'Aggregation'?**
A: Combine the outputs of all fired rules into a single fuzzy set. ✅  (usually by max/union.)

**Q16. Centre of Gravity (COG) is used in which stage?**
A: Defuzzification. ✅

**Q17. 'IF funding is adequate OR staffing is small THEN risk is low' - rule strength?**
A: Take the MAXIMUM membership value of the two inputs. ✅  (OR -> max.)

**Q18. Why is defuzzification necessary?**
A: Real-world applications usually require a single, crisp output value. ✅
(A motor or a grade sheet needs one number, not a fuzzy shape.)

---

## THE MODULE-ASSESSMENT EXAMPLE

**Q19. The two linguistic input variables?**
A: Coursework and Exam. ✅  (Output = pass possibility.)

**Q20. Coursework = 25%, 'low' set covers 0-50%. First thing the system does?**
A: Determines the degree of membership for the 25% input (Fuzzification). ✅

**Q21. Aggregated output is a trapezoid. What does COG find?**
A: The point on the x-axis that balances the area of the shape. ✅
COG = sum( x * mu(x) ) / sum( mu(x) )  -> the 'balance point', not the peak.

**Q22. Multiple rules fire ('low' pass and 'medium' pass). How is this handled?**
A: It aggregates both rule outputs into one fuzzy distribution. ✅  (then defuzzifies it.)

**Q23. x=25%, y=50% gives '35% pass possibility'. What is the 35%?**
A: A crisp numerical output produced after defuzzification. ✅

**Q24. A rule needs 'coursework is high' AND 'exam is high'. Which operator?**
A: Min. ✅  (AND -> min.)
