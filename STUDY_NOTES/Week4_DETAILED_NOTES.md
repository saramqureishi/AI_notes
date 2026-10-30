# CS5707 — WEEK 4: PROBABILISTIC INFERENCE (Complete Detailed Notes)

> Self-contained. If you read only this file, you have Week 4.
> Lecturer: Yongmin Li. Reading: Negnevitsky C3 (uncertainty), R&N C12 (quantifying
> uncertainty) & C13 (probabilistic reasoning).
> Labs use scikit-learn (Naive Bayes) and pgmpy (Bayesian networks).

---

## 0. The big idea — reasoning under UNCERTAINTY

Weeks 2-3 assumed a clean, known world. Reality is messy: information is incomplete,
inconsistent, or uncertain. **Uncertainty** = the lack of exact knowledge needed to reach a
perfectly reliable conclusion. Classical true/false logic (law of the excluded middle) can't
cope, so we use **probability** to measure degrees of belief and update them as evidence
arrives. That updating machinery is **Bayesian inference**.

Sources of uncertain knowledge (lecture):
- Weak implications (vague IF-THEN rules -> use numerical certainty factors)
- Imprecise language ("often", "sometimes", "hardly ever" -> quantify them)
- Unknown / missing data (accept "unknown" and reason approximately)
- Combining conflicting experts (weight each expert; no systematic way to set weights)

---

## 1. Basic probability (the foundations)

- Probability P is a number with **0 <= P <= 1**. 0 = impossible, 1 = certain.
- P(success) = (#successes)/(#possible outcomes). With s successes, f failures:
  p = s/(s+f), q = f/(s+f), and **p + q = 1**.
- **Conditional probability** P(B|A) = probability of B GIVEN A has occurred:
      P(B|A) = P(A and B) / P(A)
- **Joint probability** P(A and B) = P(A,B) = probability both happen.
- Rearranging the two conditionals gives the key relationship:
      P(A and B) = P(B|A) P(A) = P(A|B) P(B)

### 1.1 Bayes' Rule (THE central formula — memorise)
```
            P(A|B) * P(B)
 P(B|A) = -----------------
                P(A)
```
Named in words: posterior = (likelihood x prior) / evidence.

### 1.2 Marginalisation (summing out)
If the B_i are mutually exclusive and exhaustive, you can recover P(A):
```
 P(A) = sum_i  P(A and B_i) = sum_i  P(A|B_i) P(B_i)
```
This is how we compute the denominator of Bayes' rule when P(A) isn't given directly.

---

## 2. Bayesian reasoning in expert systems

Rules look like:  IF H is true THEN E is true {with probability p}.
- **H** = hypothesis, **E** = evidence.
- The expert supplies PRIOR p(H), p(not H) and CONDITIONALS p(E|H), p(E|not H).
- The user supplies observed evidence E.
- The system computes the POSTERIOR p(H|E).

### 2.1 Simple two-hypothesis case (H and not H)
```
             p(E|H) p(H)
p(H|E) = --------------------------------
          p(E|H) p(H) + p(E|not H) p(not H)
```

### 2.2 General case (single evidence, m hypotheses)
```
               p(E|H_i) p(H_i)
p(H_i|E) = --------------------------
            sum_k p(E|H_k) p(H_k)
```

### 2.3 Multiple evidences -> NAIVE BAYES
Computing p(E1,E2,...,En | H_i) directly is hard, so we assume the evidences are
**conditionally independent given the hypothesis**:
```
 p(E1,...,En | H_i) = p(E1|H_i) * p(E2|H_i) * ... * p(En|H_i)
```
Then:
```
                 p(E1|H_i) p(E2|H_i)...p(En|H_i) p(H_i)
p(H_i|E1..En) = -----------------------------------------------
                 sum_k p(E1|H_k)...p(En|H_k) p(H_k)
```
This is **Naive Bayes**. The independence assumption is usually NOT truly correct, BUT
Naive Bayes works surprisingly well in practice. ("Naive" = the naive independence assumption.)

---

## 3. WORKED LECTURE EXAMPLE (this is Seminar Q1 — learn to do it by hand)

Priors: p(H1)=0.40, p(H2)=0.35, p(H3)=0.25
Conditionals:
            H1    H2    H3
 p(E1|.)   0.3   0.8   0.5
 p(E2|.)   0.9   0.0   0.7
 p(E3|.)   0.6   0.7   0.9

**Without evidence**, H1 is the strongest belief (0.40).

### After E3 only  (use p(E3|H_i) p(H_i), normalise)
Numerators: H1=0.6*0.40=0.24, H2=0.7*0.35=0.245, H3=0.9*0.25=0.225; sum=0.71(approx 0.710)
 p(H1|E3) = 0.24/0.71  = 0.34
 p(H2|E3) = 0.245/0.71 = 0.34
 p(H3|E3) = 0.225/0.71 = 0.32
Belief in H2 and H3 rose to roughly equal H1.

### After E1 and E3  (multiply the two likelihoods)
Numerators: H1=0.3*0.6*0.40=0.072, H2=0.8*0.7*0.35=0.196, H3=0.5*0.9*0.25=0.1125
 p(H1|E1E3)=0.19, p(H2|E1E3)=0.52, p(H3|E1E3)=0.29.  H2 is now most likely.

### After E1, E2 and E3  (multiply all three)
Numerators: H1=0.3*0.9*0.6*0.40=0.0648, H2=0.8*0.0*0.7*0.35=0 (E2|H2=0 kills it),
            H3=0.5*0.7*0.9*0.25=0.07875
 p(H1|..)=0.45, p(H2|..)=0.00, p(H3|..)=0.55.
LESSON: one zero conditional (p(E2|H2)=0) permanently eliminates H2. Ranking flipped from
H1,H2,H3 at the start to H3 > H1, H2 gone.

---

## 4. CONSISTENCY OF EXPERT JUDGEMENT (Seminar Q2 - the "car won't start" example)

Expert says:   p(StarterFaulty|OddNoises) = 0.70,  p(StarterFaulty) = 0.05
Empirical:     p(OddNoises|StarterFaulty) = 0.85,  p(OddNoises|not StarterFaulty) = 0.20

Apply Bayes:
```
                              0.85 * 0.05
p(Faulty|OddNoises) = --------------------------------- = 0.0425/0.235 = 0.18
                        0.85*0.05 + 0.20*0.95
```
Result 0.18 is FAR below the expert's claimed 0.70 -> the expert's numbers are INCONSISTENT.
Implication for knowledge acquisition: humans are bad at giving probabilities that obey
Bayes' rule, so elicited probabilities must be cross-checked. This is the "bias of the
Bayesian method".

---

## 5. BAYESIAN NETWORKS

A **Bayesian network** is a compact graphical way to represent a full joint distribution
using independence structure.
- **Nodes** = random variables.
- **Directed, ACYCLIC graph** (arrows = direct causal/dependency influence).
- Each node has a **Conditional Probability Table (CPT)** giving P(node | its parents).

### 5.1 The classic Alarm example (causal knowledge)
 Burglary -> Alarm <- Earthquake ;  Alarm -> JohnCalls ;  Alarm -> MaryCalls.
A burglar or earthquake can trigger the alarm; the alarm can make John/Mary call.

### 5.2 Full joint distribution = product of local CPTs
```
 P(x1,...,xn) = product_i P(x_i | parents(x_i))
```
This factorisation is why Bayes nets are compact: instead of one gigantic joint table, you
store small CPTs per node.

### 5.3 Inference methods in Bayes nets
- **Exact: enumeration** - evaluate the full joint via the CPTs. Correct but INEFFICIENT
  (repeats the same sub-computations).
- **Exact: variable elimination** - same principle but smarter: sum out variables from a
  product of factors, move constants outside, reuse intermediate factors. Much faster.
- **Approximate: stochastic simulation** (sampling).
- **Approximate: Markov Chain Monte Carlo (MCMC)**.

---

## 6. THE THREE LABS (all run & verified)

Tools: sklearn's GaussianNB for Naive Bayes; pgmpy for the Bayesian network.

### 6.1 Task 4.1 - Iris classification (Naive Bayes)
- Predict species (3 classes) from flower features. 150 records, 50 per species.
- Formulation: P(c_i | f1,f2) = alpha * P(f1|c_i) P(f2|c_i) P(c_i); pick the highest posterior.
- GaussianNB assumes each feature is Gaussian within each class.
- RESULT (verified, seed=1, 80/20 split):
   - 2 features: accuracy ~0.97 ;  all 4 features: accuracy ~0.97.
   - Errors occur on the boundary between the two species that are NOT linearly separable
     (versicolor vs virginica). The confusion matrix shows 1 such misclassification.

### 6.2 Task 4.2 - Diabetes diagnosis (Naive Bayes)
- 10 features; continuous progression target bucketed into 3 classes (0-150,150-300,300+).
- RESULT (verified, 75/25 split): accuracy ~0.68 - much lower than Iris.
- LESSON: real, noisy, overlapping medical data is far harder than the clean Iris set; the
  tiny 300+ class is barely predicted. Naive Bayes' independence assumption is weaker here.

### 6.3 Task 4.3 - Monty Hall (Bayesian network, pgmpy)
- 3 doors: car + 2 goats. You pick door 0; host opens a goat door; stick or switch?
- Variables: Prize P, Contestant C (both uniform priors 1/3), Host H (CPT depends on P & C:
  host must open a door that is neither your pick nor the prize, revealing a goat).
- Network: Contestant -> Host <- Prize.
- Query P(Prize | Contestant=0, Host=2). RESULT (verified):
      Prize(0)=0.3333  (your original door - STICK)
      Prize(1)=0.6667  (the other closed door - SWITCH)
      Prize(2)=0.0000  (opened door)
  => SWITCHING DOUBLES your winning chance from 1/3 to 2/3.
- 4-door extension (Task 4.3 part 3), verified:
      Prize(0)=0.25, Prize(1)=0.375, Prize(2)=0.0, Prize(3)=0.375
  => sticking = 1/4; each remaining closed door = 3/8 > 1/4, so switching still wins.

---

## 7. Exam-style checklist
- [ ] State Bayes' rule and name each term (posterior, likelihood, prior, evidence).
- [ ] Do marginalisation to compute the denominator.
- [ ] Compute a posterior for 1 evidence, 3 hypotheses (Seminar Q1 part 1) by hand.
- [ ] Update posteriors sequentially as E1, then E1E2, then E1E2E3 arrive (Naive Bayes).
- [ ] Explain the Naive Bayes conditional-independence assumption and why it still works.
- [ ] Check an expert's judgement for consistency with Bayes (the StarterFaulty example).
- [ ] Define a Bayesian network: nodes, acyclic directed graph, CPTs.
- [ ] Write full joint = product of P(node | parents).
- [ ] Name the 4 inference methods (enumeration, variable elimination, sampling, MCMC) and
      say why enumeration is inefficient.
- [ ] Explain the Monty Hall result: stick 1/3 vs switch 2/3, and generalise to n doors.

---

## 8. Seminar questions (practise)
Q1. 3 hypotheses, 3 independent evidences: compute posteriors after E1, after E1E2, after
    E1E2E3. (Worked pattern in Section 3.)
Q2. Fault diagnosis: compute p(H1|E) from empirical conditionals + expert prior; check if the
    expert's p(H1|E)=0.70 is consistent. (Section 4: answer ~0.18, inconsistent.)
Q3. Fault diagnosis with 2 independent evidences (noisy, overheating): posteriors after E1,
    then after E1E2. Use the Naive Bayes product form.
Q4. Monty Hall via a Bayesian network: compute win chance for stick vs switch. (2/3 switch.)
