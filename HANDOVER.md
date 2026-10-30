# HANDOVER — AI Study Assistant for CS5707 Artificial Intelligence

> **To the next AI:** read this whole file before you reply to Sara. It explains who she
> is, how she wants to work, what has been done, and what comes next. Follow the workflow
> in Section 4 every week.

---

## 1. Who the student is

- **Name:** Sara. MSc student at **Brunel University London**.
- **Module:** **CS5707 Artificial Intelligence** (module leader Prof. Yongmin Li). The assessment block code is CS5807.
- **Intake:** January intake. She **already completed Machine Learning and Deep Learning** last
  semester, and she **already knows Google Colab / Jupyter**. Don't re-teach ML/DL basics or
  Colab. This module is *classical AI* and leaves out ML/DL on purpose.
- **Framing she asked for:** "You are a teacher with only 100 hours with me. Make me
  functional. No theory without practical use, no generic lists."
- She is often **in class or short on attention**. She relies on the notes being detailed
  enough to replace the lecture.

## 2. How the module is assessed (drives everything)

- **100% = one 2-hour WRITTEN exam**, held in the winter exam period. **There is no graded coursework.**
- Labs are **ungraded practice**, but they build the intuition the exam rewards.
- Learning outcomes:
  - **LO1:** understand key AI concepts and methods.
  - **LO2:** apply them to problems, justify the approach, and critically evaluate it.
- What the exam asks for: problem formulation, **hand-tracing algorithms**, hand calculations
  (fitness, Bayes posteriors, fuzzy min/max), and comparing or evaluating methods. She does **not**
  write whole programs from memory.
- The study guide encourages using AI tools for study, but copy-paste answers won't pass.

## 3. Course schedule

| Week | Topic | Status |
|---|---|---|
| 1 | Introduction | ✅ Done. Low exam value: skip the PDFs, keep 4 definitions |
| 2 | Search (BFS, DFS, UCS, Greedy, A*) | ✅ Notes + labs + quiz |
| 3 | Genetic Algorithms | ✅ Notes + labs + quiz (no lecture PDF, taught via seminar) |
| 4 | Probabilistic Inference (Bayes, Naive Bayes, Bayes nets) | ✅ Notes + labs + quiz |
| 5 | Fuzzy Systems | ⚠️ **Quiz answered only.** Lecture/lab not yet in the repo |
| 6 | Reading week | — |
| 7 | Image Processing | ⏳ Not started |
| 8 | Natural Language Processing | ⏳ Not started |
| 9 | Large Language Models | ⏳ Not started |
| 10 | Generative AI | ⏳ Not started |

Sara uploads new course PDFs and notebooks to **this repo** (`saramqureishi/AI_notes`) each week.
They go in the repo root and are named like `Lecture W5 ...pdf`, `Lab W5 ...pdf`,
`Seminar W5 ...pdf`, plus `.ipynb` notebooks.

## 4. THE WEEKLY WORKFLOW she wants (follow exactly)

When she says "added week N" or shares the repo link:

1. **Check the repo for new files.** The quickest way is
   `https://api.github.com/repos/saramqureishi/AI_notes/git/trees/HEAD?recursive=1`.
2. **Read everything for that week**: the lecture PDF, seminar PDF, lab PDF, and every notebook.
   Extract the full text and **don't skip slides**.
3. **Write VERY DETAILED lecture notes** to `STUDY_NOTES/WeekN_DETAILED_NOTES.md`.
   She has said "don't miss anything" more than once. Use the structure in Section 5.
4. **Do ALL the labs in one go.** Don't walk through them one task at a time; she asked for this
   explicitly. Actually **run** the code, report the **real verified outputs**, and save clean
   runnable scripts to `STUDY_NOTES/weekN/labN_*.py`. Write the "Discussion" lessons
   (what the results prove) because those match exam-style evaluation.
5. **Quiz:** she pastes the official weekly quiz (multiple choice). For each question give the
   **correct option plus a short explanation of why it's right and why the main trap option
   is wrong**. Save it to `STUDY_NOTES/WeekN_QUIZ_with_answers.md` with:
   - a **one-line score key** at the top (e.g. `1-A 2-B 3-C ...`)
   - a **"Most-missed"** line naming the 2–3 trickiest questions
   - questions grouped by sub-topic
   Answer the quiz questions she gives you. **Don't invent your own practice quiz instead.**
   (I did that once for Week 3, she corrected me, and I replaced it with the real one.)
6. Keep every week's notes in the **same format** so they're consistent.

## 5. Notes format and style (match the existing files)

Use these sections, in this order:
- `## 0. The big idea`: why this topic exists, in plain words.
- Numbered sections for each concept, with **formulas in code blocks** and **tables** for comparisons.
- A **"WORKED EXAMPLE"** section that does the seminar or lecture example **by hand, step by step**.
  This is the most valuable part for a written exam.
- A **"THE LABS (all run & verified)"** section with encoding/formulation, parameters, the real
  results, and the lesson.
- **"Exam-style checklist"** using `- [ ]` items.
- **"Seminar questions (practise)"** with short answer outlines.

Style rules:
- Direct and practical. Mark what is high exam value and what can be ignored.
- **Be honest about gaps.** Say "no lecture PDF this week" or "not verified" rather than guessing.
- Use markdown tables and ASCII. Short paragraphs.
- When something earns marks in a "critically evaluate" question, say so.

## 6. Key content already covered (so you can stay consistent)

- **Week 2 Search:** 4-part formulation (state/action/goal test/path cost). On the 6-city route 1→5,
  UCS gives 1-3-6-5 at cost **20**, and BFS/DFS give 1-6-5 at cost 23. A*: f=g+h. Admissible =
  never overestimates. Consistent ⇒ admissible. On the 8-puzzle, A*(h2 Manhattan) solves
  depth 26 and A*(h1 misplaced) stalls; DFS blows up.
- **Week 3 GA:** design = encoding, fitness, selection, crossover, mutation. The seminar Q1
  bitstring trace: total fitness goes 48 → 81. TSP hit the optimum of 2085. 8-Queens reached
  0 clashes. Map colouring had 1 conflict with pop=10 and 0 with pop=100. This shows GAs are
  not guaranteed to find the optimum.
- **Week 4 Inference:** Bayes rule, marginalisation, Naive Bayes (conditional independence),
  Bayes nets (DAG + CPTs, joint = product). Lecture example: H2 eliminated because p(E2|H2)=0.
  Expert consistency check: Bayes gives 0.18 against the expert's 0.70. Iris NB accuracy is
  about 97% and diabetes about 68%. Monty Hall: stick 1/3 vs switch 2/3, and with 4 doors
  1/4 vs 3/8.
- **Week 5 Fuzzy (quiz only):** NOT = 1−A, AND = min, OR = max, "very" = A².
  Mamdani steps: fuzzification → rule evaluation (clipping) → aggregation → defuzzification (COG).

## 7. Environment and technical gotchas (sandbox-specific)

- `/root/.local` is **read-only** in the sandbox, so `pip install` fails there. Create a
  workspace venv (`python3 -m venv /projects/sandbox/venv`) and install into it.
- Packages used: `pymupdf` (PDF text extraction), `simpleai` (Week 2 search), `deap` (Week 3 GA),
  `scikit-learn` + `pgmpy` (Week 4), plus `numpy`, `matplotlib`.
- pgmpy now calls the class `DiscreteBayesianNetwork`. The lab PDF uses the old `BayesianNetwork`.
- When the 8-puzzle uses weak heuristics or DFS, it can hang. Run each algorithm with a timeout
  (`timeout 20 ...`); the blow-up itself is a lesson worth noting.
- The sandbox can reset between sessions. Everything that matters is committed here.
- Sara reads in a **browser** with no local filesystem. Show key results in chat, and
  push notes to GitHub so she can read them.

## 8. Other modules (keep them separate)

- Sara also takes **CS5702 Modern Data** (R / RMarkdown, Prof. Martin Shepperd). Its materials
  are in a **different repo**: `saramqureishi/modern_data`. Assessment is shared with CS5701
  through CS5801 (coursework) and CS5802 (2-hour closed-book exam).
  I started reading its Week 3 ("Engineering or Hacking?") but **did not finish the notes or labs**.
  If she asks about it, **keep those notes in that repo**, not mixed into AI_notes.
- **Always confirm which module/repo** a "Week N" request refers to before writing notes.

## 9. Pending / next steps

1. **Week 5 Fuzzy Systems:** when the lecture, lab and seminar arrive, write the detailed notes and run
   the labs (likely `scikit-fuzzy`). Include a fully worked coursework/exam → pass-possibility
   example with fuzzification, min/max rule strengths, clipping, aggregation and a COG calculation
   by hand. The quiz is already answered in `STUDY_NOTES/Week5_QUIZ_with_answers.md`.
2. Weeks 7–10 as they're released, using the same workflow.
3. Optional, high value: practise exam questions where she answers and the AI marks her
   answers (hand traces of UCS/A*, the GA bitstring, Bayes posteriors, fuzzy COG).

## 10. File map

```
HANDOVER.md                      <- this file
STUDY_NOTES/
  README.md                      <- index of all notes
  00_How_Assessed.md
  01_Week1_Introduction.md
  02_Week2_Search.md             (short version)
  03_Week2_Lab_Results.md
  Week2_DETAILED_NOTES.md  + Week2_QUIZ_with_answers.md
  Week3_DETAILED_NOTES.md  + Week3_QUIZ_with_answers.md
  Week4_DETAILED_NOTES.md  + Week4_QUIZ_with_answers.md
  Week5_QUIZ_with_answers.md
  week1/ week2/ week3/ week4/    <- runnable verified lab scripts
```
