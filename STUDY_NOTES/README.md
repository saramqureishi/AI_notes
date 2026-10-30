# CS5707 Artificial Intelligence — Study Notes Index

Assessment: **100% two-hour written exam.** See `00_How_Assessed.md`.
If you're an AI assistant picking this up, read `../HANDOVER.md` first.

| Week | Topic | Detailed notes | Quiz (answers) | Labs |
|---|---|---|---|---|
| 1 | Introduction | `01_Week1_Introduction.md` | — | `week1/` |
| 2 | Search | `Week2_DETAILED_NOTES.md` (+ `02_Week2_Search.md`, `03_Week2_Lab_Results.md`) | `Week2_QUIZ_with_answers.md` | `week2/` |
| 3 | Genetic Algorithms | `Week3_DETAILED_NOTES.md` | `Week3_QUIZ_with_answers.md` | `week3/` |
| 4 | Probabilistic Inference | `Week4_DETAILED_NOTES.md` | `Week4_QUIZ_with_answers.md` | `week4/` |
| 5 | Fuzzy Systems | *pending (lecture not yet uploaded)* | `Week5_QUIZ_with_answers.md` | — |

## Running the lab scripts
```
python3 -m venv venv && . venv/bin/activate
pip install simpleai deap scikit-learn pgmpy numpy matplotlib
python week2/lab2_route.py
python week3/lab3_tsp.py
python week4/lab4_montyhall.py
```
