"""
Lab 4 - Task 4.3 Monty Hall using a Bayesian Network (pgmpy)
Variables (doors indexed 0..n-1):
  Prize P      - where the car is  (prior uniform)
  Contestant C - initial pick      (prior uniform)
  Host H       - door host opens   (depends on P and C: must differ from both, reveal a goat)
Structure:  Contestant -> Host <- Prize
Query: P(Prize | Contestant=0, Host=2). If > 1/n for a door other than 0 -> switch wins.
"""
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def three_door():
    model = DiscreteBayesianNetwork([('Contestant', 'Host'), ('Prize', 'Host')])
    cpd_c = TabularCPD('Contestant', 3, [[1/3], [1/3], [1/3]])
    cpd_p = TabularCPD('Prize', 3, [[1/3], [1/3], [1/3]])
    cpd_h = TabularCPD('Host', 3,
                       [[0,  0,  0,  0,  0.5, 1, 0, 1, 0.5],
                        [0.5, 0, 1, 0, 0,  0,  1, 0, 0.5],
                        [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]],
                       evidence=['Contestant', 'Prize'], evidence_card=[3, 3])
    model.add_cpds(cpd_c, cpd_p, cpd_h)
    infer = VariableElimination(model)
    post = infer.query(['Prize'], evidence={'Contestant': 0, 'Host': 2},
                       show_progress=False, joint=False)
    print("=== 3 DOORS: P(Prize | Contestant=0, Host=2) ===")
    print(post['Prize'])


def four_door():
    model = DiscreteBayesianNetwork([('Contestant', 'Host'), ('Prize', 'Host')])
    cpd_c = TabularCPD('Contestant', 4, [[0.25]] * 4)
    cpd_p = TabularCPD('Prize', 4, [[0.25]] * 4)
    cpd_h = TabularCPD('Host', 4,
        [[0,   0,   0,   0,   0,   1/3, 0.5, 0.5, 0,   0.5, 1/3, 0.5, 0,   0.5, 0.5, 1/3],
         [1/3, 0,   0.5, 0.5, 0,   0,   0,   0,   0.5, 0,   1/3, 0.5, 0.5, 0,   0.5, 1/3],
         [1/3, 0.5, 0,   0.5, 0.5, 1/3, 0,   0.5, 0,   0,   0,   0,   0.5, 0.5, 0,   1/3],
         [1/3, 0.5, 0.5, 0,   0.5, 1/3, 0.5, 0,   0.5, 0.5, 1/3, 0,   0,   0,   0,   0]],
        evidence=['Contestant', 'Prize'], evidence_card=[4, 4])
    model.add_cpds(cpd_c, cpd_p, cpd_h)
    infer = VariableElimination(model)
    post = infer.query(['Prize'], evidence={'Contestant': 0, 'Host': 2},
                       show_progress=False, joint=False)
    print("=== 4 DOORS: P(Prize | Contestant=0, Host=2) ===")
    print(post['Prize'])


if __name__ == '__main__':
    three_door()
    print()
    four_door()
