
from fava.decomposers.decomposer import *


def test_all_subsets():
    selected = [1, 4, 0]
    q = 1
    V_q = all_subsets(selected, q, False)
    V_q = list(V_q)
    assert len(V_q) == 3
