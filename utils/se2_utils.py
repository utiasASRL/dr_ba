import numpy as np

def circle_dot_operator(p):
    assert p.shape == (2,1)
    S = np.array([[0, -1],
                  [1, 0]])
    res = np.zeros((3,3))
    res[0:2,0:2] = np.eye(2)
    res[0:2,2:3] = S @ p

    return res