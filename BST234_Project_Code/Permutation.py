import numpy as np
import time
import random
from scipy import sparse
import random
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as lg


def prep_data():
    x = np.loadtxt("simulated_genos", delimiter=" ", dtype="float32")
    y = np.array([[1] * 10000 + [0] * 10000], dtype="float32")
    y_c = y - 0.5
    return x, y, y_c

def gener_permut(m = 1000):
    q = np.empty((m, 1))
    for i in range(m):
        y_cc = np.zeros((20000, 1)) - 0.5
        y_cc[np.random.choice(20000, size=10000, replace=False)] = 0.5
        q[i, 0] = y_cc.T @ xxt_sparse @ y_cc
    return q

def inverse_percentile(arr, num):
    arr = sorted(arr)
    i_arr = [i for i, x in enumerate(arr) if x > num]
    return i_arr[0] / len(arr) if len(i_arr) > 0 else 1

def p_cal(Q_distribution, q = 223.25):
    abs_Q_distribution = abs(Q_distribution - np.average(Q_distribution))
    q_abs = abs(q - np.average(Q_distribution))
    p = 1 - inverse_percentile(abs_Q_distribution, q_abs)
    return p

if __name__ == "__main__":
    # Specify file path here. Default is current working directory. 
    x, y, y_c = prep_data()
    xxt = np.dot(x, x.T)
    xxt_sparse = sparse.csr_matrix(xxt)

    # This is the function to generate permutation samples.
    start_time = time.clock()
    q_permute = gener_permut()
    end_time = time.clock()
    print("The time of 1000 Permutation is: ", end_time - start_time)

    print("The p-value of Permutation is:", p_cal(q_permute))