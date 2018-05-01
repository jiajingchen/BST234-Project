import numpy as np
import time
import scipy as sp
import scipy.linalg
from scipy import sparse
import scipy.sparse.linalg
import random
import matplotlib.pyplot as plt
import warnings


def prep_data():
    x = np.loadtxt("simulated_genos", delimiter=" ", dtype="float32")
    y = np.array([[1] * 10000 + [0] * 10000], dtype="float32")
    y_c = y - 0.5
    return x, y, y_c

#############################################
#IRLB


def mult(A,x,t=False):
  if(sparse.issparse(A)):
    m = A.shape[0]
    n = A.shape[1]
    if(t):
      return(sparse.csr_matrix(x).dot(A).transpose().todense().A[:,0])
    return(A.dot(sparse.csr_matrix(x).transpose()).todense().A[:,0])
  if(t):
    return(x.dot(A))
  return(A.dot(x))

def orthog(Y,X):
  dotY = mult(X,Y,t=True)
  return (Y - mult(X,dotY))


def invcheck(x):
  eps2  = 2*np.finfo(np.float).eps
  if(x>eps2):
    x = 1/x
  else:
    x = 0
  return(x)

def irlb(A,n,tol=0.0001,maxit=50):
  nu     = n
  m      = A.shape[0]
  n      = A.shape[1]
  m_b    = min((nu+20, 3*nu, n))
  mprod  = 0
  it     = 0
  j      = 0
  k      = nu
  smax   = 1
  ifsparse = sparse.issparse(A)

  V  = np.zeros((n,m_b))
  W  = np.zeros((m,m_b))
  F  = np.zeros((n,1))
  B  = np.zeros((m_b,m_b))

  V[:,0]  = np.random.randn(n)
  V[:,0]  = V[:,0]/np.linalg.norm(V)

  while(it < maxit):
    if(it>0): j=k
    W[:,j] = mult(A,V[:,j])
    mprod+=1
    if(it>0):
      W[:,j] = orthog(W[:,j],W[:,0:j]) # NB W[:,0:j] selects columns 0,1,...,j-1
    s = np.linalg.norm(W[:,j])
    sinv = invcheck(s)
    W[:,j] = sinv*W[:,j]
    while(j<m_b):
      F = mult(A,W[:,j],t=True)
      mprod+=1
      F = F - s*V[:,j]
      F = orthog(F,V[:,0:j+1])
      fn = np.linalg.norm(F)
      fninv= invcheck(fn)
      F  = fninv * F
      if(j<m_b-1):
        V[:,j+1] = F
        B[j,j] = s
        B[j,j+1] = fn
        W[:,j+1] = mult(A,V[:,j+1])
        mprod+=1
        W[:,j+1] = W[:,j+1] - fn*W[:,j]
        W[:,j+1] = orthog(W[:,j+1],W[:,0:(j+1)])
        s = np.linalg.norm(W[:,j+1])
        sinv = invcheck(s)
        W[:,j+1] = sinv * W[:,j+1]
      else:
        B[j,j] = s
      j+=1
    S    = np.linalg.svd(B)
    R    = fn * S[0][m_b-1,:]
    if(it<1):
      smax = S[1][0]
    else:
      smax = max((S[1][0],smax))

    conv = sum(np.abs(R[0:nu]) < tol*smax)
    if(conv < nu):
      k = max(conv+nu,k)
      k = min(k,m_b-3)
    else:
      break
    V[:,0:k] = V[:,0:m_b].dot(S[2].transpose()[:,0:k])
    V[:,k] = F
    B = np.zeros((m_b,m_b))
    for l in range(0,k):
      B[l,l] = S[1][l]
    B[0:k,k] = R[0:k]
    W[:,0:k] = W[:,0:m_b].dot(S[0][:,0:k])
    it+=1

  U = W[:,0:m_b].dot(S[0][:,0:nu])
  V = V[:,0:m_b].dot(S[2].transpose()[:,0:nu])
  return((U,S[1][0:nu],V,it,mprod))


####################################################
# randomized SVD


def randomized_svd(M, k=10):
    m, n = M.shape
    transpose = False
    if m < n:
        transpose = True
        M = M.T

    rand_matrix = np.random.normal(size=(M.shape[1], k))  # short side by k
    Q, _ = np.linalg.qr(M @ rand_matrix, mode='reduced')  # long side by k
    smaller_matrix = Q.T @ M  # k by short side
    U_hat, s, V = np.linalg.svd(smaller_matrix, full_matrices=False)
    U = Q @ U_hat

    if transpose:
        return V.T, s.T, U.T
    else:
        return U, s, V


def generate_Q(eig, N=1000000):
    Q_distribution = []
    for i in range(N):
        # generate 50 chi-square r.v
        sample = np.random.chisquare(1, size=50)
        Q_stat = np.sum(eig * sample) / 4
        Q_distribution.append(Q_stat)
    return Q_distribution


def inverse_percentile(arr, num):
    arr = sorted(arr)
    i_arr = [i for i, x in enumerate(arr) if x > num]
    return i_arr[0] / len(arr) if len(i_arr) > 0 else 1


def p_cal(Q_distribution, q=223.25):
    abs_Q_distribution = abs(Q_distribution - np.average(Q_distribution))
    q_abs = abs(q - np.average(Q_distribution))
    p = 1 - inverse_percentile(abs_Q_distribution, q_abs)
    return p


if __name__ == "__main__":
    # Specify file path here. Default is current working directory.
    x, y, y_c = prep_data()
    q = 223.25
    n = 20000
    X = x[:, :]
    n = X.shape[0]
    p = X.shape[1]
    X_sparse = sp.sparse.csr_matrix(X)

    # IRLB: This is the first implementation. Slow.
    eig_irlb = irlb(X, 50)[1] ** 2

    # RSVD: This is the second implementation. Fast.
    # If need to test speed, use this one.
    start_time = time.clock()
    eig_rsvd = randomized_svd(x, 50)[1] ** 2
    end_time = time.clock()
    print("The time of RSVD is: ", end_time - start_time)

    # Calculate p-value
    Q_distribution_1 = generate_Q(eig=eig_irlb)
    print("The p-value of IRLB is:", p_cal(Q_distribution_1, q))
    Q_distribution_2 = generate_Q(eig=eig_rsvd)
    print("The p-value of RSVD is:", p_cal(Q_distribution_2, q))