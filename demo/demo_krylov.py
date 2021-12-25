#########################################################################
# File Name: demo_krylov.py
# Author: Gang Zhao
# Mail: zhaog6@lsec.cc.ac.cn
# Created Time: 12/24/2021 Friday 15:19:26
# Description: Used to test the TF-QMR implementation
#########################################################################


import numpy as np
from numpy.testing import assert_equal
import time
from scipy import sparse

import sys
sys.path.append("..")
from src import *


#------------------------------------
#   Buiding 1-D Poisson equations
#------------------------------------
def _create_sparse_poisson1d(n):
    # Make Gilbert Strang's favorite matrix
    # http://www-math.mit.edu/~gs/PIX/cupcakematrix.jpg
    P1d = sparse.diags([[-1]*(n-1), [2]*n, [-1]*(n-1)], [-1, 0, 1])
    assert_equal(P1d.shape, (n, n))
    return P1d

#------------------------------------
#   Buiding 2-D Poisson equations
#------------------------------------
def _create_sparse_poisson2d(n):
    P1d = _create_sparse_poisson1d(n)
    P2d = sparse.kronsum(P1d, P1d)
    assert_equal(P2d.shape, (n*n, n*n))
    return P2d.tocsr()

#---------------------------------
#   Record convergence history
#---------------------------------
def cb(x):
    residNorm = np.linalg.norm(b - A@x) / bnorm
    residuals.append(residNorm)

def cb_gmres(x):
    residuals.append(x / bnorm)


n = 16
A = _create_sparse_poisson2d(n)
b = np.ones(n*n)
bnorm = np.linalg.norm(b)

#-------------------------
#   SciPy/TFQMR
#-------------------------
print("#-------------------------")
print("#   Transpose-Free QMR")
print("#-------------------------")
print("TFQMR: Beginning solve...")
residuals = []

cpuTime = time.time()
(x, info) = tfqmr(A, b, callback=cb, show=True)
print(f" --- system solved with SciPy/TFQMR (in {time.time() - cpuTime} s)")
print()
print("####################################################\n"
      "TFQMR: The convergence history of the residual norm: \n"
      "####################################################\n"
      f"{residuals}")
print()

#-----------------------
#   SciPy/Fast GMRES
#-----------------------
print("#---------------------------")
print("#   Fast restarted GMRES")
print("#---------------------------")
print("GMRES: Beginning solve...")
residuals = []

cpuTime = time.time()
(x, info) = gmres(A, b, restart=100, callback=cb_gmres, callback_type='prnorm',
                  version='fast', show=True)
print(" --- system solved with SciPy/Fast GMRES (in "
      f"{time.time() - cpuTime} s)")
print()
print("####################################################\n"
      "GMRES: The convergence history of the residual norm: \n"
      "####################################################\n"
      f"{residuals}")
print()

#-------------------------
#   SciPy/CR
#-------------------------
print("#-------------------------")
print("#   Conjugate Residual")
print("#-------------------------")
print("CR: Beginning solve...")
residuals = []

cpuTime = time.time()
(x, info) = cr(A, b, callback=cb, show=True)
print(f" --- system solved with SciPy/CR (in {time.time() - cpuTime} s)")
print()
print("####################################################\n"
      "CR: The convergence history of the residual norm: \n"
      "####################################################\n"
      f"{residuals}")
print()
