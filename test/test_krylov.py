#########################################################################
# File Name: test_krylov.py
# Author: Zhao Gang
# Mail: zhaog6@lsec.cc.ac.cn
# Created Time: 2021年12月24日 星期五 17:19:46
#########################################################################

import itertools
import platform
import numpy as np

from numpy.testing import (assert_equal, assert_array_equal, assert_,
                           assert_allclose, suppress_warnings)
import pytest
from pytest import raises as assert_raises

from numpy import zeros, arange, array, ones, eye, iscomplexobj
from scipy import __version__ as VERSION
from scipy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, SparseEfficiencyWarning, kronsum

from scipy.sparse.linalg import LinearOperator, aslinearoperator

import sys
sys.path.append("..")
from src import *


class Case:
    def __init__(self, name, A, b=None, skip=None, nonconvergence=None):
        self.name = name
        self.A = A
        if b is None:
            self.b = arange(A.shape[0], dtype=float)
        else:
            self.b = b
        if skip is None:
            self.skip = []
        else:
            self.skip = skip
        if nonconvergence is None:
            self.nonconvergence = []
        else:
            self.nonconvergence = nonconvergence

    def __repr__(self):
        return f"{self.name}"


class IterativeParams:
    def __init__(self):
        # list of tuples (solver, symmetric, positive_definite )
        solvers = [tfqmr, gmres, cr]
        sym_solvers = [cr]

        self.solvers = solvers

        # list of tuples (A, symmetric, positive_definite )
        self.cases = []

        # Symmetric and Positive Definite
        N = 40
        data = ones((3,N))
        data[0,:] = 2
        data[1,:] = -1
        data[2,:] = -1
        Poisson1D = spdiags(data, [0,-1,1], N, N, format='csr')
        self.Poisson1D = Case("poisson1d", Poisson1D)
        self.cases.append(Case("poisson1d", Poisson1D))
        # note: minres fails for single precision
        self.cases.append(Case("poisson1d", Poisson1D.astype('f')))

        # Symmetric and Negative Definite
        self.cases.append(Case("neg-poisson1d", -Poisson1D))
        # note: minres fails for single precision
        self.cases.append(Case("neg-poisson1d", (-Poisson1D).astype('f')))

        # 2-dimensional Poisson equations
        Poisson2D = kronsum(Poisson1D, Poisson1D)
        self.Poisson2D = Case("poisson2d", Poisson2D)
        # note: minres fails for 2-d poisson problem, it will be fixed in the future PR
        self.cases.append(Case("poisson2d", Poisson2D))
        # note: minres fails for single precision
        self.cases.append(Case("poisson2d", Poisson2D.astype('f')))

        # Symmetric and Indefinite
        data = array([[6, -5, 2, 7, -1, 10, 4, -3, -8, 9]], dtype='d')
        RandDiag = spdiags(data, [0], 10, 10, format='csr')
        self.cases.append(Case("rand-diag", RandDiag))
        self.cases.append(Case("rand-diag", RandDiag.astype('f')))

        # Random real-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4)
        self.cases.append(Case("rand", data, skip=sym_solvers))
        self.cases.append(Case("rand", data.astype('f'), skip=sym_solvers))

        # Random symmetric real-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4)
        data = data + data.T
        self.cases.append(Case("rand-sym", data))
        self.cases.append(Case("rand-sym", data.astype('f')))

        # Random pos-def symmetric real
        np.random.seed(1234)
        data = np.random.rand(9, 9)
        data = np.dot(data.conj(), data.T)
        self.cases.append(Case("rand-sym-pd", data))
        # note: minres fails for single precision
        self.cases.append(Case("rand-sym-pd", data.astype('f')))

        # Random complex-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
        self.cases.append(Case("rand-cmplx", data, skip=sym_solvers))
        self.cases.append(Case("rand-cmplx", data.astype('F'),
                          skip=sym_solvers))

        # Random hermitian complex-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
        data = data + data.T.conj()
        self.cases.append(Case("rand-cmplx-herm", data))
        self.cases.append(Case("rand-cmplx-herm", data.astype('F')))

        # Random pos-def hermitian complex-valued
        np.random.seed(1234)
        data = np.random.rand(9, 9) + 1j*np.random.rand(9, 9)
        data = np.dot(data.conj(), data.T)
        self.cases.append(Case("rand-cmplx-sym-pd", data))
        self.cases.append(Case("rand-cmplx-sym-pd", data.astype('F')))

        # Non-symmetric and Positive Definite
        #
        # cgs, qmr, and bicg fail to converge on this one
        #   -- algorithmic limitation apparently
        data = ones((2,10))
        data[0,:] = 2
        data[1,:] = -1
        A = spdiags(data, [0,-1], 10, 10, format='csr')
        self.cases.append(Case("nonsymposdef", A,
                               skip=sym_solvers+[tfqmr]))
        self.cases.append(Case("nonsymposdef", A.astype('F'),
                               skip=sym_solvers+[tfqmr]))

        # Symmetric, non-pd, hitting cgs/bicg/bicgstab/qmr/cr breakdown
        A = np.array([[0, 0, 0, 0, 0, 1, -1, -0, -0, -0, -0],
                      [0, 0, 0, 0, 0, 2, -0, -1, -0, -0, -0],
                      [0, 0, 0, 0, 0, 2, -0, -0, -1, -0, -0],
                      [0, 0, 0, 0, 0, 2, -0, -0, -0, -1, -0],
                      [0, 0, 0, 0, 0, 1, -0, -0, -0, -0, -1],
                      [1, 2, 2, 2, 1, 0, -0, -0, -0, -0, -0],
                      [-1, 0, 0, 0, 0, 0, -1, -0, -0, -0, -0],
                      [0, -1, 0, 0, 0, 0, -0, -1, -0, -0, -0],
                      [0, 0, -1, 0, 0, 0, -0, -0, -1, -0, -0],
                      [0, 0, 0, -1, 0, 0, -0, -0, -0, -1, -0],
                      [0, 0, 0, 0, -1, 0, -0, -0, -0, -0, -1]], dtype=float)
        b = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float)
        assert (A == A.T).all()
        self.cases.append(Case("sym-nonpd", A, b, nonconvergence=[tfqmr, cr]))


params = IterativeParams()


def check_maxiter(solver, case):
    A = case.A
    tol = 1e-12

    b = case.b
    x0 = 0*b

    # To test for the fast version of GMRES
    residuals = []

    def callback(x):
        residuals.append(norm(b - case.A*x))

    if solver is gmres:
        x, info = solver(A, b, x0=x0, tol=tol, maxiter=1, callback=callback,
                         callback_type='x', version='fast')
        assert_equal(len(residuals), 2)
        assert_equal(info, 1)


def test_maxiter():
    case = params.Poisson1D
    for solver in params.solvers:
        if solver in case.skip:
            continue
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            check_maxiter(solver, case)


def assert_normclose(a, b, tol=1e-8):
    residual = norm(a - b)
    tolerance = tol * norm(b)
    msg = f"residual ({residual}) not smaller than tolerance ({tolerance})"
    assert_(residual < tolerance, msg=msg)


def check_convergence(solver, case):
    A = case.A

    if A.dtype.char in "dD":
        tol = 1e-8
    else:
        tol = 1e-2

    b = case.b
    x0 = 0*b

    if solver is not gmres:
        x, info = solver(A, b, x0=x0, tol=tol)

        assert_array_equal(x0, 0*b)  # ensure that x0 is not overwritten
        if solver not in case.nonconvergence:
            assert_equal(info,0)
            assert_normclose(A.dot(x), b, tol=tol)
        else:
            assert_(info != 0)
            assert_(np.linalg.norm(A.dot(x) - b) <= np.linalg.norm(b))
    else:
        is_64bit = np.dtype(np.intp).itemsize == 8
        if is_64bit == True or case != params.cases[20]:
            x, info = solver(A, b, x0=x0, tol=tol, version='fast')
            assert_array_equal(x0, 0*b)  # ensure that x0 is not overwritten
            if solver not in case.nonconvergence:
                assert_equal(info,0)
                assert_normclose(A.dot(x), b, tol=tol)
            else:
                assert_(info != 0)
                assert_(np.linalg.norm(A.dot(x) - b) <= np.linalg.norm(b))


def test_convergence():
    for solver in params.solvers:
        for case in params.cases:
            if solver in case.skip:
                continue
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning, ".*called without specifying.*")
                check_convergence(solver, case)


def check_precond_dummy(solver, case):
    tol = 1e-8

    def identity(b,which=None):
        """trivial preconditioner"""
        return b

    A = case.A

    M,N = A.shape
    spdiags([1.0/A.diagonal()], [0], M, N)

    b = case.b
    x0 = 0*b

    precond = LinearOperator(A.shape, identity, rmatvec=identity)

    if solver is not gmres:
        x, info = solver(A, b, M=precond, x0=x0, tol=tol)
    else:
        x, info = solver(A, b, M=precond, x0=x0, tol=tol, version='fast')
    assert_equal(info, 0)
    assert_normclose(A.dot(x), b, tol)

    A = aslinearoperator(A)
    A.psolve = identity
    A.rpsolve = identity

    if solver is not gmres:
        x, info = solver(A, b, x0=x0, tol=tol)
    else:
        x, info = solver(A, b, x0=x0, tol=tol, version='fast')
    assert_equal(info, 0)
    assert_normclose(A@x, b, tol=tol)

    # To test for the fast version of GMRES with preconditioner
    if solver is gmres:
        x, info = solver(A, b, M=precond, x0=x0, tol=tol, version='fast')
        assert_equal(info,0)
        assert_normclose(A.dot(x), b, tol)


def test_precond_dummy():
    case = params.Poisson1D
    for solver in params.solvers:
        if solver in case.skip:
            continue
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            check_precond_dummy(solver, case)


def check_precond_inverse(solver, case):
    tol = 1e-8

    def inverse(b,which=None):
        """inverse preconditioner"""
        A = case.A
        if not isinstance(A, np.ndarray):
            A = A.toarray()
        return np.linalg.solve(A, b)

    def rinverse(b,which=None):
        """inverse preconditioner"""
        A = case.A
        if not isinstance(A, np.ndarray):
            A = A.toarray()
        return np.linalg.solve(A.T, b)

    matvec_count = [0]

    def matvec(b):
        matvec_count[0] += 1
        return case.A.dot(b)

    def rmatvec(b):
        matvec_count[0] += 1
        return case.A.T.dot(b)

    b = case.b
    x0 = 0*b

    A = LinearOperator(case.A.shape, matvec, rmatvec=rmatvec)
    precond = LinearOperator(case.A.shape, inverse, rmatvec=rinverse)

    # Solve with preconditioner
    matvec_count = [0]
    if solver is not gmres:
        x, info = solver(A, b, M=precond, x0=x0, tol=tol)
    else:
        # To test for the fast version of GMRES
        x, info = solver(A, b, M=precond, x0=x0, tol=tol, version='fast')

    assert_equal(info, 0)
    assert_normclose(case.A.dot(x), b, tol)

    # Solution should be nearly instant
    assert_(matvec_count[0] <= 3, repr(matvec_count))


def test_precond_inverse():
    case = params.Poisson1D
    for solver in params.solvers:
        if solver in case.skip:
            continue
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")


def test_gmres_basic():
    A = np.vander(np.arange(10) + 1)[:, ::-1]
    b = np.zeros(10)
    b[0] = 1
    np.linalg.solve(A, b)

    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, ".*called without specifying.*")
        x_gm, err = gmres(A, b, restart=5, maxiter=1, version='fast')

    assert_allclose(x_gm[0], 0.359, rtol=1e-2)


def test_reentrancy():
    reentrant = [tfqmr, gmres, cr]
    for solver in reentrant:
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            _check_reentrancy(solver)


def _check_reentrancy(solver):
    def matvec(x):
        A = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
        if solver is not gmres:
            y, info = solver(A, x)
        else:
            y, info = solver(A, x, version='fast')
        assert_equal(info, 0)
        return y
    b = np.array([1, 1./2, 1./3])
    op = LinearOperator((3, 3), matvec=matvec, rmatvec=matvec,
                        dtype=b.dtype)

    if solver is not gmres:
        y, info = solver(op, b)
    else:
        y, info = solver(op, b, version='fast')
    assert_equal(info, 0)
    assert_allclose(y, [1, 1, 1])


@pytest.mark.parametrize("solver", [gmres, tfqmr, cr])
def test_zero_rhs(solver):
    np.random.seed(1234)
    A = np.random.rand(10, 10)
    A = A.dot(A.T) + 10 * np.eye(10)

    b = np.zeros(10)
    tols = np.r_[np.logspace(np.log10(1e-10), np.log10(1e2), 7)]

    for tol in tols:
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            
            if solver is not gmres:
                x, info = solver(A, b, tol=tol)
            else:
                x, info = solver(A, b, tol=tol, version='fast')
            assert_equal(info, 0)
            assert_allclose(x, 0, atol=1e-15)

            if solver is not gmres:
                x, info = solver(A, b, tol=tol, x0=ones(10))
            else:
                x, info = solver(A, b, tol=tol, x0=ones(10), version='fast')
            assert_equal(info, 0)
            assert_allclose(x, 0, atol=tol)

            if solver is not gmres:
                x, info = solver(A, b, tol=tol, atol=0, x0=ones(10))
            else:
                x, info = solver(A, b, tol=tol, atol=0, x0=ones(10), version='fast')
            if info == 0:
                assert_allclose(x, 0)

            if solver is not gmres:
                x, info = solver(A, b, tol=tol, atol=tol)
            else:
                x, info = solver(A, b, tol=tol, atol=tol, version='fast')
            assert_equal(info, 0)
            assert_allclose(x, 0, atol=1e-300)

            if solver is not gmres:
                x, info = solver(A, b, tol=tol, atol=0)
            else:
                x, info = solver(A, b, tol=tol, atol=0, version='fast')
            assert_equal(info, 0)
            assert_allclose(x, 0, atol=1e-300)


@pytest.mark.parametrize("solver", [gmres, tfqmr, cr])
def test_x0_working(solver):
    # Easy problem
    np.random.seed(1)
    n = 10
    A = np.random.rand(n, n)
    A = A.dot(A.T)
    b = np.random.rand(n)
    x0 = np.random.rand(n)

    kw = dict(atol=0, tol=1e-6)

    if solver is not gmres:
        x, info = solver(A, b, **kw)
    else:
        x, info = solver(A, b, **kw, version='fast')
    assert_equal(info, 0)
    assert_(np.linalg.norm(A.dot(x) - b) <= 1e-6*np.linalg.norm(b))

    if solver is not gmres:
        x, info = solver(A, b, x0=x0, **kw)
    else:
        x, info = solver(A, b, x0=x0, **kw, version='fast')
    assert_equal(info, 0)
    assert_(np.linalg.norm(A.dot(x) - b) <= 1e-6*np.linalg.norm(b))


#========================================================================
#   To test the new feature written by Gang Zhao in SciPy version 1.8
#   Remark: Your SciPy version should be >= 1.8
#========================================================================
if VERSION >= "1.8.0":
    @pytest.mark.parametrize('solver', [gmres, tfqmr, cr])
    def test_x0_equals_Mb(solver):
        for case in params.cases:
            if solver in case.skip:
                continue
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning, ".*called without specifying.*")
                A = case.A
                b = case.b
                x0 = 'Mb'
                tol = 1e-8
                x, info = solver(A, b, x0=x0, tol=tol)

                assert_array_equal(x0, 'Mb')  # ensure that x0 is not overwritten
                assert_equal(info, 0)
                assert_normclose(A.dot(x), b, tol=tol)


#------------------------------------------------------------------------------

class TestGMRES:
    def test_abi(self):
        # Check we don't segfault on gmres with complex argument
        A = eye(2)
        b = ones(2)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            r_x, r_info = gmres(A, b, version='fast')
            r_x = r_x.astype(complex)

            x, info = gmres(A.astype(complex), b.astype(complex), version='fast')

        assert_(iscomplexobj(x))
        assert_allclose(r_x, x)
        assert_(r_info == info)

    def test_defective_precond_breakdown(self):
        # Breakdown due to defective preconditioner
        M = np.eye(3)
        M[2,2] = 0

        b = np.array([0, 1, 1])
        x = np.array([1, 0, 0])
        A = np.diag([2, 3, 4])

        x, info = gmres(A, b, x0=x, M=M, tol=1e-15, atol=0, version='fast')

        # Should not return nans, nor terminate with false success
        # The new version of GMRES takes the preconditioned norm as the
        # convergence criterion
        assert_(not np.isnan(x).any())
        if info == 0:
            assert_(np.linalg.norm(M.dot(A.dot(x) - b)) <= 1e-15*np.linalg.norm(b))

    def test_defective_matrix_breakdown(self):
        # Breakdown due to defective matrix
        A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        b = np.array([1, 0, 1])
        x, info = gmres(A, b, tol=1e-8, atol=0, version='fast')

        # Should not return nans, nor terminate with false success
        assert_(not np.isnan(x).any())
        if info == 0:
            assert_(np.linalg.norm(A.dot(x) - b) <= 1e-8*np.linalg.norm(b))

        # The solution should be OK outside null space of A
        assert_allclose(A.dot(A.dot(x)), A.dot(b))

    def test_callback_type(self):
        # The legacy callback type changes meaning of 'maxiter'
        np.random.seed(1)
        A = np.random.rand(20, 20)
        b = np.random.rand(20)

        cb_count = [0]

        def prnorm_cb(r):
            cb_count[0] += 1
            assert_(isinstance(r, float))

        def x_cb(x):
            cb_count[0] += 1
            assert_(isinstance(x, np.ndarray))

        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            # 2 iterations is not enough to solve the problem
            cb_count = [0]
            x, info = gmres(A, b, tol=1e-6, atol=0, callback=prnorm_cb, maxiter=2, restart=50, version='fast')
            assert info == 0
            assert cb_count[0] == 21

        # 2 restart cycles is enough to solve the problem
        cb_count = [0]
        x, info = gmres(A, b, tol=1e-6, atol=0, callback=prnorm_cb, maxiter=2, restart=50,
                        callback_type='prnorm', version='fast')
        assert info == 0
        assert cb_count[0] > 2

        # 2 restart cycles is enough to solve the problem
        cb_count = [0]
        x, info = gmres(A, b, tol=1e-6, atol=0, callback=x_cb, maxiter=2, restart=50,
                        callback_type='x', version='fast')
        assert info == 0
        assert cb_count[0] == 2

    def test_callback_x_monotonic(self):
        # Check that callback_type='x' gives monotonic norm decrease
        np.random.seed(1)
        A = np.random.rand(20, 20) + np.eye(20)
        b = np.random.rand(20)

        prev_r = [np.inf]
        count = [0]

        def x_cb(x):
            r = np.linalg.norm(A.dot(x) - b)
            assert r <= prev_r[0]
            prev_r[0] = r
            count[0] += 1

        x, info = gmres(A, b, tol=1e-6, atol=0, callback=x_cb, maxiter=20, restart=10,
                        callback_type='x', version='fast')
        assert info == 20
        assert count[0] == 21
        x_cb(x)
