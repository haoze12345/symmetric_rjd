from __future__ import print_function

from time import time
from numpy import diag, dot, eye, zeros
from numpy.linalg import norm
from scipy.linalg import expm
import numpy as np
from pyriemann.utils.ajd import ajd_pham, rjd

def ffdiag_update(R_tau,ortho):
    '''
    Adapted from https://github.com/archimonde1308/pybss/blob/master/bss.py
    Single update for the non-orthogonal FFDIAG algorithm.  Set ortho = True to
    do the proper update for the orthogonal version of the algorithm.
    '''
    Dk = {}
    Ek = {}
    dim = R_tau.shape[1]
    n_lags = R_tau.shape[0]
    for tau in range(n_lags):
        Dk[tau] = diag(diag(R_tau[tau]))
        Ek[tau] = R_tau[tau] - Dk[tau]
    W = zeros((dim,dim))
    if ortho is False:
        z = zeros(W.shape)
        y = zeros(W.shape)
        for i in range(0,dim):
            for j in range(0,dim):
                for tau in range(0,n_lags):
                    z[i,j] += Dk[tau][i,i]*Dk[tau][j,j]
                    y[i,j] += Dk[tau][j,j]*Ek[tau][i,j]
        # compute W
        for i in range(0,dim):
            for j in range(i+1,dim):
                W[i][j] = (z[i,j]*y[j,i]-z[i,i]*y[i,j])/(z[j,j]*z[i,i]-z[i,j]*z[i,j])
                W[j][i] = (z[i,j]*y[i,j]-z[j,j]*y[j,i])/(z[j,j]*z[i,i]-z[i,j]*z[i,j])
    else:
        num = zeros((dim,dim))
        den = zeros((dim,dim))
        for i in range(0,dim):
            for j in range(i+1,dim):
                for tau in range(0,n_lags):
                    num[i,j] +=  Ek[tau][i,j]*(Dk[tau][i,i] - Dk[tau][j,j])
                    den[i,j] += (Dk[tau][i,i]-Dk[tau][j,j])**2
                if i != j:
                    W[i,j] = num[i,j]/den[i,j]
                    # W must be skew-symmetric (W = -W^T)
                    W[j,i] = -W[i,j]
    return W

def ortho_ffdiag(X, eps = 1.0e-08, max_iter = 100):
    '''
    Adapted from https://github.com/archimonde1308/pybss/blob/master/bss.py
    Blind source separation using FFDIAG.  This version (like SOBI, AMUSE, etc.)
    finds an orthogonal mixing matrix.
    INPUT:
    ------
    X : array, required
        d x k x k: d matrices of size kxk to be jointly diagonalized
    max_lag : int, optional
        maximum lag (in samples) for covariance matrix calculation
    eps : double, optional
        convergence criterion for matrix updates
    max_iter : int, optional
        maximum number of iterations/updates
    '''
    dim = X.shape[1]
    n_lags = X.shape[0]
    W = zeros((dim,dim))
    V = eye(dim)
    C = X
    n_iter = 0
    theta = 0.9
    iter_eps = 1.0

    while iter_eps > eps and n_iter < max_iter:
        n_iter += 1
        Vn1 = V
        Q = expm(W)
        C = Q @ C @ Q.T
        #for tau in range(0,n_lags):
        #    C[tau] = multi_dot([eye(dim) + W,C[tau],(eye(dim)+W).T])
        W = ffdiag_update(C,True)
        if norm(W) > theta:
            W = (W*theta)/norm(W)
        # update V
        V = dot(expm(W),V)
        delta = np.linalg.norm(V - Vn1,'fro')**2
        #for i in range(0,dim):
        #    for j in range(0,dim):
        #        if i != j:
        #            delta += (V[i][j]-Vn1[i][j])**2
        iter_eps = (delta/(dim*(dim-1)))
    return V


def qndiag(C, B0=None, weights=None, max_iter=1000, tol=1e-6,
           lambda_min=1e-4, max_ls_tries=10, diag_only=False,
           return_B_list=False, check_sympos=True, verbose=False, ortho=False):
    """
    Take from https://github.com/pierreablin/qndiag/blob/master/qndiag/qndiag.py
    Joint diagonalization of matrices using the quasi-Newton method
    Parameters
    ----------
    C : array-like, shape (n_samples, n_features, n_features)
        Set of matrices to be jointly diagonalized. C[0] is the first matrix,
        etc...
    B0 : None | array-like, shape (n_features, n_features)
        Initial point for the algorithm. If None, a whitener is used.
    weights : None | array-like, shape (n_samples,)
        Weights for each matrix in the loss:
        L = sum(weights * KL(C, C')) / sum(weights).
        No weighting (weights = 1) by default.
    max_iter : int, optional
        Maximum number of iterations to perform.
    tol : float, optional
        A positive scalar giving the tolerance at which the
        algorithm is considered to have converged. The algorithm stops when
        `|gradient| < tol`.
    lambda_min : float, optional
        A positive regularization scalar. Each eigenvalue of the Hessian
        approximation below lambda_min is set to lambda_min.
    max_ls_tries : int, optional
        Maximum number of line-search tries to perform.
    diag_only : bool, optional
        If true, the line search is done by computing only the diagonals of the
        dataset. The dataset is then computed after the line search.
        Taking diag_only = True might be faster than diag_only=False
        when the matrices are large (n_features > 200)
    return_B_list : bool, optional
        Chooses whether or not to return the list of iterates.
    check_sympos : bool, optional
        Chooses whether to check that the provided dataset contains only
        symmetric positive definite matrices, as should be.
    verbose : bool, optional
        Prints informations about the state of the algorithm if True.
    ortho : bool, optional
        If true, performs joint diagonlization under orthogonal constrains.
    Returns
    -------
    D : array-like, shape (n_samples, n_features, n_features)
        Set of matrices jointly diagonalized
    B : array, shape (n_features, n_features)
        Estimated joint diagonalizer matrix.
    infos : dict
        Dictionnary of monitoring informations, containing the times,
        gradient norms and objective values.
    References
    ----------
    P. Ablin, J.F. Cardoso and A. Gramfort. Beyond Pham's algorithm
    for joint diagonalization. Proc. ESANN 2019.
    https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2019-119.pdf
    https://hal.archives-ouvertes.fr/hal-01936887v1
    https://arxiv.org/abs/1811.11433
    """
    t0 = time()
    if not isinstance(C, np.ndarray):
        raise TypeError('Input tensor C should be a numpy array.')
    if C.ndim != 3:
        raise ValueError(
            f'Input tensor C should have 3 dimensions (Got {C.ndim})'
        )
    if C.shape[1] != C.shape[2]:
        raise ValueError('The last two dimensions of C should be the same')
    if check_sympos:
        if not np.allclose(C, C.swapaxes(-1, -2)):
            raise ValueError('C does not contain only symmetric matrices')
        try:  # Check positivity using Cholesky
            np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            raise ValueError('C contains some non positive matrices')

    n_samples, n_features, _ = C.shape
    if B0 is None:
        C_mean = np.mean(C, axis=0)
        d, p = np.linalg.eigh(C_mean)
        B = p.T / np.sqrt(d[:, None])
    else:
        B = B0
    if weights is not None:  # normalize
        weights_ = weights / np.mean(weights)
    else:
        weights_ = None

    Bu, _, Bv = np.linalg.svd(B, full_matrices=False)
    B = Bu.dot(Bv)
    D = transform_set(B, C)
    current_loss = None

    # Monitoring
    if return_B_list:
        B_list = []
    t_list = []
    gradient_list = []
    loss_list = []
    if verbose:
        print('Running quasi-Newton for joint diagonalization')
        print(' | '.join([name.center(8) for name in
                         ["iter", "obj", "gradient"]]))

    for t in range(max_iter):
        if return_B_list:
            B_list.append(B.copy())
        t_list.append(time() - t0)
        diagonals = np.diagonal(D, axis1=1, axis2=2)
        # Gradient
        G = np.average(D / diagonals[:, :, None], weights=weights_,
                       axis=0) - np.eye(n_features)
        if ortho:
            G = G - G.T
        g_norm = np.linalg.norm(G)
        if g_norm < tol * np.sqrt(n_features):  # rescale by identity
            break


        # Hessian coefficients
        h = np.average(
            diagonals[:, None, :] / diagonals[:, :, None],
            weights=weights_, axis=0)
        if ortho:
            det = h + h.T - 2
            det[det < lambda_min] = lambda_min  # Regularize
            direction = -G / det
        else:
            # Quasi-Newton's direction
            det = h * h.T - 1.
            det[det < lambda_min] = lambda_min  # Regularize
            direction = -(G * h.T - G.T) / det

        # Line search
        success, new_D, new_B, new_loss, direction =\
            _linesearch(D, B, direction, current_loss, max_ls_tries, diag_only,
                        weights_, ortho=ortho)
        D = new_D
        B = new_B
        current_loss = new_loss

        # Monitoring
        gradient_list.append(g_norm)
        loss_list.append(current_loss)
        if verbose:
            print(' | '.join([("%d" % (t + 1)).rjust(8),
                              ("%.2e" % current_loss).rjust(8),
                              ("%.2e" % g_norm).rjust(8)]))
    infos = {'t_list': t_list, 'gradient_list': gradient_list,
             'loss_list': loss_list}
    if return_B_list:
        infos['B_list'] = B_list
    return B, infos


def transform_set(M, D, diag_only=False):
    """Transform a set of matrices
    Returns matrices D' such that
    D'[i] = M x D[i] x M.T
    Parameters
    ----------
    M : array-like, shape (n_features, n_features)
        The transform matrix
    D : array-like, shape (n_samples, n_features, n_features)
        The set of covariance matrices
    diag_only : bool, optional
        Whether to return the diagonal of the dataset only
    Returns
    -------
    op : array-like
        Array of shape (n_samples, n_features, n_features)
        if diag_only is False, else (n_samples, n_features)
        The transformed set of covariances
    """
    n, p, _ = D.shape
    if not diag_only:
        op = np.zeros((n, p, p))
        for i, d in enumerate(D):
            op[i] = M.dot(d.dot(M.T))
    else:
        op = np.zeros((n, p))
        for i, d in enumerate(D):
            op[i] = np.sum(M * d.dot(M.T), axis=0)
    return op


def loss(B, D, is_diag=False, weights=None):
    n, p = D.shape[:2]
    if not is_diag:
        diagonals = np.diagonal(D, axis1=1, axis2=2)
    else:
        diagonals = D
    logdet = -np.linalg.slogdet(B)[1]
    if weights is None:
        return logdet + 0.5 * np.sum(np.log(diagonals)) / n
    else:
        return logdet + 0.5 * np.sum(weights[:, None] * np.log(diagonals)) / n


def gradient(D, weights=None):
    n, p, _ = D.shape
    diagonals = np.diagonal(D, axis1=1, axis2=2)
    grad = np.average(D / diagonals[:, :, None], weights=weights, axis=0)
    grad.flat[::p + 1] -= 1  # equivalent to - np.eye(p)
    return grad


def _linesearch(D, B, direction, current_loss, n_ls_tries, diag_only,
                weights, ortho):
    n, p, _ = D.shape
    step = 1.
    if current_loss is None:
        current_loss = loss(B, D)
    for n in range(n_ls_tries):
        if ortho:
            M = expm(step * direction)
        else:
            M = np.eye(p) + step * direction

        new_D = transform_set(M, D, diag_only=diag_only)
        new_B = np.dot(M, B)
        new_loss = loss(new_B, new_D, diag_only, weights)
        if new_loss < current_loss:
            success = True
            break
        step /= 2.
    else:
        success = False
    # Compute new value of D if only its diagonal was computed
    if diag_only:
        new_D = transform_set(M, D, diag_only=False)
    return success, new_D, new_B, new_loss, step * direction