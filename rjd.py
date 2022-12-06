import numpy as np
from rjd_util import *

def randomized_jd_deflat(AA, trails):
    thr = search_threshold(AA, trails)
    return randomized_jd_deflat_threshold(AA, thr , trails)

def randomized_jd(AA,trails=1):
    n = AA.shape[0]
    mu = np.random.normal(0,1,n)
    A_mu = np.einsum('ijk,i->jk',AA, mu)
    _, Q = np.linalg.eigh(A_mu)
    best_Q = Q
    
    if trails == 1:
        return best_Q

    D = best_Q.T @ AA @ best_Q
    max_diag = sum_diagonal_squared(D)
    
    for _ in range(trails-1):
        mu = np.random.normal(0,1,n)
        A_mu = np.einsum('ijk,i->jk',AA, mu)
        _, Q = np.linalg.eigh(A_mu)

        D = Q.T @ AA @ Q
        diag = sum_diagonal_squared(D)
        if diag > max_diag:
            max_diag = diag
            best_Q = Q
    return best_Q

def randomized_jd_complex(AA,trails=1):
    n = AA.shape[0]
    mu_r = np.random.normal(0,1,n)
    mu_c = np.random.normal(0,1,n)
    mu = mu_r + mu_c * 1j
    A_mu = np.einsum('ijk,i->jk',AA, mu)
    
    _, Q = np.linalg.eig(A_mu)
    Q,_ = np.linalg.qr(np.real(Q))
    best_Q = Q
    if trails == 1:
        return best_Q

    D = best_Q.T @ AA @ best_Q
    #print(np.conjugate(best_Q).T @ best_Q)
    max_diag = sum_diagonal_squared(D)
    
    for _ in range(trails-1):
        mu_r = np.random.normal(0,1,n)
        mu_c = np.random.normal(0,1,n)
        mu = mu_r + mu_c * 1j
        A_mu = np.einsum('ijk,i->jk',AA, mu)
        
        _, Q = np.linalg.eigh(A_mu)
        Q,_ = polar(np.real(Q))

        D = Q.T @ AA @ Q
        diag = sum_diagonal_squared(D)
        if diag > max_diag:
            max_diag = diag
            best_Q = Q
    return best_Q

def randomized_jd_deflat_complex(AA, trails):
    thr = search_threshold(AA, trails)
    return randomized_jd_deflat_threshold_complex(AA, thr , trails)