from rjd_util import *

def randomized_jd_deflat(AA, trails):
    thr = search_threshold(AA, trails)
    return randomized_jd_deflat_threshold(AA, thr , trails)

def randomized_jd(AA,trails=1):
    n = AA.shape[0]
    Q = diagonalize_random_combination(AA, n)
    best_Q = Q
    
    if trails == 1:
        return best_Q

    D = best_Q.T @ AA @ best_Q
    max_diag = sum_diagonal_squared(D)
    
    for _ in range(trails-1):
        Q = diagonalize_random_combination(AA, n)
        D = Q.T @ AA @ Q
        diag = sum_diagonal_squared(D)
        if diag > max_diag:
            max_diag = diag
            best_Q = Q
    return best_Q