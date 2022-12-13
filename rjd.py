from rjd_util import *

def randomized_jd(AA,trails=1):
    n = AA.shape[0]
    max_diag = 0
    best_Q = None
    for _ in range(trails):
        Q = diagonalize_random_combination(AA, n)
        D = Q.T @ AA @ Q
        diag = sum_diagonal_squared(D)
        if diag > max_diag:
            max_diag = diag
            best_Q = Q
    return best_Q

def randomized_jd_deflat(AA, scale = 2, trails=1, threshold = 0):
    d,n,_ = AA.shape
    if n == 1:
        return np.array([[1]])

    Q_list = []
    AA_new_list = []
    error_cols_list = []
    best_i, success_indices, best_num_cols = 0, None, 0

    final_min =np.inf
    if threshold==0:
        for i in range(trails):
            Q = diagonalize_random_combination(AA,d)
            AA_new = Q.T @ AA @ Q
            error_cols = offdiag_frobenius_square_by_column(AA_new)
            final_min = min(final_min,np.min(error_cols))
            AA_new_list.append(AA_new)
            error_cols_list.append(error_cols)
            Q_list.append(Q)

    else:
        for i in range(trails):
            Q = diagonalize_random_combination(AA,d)
            AA_new = Q.T @ AA @ Q
            error_cols = offdiag_frobenius_square_by_column(AA_new)
            final_min = min(final_min,np.min(error_cols))
            cur_success_indices = error_cols <= threshold
            num_cols = np.sum(cur_success_indices)
            if num_cols>=2:
                if num_cols == n:
                    return Q
                success_indices = [i for i in range(n) if cur_success_indices[i]!=0 ]
                AA_deflated = np.delete(np.delete(AA_new, success_indices, axis=1),success_indices,axis=2)
                Q_suc =  Q[:,success_indices]
                Q_left = np.delete(Q,success_indices,axis = 1)
                return np.column_stack([Q_suc,Q_left @ randomized_jd_deflat(AA_deflated, scale, trails, threshold)])
            AA_new_list.append(AA_new)
            error_cols_list.append(error_cols)
            Q_list.append(Q)

    threshold = scale * final_min
    for i in range(len(AA_new_list)):
        cur_success_indices = error_cols_list[i] <= threshold
        num_cols = np.sum(cur_success_indices)
        if num_cols > best_num_cols:
            best_i = i
            best_num_cols = num_cols
            success_indices = cur_success_indices
    best_Q = Q_list[best_i]
    if best_num_cols == n:
        return best_Q
    success_indices = [i for i in range(n) if success_indices[i]!=0 ]
    AA_deflated = np.delete(np.delete(AA_new_list[best_i], success_indices, axis=1),success_indices,axis=2)
    Q_suc =  best_Q[:,success_indices]
    Q_left = np.delete(best_Q,success_indices,axis = 1)
    return np.column_stack([Q_suc,Q_left @ randomized_jd_deflat(AA_deflated, scale, trails, threshold)])


