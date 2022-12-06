import numpy as np
from scipy.linalg import polar

def offdiag_frobenius_square_by_column(A):
    col_sum = np.sum(np.square(A),axis=(0,1))
    diags = np.diagonal(A, axis1=1, axis2=2)
    return col_sum - np.sum(np.square(diags),axis =0)

def sum_diagonal_squared(D):
    diagonals = np.abs(np.diagonal(D, axis1=1, axis2=2))
    loss = np.sum(np.square(diagonals))
    return loss

def search_threshold(AA,  trails):
    d,n,_ = AA.shape
    final_min =np.inf
    for i in range(trails):
        mu = np.random.normal(0,1,d)
        A_mu = np.einsum('ijk,i->jk',AA, mu)
        _, Q = np.linalg.eigh(A_mu)
        AA_new = Q.T @ AA @ Q
        error_cols = offdiag_frobenius_square_by_column(AA_new)
        final_min = min(final_min,np.min(error_cols))
    #print(n, 2 * final_min)
    return 2 * final_min

def randomized_jd_deflat_threshold(AA, threshold = 1e-5, max_trials = 3):
    d,n,_ = AA.shape
    if n == 1:
        return np.array([[1]])
    
    min_error = np.inf
    best_Q, final_AA, success_indices, error_sum, Q, min_indices = None, None, None, None, None, None
    best_num_cols = 0
    success = False

    for i in range(max_trials):
        mu = np.random.normal(0,1,d)
        A_mu = np.einsum('ijk,i->jk',AA, mu)
        _, Q = np.linalg.eigh(A_mu)
        AA_new = Q.T @ AA @ Q
        error_cols = offdiag_frobenius_square_by_column(AA_new)
        error_col_index = np.argmin(error_cols)
        error_col_index = [error_col_index]
        cur_min_error = np.sum(error_cols[error_col_index])
        cur_success_indices = error_cols <= threshold
        num_cols = np.sum(cur_success_indices)
        if cur_min_error < min_error:
            final_AA = AA_new
            min_indices = error_col_index
            best_Q = Q
            min_error = cur_min_error
        if num_cols > 0:
            best_num_cols = num_cols
            final_AA = AA_new
            best_Q = Q
            success = True
            success_indices = cur_success_indices
            break

    if np.sum(success_indices) == n:
        return best_Q
    if not success:
        success_indices = min_indices
    else:
        success_indices = [i for i in range(n) if success_indices[i]!=0 ]
    AA_deflated = np.delete(np.delete(final_AA, success_indices, axis=1),success_indices,axis=2)
    Q_deflated = randomized_jd_deflat_threshold(AA_deflated, threshold, max_trials)
    Q_suc = best_Q[:,success_indices]
    Q_left = np.delete(best_Q,success_indices,axis = 1)
    Q_left = Q_left @ Q_deflated
    return np.column_stack([Q_suc,Q_left])


def randomized_jd_deflat_threshold_complex(AA, threshold = 1e-5, max_trials = 3):
    d,n,_ = AA.shape
    if n == 1:
        return np.array([[1]])
    
    min_error = np.inf
    best_Q, final_AA, success_indices, error_sum, Q, min_indices = None, None, None, None, None, None
    best_num_cols = 0
    success = False

    for i in range(max_trials):
        mu_r = np.random.normal(0,1,d)
        mu_c = np.random.normal(0,1,d)
        mu = mu_r + mu_c * 1j;
        A_mu = np.einsum('ijk,i->jk',AA, mu)
        _, Q = np.linalg.eig(A_mu)
        Q,_ = polar(np.real(Q))
        AA_new = Q.T @ AA @ Q
        error_cols = offdiag_frobenius_square_by_column(AA_new)
        error_col_index = np.argmin(error_cols)
        cur_min_error = error_cols[error_col_index]
        cur_success_indices = error_cols <= threshold
        num_cols = np.sum(cur_success_indices)
        if cur_min_error < min_error:
            final_AA = AA_new
            min_indices = [error_col_index]
            best_Q = Q
            min_error = cur_min_error
        if num_cols > 0:
            best_num_cols = num_cols
            final_AA = AA_new
            best_Q = Q
            success = True
            success_indices = cur_success_indices
            break

    if np.sum(success_indices) == n:
        return best_Q
    if not success:
        success_indices = min_indices
    else:
        success_indices = [i for i in range(n) if success_indices[i]!=0 ]

    AA_deflated = np.delete(np.delete(final_AA, success_indices, axis=1),success_indices,axis=2)
    _,n_new,_ = AA_deflated.shape
    Q_deflated = randomized_jd_deflat_threshold(AA_deflated,threshold , max_trials)
    Q_suc = best_Q[:,success_indices]
    Q_left = np.delete(best_Q,success_indices,axis = 1)
    Q_left = Q_left @ Q_deflated
    return np.column_stack([Q_suc,Q_left])