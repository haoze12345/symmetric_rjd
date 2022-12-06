from experiments_util import *

def overall_experiments(repeats = 100, trails = 3, MA= False, norm_type = 2):
    settings = [(10,10), (10,100),  (30,30)]
    error_levels = [0, 1e-5, 1e-1]
    #settings = [(10,100)]
    #error_levels = [1e-5, 1e-5,1e-5]
    for n,p in settings:
        diagonals =np.random.uniform(size=(n, p)) + 0.01
        A = np.random.randn(p, p)
        Q, R = np.linalg.qr(A)# mixing matrix
        C = np.array([Q.dot(d[:, None] * Q.T) for d in diagonals])  # dataset
        if not MA:
            experiment_helper(C, repeats, trails= trails, error_levels=error_levels,\
                            with_error = False, d = n, n = p, norm_type = norm_type)
        else:
            experiment_helper_MA(C, repeats, trails= trails, error_levels=error_levels,\
                            with_error = False, d = n, n = p,mixing_matrix=Q)
overall_experiments(1,3, False, 'fro')