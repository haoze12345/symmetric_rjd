from time import time
import numpy as np
from other_jd_algorithms import *
from rjd import *


def random_error(AA, eps = 1e-5, norm_type = 2):
    n = AA.shape[0]
    p = AA.shape[1]
    result = []
    for A in AA:
        E = np.random.chisquare(1,(p,p))
        E = (E + E.T) / 2
        E = eps * (1/(np.linalg.norm(E,norm_type) * np.sqrt(n))) * E
        result.append(((A + E) + (A + E).T) / 2.0)
    return np.array(result)

def print_time_error(name, times,errors, bold = False):
    if not bold:
        output_str = name + " & " + "$\\num{%.3g}$" + " & " + "$\\num{%.2g}$"  + " & " \
                    + "$\\num{%.3g}$" + " & " + "$\\num{%.2g}$" + " & " \
                    + "$\\num{%.3g}$" + " & " + "$\\num{%.2g}$" + "\\\\\n" 
    else:
        output_str = "{\\bf " + name + "}" + " & " + "$\\num{%.2e}$" + " & " + "$\\num{%.2e}$"  + " & " \
                    + "$\\num{%.2e}$" + " & " + "$\\num{%.2e}$" + " & " \
                    + "$\\num{%.2e}$" + " & " + "$\\num{%.2e}$" + "\\\\\n"
    print(output_str % (times[0], errors[0], times[1], errors[1], times[2], errors[2]))

def offdiagonal_frobenius_square(A, by_column = False):
    """
    computes the frobenius norm of the off diagonal elements
    of the tensor A (k x m x m)
    Args:
        A: np.ndarray
            of shape k x m x m
    Returns:
        norm: np.ndarray
            the frobenius norm square of the offdiagonal of A
    """
    shape = A.shape
    identity_3d = np.zeros(shape)
    idx = np.arange(shape[1])
    identity_3d[:, idx, idx] = 1 
    mask = np.array(1 - identity_3d, dtype = np.int0)
    offdiag = A * mask
    if not by_column:
        loss = np.sum(np.power(offdiag,2))
        return loss
    else:
        col_loss = np.sum(np.sum(np.power(offdiag,2),axis=1),axis = 0)
        return col_loss

def pham_loss(AAs):
    n = AAs.shape[1]
    res = 0
    for AA in AAs:
        log_diagdet = np.log(np.linalg.det(np.diag(np.diag(AA))))
        log_det = np.log(np.linalg.det(AA))
        res += log_diagdet - log_det
    return res / (2*n)


def experiment_helper(input_arrays, repeats, with_error, error_levels, \
                      trails, d,n, norm_type):
    eps = np.finfo(float).eps
    n_l = len(error_levels)
    times_qndiag, times_rand, times_rand_de, times_pham, times_jade, times_ffdiag \
        = np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l)
    errors_qndiag, errors_rand, errors_rand_de, errors_pham, errors_jade, errors_ffdiag \
        = np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l)

    for i, error_level in enumerate(error_levels):
        test_array = random_error(input_arrays,error_level,norm_type)

        for _ in range(repeats):
            start = time()
            n = input_arrays.shape[2]
            B0 = np.eye(n)
            B, _ = qndiag(test_array, B0 = B0, ortho=True, tol = 1e-6, check_sympos = True)  # use the algorithm
            end = time()
            times_qndiag[i] += end - start
            errors_qndiag[i] += offdiagonal_frobenius_square(B @ test_array @ B.T)

            start = time()
            Q = randomized_jd(test_array,trails)
            end = time()
            times_rand[i] += end - start
            errors_rand[i] += offdiagonal_frobenius_square(Q.T @ test_array @ Q)
            
            start = time()
            Q = randomized_jd_deflat_complex(test_array,trails)
            end = time()
            times_rand_de[i] += end - start
            errors_rand_de[i] += offdiagonal_frobenius_square(Q.T @ test_array @ Q)

            start = time()
            V,_ = rjd(test_array)
            end = time()
            times_jade[i] += end - start
            errors_jade[i] += offdiagonal_frobenius_square(V.T @ test_array @ V)

            start = time()
            V,_ = ajd_pham(test_array,eps=1e-10)
            end = time()
            times_pham[i] += end - start
            errors_pham[i] += offdiagonal_frobenius_square(V @ test_array @ V.T)

            start = time()
            Q_ffdiag = ortho_ffdiag(test_array,eps = 1e-8)
            end = time()
            times_ffdiag[i] += end - start
            errors_ffdiag[i] += offdiagonal_frobenius_square(Q_ffdiag @ test_array @ Q_ffdiag.T)

    title_str = "\\begin{table}[!hbt!]\n" + "\\begin{center}\n" + \
                "\\caption{Runtime and Accuracy Comparison for " + \
                "$d={d}, n={n}$".format(d = d, n=n) +"}\n" +"\\begin{tabular}{||c|c|c|c|c|c|c||}\n" \
                + "\\hline\n"
    title_str += "Name & Time $\\epsilon_1$ & Error $\\epsilon_1$ & Time $\\epsilon_2$ & Error $\\epsilon_2$ &Time $\\epsilon_3$ &Error $\\epsilon_3$\\\\\n"\
         + "\\hline"
    print(title_str)
    print_time_error("JADE", 1000*times_jade / repeats, np.sqrt(errors_jade / repeats))
    print_time_error("FFDIAG", 1000*times_ffdiag / repeats, np.sqrt(errors_ffdiag / repeats))
    print_time_error("PHAM", 1000*times_pham / repeats, np.sqrt(errors_pham / repeats))
    print_time_error("QNDIAG", 1000*times_qndiag / repeats, np.sqrt(errors_qndiag / repeats))
    print_time_error("RJD", 1000*times_rand / repeats, np.sqrt(errors_rand / repeats))
    print_time_error("DRJD", 1000*times_rand_de / repeats, np.sqrt(errors_rand_de / repeats))
    print("\\hline")
    if not with_error:
        error_level = 0
        closing_str = "\\end{tabular}\n" + "\\end{center}\n" + \
        "\\end{table}"
    else:
        closing_str = "\\end{tabular}\n" + "\\end{center}\n" + \
        "\\end{table}"
    print(closing_str + '\n')

def MA_index(M):
    n = M.shape[0]
    s = 0
    for p in range(n):
        p_col = np.abs(M[:,p])
        p_row = np.abs(M[p,:])
        s+= np.sum(p_col)/ np.max(p_col)
        s+= np.sum(p_row)/ np.max(p_row)
        s-=2
    return s/(2*n*(n-1))

def experiment_helper_MA(input_arrays, repeats, with_error, error_levels, \
                      trails, d,n, mixing_matrix):
    eps = np.finfo(float).eps
    n_l = len(error_levels)
    times_qndiag, times_rand, times_rand_de, times_pham, times_jade, times_ffdiag \
        = np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l)
    errors_qndiag, errors_rand, errors_rand_de, errors_pham, errors_jade, errors_ffdiag \
        = np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l)

    for i, error_level in enumerate(error_levels):
        test_array = random_error(input_arrays,error_level)

        for _ in range(repeats):
            start = time()
            n = input_arrays.shape[2]
            B0 = np.eye(n)
            B, _ = qndiag(test_array, B0 = B0, ortho=True, tol = 1e-8)  # use the algorithm
            end = time()
            times_qndiag[i] += end - start
            errors_qndiag[i] += MA_index(B @ mixing_matrix)

            start = time()
            Q = randomized_jd(test_array,trails)
            end = time()
            times_rand[i] += end - start
            errors_rand[i] += MA_index(Q.T @ mixing_matrix)
            
            start = time()
            Q = randomized_jd_deflat(test_array,trails)
            end = time()
            times_rand_de[i] += end - start
            errors_rand_de[i] += MA_index(Q.T @ mixing_matrix)

            start = time()
            V,_ = rjd(test_array)
            end = time()
            times_jade[i] += end - start
            errors_jade[i] += MA_index(V.T @ mixing_matrix)

            start = time()
            Q_ffdiag = ortho_ffdiag(test_array,eps = 1e-8)
            end = time()
            times_ffdiag[i] += end - start
            errors_ffdiag[i] += MA_index(Q_ffdiag @ mixing_matrix)

    title_str = "\\begin{table}[!hbt!]\n" + "\\begin{center}\n" + \
                "\\caption{Runtime and Accuracy Comparison for " + \
                "$d={d}, n={n}$".format(d = d, n=n) +"}\n" +"\\begin{tabular}{||c|c|c|c|c|c|c||}\n" \
                + "\\hline\n"
    title_str += "Name & Time $\\epsilon_1$ & Error $\\epsilon_1$ & Time $\\epsilon_2$ & Error $\\epsilon_2$ &Time $\\epsilon_3$ &Error $\\epsilon_3$\\\\\n"\
         + "\\hline"
    print(title_str)
    print_time_error("QNDIAG", 1000*times_qndiag / repeats, errors_qndiag / repeats)
    print_time_error("FFDIAG", 1000*times_ffdiag / repeats, errors_ffdiag / repeats)
    print_time_error("JADE", 1000*times_jade / repeats, errors_jade / repeats)
    print_time_error("RJD", 1000*times_rand / repeats, errors_rand / repeats)
    print_time_error("DRJD", 1000*times_rand_de / repeats, errors_rand_de / repeats)
    print("\\hline")
    if not with_error:
        error_level = 0
        closing_str = "\\end{tabular}\n" + "\\end{center}\n" + \
        "\\end{table}"
    else:
        closing_str = "\\end{tabular}\n" + "\\end{center}\n" + \
        "\\end{table}"
    print(closing_str + '\n')

def experiment_helper_pham(input_arrays, repeats, with_error, error_levels, \
                      trails, d,n):
    eps = np.finfo(float).eps
    n_l = len(error_levels)
    times_qndiag, times_rand, times_rand_de, times_pham, times_jade, times_ffdiag \
        = np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l)
    errors_qndiag, errors_rand, errors_rand_de, errors_pham, errors_jade, errors_ffdiag \
        = np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l), np.zeros(n_l)

    for i, error_level in enumerate(error_levels):
        test_array = random_error(input_arrays,error_level)

        for _ in range(repeats):
            start = time()
            n = input_arrays.shape[2]
            B0 = np.eye(n)
            B, _ = qndiag(test_array, B0 = B0, ortho=True)  # use the algorithm
            end = time()
            times_qndiag[i] += end - start
            errors_qndiag[i] += pham_loss(B @ test_array @ B.T)

            start = time()
            Q = randomized_jd(test_array,trails)
            end = time()
            times_rand[i] += end - start
            errors_rand[i] += pham_loss(Q.T @ test_array @ Q)
            
            start = time()
            Q = randomized_jd_deflat(test_array,trails)
            end = time()
            times_rand_de[i] += end - start
            errors_rand_de[i] += pham_loss(Q.T @ test_array @ Q)

            start = time()
            V,_ = rjd(test_array)
            end = time()
            times_jade[i] += end - start
            errors_jade[i] += pham_loss(V.T @ test_array @ V)

            start = time()
            Q_ffdiag = ortho_ffdiag(test_array,eps = 1e-8)
            end = time()
            times_ffdiag[i] += end - start
            errors_ffdiag[i] += pham_loss(Q_ffdiag @ test_array @ Q_ffdiag.T)

    title_str = "\\begin{table}[!hbt!]\n" + "\\begin{center}\n" + \
                "\\caption{Runtime and Accuracy Comparison for " + \
                "$d={d}, n={n}$".format(d = d, n=n) +"}\n" +"\\begin{tabular}{||c|c|c|c|c|c|c||}\n" \
                + "\\hline\n"
    title_str += "Name & Time $\\epsilon_1$ & Error $\\epsilon_1$ & Time $\\epsilon_2$ & Error $\\epsilon_2$ &Time $\\epsilon_3$ &Error $\\epsilon_3$\\\\\n"\
         + "\\hline"
    print(title_str)
    print_time_error("QNDIAG", 1000*times_qndiag / repeats, np.sqrt(errors_qndiag / repeats))
    print_time_error("FFDIAG", 1000*times_ffdiag / repeats, np.sqrt(errors_ffdiag / repeats))
    print_time_error("JADE", 1000*times_jade / repeats, np.sqrt(errors_jade / repeats))
    print_time_error("RRJD", 1000*times_rand / repeats, np.sqrt(errors_rand / repeats))
    print_time_error("DRRJD", 1000*times_rand_de / repeats, np.sqrt(errors_rand_de / repeats))
    print("\\hline")
    if not with_error:
        error_level = 0
        closing_str = "\\end{tabular}\n" + "\\end{center}\n" + \
        "\\end{table}"
    else:
        closing_str = "\\end{tabular}\n" + "\\end{center}\n" + \
        "\\end{table}"
    print(closing_str + '\n')