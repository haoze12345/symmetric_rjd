import sys 
sys.path.append('..')
import sktensor as skt
import numpy as np
from rjd import *

#Most of the code of this file is taken from https://github.com/mruffini/SpectralMethod
#

# Joint Diagonalization for Learning Latent Variable Models
def learn_LVM_RJD(M1, M2, M3, Whiten, k, N=0, deflat = True):
    """
    @param M2,M3: the symmetric moments
    @param k: the number of latent states
    @param L: the number of etas
    @param N: the number of trails of random jd
    """
    n, col = M2.shape

    W = Whiten
    H = M3

    #for r in range(0, n):
    #    H[r, :, :] = np.dot(np.dot(W.T,M3[:,:,r]),W)

    if N == 0:
        N = k * n
    Q = randomized_jd_deflat(H, N)

    M = np.linalg.pinv(W.T) @ Q
    M = M / np.sum(M, 0)

    x = np.linalg.lstsq(M, M1)
    omega = x[0]
    omega = omega / sum(omega)

    return M, omega
# Joint Diagonalization for Learning Latent Variable Models


# Tensor Decompositions for Learning Latent Variable Models
def learn_LVM_Tensor14(M2, M3, Whiten, k, L=25, N=20):
    """
    Theorem 4.3 from  "Tensor Decompositions for Learning Latent Variable Models"
    @param M2,M3: the symmetric moments
    @param k: the number of latent states
    @param L,N: number of iterations
    """
    d, col, _ = M3.shape
    W = Whiten
    I = np.eye(col)

    T = skt.dtensor(M3).ttm([W.T, I, I])

    Thetas = []
    Lambdas = []

    wT = T.copy()
    for i in range(k):
        theta, Lambda, wT = RobustTPM(wT.copy(),k,L,N)
        Thetas.append(theta)
        Lambdas.append(Lambda)

    Thetas = np.array(Thetas)
    Lambdas = np.array(Lambdas)

    B = np.linalg.pinv(W.T)
    pi = 1 / Lambdas ** 2

    M = np.zeros((d, k))

    for j in range(k):
        M[:, j] = Lambdas[j] * B.dot(Thetas[j, :].reshape(k,1)).reshape(d)
    M = M/np.sum(M,0)

    omega = pi
    omega = omega / sum(omega)

    return M, omega


def RobustTPM(T,k,L=25, N=20):
    """
    Algorithm 1 from   "Tensor Decompositions for Learning Latent Variable Models"
    @param T: symmetric tensor
    @param k: the number of latent states
    @param L,N: number of iterations
    """
    Thetas = []

    for tau in range(L):
        Theta = np.random.randn(1,k)
        Theta = Theta / np.linalg.norm(Theta)

        for t in range(N):
            Theta = T.ttm([Theta, Theta], [1, 2]).reshape(1, k)
            Theta = Theta / np.linalg.norm(Theta)

        Thetas.append(Theta)

    ThetaFinal_idx = np.argmax([T.ttm([theta, theta, theta], [0, 1, 2]) for theta in Thetas])

    Theta = Thetas[ThetaFinal_idx]

    for t in range(N):
        Theta = T.ttm([Theta, Theta], [1, 2]).reshape(1, k)
        Theta = Theta / np.linalg.norm(Theta)

    Lambda = T.ttm([Theta, Theta, Theta], [0, 1, 2]).squeeze()

    return Theta, Lambda, T - skt.ktensor([Theta.T, Theta.T, Theta.T]).totensor() * Lambda

def learn_LVM_SVTD(M1, M2, M3, Whiten, k):

    """
    Implementation of Algorithm 1 to learn the Sinlge topic model from a sample.
    Returns:
    the topic-word probability matrix M, with n rows an k columns;
        at position (i,j) we have the probability of the word i under topic k.
    the topic probability array omega, with k entries.
        at position (i) we have the probability of drawing topic i.
    @params M1, M2, M3: to be used to learn the Single Topic Model,
        from in theorem 2.1 (retrieved from RetrieveTensorsST)
    """
    n, col = M2.shape
    #Step 1
    #u,s,v = np.linalg.svd(M2)
    #Step 2
    #E = u[:, 0:k].dot((np.diag(np.sqrt(s[0:k]))))
    E = Whiten
    pE = np.linalg.pinv(E)

    HMin = 0
    H = np.zeros([k,k,n])
    M = np.zeros([n,k])

    #We select the feature with the most different singular vectors
    for r in range(0,n):
        H[:,:,r] =  M3[r,:,:]
        t = H[:,:,r]
        Or,s,v = np.linalg.svd(t)
        if np.min(-np.diff(s))>HMin:
            HMin = np.min(-np.diff(s))
            # Step 4
            O = Or

    #Step 5
    for r in range(0, n):
        # Step 7
        mur = np.diag(np.transpose(O).dot(H[:, :, r]).dot(O))
        M[r,:] = mur

    M = M/np.sum(M,0)

    #Step 9
    x = np.linalg.lstsq(M, M1)
    omega = x[0]
    omega = omega / sum(omega)

    return M, omega

def RetrieveTensorsST(X, k):
    """
    Returns a the three tensors M1, M2 and M3 to be used
    to learn the Single Topic Model, as in theorem 2.1
    @param X: a bag-of-words documents distributed
        as a Single Topic Model, with N rows an n columns;
        at position (i,j) we have the number of times the word j appeared in doc. i,
    """

    (N, n) = np.shape(X)

    M1 = np.sum(X,0)/np.sum(X)

    W = X - 1
    W[W < 0] = 0
    W2 = X - 2
    W2[W2 < 0] = 0
    #print(W.shape)
    #print(X.shape)
    Num = X * W
    Den = np.sum(X, 1)
    wDen = Den - 1
    wDen[wDen < 0] = 0
    wwDen = Den - 2
    wwDen[wwDen < 0] = 0

    Den1 = sum(Den * wDen)
    Den2 = sum(Den * wDen * wwDen)

    Diag = np.sum(Num, 0) / Den1

    M2 = (X.T @ X) / Den1
    M2[range(n), range(n)] = Diag
    u, s, _ = np.linalg.svd(M2)
    u = u[:, :k]
    s = s[:k]
    Whiten = u.dot(np.diag(1 / np.sqrt(s)))

    M3 = np.zeros((n, k, k))
    for j in range(n):
        Y = X[:, j].reshape((N, 1))
        Num = X * Y * W
        Diag = np.sum(Num, 0) / Den2
        wM3 = (Y * X).T.dot(X) / Den2
        wM3[range(n), range(n)] = Diag
        rr = np.sum(Y * W[:, j].reshape((N, 1)) * X,0) / Den2
        wM3[j, :] = rr
        wM3[:, j] = rr
        wM3[j, j] = np.sum(Y * W[:, j].reshape((N, 1)) * W2[:, j].reshape((N, 1))) / Den2
        M3[j] = Whiten.T @ wM3 @ Whiten

    return M1, M2, M3, Whiten

# Coherence score calculation
def coherence(X, M, l=20):
    k = M.shape[1]
    coherences = []
    sorted_indices = np.argsort(M,0)
    sorted_indices = np.flip(sorted_indices, axis=0)
    sorted_indices = sorted_indices[:l,:]

    for n in range(k):
        coherence = 0
        for j in range(1,l):
            d_i = 0
            d_ij = 0
            for i in range(j):
                w_i = sorted_indices[i,n]
                w_j = sorted_indices[j,n]
                for doc in X:
                    if doc[w_i] != 0:
                        d_i+=1
                        if doc[w_j] != 0:
                            d_ij += 1
                coherence += np.log((d_ij + 1) / d_i)
        coherences.append(coherence)
    return np.mean(np.array(coherences))


def print_top_words_table(M, omega, num_words, num_topics, id2word):
    """
    Print the table of top words according to probability for each topic
    @param M: The probabiliy matrix of each topic, M_ij is the prob of word i for topic j
    @param omega: The probability vector for the topics
    @num_words: The number of top words we want to show for each topic
    @num_topics: Number of topics we want to print in one row
    @id2word: The dictionary of id<>word
    """
    topic_order_desc = np.flip(np.argsort(omega))
    M_sorted = M[:,topic_order_desc]
    #M_sorted = M
    sorted_indices = np.argsort(M_sorted,0)
    sorted_indices = sorted_indices[:,:num_topics]
    results = np.empty([num_words,num_topics], dtype='object')
    for i,ids in enumerate(sorted_indices.T):
        ids = np.flip(ids)
        for j,id in enumerate(ids[:num_words]):
            results[j,i] = id2word[id] 
    print("\\begin{center}" +'\n')
    for k in range(num_topics):
        print("\\begin{tabular}{|c|c|c|c|c|c|}" +"\n" + "\\hline")
        for i in range(num_words):
            output_str = " & ".join(list(results[i,k*10: (k+1)*10])) + "\\\\"
            print(output_str)
        print("\\hline" + "\n" + "\\end{tabular}" + "\n")
    print("\\end{center}")
    print(results)