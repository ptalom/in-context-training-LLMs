import torch 
import numpy as np
import math
import time
        



########################################################################################
########################################################################################
######### Singular value decompotision

"""
For a matrix M, SVD gives Σ, U and V such M = UΣV^T.
The SVD of a matrix M can produce singular vectors with flipped signs and still be mathematically correct because UΣV^T =(−U)Σ(−V^T).
To ensure consistency between our custom SVD implementation and the sklearn PCA, we enforce a consistent sign convention 
for the principal components : we ensure that the largest absolute value in each principal component has a positive sign. 
"""
def enforce_sign_convention_V(V):
    for i in range(V.shape[1]):
        if np.sign(V[np.abs(V[:, i]).argmax(), i]) < 0:
            V[:, i] = -V[:, i]
    return V

def enforce_sign_convention_UV(U, V):
    for i in range(V.shape[1]):
        if np.sign(V[np.abs(V[:, i]).argmax(), i]) < 0:
            V[:, i] = -V[:, i]
            U[:, i] = -U[:, i]
    return U, V

def SVD(M, type_svd="full", r=None, verbose=False):
    """
    Singular value decomposition of M ~ (m, n)
    Return : Lambda, U, V
    With M = U @ Lambda @ V^T
    - Full SVD : |U| = (m, m), |Lambda| = (m, n) and |V| = (n, n)
    - Thin SVD : |U| = (m, min(m, n)), |Lambda| = (min(m, n), min(m, n)) and |V| = (n, min(m, n))
    - Compact SVD : |U| = (m, rank(M)), |Lambda| = (rank(M), rank(M)) and |V| = (n, rank(M))
    - Truncated SVD : |U| = (m, r), |Lambda| = (r, r) and |V| = (n, r)
    """
    assert type_svd in ["full", "thin", "compact", "truncated"]
    
    ##### With this decomposition, |U| = (m, m), |Lambda| = (m, n) and |V| = (n, n)
    if type_svd=="full" :
        U, Lambda_, VT = torch.linalg.svd(M) # (m, m), (min(n,m),), (n, n)
        V = VT.T # (n, n)
        m, n = M.shape

        # Lambda = torch.diag(Lambda_) # (min(n,m), min(n,m))
        # if m > n: # min(n,m) = n, so we whant to move from (n, n) to (m, n) = (n+m-n, n)
        #     # add rows with value 0
        #     #Lambda = torch.concatenate((Lambda, torch.zeros((m-n, n))), axis=0) # (m, n)
        #     Lambda = torch.cat((Lambda, torch.zeros((m-n, n))), axis=0) # (m, n)
        # elif m < n: # min(n,m) = m, so we whant to move from (m, m) to (m, n) = (m, m+n-m)
        #     # add columns with value 0
        #     #Lambda = torch.concatenate((Lambda, torch.zeros((m, n-m))), axis=1) # (m, n)
        #     Lambda = torch.cat((Lambda, torch.zeros((m, n-m))), axis=1) # (m, n)
        
        Lambda = torch.zeros(m, n, dtype=U.dtype) # (m, n)
        Lambda[:min(m, n), :min(m, n)] = torch.diag(Lambda_) # (min(n,m), min(n,m))

    ##### With this decomposition, |U| = (m, min(m, n)), |Lambda| = (min(m, n), min(m, n)) and |V| = (n, min(m, n))
    elif type_svd=="thin" :
        #U, Lambda, VT = torch.linalg.svd(M, full_matrices=False)
        U, Lambda, V = torch.svd(M) # (m, min(m, n)), (min(m, n),), (n, min(m, n))
        VT = V.T # (min(m, n), n)
        Lambda = torch.diag(Lambda) # (min(m, n), min(m, n))

    ##### |U| = (m, r), |Lambda| = (r, r) and |V| = (n, r)
    elif type_svd in ["compact", "truncated"] : # =="compact":
        #U, Lambda, VT = torch.linalg.svd(M, full_matrices=False)
        U, Lambda, V = torch.svd(M) # (m, min(m, n)), (min(m, n),), (n, min(m, n))
        if type_svd=="truncated": assert r is not None
        else : 
            # rank(M) = number of non zero eigen values <= min(m, n)
            r = torch.linalg.matrix_rank(M).item()
            #r = (Lambda > 0).sum().item() # singular values are non-negative real numbers
        U = U[:,:r] # (m, r)
        Lambda = torch.diag(Lambda[:r]) # (r, r)
        V = V[:,:r] # (n, r)
        VT = V.T # (r, n)

    if verbose : 
        print(f"SVD Error : {torch.dist(U @ Lambda @ VT, M).item()}")

    return Lambda, U, V

"""
# from sklearn.decomposition import TruncatedSVD
# from scipy.sparse import csr_matrix
# X_dense = model_repr + 0
# X_dense[:, 2 * np.arange(50)] = 0
# X = csr_matrix(X_dense)
# svd = TruncatedSVD(n_components=K, n_iter=7, random_state=42)
# svd.fit(X)
# print(svd.explained_variance_ratio_)
# print(svd.explained_variance_ratio_.sum())
# print(svd.singular_values_)
"""

def np_SVD(M, type_svd="full", r=None, verbose=False):
    """
    Singular value decomposition of M ~ (m, n)
    Return : Lambda, U, V
    With M = U @ Lambda @ V^T : |U| = (m, m), |Lambda| = (m, n) and |V| = (n, n)
    """
    assert type_svd in ["full", "thin", "compact", "truncated"]

    U, Lambda_, VT = np.linalg.svd(M) # (m, m), (min(n,m),), (n, n)
    V = VT.T # (n, n)
    m, n = M.shape 
    K = min(m, n)
    Lambda = np.zeros((m, n), dtype=float) # (m, n)
    Lambda[:K, :K] = np.diag(Lambda_) # (min(n,m), min(n,m))
    
    #### With this decomposition, |U| = (m, m), |Lambda| = (m, n) and |V| = (n, n)
    if type_svd=="full":
        #return Lambda, U, V
        pass

    ##### With this decomposition, |U| = (m, min(m, n)), |Lambda| = (min(m, n), min(m, n)) and |V| = (n, min(m, n))
    elif type_svd=="thin" :
        Lambda, U, V = Lambda[:K, :K], U[:, :K], V[:, :K] # (min(m, n), min(m, n)), (m, min(m, n)), (n, min(m, n))

    ##### |U| = (m, r), |Lambda| = (r, r) and |V| = (n, r)
    elif type_svd in ["compact", "truncated"] : # =="compact":
        if type_svd=="truncated": assert r is not None
        else : 
            # rank(M) = number of non zero eigen values <= min(m, n)
            r = np.linalg.matrix_rank(M)
            #r = (Lambda > 0).sum().item() # singular values are non-negative real numbers
    
        U = U[:,:r] # (m, r)
        Lambda = Lambda[:r, :r] # (r, r)
        V = V[:,:r] # (n, r)

    if verbose : print(f"SVD Error : {np.linalg.norm(U @ Lambda @ V.T - M)}")

    return Lambda, U, V

def np_component(M, verbose=False):
    """
    Get the right singular vector and the square of the singular value of M ~ (m, n)
    Assume the svd M = U @ Lambda @ V^T : |U| = (m, m), |Lambda| = (m, n) and |V| = (n, n)
    Then M.T @ M = V @ Lambda^2 @ V^T ~ (n, n)
    If n <<<< m, this significantly reduce the cost of obtaining the right singular vectors
    """
    m, n = M.shape
    Gram = M.T @ M # V @ Lambda^2 @ V^T (n, n)
    try :
        R = np.linalg.eigh(Gram)
        Lambda_square, V = R.eigenvalues, R.eigenvectors # (n,), (n, n)
    except AttributeError: #'tuple' object has no attribute 'eigenvalues'
        Lambda_square, V = np.linalg.eig(Gram)

    Lambda_square = np.diag(Lambda_square) # (n, n)
    if verbose : print(f"EVD Error : {np.linalg.norm(V @ Lambda_square @ V.T - Gram)}")
    
    #Lambda = Lambda_square**0.5 # (n, n)
    if n < m :
        Lambda = np.zeros((m, n), dtype=float) # (m, n)
        Lambda[:n, :n] = Lambda_square**0.5 # (n, n)
    else :
        Lambda = Lambda_square[:m]**0.5 # (m, n)

    return Lambda, V # (m, n), (n, n)




if __name__ == "__main__":
    M = torch.tensor([
        [1, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 0.0],
    ]) # (3, 4), r=2
    #M=M.T # (4, 3), r=2

    # Lambda, U, V = SVD(M, type_svd="full", verbose=True)
    # Lambda, U, V = SVD(M, type_svd="thin", verbose=True)
    # Lambda, U, V = SVD(M, type_svd="compact", verbose=True)
    # Lambda, U, V = SVD(M, type_svd="truncated", r=1, verbose=True)

    M = M.numpy()
    # Lambda, U, V = np_SVD(M, type_svd="full", verbose=True)
    # Lambda, U, V = np_SVD(M, type_svd="thin", verbose=True)
    Lambda, U, V = np_SVD(M, type_svd="compact", verbose=True)
    # Lambda, U, V = np_SVD(M, type_svd="truncated", r=1, verbose=True)

    print(U.shape, Lambda.shape, V.shape)



# if __name__ == "__main__":

#     N, d = 2, 3 # 10**2, 10
#     #N, d = 10**3, 28*28*2 # (m, n)
#     #N, d = 10**4, 28*28*2 # (m, n)
#     print("(m, n) :", N, d)
#     H = np.random.normal(size=(N, d))    

#     start_time = time.time()
#     Lambda, U, V = np_SVD(H, verbose=True)
#     print(Lambda.shape, V.shape, U.shape)
#     print(f"Sklearn method execution time: {time.time() - start_time:.4f} seconds")
#     #U, V = enforce_sign_convention_UV(U, V)
#     print(Lambda)
#     print(V)

#     print("="*60)

#     start_time = time.time()
#     Lambda, V = np_component(H, verbose=True)
#     print(Lambda.shape, V.shape)
#     print(f"Custom method execution time: {time.time() - start_time:.4f} seconds")
#     #V = enforce_sign_convention_V(V)
#     print(Lambda)
#     print(V)