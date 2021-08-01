import numpy as np
import operator


def dct_ii(N: int) -> np.ndarray:
    """
    Generate the 1D DCT transform matrix of size N.

    Parameters:
    N (int): Size of DCT matrix required

    Returns:
    C (2D np array): 1D DCT transform matrix

    Uses an orthogonal Type-II DCT.
    Y = C * X tranforms N-vector X into Y.
    """
    C = np.ones((N, N)) / np.sqrt(N)
    theta = (np.arange(N) + 0.5) * (np.pi/N)
    g = np.sqrt(2/N)
    for i in range(1, N):
        C[i, :] = g * np.cos(theta*i)

    return C


def dct_iv(N: int) -> np.ndarray:
    """
    Generate the 1D DCT transform matrix of size N.

    Parameters:
    N (int): Size of DCT matrix required

    Returns:
    C (2D np array): 1D DCT transform matrix

    Uses an orthogonal Type-IV DCT.
    Y = C * X tranforms N-vector X into Y.
    """
    C = np.ones((N, N)) / np.sqrt(N)
    theta = (np.arange(N) + 0.5) * (np.pi/N)
    g = np.sqrt(2/N)
    for i in range(N):
        C[i, :] = g * np.cos(theta*(i+0.5))

    return C


def colxfm(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Transforms the columns of X using the tranformation in C.

    Parameters:
        X: Image whose columns are to be transformed
        C: N-size 1D DCT coefficients obtained using dct_ii(N)
    Returns:
        Y: Image with transformed columns

    PS: The height of X must be a multiple of the size of C (N).
    """
    N = len(C)
    m, n = X.shape

    # catch mismatch in size of X
    if m % N != 0:
        raise ValueError('colxfm error: height of X not multiple of size of C')

    Y = np.zeros((m, n))
    # transform columns of each horizontal stripe of pixels, N*n
    for i in range(0, m, N):
        Y[i:i+N, :] = C @ X[i:i+N, :]

    return Y


def regroup(X, N):
    """
    Regroup the rows and columns in X.
    Rows/Columns that are N apart in X are adjacent in Y.

    Parameters:
    X (np.ndarray): Image to be regrouped
    N (list): Size of 1D DCT performed (could give int)

    Returns:
    Y (np.ndarray): Regoruped image
    """
    # if N is a 2-element list, N[0] is used for columns and N[1] for rows.
    # if a single value is given, a square matrix is assumed
    try:
        N_m = N_n = operator.index(N)
    except TypeError:
        N_m, N_n = N

    m, n = X.shape

    if m % N_m != 0 or n % N_n != 0:
        raise ValueError('regroup error: X dimensions not multiples of N')

    X = X.reshape(m // N_m, N_m, n // N_n, N_n)  # subdivide the axes
    X = X.transpose((1, 0, 3, 2))                # permute them
    return X.reshape(m, n)                       # and recombine


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from .familiarisation import load_mat_img
    from .familiarisation import prep_cmap_array_plt
    from .familiarisation import plot_image

    # code testing dct_iv
    print(dct_iv(8))
    # code for images
    '''
    N = 8
    C8 = dct_ii(N)
    # print(len(C8))
    # print(C8.shape)
    img = 'lighthouse.mat'
    img_info = 'X'
    cmap_info = {'map', 'map2'}
    X, cmaps_dict = load_mat_img(img, img_info, cmap_info)
    # print(X)
    X = X - 128
    # print(X)
    # Y = colxfm(X, C8)
    Y = colxfm(colxfm(X, C8).T, C8).T
    # plot_image(Y)

    cmap_array = cmaps_dict['map']
    cmap_plt = prep_cmap_array_plt(cmap_array, 'map')
    # plot_image(X, cmap_plt='gray')
    # plot_image(Y)
    print(regroup(Y, N)/N)
    # plot_image(regroup(Y, N)/N, cmap_plt)
    '''
    # code to check produced matrices are the same
    '''
    X = np.array([[1,2,3,4,5,6,7,8],
        [10,20,30,40,50,60,70,80],
        [3,6,9,13,15,17,22,32],
        [4,7,88,97,23,45,34,54],
        [1,2,3,4,5,6,7,8],
        [10,20,30,40,50,60,70,80],
        [3,6,9,13,15,17,22,32],
        [4,7,88,97,23,45,34,54]])
    #print(X)
    C4 = dct_ii(4)
    #print(C4)
    Y = colxfm(colxfm(X, C4).T, C4).T
    #print(Y)
    print(regroup(Y,4)/4)
    #plot_image(Y)
    '''
