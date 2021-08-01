"""
With thanks to 2019 SF2 Group 7 (Jason Li - jl944@cam.ac.uk, Karthik Suresh -
ks800@cam.ac.uk), who did the bulk of the porting from matlab to Python.
"""
import warnings
from typing import Tuple, Optional
import numpy as np
from .laplacian_pyramid import quant1, quant2
from .dct import dct_ii, colxfm, regroup
from .bitword import bitword
import warnings
import numpy as np
from .laplacian_pyramid import quant1, quant2
from .dct import dct_ii, colxfm, regroup
from .lbt import pot_ii
from cued_sf2_lab.dwt import idwt
from cued_sf2_lab.dwt import dwt


def nlevdwt(X, n):
    # your code here
    m = X.shape[0]
    Q = X.copy()
    for i in range(n):
        Q[:m,:m]=dwt(Q[:m,:m])
        m=m//2
    return Q

def nlevidwt(Y, n):
    m = Y.shape[0]//(2**(n-1))
    Q = Y.copy()
    for i in range(n):
        Q[:m,:m] = idwt(Q[:m,:m])
        m*=2
    return Q

def quantdwt(Y, dwtstep,rise_ratio = None):
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    n = dwtstep.shape[1]-1
    m = Y.shape[0]
    dwtent = np.zeros((3, n+1));
    Q = Y.copy()
    if rise_ratio is None:
        rise_ratio = 0.5*np.ones(dwtstep.shape)
#     print(rise_ratio)
    for i in range(n):
        m=m//2
        Q[:m,m:2*m] = quantise(Y[:m,m:2*m], dwtstep[0,i],rise_ratio[0,i]*dwtstep[0,i])
        dwtent[0,i]=bpp(Q[:m,m:2*m])
        
        Q[m:2*m,:m] = quantise(Y[m:2*m,:m], dwtstep[1,i],rise_ratio[1,i]*dwtstep[1,i])
        dwtent[1,i]=bpp(Q[m:2*m,:m])
        
        Q[m:2*m,m:2*m] = quantise(Y[m:2*m,m:2*m], dwtstep[2,i],rise_ratio[2,i]*dwtstep[2,i])
        dwtent[2,i]=bpp(Q[m:2*m,m:2*m])
    Q[:m,:m] = quantise(Y[:m,:m], dwtstep[0,n],rise_ratio[0,n]*dwtstep[0,n])
    dwtent[0,n] = bpp(Q[:m,:m])
    return Q, dwtent

def quantdwt1(Y, dwtstep,rise_ratio = None):
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    n = dwtstep.shape[1]-1
    m = Y.shape[0]
    dwtent = np.zeros((3, n+1));
    Q = Y.copy()
    if rise_ratio is None:
        rise_ratio = 0.5*np.ones(dwtstep.shape)
#     print(rise_ratio)
    for i in range(n):
        m=m//2
        Q[:m,m:2*m] = quant1(Y[:m,m:2*m], dwtstep[0,i],rise_ratio[0,i]*dwtstep[0,i])
#         dwtent[0,i]=bpp(Q[:m,m:2*m])
        
        Q[m:2*m,:m] = quant1(Y[m:2*m,:m], dwtstep[1,i],rise_ratio[1,i]*dwtstep[1,i])
#         dwtent[1,i]=bpp(Q[m:2*m,:m])
        
        Q[m:2*m,m:2*m] = quant1(Y[m:2*m,m:2*m], dwtstep[2,i],rise_ratio[2,i]*dwtstep[2,i])
#         dwtent[2,i]=bpp(Q[m:2*m,m:2*m])
    Q[:m,:m] = quant1(Y[:m,:m], dwtstep[0,n],rise_ratio[0,n]*dwtstep[0,n])
#     dwtent[0,n] = bpp(Q[:m,:m])
    return Q

def quantdwt2(Y, dwtstep,rise_ratio = None):
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    n = dwtstep.shape[1]-1
    m = Y.shape[0]
    dwtent = np.zeros((3, n+1));
    Q = Y.copy()
    if rise_ratio is None:
        rise_ratio = 0.5*np.ones(dwtstep.shape)
#     print(rise_ratio)
    for i in range(n):
        m=m//2
        Q[:m,m:2*m] = quant2(Y[:m,m:2*m], dwtstep[0,i],rise_ratio[0,i]*dwtstep[0,i])
#         dwtent[0,i]=bpp(Q[:m,m:2*m])
        
        Q[m:2*m,:m] = quant2(Y[m:2*m,:m], dwtstep[1,i],rise_ratio[1,i]*dwtstep[1,i])
#         dwtent[1,i]=bpp(Q[m:2*m,:m])
        
        Q[m:2*m,m:2*m] = quant2(Y[m:2*m,m:2*m], dwtstep[2,i],rise_ratio[2,i]*dwtstep[2,i])
#         dwtent[2,i]=bpp(Q[m:2*m,m:2*m])
    Q[:m,:m] = quant2(Y[:m,:m], dwtstep[0,n],rise_ratio[0,n]*dwtstep[0,n])
#     dwtent[0,n] = bpp(Q[:m,:m])
    return Q

def get_quantisation_step_ratio(n):
    ''' Returns the quantisation ratios between successive images in the pyramid needed to 
        ensure equal MSE contribution'''
    test_image = np.zeros((256,256))
    Y = nlevdwt(test_image,n)
    energies = np.zeros((3,n+1))
    dwtstep = np.zeros((3,n+1))
    m,x = Y.shape[0],Y.shape[1]
    for i in range(n):
        Y = nlevdwt(test_image,n)
        # top right -> k = 0
#         print(np.nonzero(Y))
        Y[:m//2,x//2:x][Y[:m//2,x//2:x].shape[0]//2-1,Y[:m//2,x//2:x].shape[1]//2-1] = 100
        
        Z = nlevidwt(Y,n)
        energy_tr = np.sum(Z**2)
        energies[0,i] = energy_tr
        Y[:m//2,x//2:x][Y[:m//2,x//2:x].shape[0]//2-1,Y[:m//2,x//2:x].shape[1]//2-1] = 0

        
        # bottom right -> k = 1
        Y = nlevdwt(test_image,n)
#         Ybr = Y[m//2:m,x//2:x]
#         Ybr[Ybr.shape[0]//2-1,Ybr.shape[1]//2-1] = 100
#         Y[m//2:m,x//2:x] = Ybr
        
        Y[m//2:m,x//2:x][Y[m//2:m,x//2:x].shape[0]//2-1,Y[m//2:m,x//2:x].shape[1]//2-1] = 100
        
        Z = nlevidwt(Y,n)
        energy_br = np.sum(Z**2)
        energies[1,i] = energy_br
#         Ybr[Ybr.shape[0]//2-1,Ybr.shape[1]//2-1] = 0
#         Y[m//2:m,x//2:x] = Ybr
        Y[m//2:m,x//2:x][Y[m//2:m,x//2:x].shape[0]//2-1,Y[m//2:m,x//2:x].shape[1]//2-1] = 0
        
        
        # bottom left -> k = 2
        Y = nlevdwt(test_image,n)
        
        Y[m//2:m,:x//2][Y[m//2:m,:x//2].shape[0]//2-1,Y[m//2:m,:x//2].shape[1]//2-1] = 100
        
#         Ybr = Y[m//2:m,:x//2]
#         Y[m//2:m,:x//2]
#         Ybr[Ybr.shape[0]//2-1,Ybr.shape[1]//2-1] = 100
#         Y[m//2:m,:x//2] = Ybr
        Z = nlevidwt(Y,n)
        energy_br = np.sum(Z**2)
        energies[2,i] = energy_br
#         Ybr[Ybr.shape[0]//2-1,Ybr.shape[1]//2-1] = 0
#         Y[m//2:m,:x//2] = Ybr
        Y[m//2:m,:x//2][Y[m//2:m,:x//2].shape[0]//2-1,Y[m//2:m,:x//2].shape[1]//2-1] = 0
        
        m //= 2
        x //= 2
        
    Y = nlevdwt(test_image,n)
    Y[:m,:x][Y[:m,:x].shape[0]//2-1,Y[:m,:x].shape[1]//2-1] = 100
    Z = nlevidwt(Y,n)
    energy_tr = np.sum(Z**2)
    energies[0,n] = energy_tr
    ratios = []
    ratios = np.sqrt(energies[0,0]/energies)
    Y[:m,:x][Y[:m,:x].shape[0]//2-1,Y[:m,:x].shape[1]//2-1] = 0
#     for i in range(1,n):
#         ratios.append([np.sqrt(energies[0,i-1]/energies[0,i]),np.sqrt(energies[1,i-1]/energies[1,i]),np.sqrt(energies[2,i-1]/energies[2,i])])
#     ratios.append([np.sqrt(energies[0,n-1]/energies[0,n]),1,1])
    return ratios

def custom_quant1(x, N, step,quantisation_matrix, rise_ratio=None):
    """
    Custom Quantisation with a defined N*N matrix
    For a given value of step, we quantise in terms of step*quantisation matrix
    rise_ratio should be given in terms of the ratio between the absolute rise and step size. The final absolute 
    rise values will be rise = rise_ratio*step*quantisation_matrix
    """
    
    if rise_ratio is None:
        rise_ratio = 1
    step *= quantisation_matrix
    rise = rise_ratio*step
    out = np.zeros(x.shape)
    indices = np.arange(0,x.shape[0])
    indices = np.reshape(np.reshape(indices,(x.shape[0]//N,N)).T,x.shape[0])

    for i in range(N):
        k = indices[i*(x.shape[0]//N):(i+1)*(x.shape[0]//N)]
        for j in range(N):
            l = indices[j*(x.shape[1]//N):(j+1)*(x.shape[1]//N)]
            u,v = np.meshgrid(k,l) # Relevant indices (0,8,16,.....)
            temp = np.ceil((np.abs(x[u,v]) - rise[i,j])/step[i,j])
            q = temp*(temp > 0)*np.sign(x[u,v])
            out[u,v] = q
    return out

def custom_quant2(q, N, step,quantisation_matrix, rise_ratio=None):
    """
    Custom Quantisation with a defined N*N matrix
    For a given value of step, we quantise in terms of step*quantisation matrix
    rise_ratio should be given in terms of the ratio between the absolute rise and step size. It can be a matrix of size N*N or a float
    The final absolute rise values will be rise = rise_ratio*step*quantisation_matrix
    """
    
    if rise_ratio is None:
        rise_ratio = 1
    step *= quantisation_matrix
    rise = rise_ratio*step
    out = np.zeros(q.shape)
    indices = np.arange(0,q.shape[0])
    indices = np.reshape(np.reshape(indices,(q.shape[0]//N,N)).T,q.shape[0])
    for i in range(N):
        k = indices[i*(q.shape[0]//N):(i+1)*(q.shape[0]//N)]
        for j in range(N):
            l = indices[j*(q.shape[1]//N):(j+1)*(q.shape[1]//N)]
            u,v = np.meshgrid(k,l) # Relevant indices (0,8,16,.....)
            y = q[u,v] * step[i,j] + np.sign(q[u,v]) * (rise[i,j] - step[i,j]/2.0)
            out[u,v] = y
    return out

def lbt_enc(X,N,s=None):
    if s:
        Pf,Pr = pot_ii(N,s)
    else:
        Pf,Pr = pot_ii(N)

    C = dct_ii(N)
    t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
    Xp = X.copy()  # copy the non-transformed edges directly from X
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    Y = colxfm(colxfm(Xp, C).T, C).T
    return Y

def lbt_dec(Y,N,s=None):
    if s:
        Pf,Pr = pot_ii(N,s)
    else:
        Pf,Pr = pot_ii(N)
    C = dct_ii(N)
    t = np.s_[N//2:-N//2]
    Z = colxfm(colxfm(Y.T, C.T).T, C.T)
    Zp = Z.copy()
    Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
    Zp[t,:] = colxfm(Zp[t,:], Pr.T)
    return Zp

def dct_enc(X, N):
    C = dct_ii(N)
    Y = colxfm(colxfm(X, C).T, C).T
    return Y

def dct_dec(Y,N):
    C = dct_ii(N)
    Z = colxfm(colxfm(Y.T, C.T).T, C.T)
    return Z


def diagscan(N: int) -> np.ndarray:
    '''
    Generate diagonal scanning pattern
    Returns:
        A diagonal scanning index for a flattened NxN matrix
        The first entry in the matrix is assumed to be the DC coefficient
        and is therefore not included in the scan
    '''
    if N <= 1:
        raise ValueError('Cannot generate a scan pattern for a {}x{} matrix'.format(N, N))

    # Copied from matlab without accounting for indexing.
    slast = N + 1
    scan = [slast]
    while slast != N * N:
        while slast > N and slast % N != 0:
            slast = slast - (N - 1)
            scan.append(slast)
        if slast < N:
            slast = slast + 1
        else:
            slast = slast + N
        scan.append(slast)
        if slast == N * N:
            break
        while slast < (N * N - N + 1) and slast % N != 1:
            slast = slast + (N - 1)
            scan.append(slast)
        if slast == N * N:
            break
        if slast < (N * N - N + 1):
            slast = slast + N
        else:
            slast = slast + 1
        scan.append(slast)
    # Python indexing
    return np.array(scan) - 1


def runampl(a: np.ndarray) -> np.ndarray:
    '''
    Create a run-amplitude encoding from input stream of integers
    Parameters:
        a: array of integers to encode
    Returns:
        ra: (N, 3) array
            ``ra[:, 0]`` gives the runs of zeros between each non-zero value.
            ``ra[:, 1]`` gives the JPEG sizes of the non-zero values (no of
            bits excluding leading zeros).
            ``ra[:, 2]`` gives the values of the JPEG remainder, which
            is normally coded in offset binary.
    '''
    # Check for non integer values in a
    if not np.issubdtype(a.dtype, np.integer):
        raise TypeError(f"Arguments to runampl must be integers, got {a.dtype}")
    b = np.where(a != 0)[0]
    if len(b) == 0:
        ra = np.array([[0, 0, 0]])
        return ra

    # List non-zero elements as a column vector
    c = a[b]
    # Generate JPEG size vector ca = floor(log2(abs(c)) + 1)
    ca = np.zeros(c.shape, dtype=np.int)
    k = 1
    cb = np.abs(c)
    maxc = np.max(cb)

    ka = [1]
    while k <= maxc:
        ca += (cb >= k)
        k = k * 2
        ka.append(k)
    ka = np.array(ka)

    cneg = np.where(c < 0)[0]
    # Changes expression for python indexing
    c[cneg] = c[cneg] + ka[ca[cneg]] - 1
    # appended -1 instead of 0.
    col1 = np.diff(np.concatenate((np.array([-1]), b))) - 1
    ra = np.stack((col1, ca, c), axis=1)
    ra = np.concatenate((ra, np.array([[0, 0, 0]])))
    return ra


def huffdflt(typ: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates default JPEG huffman tables
    Parameters:
        typ: whether to produce luminance (1) or chrominance (2) tables.
    Returns:
        bits: The number of values per bit level, shape ``(16,)``.
        huffval: The codes sorted by bit length, shape ``(162,)``.
    """
    if typ == 1:
        vals = [
            [],  # 1-bit
            [1, 2],  # 2-bit
            [3],  # 3-bit
            [0, 4, 17],  # 4-bit
            [5, 18, 33],  # 5-bit
            [49, 65],  # 6-bit
            [6, 19, 81, 97],  # 7-bit
            [7, 34, 113],  # 8-bit
            [20, 50, 129, 145, 161],  # 9-bit
            [8, 35, 66, 177, 193],  # 10-bit
            [21, 82, 209, 240],  # 11-bit
            [36, 51, 98, 114],  # 12-bit
            [],  # 13-bit
            [],  # 14-bit
            [130],  # 15-bit
            [9, 10,  22,  23,  24,  25,  26,  37,  38,  39,
             40,  41,  42,  52,  53,  54,  55,  # 16-bit
             56,  57,  58,  67,  68,  69,  70,  71,  72,
             73,  74,  83,  84,  85,  86,  87,  88,  89,
             90,  99, 100, 101, 102, 103, 104, 105, 106,
             115, 116, 117, 118, 119, 120, 121, 122, 131,
             132, 133, 134, 135, 136, 137, 138, 146, 147,
             148, 149, 150, 151, 152, 153, 154, 162, 163,
             164, 165, 166, 167, 168, 169, 170, 178, 179,
             180, 181, 182, 183, 184, 185, 186, 194, 195,
             196, 197, 198, 199, 200, 201, 202, 210, 211,
             212, 213, 214, 215, 216, 217, 218, 225, 226,
             227, 228, 229, 230, 231, 232, 233, 234, 241,
             242, 243, 244, 245, 246, 247, 248, 249, 250]]
    else:
        vals = [
            [],  # 1-bit
            [0, 1],  # 2-bit
            [2],  # 3-bit
            [3, 17],  # 4-bit
            [4, 5,  33,  49],  # 5-bit
            [6, 18,  65,  81],  # 6-bit
            [7,  97, 113],  # 7-bit
            [19,  34,  50, 129],  # 8-bit
            [8,  20,  66, 145, 161, 177, 193],  # 9-bit
            [9,  35,  51,  82, 240],  # 10-bit
            [21,  98, 114, 209],  # 11-bit
            [10,  22,  36,  52],  # 12-bit
            [],  # 13-bit
            [225],  # 14-bit
            [37, 241],  # 15-bit
            [23,  24,  25,  26,  38,  39,  40,  41,  42,  53,  54,  # 16-bit
             55, 56,  57,  58,  67,  68,  69,  70,  71,
             72,  73,  74,  83,  84,  85,  86,  87,  88,
             89,  90,  99, 100, 101, 102, 103, 104, 105,
             106, 115, 116, 117, 118, 119, 120, 121, 122,
             130, 131, 132, 133, 134, 135, 136, 137, 138,
             146, 147, 148, 149, 150, 151, 152, 153, 154,
             162, 163, 164, 165, 166, 167, 168, 169, 170,
             178, 179, 180, 181, 182, 183, 184, 185, 186,
             194, 195, 196, 197, 198, 199, 200, 201, 202,
             210, 211, 212, 213, 214, 215, 216, 217, 218,
             226,  227,  228,  229,  230,  231,  232,  233,  234,
             242,  243,  244,  245,  246,  247,  248,  249,  250]]

    # Flatten the nested list alongside the length of each sublist, to make it
    # clear how many bytes of data it takes to store these tables.
    bits = np.array([len(v) for v in vals], dtype=np.uint8)
    huffval = np.concatenate([np.array(v, dtype=np.uint8) for v in vals])

    return bits, huffval


def huffgen(bits: np.ndarray, huffval: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate huffman codes from a huffman table
    Parameters:
        bits, huffval: results from `huffdflt` or `huffdes`.
    Returns:
        huffcode: the valid codes in ascending order of length.
        ehuf: a two-column vector, with one entry per possible 8-bit value. The
            first column lists the code for that value, and the second lists
            the length in bits.
    """
    ncodes = len(huffval)
    if np.sum(bits) != ncodes:
        raise ValueError("bits and huffvals disagree")

    # Generate huffman size table (JPEG fig C1, p78):
    k = 0
    huffsize = np.zeros(ncodes, dtype=int)
    for i, b in enumerate(bits):
        huffsize[k:k+b] = i + 1
        k += b

    huffcode = np.zeros(ncodes, dtype=int)
    code = 0
    si = huffsize[0]

    # Generate huffman code table (JPEG fig C2, p79)
    for k in range(ncodes):
        while huffsize[k] > si:
            code = code * 2
            si += 1
        huffcode[k] = code
        code += 1

    # Reorder the code tables according to the data in
    # huffval to yield the encoder look-up tables.
    ehuf = np.zeros((256, 2), dtype=int)
    ehuf[huffval] = np.stack((huffcode, huffsize), axis=1)

    return huffcode, ehuf


def huffdes(huffhist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a JPEGhuffman table from a 256-point histogram of values.
    This is based on the algorithms in the JPEG Book Appendix K.2.
    Parameters:
        huffhist: the histogram of values
    Returns:
        bits: The number of values per bit level, shape ``(16,)``.
        huffval: The codes sorted by bit length, shape ``(162,)``.
    """

    # Scale huffhist to sum just less than 32K, allowing for
    # the 162 ones we are about to add.
    huffhist = huffhist * (127 * 256)/np.sum(huffhist)

    # Add 1 to all valid points so they are given a code word.
    # With the scaling, this should ensure that all probabilities exceed
    # 1 in 2^16 so that no code words exceed 16 bits in length.
    # This is probably a better method with poor statistics
    # than the JPEG method of fig K.3.
    # Every 16 values made column.
    freq = np.reshape(huffhist, (16, 16), 'F')
    freq[1:11, :] = freq[1:11, :] + 1.1
    freq[0, [0, 15]] = freq[0, [0, 15]] + 1.1

    # Reshape to a vector and add a 257th point to reserve the FFFF codeword.
    # Also add a small negative ramp so that min() always picks the
    # larger index when 2 points have the same probability.
    freq = (np.append(freq.flatten('F'), 1) -
            np.arange(1, 258, 1) * (10 ** -6))

    codesize = np.zeros(257, dtype=int)
    others = -np.ones(257, dtype=int)

    # Find Huffman code sizes: JPEG fig K.1, procedure Code_size

    # Find non-zero entries in freq and loop until 1 entry left.
    nz = np.where(freq > 0)[0]  # np.where output is 2 element tuple.
    while len(nz) > 1:

        # Find v1 for least value of freq(v1) > 0.
        # min in each column
        i = np.argmin(freq[nz])  # freq[nz] and nz index
        v1 = nz[i]

        # Find v2 for next least value of freq(v2) > 0.
        nz = np.delete(nz, i)  # Remove v1 from nz.
        i = np.argmin(freq[nz])  # freq[nz] and nz index
        v2 = nz[i]

        # Combine frequency values to gradually reduce the code table size.
        freq[v1] = freq[v1] + freq[v2]
        freq[v2] = 0

        # Increment the codeword lengths for all codes in the two combined sets.
        # Set 1 is v1 and all its members, stored as a linked list using
        # non-negative entries in vector others(). The members of set 1 are the
        # frequencies that have already been combined into freq(v1).
        codesize[v1] += 1
        while others[v1] > -1:
            v1 = others[v1]
            codesize[v1] += 1

        others[v1] = v2  # Link set 1 with set 2.

        # Set 2 is v2 and all its members, stored as a linked list using
        # non-negative entries in vector others(). The members of set 2 are the
        # frequencies that have already been combined into freq(v2).
        codesize[v2] = codesize[v2] + 1
        while others[v2] > -1:
            v2 = others[v2]
            codesize[v2] = codesize[v2] + 1

        nz = np.where(freq > 0)[0]

    # Find no. of codes of each size: JPEG fig K.2, procedure Count_BITS

    bits = np.zeros(max(16, max(codesize)), dtype=int)
    for i in range(256):
        if codesize[i] > 0:
            bits[codesize[i]-1] = bits[codesize[i]-1] + 1

    # Code length limiting not needed since 1's added earlier to all valid
    # codes.
    assert max(codesize) <= 16

    # Sort codesize values into ascending order and generate huffval:
    # JPEG fig K.4, procedure Sort_input.

    huffval = np.array([], dtype=int)
    t = np.arange(0, 256, 1)
    for i in range(1, 17):
        ii = np.where(codesize[t] == i)[0]
        huffval = np.concatenate((huffval, ii))

    assert len(huffval) == sum(bits)

    return bits, huffval


def huffenc(huffhist: np.ndarray, rsa: np.ndarray, ehuf: np.ndarray
        ) -> np.ndarray:
    """
    Convert a run-length encoded stream to huffman coding.
    Parameters:
        rsa: run-length information as provided by `runampl`.
        ehuf: the huffman codes and their lengths
        huffhist: updated in-place for use in `huffgen`.
    Returns:
        vlc: Variable-length codewords, consisting of codewords in ``vlc[:,0]``
            and corresponding lengths in ``vlc[:,1]``.
    """
    if max(rsa[:, 1]) > 10:
        print("Warning: Size of value in run-amplitude " +
              "code is too large for Huffman table")
        rsa[np.where(rsa[:, 1] > 10), 2] = (2 ** 10) - 1
        rsa[np.where(rsa[:, 1] > 10), 1] = 10

    r, c = rsa.shape

    vlc = []
    for i in range(r):
        run = rsa[i, 0]
        # If run > 15, use repeated codes for 16 zeros.
        while run > 15:
            # Got rid off + 1 to suit python indexing.
            code = 15 * 16
            huffhist[code] = huffhist[code] + 1
            vlc.append(ehuf[code])
            run = run - 16
        # Code the run and size.
        # Got rid off + 1 to suit python indexing.
        code = run * 16 + rsa[i, 1]
        huffhist[code] = huffhist[code] + 1
        vlc.append(ehuf[code])
        # If size > 0, add in the remainder (which is not coded).
        if rsa[i, 1] > 0:
            vlc.append(rsa[i, [2, 1]])
    return np.array(vlc)


def dwtgroup(X: np.ndarray, n: int) -> np.ndarray:
    '''
    Regroups the rows and columns of ``X``, such that an
    n-level DWT image composed of separate subimages is regrouped into 2^n x
    2^n blocks of coefs from the same spatial region (like the DCT).
    If n is negative the process is reversed.
    '''
    Y = X.copy()

    if n == 0:
        return Y
    elif n < 0:
        n = -n
        invert = 1
    else:
        invert = 0

    sx = X.shape
    N = np.round(2**n)

    if sx[0] % N != 0 or sx[1] % N != 0:
        raise ValueError(
            'Error in dwtgroup: X dimensions are not multiples of 2^n')

    if invert == 0:
        # Determine size of smallest sub-image.
        sx = sx // N

        # Regroup the 4 subimages at each level, starting with the smallest
        # subimages in the top left corner of Y.
        k = 1  # Size of blocks of pels at each level.
        # tm = 1:sx[0];
        # tn = 1:sx[1];
        tm = np.arange(sx[0])
        tn = np.arange(sx[1])

        # Loop for each level.
        for _ in range(n):
            tm2 = np.block([
                [np.reshape(tm, (k, sx[0]//k), order='F')],
                [np.reshape(tm+sx[0], (k, sx[0]//k), order='F')]
            ])
            tn2 = np.block([
                [np.reshape(tn, (k, sx[1]//k), order='F')],
                [np.reshape(tn+sx[1], (k, sx[1]//k), order='F')]
            ])

            sx = sx * 2
            k = k * 2
            tm = np.arange(sx[0])
            tn = np.arange(sx[1])
            Y[np.ix_(tm, tn)] = Y[np.ix_(
                tm2.flatten('F'), tn2.flatten('F'))]

    else:
        # Invert the grouping:
        # Determine size of largest sub-image.
        sx = np.array(X.shape) // 2

        # Regroup the 4 subimages at each level, starting with the largest
        # subimages in Y.
        k = N // 2  # Size of blocks of pels at each level.

        # Loop for each level.
        for _ in np.arange(n):
            tm = np.arange(sx[0])
            tn = np.arange(sx[1])
            tm2 = np.block([
                [np.reshape(tm, (k, sx[0]//k), order='F')],
                [np.reshape(tm+sx[0], (k, sx[0]//k), order='F')]
            ])
            tn2 = np.block([
                [np.reshape(tn, (k, sx[1]//k), order='F')],
                [np.reshape(tn+sx[1], (k, sx[1]//k), order='F')]
            ])

            Y[np.ix_(tm2.flatten('F'), tn2.flatten('F'))] = Y[np.ix_(
                np.arange(sx[0]*2), np.arange(sx[1]*2))]

            sx = sx // 2
            k = k // 2

    return Y


def jpegenc(X: np.ndarray, qstep: float, N: int = 8, s = 1.33, M: int = 8,
        opthuff: bool = False, quantisation_matrix = None, dcbits: int = 8, log: bool = True, frequency_quant = True, enc_type = 'lbt'):
    '''
    Encodes the image in X to generate a variable length bit stream.
    Parameters:
        X: the input greyscale image
        qstep: the quantisation step to use in encoding
        N: the width of the DCT block (defaults to 8)
        M: the width of each block to be coded (defaults to N). Must be an
            integer multiple of N - if it is larger, individual blocks are
            regrouped.
        opthuff: if true, the Huffman table is optimised based on the data in X
        dcbits: the number of bits to use to encode the DC coefficients
            of the DCT.
    Returns:
        vlc: variable length output codes, where ``vlc[:,0]`` are the codes and
            ``vlc[:,1]`` the number of corresponding valid bits, so that
            ``sum(vlc[:,1])`` gives the total number of bits in the image
        bits, huffval: optional outputs containing the Huffman encoding
            used in compression when `opthuff` is ``True``.
    '''
    if enc_type != 'dwt':
        if M % N != 0:
            raise ValueError('M must be an integer multiple of N!')

    # DCT on input image X.
    if log:
        print('Forward {} x {} DCT'.format(N, N))
    if enc_type == 'dct':
        C8 = dct_ii(N)
        Y = colxfm(colxfm(X, C8).T, C8).T
    elif enc_type == 'lbt':
        Y = lbt_enc(X,N,s)
    elif enc_type == 'dwt':
        Y = nlevdwt(X,N)
        M = 2**N
    rat = 0.9
    # Quantise to integers.
    if log:
        print('Quantising to step size of {}'.format(qstep))
    if not frequency_quant:
        if enc_type != 'dwt':
            Yq = quant1(Y, qstep, rat*qstep).astype('int')
        else:
            ratios = get_quantisation_step_ratio(N)
            Yq = quantdwt1(Y,ratios*qstep,rat*np.ones(ratios.shape)).astype('int')
            Yq = dwtgroup(Yq,N)
#             Yq = quant1(Yr,qstep,rat*qstep).astype('int')
#             Yq = dwtgroup(Yq,N)
    else:
        if enc_type != 'dwt':
            Yq = custom_quant1(Y, N, qstep, quantisation_matrix,rat).astype('int')
        else:
            ratios = get_quantisation_step_ratio(N)
            Yq = quantdwt1(Y,ratios*qstep,rat*np.ones(ratios.shape)).astype('int')
            Yq = dwtgroup(Yq,N)
#             Yq = quant1(Yr,qstep,rat*qstep).astype('int')
#             Yr = dwtgroup(Y,N)

    
        
#     Yq = quant1(Y, qstep, qstep).astype('int')

    # Generate zig-zag scan of AC coefs.
    scan = diagscan(M)

    # On the first pass use default huffman tables.
    if log:
        print('Generating huffcode and ehuf using default tables')
    dbits, dhuffval = huffdflt(1)  # Default tables.
    huffcode, ehuf = huffgen(dbits, dhuffval)

    # Generate run/ampl values and code them into vlc(:,1:2).
    # Also generate a histogram of code symbols.
    if log:
        print('Coding rows')
    sy = Yq.shape
    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yq = Yq[r:r+M,c:c+M]
            # Possibly regroup
            if enc_type != 'dwt':
                if M > N:
                    yq = regroup(yq, N)
            yqflat = yq.flatten('F')
            # Encode DC coefficient first
            dccoef = yqflat[0] + 2 ** (dcbits-1)
            if dccoef not in range(2**dcbits):
                raise ValueError(
                    'DC coefficients too large for desired number of bits')
            vlc.append(np.array([[dccoef, dcbits]]))
            # Encode the other AC coefficients in scan order
            # huffenc() also updates huffhist.
            ra1 = runampl(yqflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    # (0, 2) array makes this work even if `vlc == []`
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    # Return here if the default tables are sufficient, otherwise repeat the
    # encoding process using the custom designed huffman tables.
    if not opthuff:
        bits = dbits
        huffval = dhuffval
        if log:
            print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
        return vlc, bits, huffval

    # Design custom huffman tables.
    if log:
        print('Generating huffcode and ehuf using custom tables')
    dbits, dhuffval = huffdes(huffhist)
    huffcode, ehuf = huffgen(dbits, dhuffval)

    # Generate run/ampl values and code them into vlc(:,1:2).
    # Also generate a histogram of code symbols.
    if log:
        print('Coding rows (second pass)')
    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yq = Yq[r:r+M, c:c+M]
            # Possibly regroup
            if enc_type != 'dwt':
                if M > N:
                    yq = regroup(yq, N)
            yqflat = yq.flatten('F')
            # Encode DC coefficient first
            dccoef = yqflat[0] + 2 ** (dcbits-1)
            vlc.append(np.array([[dccoef, dcbits]]))
            # Encode the other AC coefficients in scan order
            # huffenc() also updates huffhist.
            ra1 = runampl(yqflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    # (0, 2) array makes this work even if `vlc == []`
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    if log:
        print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
        print('Bits for huffman table = {}'.format(
            (16 + max(dhuffval.shape))*8))

    bits = dbits
    huffval = dhuffval

    return vlc, bits, huffval


def jpegdec(vlc: np.ndarray, qstep: float, N: int = 8, s = 1.33, M: int = 8,
        bits: Optional[np.ndarray] = None, quantisation_matrix = None, huffval: Optional[np.ndarray] = None,
        dcbits: int = 8, W: int = 256, H: int = 256, log: bool = True, frequency_quant = True, enc_type = 'lbt') -> np.ndarray:
    '''
    Decodes a (simplified) JPEG bit stream to an image
    Parameters:
        vlc: variable length output code from jpegenc
        qstep: quantisation step to use in decoding
        N: width of the DCT block (defaults to 8)
        M: width of each block to be coded (defaults to N). Must be an
            integer multiple of N - if it is larger, individual blocks are
            regrouped.
        bits, huffval: if supplied, these will be used in Huffman decoding
            of the data, otherwise default tables are used
        dcbits: the number of bits to use to decode the DC coefficients
            of the DCT
        W, H: the size of the image (defaults to 256 x 256)
    Returns:
        Z: the output greyscale image
    '''
    if enc_type == 'dwt':
        M = 2**N
        
    opthuff = (huffval is not None and bits is not None)
    if enc_type != 'dwt':
        if M % N != 0:
            raise ValueError('M must be an integer multiple of N!')

    # Set up standard scan sequence
    scan = diagscan(M)

    if opthuff:
        if len(bits.shape) != 1:
            raise ValueError('bits.shape must be (len(bits),)')
        if log:
            print('Generating huffcode and ehuf using custom tables')
    else:
        if log:
            print('Generating huffcode and ehuf using default tables')
        bits, huffval = huffdflt(1)

    # Define starting addresses of each new code length in huffcode.
    # 0-based indexing instead of 1
    huffstart = np.cumsum(np.block([0, bits[:15]]))
    # Set up huffman coding arrays.
    huffcode, ehuf = huffgen(bits, huffval)

    # Define array of powers of 2 from 1 to 2^16.
    k = 2 ** np.arange(17)

    # For each block in the image:

    # Decode the dc coef (a fixed-length word)
    # Look for any 15/0 code words.
    # Choose alternate code words to be decoded (excluding 15/0 ones).
    # and mark these with vector t until the next 0/0 EOB code is found.
    # Decode all the t huffman codes, and the t+1 amplitude codes.

    eob = ehuf[0]
    run16 = ehuf[15 * 16]
    i = 0
    Zq = np.zeros((H, W))

    if log:
        print('Decoding rows')
    for r in range(0, H, M):
        for c in range(0, W, M):
            yq = np.zeros(M**2)

            # Decode DC coef - assume no of bits is correctly given in vlc table.
            cf = 0
            if vlc[i, 1] != dcbits:
                raise ValueError(
                    'The bits for the DC coefficient does not agree with vlc table')
            yq[cf] = vlc[i, 0] - 2 ** (dcbits-1)
            i += 1

            # Loop for each non-zero AC coef.
            while np.any(vlc[i] != eob):
                run = 0

                # Decode any runs of 16 zeros first.
                while np.all(vlc[i] == run16):
                    run += 16
                    i += 1

                # Decode run and size (in bits) of AC coef.
                start = huffstart[vlc[i, 1] - 1]
                res = huffval[start + vlc[i, 0] - huffcode[start]]
                run += res // 16
                cf += run + 1
                si = res % 16
                i += 1

                # Decode amplitude of AC coef.
                if vlc[i, 1] != si:
                    raise ValueError(
                        'Problem with decoding .. you might be using the wrong bits and huffval tables')
                ampl = vlc[i, 0]

                # Adjust ampl for negative coef (i.e. MSB = 0).
                thr = k[si - 1]
                yq[scan[cf-1]] = ampl - (ampl < thr) * (2 * thr - 1)

                i += 1

            # End-of-block detected, save block.
            i += 1

            yq = yq.reshape((M, M)).T

            # Possibly regroup yq
            if enc_type != 'dwt':
#                 print(enc_type)
                if M > N:
                    yq = regroup(yq, M//N)
            Zq[r:r+M, c:c+M] = yq

    if log:
        print('Inverse quantising to step size of {}'.format(qstep))
    rat = 0.9
    if not frequency_quant:
        if enc_type != 'dwt':   
            Zi = quant2(Zq, qstep, rat*qstep)
        else:
            ratios = get_quantisation_step_ratio(N)
            Zi = dwtgroup(Zq,-N)
            Zi = quantdwt2(Zi,ratios*qstep,rat*np.ones(ratios.shape)).astype('int')
            
#             N = n
#             Zi = dwtgroup(Zq,-N)
#             Zi = quant2(Zi, qstep, rat*qstep)
    else:
        if enc_type != 'dwt':
            Zi = custom_quant2(Zq, N, qstep, quantisation_matrix,rat)
        else:
            ratios = get_quantisation_step_ratio(N)
            Zi = dwtgroup(Zq,-N)
            Zi = quantdwt2(Zi,ratios*qstep,rat*np.ones(ratios.shape)).astype('int')
#             N = n
#             Zi = dwtgroup(Zq,-N)
#             Zi = quant2(Zi, qstep, rat*qstep)
            
        
    if log:
        print('Inverse {} x {} DCT\n'.format(N, N))
#     C8 = dct_ii(N)
#     Z = colxfm(colxfm(Zi.T, C8.T).T, C8.T)
    if enc_type == 'dct':
        C8 = dct_ii(N)
        Z = colxfm(colxfm(Zi.T, C8.T).T, C8.T)
    elif enc_type == 'lbt':
        Z = lbt_dec(Zi,N,s)
    elif enc_type == 'dwt':
        Z = nlevidwt(Zi,N)
    return Z


def vlctest(vlc: np.ndarray) -> int:
    """ Test the validity of an array of variable-length codes.
    Returns the total number of bits to code the vlc data. """
    from numpy.lib.recfunctions import (
        structured_to_unstructured, unstructured_to_structured)
    if not np.all(vlc[:,1] >= 0):
        raise ValueError("Code words must be non-negative")
    bitwords = unstructured_to_structured(vlc, dtype=bitword.dtype)
    bitword.verify(bitwords)
    return bitwords['bits'].sum(dtype=np.intp)