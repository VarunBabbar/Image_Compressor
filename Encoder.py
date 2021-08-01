import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab import laplacian_pyramid
import warnings
import inspect
import matplotlib.pyplot as plt
import matplotlib as mpl
import IPython.display
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
import numpy as np
from typing import Tuple
from cued_sf2_lab.laplacian_pyramid import bpp, quantise
from cued_sf2_lab.dwt import idwt
from cued_sf2_lab.dwt import dwt
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.dct import regroup, dct_ii
from cued_sf2_lab.lbt import pot_ii
import math
from skimage.metrics import structural_similarity as ssim
from cued_sf2_lab.jpeg2 import get_quantisation_step_ratio
from cued_sf2_lab.jpeg2 import jpegenc,jpegdec,dwtgroup
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.filters import unsharp_mask
from scipy.signal import convolve2d
from scipy import signal
from skvideo.measure import msssim
from cued_sf2_lab.jpeg2 import quant1,quant2
from cued_sf2_lab.jpeg2 import custom_quant1,custom_quant2,diagscan
from cued_sf2_lab import arithmetic
from cued_sf2_lab import pyae
import rle
import objsize
import argparse
from scipy.io import savemat

# lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})
# bridge, _ = load_mat_img(img='bridge.mat', img_info='X', cmap_info={'map'})
# flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')
# lighthouse = lighthouse-128.0
# bridge = bridge-128.0
# flamingo = flamingo-128.0
# img = bridge

class Encode:
    def __init__(self,curr_min,curr_max,opthuff,s_range,Ns,enc_type,fmin,fmax):
        quantisation_matrix1 =[[16,11,10,16,24,40,51,61], # Original JPEG
                        [12,12,14,19,26,58,60,55],
                        [14,13,16,24,40,57,69,56],
                        [14,17,22,29,51,87,80,62],
                        [18,22,37,56,68,109,103,77],
                        [24,35,55,64,81,104,113,92],
                        [49,64,78,87,103,121,120,101],
                        [72,92,95,98,112,100,103,99]]

        quantisation_matrix2 = [[8,13,16,25,39,68,95,74], # Quality 95
                            [13,19,47,17,49,65,79,69],
                            [19,15,36,23,67,95,79,58],
                            [15,39,30,85,79,127,128,77],
                            [27,47,55,75,122,174,117,76],
                            [45,87,75,86,102,167,178,105],
                            [71,96,113,115,139,156,151,122],
                            [137,131,161,140,176,115,132,125]]
        # Take the best out of them
        self.matrices = [np.array(quantisation_matrix1),np.array(quantisation_matrix2)]

        self.curr_min = curr_min
        self.curr_max = curr_max
        self.opthuff = opthuff
        self.s_range = s_range
        self.Ns = Ns
        self.enc_type = enc_type
        self.fmin = fmin
        self.fmax = fmax
        
    def binary_search(self,X,s,N,dcbits,curr_min,curr_max,frequency_quant, quantisation_matrix,opthuff,enc_type):
        delta = 0.1
        min_bits = 40900
        max_bits = 40940
        valid_params = []
        vlc, bits2, huffval = jpegenc(X, curr_max, N=N, M=N, opthuff=opthuff, dcbits=dcbits, log=False,s = s,quantisation_matrix = quantisation_matrix, frequency_quant = frequency_quant,enc_type = enc_type)
#         x_rec = jpegdec(vlc, curr_max, N=N, M=N, bits=bits, huffval=huffval, dcbits=dcbits, W=256, H=256, log=True,s=s,quantisation_matrix = quantisation_matrix,frequency_quant = frequency_quant,enc_type = enc_type)
        bits = np.sum(vlc[:,1])+1424
#         print('hi')
        while bits > max_bits or bits < min_bits:
            print(bits,curr_max,curr_min)
            qstep = (curr_max + curr_min)/2
            vlc, bits2, huffval = jpegenc(X,qstep, N=N, M=N, opthuff=opthuff, dcbits=dcbits, log=False,s = s,quantisation_matrix = quantisation_matrix, frequency_quant = frequency_quant,enc_type = enc_type)
#             x_rec = jpegdec(vlc, qstep, N=N, M=N, bits=bits, huffval=huffval, dcbits=dcbits, W=256, H=256, log=True,s=s,quantisation_matrix = quantisation_matrix,frequency_quant = frequency_quant,enc_type = enc_type)
            bits = np.sum(vlc[:,1])+1424
            if bits >= max_bits:
                curr_min = qstep
            elif bits <= min_bits:
                curr_max = qstep
        # we have ourselves an interval for curr_max and curr_min
        if frequency_quant:
            return [(dcbits,s,N,frequency_quant,round(qstep,4),quantisation_matrix,vlc,bits2,huffval)]
        valid_params.append((dcbits,s,N,frequency_quant,round(qstep,4),vlc,bits2,huffval))
        for step in range(int(curr_min/delta),int(curr_max/delta)):
            step *= delta
            vlc, bits2, huffval = jpegenc(X,step, N=N, M=N, opthuff=opthuff, dcbits=dcbits, log=False,s = s,quantisation_matrix = quantisation_matrix, frequency_quant = frequency_quant,enc_type = enc_type)
#             x_rec = jpegdec(vlc, step, N=N, M=N, bits=bits, huffval=huffval, dcbits=dcbits, W=256, H=256, log=True,s=s,quantisation_matrix = quantisation_matrix,frequency_quant = frequency_quant,enc_type = enc_type)
            bits = np.sum(vlc[:,1])+1424
            if bits <= min_bits:
                break
            elif bits >= min_bits and bits <= max_bits:
                valid_params.append((dcbits,s,N,frequency_quant,round(step,4),vlc,bits2,huffval))
        return valid_params

    def optimize_params(self,X): # Need to repurpose this for the arithmetic coding function
        curr_min = self.curr_min
        curr_max = self.curr_max
        opthuff = self.opthuff
        s_range = self.s_range
        Ns = self.Ns
        enc_type = self.enc_type
        fmax = self.fmax
        fmin = self.fmin
        quantisation_matrix1 =[[16,11,10,16,24,40,51,61], # Original JPEG
                            [12,12,14,19,26,58,60,55],
                            [14,13,16,24,40,57,69,56],
                            [14,17,22,29,51,87,80,62],
                            [18,22,37,56,68,109,103,77],
                            [24,35,55,64,81,104,113,92],
                            [49,64,78,87,103,121,120,101],
                            [72,92,95,98,112,100,103,99]]

        quantisation_matrix2 = [[8,13,16,25,39,68,95,74], # Quality 95
                                [13,19,47,17,49,65,79,69],
                                [19,15,36,23,67,95,79,58],
                                [15,39,30,85,79,127,128,77],
                                [27,47,55,75,122,174,117,76],
                                [45,87,75,86,102,167,178,105],
                                [71,96,113,115,139,156,151,122],
                                [137,131,161,140,176,115,132,125]]
        # Take the best out of them
        matrices = [np.array(quantisation_matrix1),np.array(quantisation_matrix2)]      
#         matrices = [np.array(quantisation_matrix1)]   
        valid_params = []
        possible_dcbits = [6,7,8,9,10,11,12,13,14]
        freq = [False,True]
        # enc_type = 'lbt'
        delta = 0.1
        for s in s_range:
            for N in Ns:
                N = int(N)
                print("S, N are {}, {}".format(s,N))
                for frequency_quant in freq:
                    dcflag = False # if we don't get an error we are done
                    if frequency_quant:
                        if N == 8:
                            for quantisation_matrix in matrices:
                                for dcbits in possible_dcbits:
                                    if dcflag:
                                        break
                                    try:
                                        if not dcflag:
                                            valid_params += self.binary_search(X,s,N,dcbits,fmin,fmax,frequency_quant, quantisation_matrix,opthuff,enc_type)
                                            dcflag = True
                                    except:
                                        print("Error: Trying new values!")
                                        continue
                    else:
                        for dcbits in possible_dcbits:
                            if dcflag:
                                break
                            try:
                                if not dcflag:
                                    valid_params += self.binary_search(X,s,N,dcbits,curr_min,curr_max,frequency_quant, None,opthuff,enc_type)
                                    dcflag = True
                            except:
                                print("Error: Trying new values!")
                                continue
        max_error = -float('inf')
        min_error = float('inf')
        optimum_x = 0
        opt_param = [()]
        optimum_vlc = []
        sharp = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
        for param in valid_params:
            dcbits = param[0]
            s = param[1]
            N = param[2]
            frequency_quant = param[3]
            step = param[4]
            if frequency_quant:
                quantisation_matrix = param[5]
                vlc = param[6]
                bits = param[7]
                huffval = param[8]
                x_rec = jpegdec(vlc, step, N=N, M=N, bits=bits, huffval=huffval, dcbits=dcbits, W=256, H=256, log=True,s=s,quantisation_matrix = quantisation_matrix,frequency_quant = frequency_quant,enc_type = enc_type)
            else:
                quantisation_matrix = None
                vlc = param[5]
                bits = param[6]
                huffval = param[7]
                x_rec = jpegdec(vlc, step, N=N, M=N, bits=bits, huffval=huffval, dcbits=dcbits, W=256, H=256, log=True,s=s,quantisation_matrix = quantisation_matrix,frequency_quant = frequency_quant,enc_type = enc_type)
            rec = x_rec + 128.0
            rec = (rec-rec.min())/(rec.max()-rec.min())
            rec2 = rec*255
            met = (msssim(X+128.0,rec2) + ssim(X+128.0,rec2)-np.std((X+128.0)/255-rec2/255))/3
            if met > max_error:
                max_error = met
                opt_param = param
                optimum_vlc = vlc
            if met < min_error:
                min_error = met
        return opt_param,optimum_vlc,valid_params

parser = argparse.ArgumentParser(description="")
parser.add_argument("--image_dir", default= "bridge.mat", type=str, help="Path to checkpoint (default: none)")
args = parser.parse_args()
img,_ = load_mat_img(img=args.image_dir, img_info='X', cmap_info={})
img = img - 128.0
curr_min = 1
curr_max = 200
opthuff = True
s_range = np.arange(1.2,1.6,0.05)
Ns = np.array([4,8,16])
enc_type = 'lbt'
fmin = 0.7
fmax = 6.5
encoder = Encode(curr_min,curr_max,opthuff,s_range,Ns,enc_type,fmin,fmax)
opt_params,optimum_vlc,_ = encoder.optimize_params(img)
final_params = {
    'dct_bits': opt_params[0],
    's': opt_params[1],
    'N': opt_params[2],
    'freq':opt_params[3],
    'step_ratio':opt_params[4],
    'huffval':opt_params[-1],
    'bits2': opt_params[-2],
    'vlc': np.array(optimum_vlc)
}

#opt_dict = {"params":opt_params}
total_size = sum(optimum_vlc[:,1])+1424+1+int(np.log2(opt_params[2])+1)+32
print(total_size)
savemat("Group_13_vlc_params.mat",final_params)


