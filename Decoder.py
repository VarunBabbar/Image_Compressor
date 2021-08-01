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
import argparse
from scipy.io import loadmat
from scipy.io import savemat
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv3', nn.Conv2d(64,32,3,1,1))
        self.layers.add_module('Act3' , nn.ReLU(inplace=True))    
        self.layers.add_module('Conv4', nn.Conv2d(32,16,3,1,1))
        self.layers.add_module('Act4' , nn.ReLU(inplace=True))
        self.layers.add_module('Conv5', nn.Conv2d(16,1,3,1,1))
        
    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        
        self.Conv1 = nn.Conv2d(1,16,3,1,1)
        self.Relu  = nn.ReLU(inplace=True)
        
        self.layers = nn.ModuleDict({
            'DenseConv1': nn.Conv2d(16,16,3,1,1),
            'DenseConv2': nn.Conv2d(32,16,3,1,1),
            'DenseConv3': nn.Conv2d(48,16,3,1,1)
        })
        
    def forward(self, x):
        x = self.Relu(self.Conv1(x))
        for i in range(len(self.layers)):
            out = self.layers['DenseConv'+str(i+1)](x)
            x = torch.cat([x,out],1)
        return x

class architecture(nn.Module):
    
    def __init__(self):
        super(architecture,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self,x):
        return self.decoder(self.encoder(x))
    
class DenseFuseNet(nn.Module):
    def __init__(self,directory):
        super(DenseFuseNet,self).__init__()
        self.architecture = architecture()
        self.architecture.load_state_dict(torch.load(directory,map_location=torch.device('cpu')))
        
    def forward(self,x):
        return self.architecture(x)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--model_weights", default= "ssim_lbt_trained_DenseFuseNet_15_epochs", type=str, help="Path to Model Weights")
parser.add_argument("--vlc_params", default= "Group_13_vlc_params.mat", type=str, help="Path to VLC and Optimum Parameters")
parser.add_argument("--image_dir", default= "bridge.mat", type=str, help="Path to checkpoint (default: none)")
args = parser.parse_args()
model = DenseFuseNet(args.model_weights)
opt_params = loadmat(args.vlc_params)
dcbits = opt_params['dct_bits'][0][0]
s = opt_params['s'][0][0]
N = int(opt_params['N'][0][0])
frequency_quant = opt_params['freq'][0][0]
step = opt_params['step_ratio'][0][0]
enc_type='lbt'
if frequency_quant:
    quantisation_matrix = np.array([[16,11,10,16,24,40,51,61], # Original JPEG
                        [12,12,14,19,26,58,60,55],
                        [14,13,16,24,40,57,69,56],
                        [14,17,22,29,51,87,80,62],
                        [18,22,37,56,68,109,103,77],
                        [24,35,55,64,81,104,113,92],
                        [49,64,78,87,103,121,120,101],
                        [72,92,95,98,112,100,103,99]])
    vlc = opt_params['vlc']
    bits = opt_params['bits2'][0]
    huffval = opt_params['huffval'][0]
    
    vlc = np.array(vlc)
    bits = np.array(bits)
    huffval = np.array(huffval)
    x_rec = jpegdec(vlc, step, N=N, M=N, bits=bits, huffval=huffval, dcbits=dcbits, W=256, H=256, log=True,s=s,quantisation_matrix = quantisation_matrix,frequency_quant = frequency_quant,enc_type = enc_type)
else:
    quantisation_matrix = None
    vlc = opt_params['vlc']
    bits = opt_params['bits2'][0]
    huffval = opt_params['huffval'][0]
    x_rec = jpegdec(vlc, step, N=N, M=N, bits=bits, huffval=huffval, dcbits=dcbits, W=256, H=256, log=True,s=s,quantisation_matrix = quantisation_matrix,frequency_quant = frequency_quant,enc_type = enc_type)

x_rec = (x_rec-x_rec.min())/(x_rec.max()-x_rec.min())
rec = x_rec.copy()
x_rec = torch.Tensor(x_rec)
x_rec = x_rec.unsqueeze(0).unsqueeze(0)
x_rec = x_rec*2-1
reconstructed_image = model(x_rec)
reconstructed_image = np.array(reconstructed_image.squeeze().squeeze().detach().cpu())
reconstructed_image = (reconstructed_image+1)/2
sharp = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
reconstructed_image = reconstructed_image*255
before = reconstructed_image
reconstructed_image = signal.convolve2d(reconstructed_image, sharp/16, boundary='symm', mode='same') + reconstructed_image
reconstructed_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min()) * 255
alpha = 1.025
reconstructed_image *= alpha
reconstructed_image[reconstructed_image > 255] = 255
reconstructed_image = reconstructed_image.astype(np.uint8)

img,_ = load_mat_img(img=args.image_dir, img_info='X', cmap_info={})
m1 = -np.std((img+128.0)/255-before/255)+msssim((img+128.0)/255,before/255)+ssim((img+128.0)/255,before/255)
m2 = -np.std((img+128.0)/255-reconstructed_image/255)+msssim((img+128.0)/255,reconstructed_image/255)+ssim((img+128.0)/255,reconstructed_image/255)
if m1 > m2:
    reconstructed_image = before.astype(np.uint8)

savemat("Final_Image.mat",{'X':reconstructed_image})

