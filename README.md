# Image_Compressor
A Deep Learning Based Image Compression Scheme developed as part of my 3rd Year Image Processing Project. Note that some of the code in the folder 'cued_sf2_lab' is taken from https://github.com/sigproc/cued_sf2_lab. 

## Table of contents
* [General info](#general-info)
* [Dependencies](#dependencies)
* [Running the Scripts](#running-the-scripts)
* [Useful Resources](#useful-resources)

## General info  
This repository is an extension to my 3rd Year Project, which aims to develop a scheme to compress an image to less than 5 kB whilst retaining maximum image quality. I developed a more efficient hyper-parameter search scheme for Huffman coding and added Deep Learning based JPEG artefact removal. 
The code + dataset for training this artefact removal model will be added to this repository soon. 
## Dependencies
The modules are created with:
* [PyTorch 1.6](https://pytorch.org/get-started/locally/)
* Python 3.7
* [Numpy 1.19.2](https://pypi.org/project/numpy/)
 
## Running the Scripts
 Here is the expected syntax for the Encoder.py script. 
 ```
   usage: Encoder.py [-h] [--image_dir IMAGE_DIR]

   optional arguments:
      -h, --help            show this help message and exit
      --image_dir IMAGE_DIR
                            Path to image (.mat file) (default: none)
 ```
 This will save the image vlc and optimum compression hyper-parameters in a mat file in the same directory as this code. This needs to be passed to the decoder.
 
 Here is the expected syntax for the Decoder.py script. 
 
 ```
 usage: Decoder.py [-h] [--model_weights MODEL_WEIGHTS]
                  [--vlc_params VLC_PARAMS] [--image_dir IMAGE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --model_weights MODEL_WEIGHTS
                        Path to Model Weights
  --vlc_params VLC_PARAMS
                        Path to VLC and Optimum Parameters
  --image_dir IMAGE_DIR
                        Path to image (.mat file) (default: none)
 ```
 
### Example usage:
 ```
 python3 Encoder.py --image_dir /Users/varunbabbar/Desktop/Flamingo.mat 
 ```
 This will save a file called 'Group_13_vlc_params.mat'. Now run:
 ```
 python3 Decoder.py --model_weights ssim_lbt_trained_DenseFuseNet_15_epochs —vlc_params Group_13_vlc_params.mat  --image_dir /Users/varunbabbar/Desktop/Flamingo.mat 
 ```
 This will output the final compressed and reconstructed image as a mat file in the same directory as this code. Note that the initial image passed to the Encoder has to be a greyscale image with pixel values in the range (0,255).


 ## Useful Resources
 
https://github.com/sigproc/cued_sf2_lab for the preliminary Python code on DCT, LBT, DWT and Huffman Coding. 



