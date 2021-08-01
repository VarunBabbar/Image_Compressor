# Image_Compressor
A Deep Learning Based Image Compression Scheme developed as part of my 3rd Year Image Processing Project. Note that some of the code in the folder 'cued_sf2_lab' is taken from https://github.com/sigproc/cued_sf2_lab. 


1) To run the encoder, open Terminal / Command line and type 
    python3 Encoder.py --image_dir path_to_image
     Replace path_to_image with the image path (eg ‘flamingo.mat’)
    This will save the image vlc and optimum parameters in a mat file in the same directory as the code that will be passed to the decoder

2) To run the decoder, type the following on command line / Terminal
    python3 Decoder.py --model_weights ssim_lbt_trained_DenseFuseNet_15_epochs.mat —vlc_params Group_13_vlc_params.mat  --image_dir path_to_image

    This will output the final image as a mat file in the same directory as the code
