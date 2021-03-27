# A learning Project for GANs

# MNIST Fashion gnerative model

## Starting out:
    - Wanted to use a pytorch dataset to work on image generation using a GAN
    - found larger batch size helps because models train
    - realized that relu + dropout is not best practice compared to leakyrelu
    - realized that method of upsampling was not so good along with the fact that
    - trained a model with too high of a loss ration, got poor results where the snapshots are taken every 10 epochs
    - trained model with lower loss ratio (0.0002) and got better results. 
        - there are also sample outputs (I lost them, but they are in the git history I *think*)

## Next Steps:
    - create personal dataset for image generation


## NOTE:
    - wanted to do some ML that would run better on GPU
    - single epoch using current commit:
        - CPU: 213 Seconds
        - GPU: 30 Seconds

# DND Map Maker (mapmaker)

# Outputs:

    - check out the sample_runs dir
    - note the model is built to dist/ in case you want to pip install it
    - sample outputs are in `sample_runs`:
        -


## Notes, thoughts, learning:
    - dataset is really small ~360 images (from map of the week)
        - pruned a few by hand after downloading the script
        - more data might be needed
        - though, if the gan is really stable, it may be able to use more epochs with the same dataset
        - the dataset is pretty diverse, but I am not sure if that helps or not
    - Was getting some divergences, reduced discriminator learning rate from 0.001 -> 0.0004
    - At one point, the last layer of the generator had a larger kernel size and a larger padding, changed to lower padding and kernel
    - took some online advice and the labels for the real images are "0.9" not 1
    - added batch normalization to generator model
    - using tanh activation for the image output, since images are 0-1 created with (X + 1) / 2
        - thinking about applying 2*X - 1 to training images, but undecided
    - have added ability but have not tried adding noise (15%) to train and generated images before passing to the discriminator during training
    - desired output image size may just be too large
    - had to strike a balance with making the model fit on 24GB of RAM
        - used large stride to upsample image size
        - only had latent space of 3x20x20 -> linear layer -> 16x64x64
        - so upsizing from 64x64 to ~1k over the two transpose layers
        - use one convolution to go down from ~1000x1000 to 750x750
    - not sure if large stride is best way to upsample
    - had gut feeling that I wanted to have at least one layer that reduces input -> output size
    - not sure if I should have any equal input and output layers
        - might want to add one more layer at the end of the generator for better smoothing?
    - didn't have dropout in generator seems bad
    - maybe should try smaller images
    - smaller images worked better
    - pruned the corpus I was using down to 225 images from 360 by removing maps that were of a different scale (geography not dungeon maps),
    - got real results!
    - not sure how to procceed since images are coming out looking pretty sweet, but the 128x128 size is not great
        - thinking of training another model to improve the output of this model, e.g.:
            - use one of these models as input
            - create a gan that scales up from 128x128 to e.g. 1024x1024
            - corpus would have to load in at the larger size - trivial 
            - memory might be tricky with larger images / model for larger images
                - number of layers might be able to be small