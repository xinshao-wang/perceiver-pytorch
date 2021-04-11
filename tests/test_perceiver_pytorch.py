

import torch

from perceiver_pytorch import Perceiver

model = Perceiver(
    input_channels = 3,          # number of channels for each token of the input
    input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
    num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
    max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
    depth = 6,                   # depth of net
    num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,
    latent_dim_head = 64,
    num_classes = 1000,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
)

img = torch.randn(2, 224, 224, 3) # 1 imagenet image, pixelized

output=model(img) # (2, 1000)

print(output.shape)


"""
This experimental would not work due to data is too long. 

I have also included a version of Perceiver that includes bottom-up (in addition to top-down) attention, 
using the same scheme as presented in the original Set Transformers paper as the Induced Set Attention Block.

You simply have to change the above import to

from perceiver_pytorch.experimental import Perceiver
"""