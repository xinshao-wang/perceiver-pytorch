from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4, base = 2):
    """
    Positional encodings for input x:
    Enrich the input features with fourier feature encodings.
    This is a way to tag each input units with a position and
    construct topographic maps.

    :param x:
    :param max_freq:
    :param num_bands:
    :param base:
    :return: each input unit of x is tagged with a position encoding.
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(1., log(max_freq / 2) / log(base), num_bands, base = base, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    """
    Two layer normalisation
    1. self.norm = nn.LayerNorm(dim)
    2. self.norm_context = nn.LayerNorm(context_dim)
    """
    def __init__(self, dim, fn, context_dim = None):
        """

        :param dim: the dimensionality of a latent bottleneck
        :param fn: an attention module
        :param context_dim: the dimensionality of a context input (e.g., original input)
        """
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        """
        Use layer normalisation before the attention module: fn
        Layer normalisation is non-parametric.

        :param x:
        :param kwargs:
        :return:
        """
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    """
    1. What is the chunk?
    2. What is gelu?
    """
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    """
    A subnet : a combination of Linear, GEGLU, Dropout, Linear

    The output shape is of the same as the input.
    """
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
        Attention layer with query and context (optional, a.k.a. latent bottleneck)
    """
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        """
        1. Operations
        2. in and output dimensionality configurations

        :param query_dim:
        :param context_dim:
        :param heads:
        :param dim_head:
        :param dropout:
        """
        super().__init__()
        # what scale does?
        self.scale = dim_head ** -0.5

        self.heads = heads
        # #heads x dim_head => concatenation operation
        inner_dim = dim_head * heads

        # if context_dim exist, use context_dim,
        # otherwise, use query_dim
        context_dim = default(context_dim, query_dim)

        # transformation into the attention block
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        # Note that here: inner_dim * 2 so that
        # k and v has the same-dimensional embedding
        # context_dim controls whether Q and KV has the same embedding size
        # outside the attention block
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        # go out of the attention block
        # ? No process for context when context is used?
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        """
        1. What is the context input?
            if context is used:
                context => Linear(context) => K and V
            otherwise:
                x (same as query) => Linear(x) => Q
        2. What is the mask input?

        :param x:
        :param context:
        :param mask:
        :return:
        """
        h = self.heads

        #################Linear Projection#########################
        # simply linear projection x into query: Q
        # => Q: (b,n, inner_dim)
        q = self.to_q(x)

        # if context is not None => use context to create K and V
        # first Linear into (b,n, inner_dim * 2), then use chunk to split:
        # K: (b,n, inner_dim)
        # V: (b,n, inner_dim)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        ##################Linear Projection########################

        ####################################################################################
        # rearrange the shape in q, k, v
        # for the following matrix product
        # !!! split the head: h = self.heads
        # n = #index dimensionality (#tokens)
        # In a nutshell: #batch, #query, #headsx#dim_head
        # => (#batch x #head) x #query x dim
        # => (#batch x #head) x #keys x dim
        # => (#batch x #head) x #values x dim
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        ####################################################################################
        # matrix product specified by  "b i d, b j d -> b i j"
        # q, k => get keys' attention matrix b, i, j
        # Three dimensions: (#batch x #head) x #query x #keys
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        # (#batch x #head) x #query x #keys
        attn = sim.softmax(dim = -1)


        # (#batch x #head) x #query x dim
        out = einsum('b i j, b j d -> b i d', attn, v)
        ####################################################################################

        # back to: #batch x #query x (#head x dim) or (b,n, inner_dim)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        ####################################################################################

        # And a linear layer after multi-head concatenation
        # so that the shape becomes the same as input: #batch x #query x query_dim
        return self.to_out(out)

# main class

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        freq_base = 2,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False
    ):
        """
        What configurations are necessary?

        :param num_freq_bands: Fourier feature encoding
        :param depth: The length of stacking, # depth of net
        :param max_freq: # maximum frequency, hyperparameter depending on how fine the data is
        :param freq_base:
        :param input_channels: # number of channels for each token of the input
        :param input_axis:  # number of axis for input data (2 for images, 3 for video)
        :param num_latents: # number of latents, or induced set points, or centroids. different papers giving it different names
        :param latent_dim:  # latent dimension
        :param cross_heads: # number of heads for cross attention. paper said 1
        :param latent_heads: # number of heads for latent self attention, 8
        :param cross_dim_head:
        :param latent_dim_head:
        :param num_classes: # output number of classes
        :param attn_dropout:
        :param ff_dropout:
        :param weight_tie_layers: # whether to weight tie layers (optional, as indicated in the diagram)
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base

        # context_dim = input_dim, where we iteratively query from in cross attention
        # each input unit is tagged with a position encoding using fourier feature encoding
        input_dim = input_axis * ((num_freq_bands * 2) + 1) + input_channels


        # random initialisation of latent bottleneck: i.e., latents (self attention module)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim,
                                         Attention(latent_dim, context_dim = input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout),
                                         context_dim = input_dim,
                                         )
        # for self attention, context_dim = None
        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, context_dim = None,  heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout),
                                          context_dim = None,
                                          )

        get_cross_ff = lambda: PreNorm( latent_dim, FeedForward( latent_dim, dropout=ff_dropout ) )
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        #? map of four modules
        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))


        # Stack those four modules into a deep net
        self.layers = nn.ModuleList([])
        for i in range(depth):
            # cache, and weight_tie_layers
            # weight share option: True or False
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            # append with an optional weight sharing mechanism
            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # LayerNorm + Linear Projection to num_classes
        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, data, mask = None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # for image input:
        print(data.shape)

        # calculate fourier encoded positions in the range of [-1, 1], for all axis
        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
        print(len(axis_pos)) # = 2, two axis
        print(len(axis_pos[0])) # = 224, 224 per axis

        pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base = self.freq_base)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        # repeat the positional encodings, which is the same for all input unit if length is the same.
        enc_pos = repeat(enc_pos, '... -> b ...', b = b)

        # concat to channels of data and flatten axis
        data = torch.cat((data, enc_pos), dim = -1)
        print(data.shape) # torch.Size([1, 224, 224, 29])

        data = rearrange(data, 'b ... d -> b (...) d')
        print(data.shape) # as context: torch.Size([1, 50176, 29])

        # latents are also shared by each single data point/image
        x = repeat(self.latents, 'n d -> b n d', b = b)
        print(x.shape) # as latents (learned Q): torch.Size([1, 256, 512])

        # Stacking four modules
        for cross_attn, cross_ff, latent_attn, latent_ff in self.layers:
            # cross attention with residual connection
            x = cross_attn(x, context = data, mask = mask) + x
            # post feed forward of x with residual connection
            x = cross_ff(x) + x

            # self attention with residual connection
            x = latent_attn(x) + x
            # post feed forward of x with residual connection
            x = latent_ff(x) + x

        # attention layers does not change the input shape
        print(x.shape) # torch.Size([1, 256, 512])

        # mean output over latents: the average of learned queries/centroids/cluster centers
        x = x.mean(dim = -2)
        print(x.shape) # torch.Size([1, 512])
        return self.to_logits(x)
