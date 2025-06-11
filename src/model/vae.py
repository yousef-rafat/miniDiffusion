########################################### REIMPLEMENTATION OF THE VARIATIONAL AUTOENCODER USED IN STABLE DIFFUSION ###########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import Resize

class ResnetBlock(nn.Module):
    """ Simple Resnet block for VAE with Normalization """
    def __init__(self, channels):
        super(ResnetBlock, self).__init__()

        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)

        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)

        self.silu = nn.SiLU(inplace = True)

    def forward(self, x: torch.Tensor):
        h = self.silu(self.norm1(x))
        h = self.conv1(h)

        h = self.silu(self.norm2(h))
        h = self.conv2(h)

        return x + h
    
class ResNetBlockProjection(nn.Module):
    """ Resnet to make the input and output the same dimensions with projection """
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.conv_shortcut = nn.Conv2d(input_channels, output_channels, kernel_size = 1, padding = 0)

        self.norm1 = nn.GroupNorm(32, input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1)

        self.norm2 = nn.GroupNorm(32, output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size = 3, padding = 1)

        self.silu = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor):
        shortcut = self.conv_shortcut(x)

        h = self.silu(self.norm1(x))
        h = self.conv1(h)

        h = self.silu(self.norm2(h))
        h = self.conv2(h)

        return shortcut + h
    
class AttentionBlock(nn.Module):
    """ Create a specific Attention block for VAE """
    def __init__(self, channels):   
        super().__init__()

        self.group_norm = nn.GroupNorm(32, channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, x: torch.Tensor):
        
        h = self.group_norm(x)
        B, C, H, W = x.shape

        # reshape so matmul work (512 x 512)
        h = h.view(B, H * W, C)

        # forward
        query = self.to_q(h)
        key = self.to_k(h)
        value = self.to_v(h)

        # resizing
        query = query.view(B, C, H * W)
        key = key.view(B, C, H * W)
        value = value.view(B, C, H * W)

        # attention matmul
        attn = torch.bmm(query.transpose(1, 2), key)
        attn = torch.softmax(attn, dim = -1)
        out = torch.bmm(value, attn.transpose(1, 2))

        # project and return
        out = self.to_out[0](out.transpose(1, 2))
        out = out.view(B, C, H, W)

        return out + x


# ##########################
# Reparamatrization Sampling
# ##########################
class DiagonalGaussianDistribution:
    def __init__(self, params: torch.Tensor, latent_channels: int = 4):

        # divide quant channels (8) into mean and log variance
        self.latent_channels = latent_channels
        self.mean, self.logvar = torch.chunk(params, 2, dim = 1)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):

        eps = torch.randn_like(self.std)
        z = self.mean + eps * self.std

        return z

# ///////////////////////
# Samplers /////////////
#///////////////////////

# wrapper for the downsampling convolution to match key names.
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv = True):
        super().__init__()

        # if it's the last layer, don't use conv block
        if use_conv: self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 0)
        else: self.conv = None

    def forward(self, x):

        if self.conv is not None:
            # manual asymetric padding for precision
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode = "constant", value = 0)
            x = self.conv(x)

        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv):
        super().__init__()

        # we upscale by linear interpolation instead of transposed convs
        if use_conv: self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        else: self.conv = None

    def forward(self, x):
        
        # avoid checkboard artificats from transpose convs
        x = F.interpolate(x, scale_factor = 2.0, mode = "nearest")

        if self.conv is not None:
            x = self.conv(x)

        return x
    
# ///////////////////////
# Blocks ///////////////
#///////////////////////

# down block that contains a downsampler and 2 resnet blocks.
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv):
        super().__init__()
        
        self.downsamplers = nn.ModuleList([Downsample(out_channels, out_channels, use_conv)])

        if in_channels != out_channels:
            resnet1 = ResNetBlockProjection(in_channels, out_channels)
            
        else: resnet1 = ResnetBlock(out_channels)

        resnet2 = ResnetBlock(out_channels)

        self.resnets = nn.ModuleList([resnet1, resnet2])

    def forward(self, x):

        for layer in self.resnets:
            x = layer(x)

        x = self.downsamplers[0](x)

        return x

# block that contains an upsampler and 2 resnet blocks.
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv):
        super().__init__()

        self.upsamplers = nn.ModuleList([Upsample(out_channels, out_channels, use_conv)])

        if in_channels != out_channels: 
            resnet1 = ResNetBlockProjection(in_channels, out_channels)

        else: resnet1 = ResnetBlock(out_channels)

        resnet2 = ResnetBlock(out_channels)
        resnet3 = ResnetBlock(out_channels)

        self.resnets = nn.ModuleList([resnet1, resnet2, resnet3])

    def forward(self, x):

        for block in self.resnets:
            x = block(x)

        x = self.upsamplers[0](x)

        return x
    
class MidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attentions = nn.ModuleList([AttentionBlock(channels)])
        self.resnets = nn.ModuleList([ResnetBlock(channels), ResnetBlock(channels)])

    def forward(self, hidden_states):
        
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):

            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)

        return hidden_states
    
class Encoder(nn.Module):
    def __init__(self, in_channels = 3, base_channels = 128, quant_channels = 32):
        super().__init__()

        # ////////////////////
        # Building the Encoder
        # ////////////////////

        # to comply with the checkpoint, we have to project different sizes
        # if the in and out channels are different we use the ResNetBlockProjection class
        # to make the computational possible, we need both in and out channels to be the same

        self.down_configs = [(base_channels, base_channels, True),              # block0: 128 -> 128
                            (base_channels, base_channels * 2, True),           # block1: 128 -> 256
                            (base_channels * 2, base_channels * 4, True),       # block2: 256 -> 512
                            (base_channels * 4, base_channels * 4, False)]      # block3: 512 -> 512

        # intial layer
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size = 3, padding = 1)
        self.conv_act = nn.SiLU()

        self.down_blocks = nn.ModuleList()
        channels = base_channels

        # downsampling layers for encoder_model.down_blocks
        for (in_ch, out_ch, use_conv) in self.down_configs:
            self.down_blocks.append(DownBlock(in_ch, out_ch, use_conv))
            channels = out_ch

        # expected key "encoder.mid_block"
        self.mid_block = MidBlock(channels)

        # normalization and conv layer
        self.conv_norm_out = nn.GroupNorm(num_groups = 32, num_channels = channels)
        self.conv_out = nn.Conv2d(channels, quant_channels, kernel_size = 3, padding = 1)

        self.output_channels = channels

    def forward(self, x: torch.Tensor):

        h = self.conv_in(x)
        
        for layer in self.down_blocks:
            h = layer(h)
        
        # apply mid block
        h = self.mid_block(h)
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        h = self.conv_out(h)

        return h

class Decoder(nn.Module):
    def __init__(self, output_channels = 3, latent_channels = 4, base_channels = 128, num_up = 2):
        super().__init__()

        # ////////////////////
        # Building the Decoder
        # ///////////////////

        # input
        self.conv_in = nn.Conv2d(latent_channels, base_channels * (2 ** num_up), kernel_size = 3, padding = 1)

        self.up_blocks = nn.ModuleList()
        channels = base_channels * (num_up ** 2)

        # middle
        self.mid_block = MidBlock(channels)
        
        up_config = [
            (512, 512, True),
            (512, 512, True),
            (512, 256, True),
            (256, 128, False)
        ]

        # upsampling layers for encoder_model.up_blocks
        for (in_ch, out_ch, use_conv) in up_config:
            self.up_blocks.append(UpBlock(in_ch, out_ch, use_conv))
            channels = out_ch

        # normalization and conv layer
        self.conv_norm_out = nn.GroupNorm(num_groups = 32, num_channels = channels)
        self.conv_out = nn.Conv2d(channels, output_channels, kernel_size = 3, padding = 1)

    def forward(self, x: torch.Tensor):

        h = self.conv_in(x)

        # apply mid block
        h = self.mid_block(h)

        for layer in self.up_blocks:
            h = layer(h)

        h = self.conv_norm_out(h)
        h = self.conv_out(h)

        return torch.tanh(h)
    
class VAE(nn.Module):
    # Create the variational autoencoder
    #                  RGB                 dim. of latent space  1st layers channels  no. of channels after bottleneck 
    def __init__(self, input_channels = 3, latent_channels = 16, base_channels = 128, quant_channels = 32):
        """
        Args:
            depth (int): Number of downsampling (and upsampling) blocks.
            latent_channels (int): dimensionality of the latent space
        """
        super(VAE, self).__init__()

        self.latent_channels = latent_channels
        self.shift_factor = 0.0609
        self.scaling_factor = 1.5305
        self.vae_scale_factor = 8

        self.encoder = Encoder(
            in_channels = input_channels, base_channels = base_channels,
            quant_channels = quant_channels
        )

        self.decoder = Decoder(
            latent_channels = latent_channels, base_channels = base_channels,
            output_channels = input_channels  
        )

        self.resizer = Resize(size = (384, 384))

    def encode(self, x):

        h = self.encoder(x)

        latent_dist = DiagonalGaussianDistribution(params = h, latent_channels = self.latent_channels)

        return latent_dist

    def decode(self, z):

        decoded =  self.decoder(z)
        decoded = self.resizer(decoded)

        return decoded
    

def load_vae(model: VAE, device: str = "cpu") -> VAE:

    # checkpoints could be installed automatically from encoders/get_checkpoints.py

    DEBUG = False

    import os
    path = os.path.join(os.getcwd(), os.path.join("encoders", "hub", "checkpoints", "vae.pth"))

    checkpoint = torch.load(path, map_location = device)
    missing, unexpected = model.load_state_dict(checkpoint, strict = not DEBUG)

    # for debuggging
    if DEBUG:
        print(f"Missing keys ({len(missing)}):", missing)
        print(f"\nUnexpected keys ({len(unexpected)}):", unexpected)

    model.eval()

    return model

def test_vae():

    from torchvision.transforms import ToTensor, Lambda
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    import os

    vae = VAE()
    vae = load_vae(model = vae)

    image_path = os.path.join(os.getcwd(), "assets", "cat.webp")
    image = Image.open(image_path).convert("RGB")

    image = Resize(size=(384, 384))(image)
    image = ToTensor()(image).unsqueeze(0)
    image = Lambda(lambda t: t * 2 - 1)(image)

    with torch.no_grad():
        latent_dist = vae.encode(image)
        print("Logvar stats: min={}, max={}".format(latent_dist.logvar.min().item(), latent_dist.logvar.max().item()))

        latent = latent_dist.sample() 
        print(latent.size())

        recon = vae.decode(latent)

    print(recon.size())

    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image.squeeze(0).permute(1, 2, 0).numpy() / 2 + 0.5)

    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon.squeeze(0).permute(1, 2, 0).numpy() / 2 + 0.5)
    axes[1].set_title("Reconstructed")

    axes[1].axis("off")

    plt.show()

if __name__ == '__main__':
    test_vae()