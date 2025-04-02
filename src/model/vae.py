import torch
import torch.nn as nn

class VAE(nn.Module):
    # Create the variational autoencoder
    def __init__(self, input_channels = 3, latent_dim = 4, depth = 5, latent_size = 16, output_size = 30):
        """
        Args:
            input_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            latent_dim (int): Dimensionality of the latent space.
            depth (int): Number of downsampling (and upsampling) blocks.
        """
        # the goal of the code is to get the encoder and the decoder models with some helpful layers
        super(VAE, self).__init__()
        self.latent_size = latent_size

        # ////////////////////
        # Building the Encoder
        # ////////////////////

        encoder_layers = []
        in_channels = input_channels

        # Create a sample of conv blocks each downsampling by 2
        for i in range(depth):
            output_channels = 32 if i == 0 else in_channels * 2
            encoder_layers.append(
                nn.Conv2d(in_channels, output_channels, kernel_size = 3, stride = 2, padding = 1)
            )
            in_channels = output_channels

        # turn the layers into a model
        self.encoder_model = nn.Sequential(*encoder_layers)

        # conv layers for varianece and mean
        # use conv to keep the spaial dimensions
        self.fc_mu  = nn.Conv2d(in_channels, latent_dim, kernel_size = 3, stride = 1, padding = 1) # mean
        self.fc_var = nn.Conv2d(in_channels, latent_dim, kernel_size = 3, stride = 1, padding = 1) # variance

        # adaptive average pooling to allow any input size for the images
        self.avg_pool = nn.AdaptiveAvgPool2d((latent_size, latent_size)) # ensure fixed size before passing

        # later use for reshaping in decoding
        self.final_channels = in_channels

        # layer for decoding the latents
        self.fc_decode = nn.Linear(4 * latent_size * latent_size, self.final_channels * latent_size * latent_size)

        # ////////////////////
        # Building the Decoder
        # ///////////////////

        # compute intermediate sizes
        sizes = [
            round(latent_size + (output_size - latent_size) * (i + 1) / depth)
            for i in range(depth)
        ]
        
        decoder_layers = []
        # latent dimension
        in_channels = self.final_channels
        # Create sample of up block each decreasing the depth by 2

        for size in sizes:

            # upsample to target size and apply conv
            decoder_layers.append(nn.Upsample(size = size, mode = "bilinear", align_corners = False))

            # make sure in_channels don't go below input_channels
            output_channels = max(input_channels, in_channels // 2)

            decoder_layers.append(
                # stride = 2 for increasing the dimensions of the image
                nn.ConvTranspose2d(in_channels, output_channels, kernel_size = 3, padding = 1)
            )
            # add batch norm and relu to better the vae decoder
            decoder_layers.append(nn.BatchNorm2d(output_channels))
            decoder_layers.append(nn.ReLU(inplace = True))

            in_channels = output_channels

        # last layer to get the image to the original number of channels
        decoder_layers.append(nn.Conv2d(in_channels, input_channels, kernel_size = 3))

        # we will add a sigmoid layer to ensure our output is between 0-1
        # without the sigmoid, the model will have to learn that the output is between 0 and 1
        
        decoder_layers.append(nn.Sigmoid())

        # get the decoder model
        self.decoder_model = nn.Sequential(*decoder_layers)

    def reparam_trick(self, mean: torch.Tensor, logvar: torch.Tensor):
        " Applied the reparamterization trick "
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mean + eps * std

    def encode(self, image: torch.Tensor):

        """
        Encodes the input image into a latent vector.
        Input Image = (batch, channels, height, width).
        """

        # function to encode image
        x = self.encoder_model(image.float())
        x = self.avg_pool(x) # get fixed size

        mean_u = self.fc_mu(x)
        logvar = self.fc_var(x)

        latent = self.reparam_trick(mean = mean_u, logvar = logvar)

        return mean_u, logvar, latent

    def decode(self, latent: torch.Tensor):
        " decodes a latent "

        # avoid flattening batch size
        x = torch.flatten(latent, start_dim = 1)

        x = self.fc_decode(x)

        batch_size = latent.size(0)

        x = x.reshape(batch_size, self.final_channels, self.latent_size, self.latent_size)

        image = self.decoder_model(x)

        return image
    

def load_vae(model: VAE, path: str, device: str = "cpu") -> VAE:
    # load checkpoint into the model

    checkpoint = torch.load(path, map_location = device)
    model.load_state_dict(checkpoint)

    model.eval()

    return model

def test_vae():

    vae = VAE()

    image = torch.randn(1, 3, 224, 224)

    _, _, latent = vae.encode(image)

    print(latent.size())
    image = vae.decode(latent = latent)

    print(image.size())