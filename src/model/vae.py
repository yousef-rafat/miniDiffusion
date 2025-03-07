import torch
import torch.nn as nn

class VAE(nn.Module):
    # Create the variational autoencoder
    def __init__(self, input_channels=3, latent_dim = 128, depth = 4, size = 28):
        """
        Args:
            input_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            latent_dim (int): Dimensionality of the latent space.
            depth (int): Number of downsampling (and upsampling) blocks.
        """
        # the goal of the code is to get the encoder and the decoder models with some helpful layers
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.depth = depth
        self.size = size

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

        # linear layers for varianece and mean
        self.fc_mu  = nn.Linear(in_channels * 4 * 4, latent_dim) # mean
        self.fc_var = nn.Linear(in_channels * 4 * 4, latent_dim) # variance

        # adaptive average pooling to allow any input size for the images
        self.avg_pool = nn.AdaptiveAvgPool2d((size, size)) # ensure fixed size before passing

        # later use for reshaping in decoding
        self.final_channels = in_channels

        # layer for decoding the latents
        self.fc_decode = nn.Linear(latent_dim, in_channels * 4 * 4)

        # ////////////////////
        # Building the Decoder
        # ///////////////////
        
        decoder_layers = []

        # Create sample of up block each decreasing the depth by 2

        for i in range(depth):
            # reverse the channel doubling that happened in encoding
            output_channels = in_channels // 2
            decoder_layers.append(
                nn.ConvTranspose2d(in_channels, output_channels)
            )
            # add batch norm and relu to better the vae decoder
            decoder_layers.append(nn.BatchNorm2d(output_channels))
            decoder_layers.append(nn.ReLU(inplace = True))

            in_channels = output_channels

        # last layer to get the image to the original number of channels
        decoder_layers.append(nn.Conv2d(in_channels, input_channels))

        # we will add a sigmoid layer to ensure our output is between 0-1
        # without the sigmoid, the model will have to learn that the output is between 0 and 1
        
        decoder_layers.append(nn.Sigmoid())

        # get the decoder model
        self.decoder_model = nn.Sequential(*decoder_layers)

    def reparam_trick(self, mean: torch.Tensor, logvar: torch.Tensor):
        " Applied the reparamterization trick "
        std = torch.exp(logvar * 0.5)
        eps = torch.rand_like(std)

        return mean + eps * std

    def encode(self, image: torch.Tensor):

        """
        Encodes the input image into a latent vector.
        Input Image = (batch, channels, height, width).
        """

        # function to encode image
        x = self.encoder_model(image)
        x = self.avg_pool(x) # get fixed size
        x = nn.Flatten(x)

        mean_u = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mean_u, logvar

    def decode(self, latent: torch.Tensor):
        " decodes a latent "
        x = self.fc_decode(latent)
        batch_size = latent.size(0)

        x = x.reshape(batch_size, self.final_channels, self.size, self.size)

        image = self.decoder_model(x)

        return image
    

def load_vae(model: VAE, path: str, device: str = "cpu") -> VAE:
    # load checkpoint into the model

    checkpoint = torch.load(path, map_location = device)
    model = model.load_state_dict(checkpoint)

    model.eval()

    return model